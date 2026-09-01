"""Алгоритмы обработки активной мощности станков-качалок.

Здесь решаются три задачи, которые стояли в README:

1. Холостой ход у каждого двигателя свой — оцениваем его отдельно
   по скважине (:func:`estimate_idle_power`) и нормируем сигнал
   относительно него (:func:`normalize`), чтобы графики разных
   скважин были сопоставимы.
2. Отрицательные значения — обрабатываются явной стратегией плюс
   фильтром Хампеля, который убирает одиночные выбросы счётчика
   (:func:`clean_values`).
3. Нарезка не по n отсчётам, а по времени — :func:`iter_time_windows`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from config import SignalConfig, WindowConfig

_EPS = 1e-9


# --------------------------------------------------------------------------- #
# 2. Очистка сигнала: отрицательные значения и выбросы
# --------------------------------------------------------------------------- #
def hampel_mask(
    values: np.ndarray,
    window: int = 7,
    k: float = 4.0,
    abs_floor: float = 0.05,
    rel_floor: float = 0.25,
) -> np.ndarray:
    """Маска выбросов по фильтру Хампеля (скользящие медиана и MAD).

    В отличие от отсечения по среднему и СКО, медиана и MAD сами не
    «уезжают» от выброса, поэтому одиночный провал в -32 кВт не тянет
    за собой порог и не маскирует соседние нормальные отсчёты.

    У чистого MAD есть известная беда: на ровном участке он равен нулю,
    и порог вырождается — выброс посреди «полки» холостого хода не
    находится вовсе. Поэтому масштаб снизу ограничен: разрешением
    счётчика (``abs_floor``, кВт) и долей общего разброса по ряду
    (``rel_floor``). Без второго слагаемого скользящая медиана срезала бы
    вершины нормальных колебаний как выбросы.
    """
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    s = pd.Series(values, dtype="float64")
    win = max(3, int(window) | 1)  # нечётное окно, чтобы медиана была центрирована

    # 1.4826 — приведение MAD к СКО для нормального распределения.
    global_sigma = 1.4826 * float((s - s.median()).abs().median())
    floor = max(abs_floor, rel_floor * global_sigma)

    med = s.rolling(win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(win, center=True, min_periods=1).median()
    sigma = np.maximum(1.4826 * mad.to_numpy(), floor)

    deviation = (s - med).abs().to_numpy()
    with np.errstate(invalid="ignore"):
        mask = deviation > k * sigma
    return np.nan_to_num(mask, nan=False).astype(bool)


def clean_values(values: np.ndarray, cfg: SignalConfig | None = None) -> np.ndarray:
    """Убирает выбросы и приводит отрицательные значения к выбранной стратегии.

    Стратегии (``cfg.negatives``):

    * ``keep``  — оставить как есть (мощность может быть отрицательной,
      когда на ходе вниз колонна штанг раскручивает двигатель);
    * ``clip``  — обрезать по нулю (по умолчанию: для задачи «оборван
      ремень / не оборван» важен уровень нагрузки, а не рекуперация);
    * ``abs``   — взять модуль, если у счётчика перепутан знак;
    * ``drop``  — заменить на NaN, окно потом отбракуется по покрытию.

    Выбросы (по Хампелю) всегда заменяются на NaN и восстанавливаются
    интерполяцией уже на равномерной сетке.
    """
    cfg = cfg or SignalConfig()
    out = np.asarray(values, dtype="float64").copy()

    outliers = hampel_mask(out, cfg.hampel_window, cfg.hampel_k, cfg.hampel_abs_floor, cfg.hampel_rel_floor)
    out[outliers] = np.nan

    strategy = cfg.negatives
    negative = out < 0
    if strategy == "clip":
        out[negative] = 0.0
    elif strategy == "abs":
        out[negative] = np.abs(out[negative])
    elif strategy == "drop":
        out[negative] = np.nan
    elif strategy != "keep":
        raise ValueError(f"Неизвестная стратегия для отрицательных значений: {strategy!r}")
    return out


# --------------------------------------------------------------------------- #
# 1. Холостой ход отдельно по каждому двигателю
# --------------------------------------------------------------------------- #
def _otsu_threshold(values: np.ndarray, bins: int = 128) -> float | None:
    """Порог Оцу — делит гистограмму на «холостой ход» и «работу под нагрузкой».

    Возвращает ``None``, если разделение вырожденное (весь сигнал в одном
    режиме) — тогда вызывающий код падает обратно на квантиль.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2 or np.ptp(finite) < _EPS:
        return None

    hist, edges = np.histogram(finite, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight = hist.astype("float64") / finite.size

    w0 = np.cumsum(weight)[:-1]
    w1 = 1.0 - w0
    valid = (w0 > _EPS) & (w1 > _EPS)
    if not valid.any():
        return None

    mean_total = float(np.sum(weight * centers))
    mean0 = np.cumsum(weight * centers)[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = mean0 / w0
        mu1 = (mean_total - mean0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.where(valid, between, -np.inf)
    best = int(np.argmax(between))
    if not np.isfinite(between[best]):
        return None
    return float(edges[best + 1])


@dataclass(frozen=True)
class IdleLevel:
    """Оценка режима одной скважины."""

    idle: float          #: уровень холостого хода, кВт
    scale: float         #: типовая амплитуда рабочей нагрузки над холостым ходом
    threshold: float     #: порог разделения режимов
    n_samples: int       #: сколько отсчётов участвовало в оценке
    method: str          #: "otsu" или "quantile"

    @property
    def reliable(self) -> bool:
        return self.method == "otsu" and self.scale > _EPS


def estimate_idle_power(values: np.ndarray, cfg: SignalConfig | None = None) -> IdleLevel:
    """Оценивает мощность холостого хода конкретного двигателя.

    Гистограмма мощности по скважине бимодальна: нижняя мода — холостой
    ход (двигатель крутится, ремень оборван либо станок разгружен),
    верхняя — работа под нагрузкой. Порог Оцу разделяет эти моды без
    предположений о форме распределения, и медиана нижнего кластера даёт
    устойчивый уровень холостого хода.
    """
    cfg = cfg or SignalConfig()
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return IdleLevel(0.0, 0.0, 0.0, 0, "quantile")

    threshold = _otsu_threshold(finite) if finite.size >= cfg.min_samples_for_idle else None
    if threshold is not None:
        low = finite[finite <= threshold]
        high = finite[finite > threshold]
        if low.size and high.size:
            idle = float(np.median(low))
            scale = float(np.quantile(high, 0.95) - idle)
            if scale > _EPS:
                return IdleLevel(idle, scale, float(threshold), finite.size, "otsu")

    # Запасной путь: нижний квантиль как холостой ход, межквантильный размах как масштаб.
    idle = float(np.quantile(finite, cfg.idle_quantile))
    scale = float(np.quantile(finite, 0.95) - idle)
    return IdleLevel(idle, max(scale, 0.0), idle, finite.size, "quantile")


def normalize(values: np.ndarray, level: IdleLevel) -> np.ndarray:
    """Нормирует окно относительно холостого хода своей скважины.

    0.0 — двигатель на холостом ходу, 1.0 — типовая рабочая нагрузка.

    Старый вариант делал min-max внутри окна и терял абсолютную амплитуду:
    ровная «полка» холостого хода растягивалась на всю высоту графика и
    выглядела как нормальная работа, а деление на ``max - min`` падало с
    ZeroDivisionError на константном окне. Здесь масштаб общий для
    скважины, поэтому холостой ход остаётся плоской линией внизу.
    """
    arr = np.asarray(values, dtype="float64")
    if level.scale <= _EPS:
        return np.full(arr.shape, 0.5, dtype="float64")
    return np.clip((arr - level.idle) / level.scale, -0.05, 1.05)


# --------------------------------------------------------------------------- #
# 3. Нарезка по времени вместо нарезки по n отсчётам
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Window:
    """Одно временное окно, готовое к отрисовке."""

    start: pd.Timestamp
    end: pd.Timestamp
    values: np.ndarray      #: интерполированные значения, cfg.points штук
    coverage: float         #: доля реальных отсчётов в окне


def to_regular_grid(series: pd.Series, cfg: WindowConfig) -> tuple[pd.Series, pd.Series]:
    """Ресемплит ряд на равномерную сетку.

    Возвращает пару (значения, признак реального отсчёта). Пропуски
    заполняются линейной интерполяцией, но помечаются как «не реальные»,
    чтобы окно с редкой телеметрией можно было отбраковать.
    """
    series = series[~series.index.duplicated(keep="last")].sort_index()
    binned = series.resample(cfg.resample).mean()
    observed = binned.notna()
    filled = binned.interpolate(method="time", limit_direction="both")
    return filled, observed


def iter_time_windows(series: pd.Series, cfg: WindowConfig | None = None) -> Iterator[Window]:
    """Режет ряд на окна фиксированной длительности с заданным шагом.

    Ключевое отличие от старого ``v[i * 12:(i + 1) * 12]``: длина окна
    задана во времени, поэтому окно всегда покрывает один и тот же
    физический интервал независимо от периода опроса счётчика, а разрывы
    телеметрии не «склеиваются» в один график.
    """
    cfg = cfg or WindowConfig()
    if series.empty:
        return

    grid, observed = to_regular_grid(series, cfg)
    if grid.empty:
        return

    step_freq = cfg.step or cfg.window
    bin_delta = pd.Timedelta(cfg.resample)
    size = max(1, int(pd.Timedelta(cfg.window) / bin_delta))
    stride = max(1, int(pd.Timedelta(step_freq) / bin_delta))
    if grid.size < size:
        return

    values = grid.to_numpy(dtype="float64")
    flags = observed.to_numpy(dtype=bool)
    index = grid.index
    target_x = np.linspace(0.0, size - 1, cfg.points)
    source_x = np.arange(size, dtype="float64")

    for start in range(0, values.size - size + 1, stride):
        chunk = values[start:start + size]
        if not np.isfinite(chunk).all():
            continue
        coverage = float(flags[start:start + size].mean())
        if coverage < cfg.min_coverage:
            continue
        points = chunk if size == cfg.points else np.interp(target_x, source_x, chunk)
        yield Window(
            start=index[start],
            end=index[start + size - 1],
            values=points,
            coverage=coverage,
        )
