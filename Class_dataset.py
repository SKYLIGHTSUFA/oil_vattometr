"""Подготовка датасета: выгрузка Excel -> временные окна -> картинки.

Заменяет ноутбук ``plotes.ipynb``: тот же конвейер, но нарезка по времени,
нормировка по холостому ходу своей скважины и переиспользуемая фигура
matplotlib вместо создания новой на каждый график.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib

matplotlib.use("Agg")  # без GUI: скрипт часто гоняется на сервере

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg_mod
from algorithms import IdleLevel, Window, clean_values, estimate_idle_power, iter_time_windows, normalize
from config import COL_DATE, COL_FIELD, COL_VALUE, COL_WELL, SignalConfig, WindowConfig

log = logging.getLogger(__name__)


@dataclass
class WellSeries:
    """Очищенный ряд одной скважины вместе с оценкой её холостого хода."""

    well: str
    field: str
    series: pd.Series
    level: IdleLevel


class PowerDataset:
    """Читает выгрузку активной мощности и строит из неё датасет картинок."""

    def __init__(
        self,
        source: Path | str = cfg_mod.SOURCE_XLSX,
        signal: SignalConfig | None = None,
        window: WindowConfig | None = None,
    ) -> None:
        self.source = Path(source)
        self.signal = signal or SignalConfig()
        self.window = window or WindowConfig()
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Загрузка
    # ------------------------------------------------------------------ #
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self.load()
        return self._df

    def load(self) -> pd.DataFrame:
        """Читает Excel и приводит колонки к рабочему виду.

        В исходном файле заголовок называется ``'Дата '`` — с висящим
        пробелом, поэтому имена колонок чистятся, а не берутся как есть.
        """
        if not self.source.exists():
            raise FileNotFoundError(f"Не найдена выгрузка: {self.source}")

        df = pd.read_excel(self.source)
        df.columns = [str(c).strip() for c in df.columns]

        missing = {COL_WELL, COL_DATE, COL_VALUE} - set(df.columns)
        if missing:
            raise ValueError(f"В выгрузке нет колонок: {sorted(missing)}")

        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
        df[COL_VALUE] = pd.to_numeric(df[COL_VALUE], errors="coerce")
        df[COL_WELL] = df[COL_WELL].astype(str).str.strip()
        if COL_FIELD in df.columns:
            df[COL_FIELD] = df[COL_FIELD].astype(str).str.strip()

        before = len(df)
        df = df.dropna(subset=[COL_DATE, COL_VALUE]).sort_values(COL_DATE)
        if before != len(df):
            log.info("Отброшено строк с пустой датой или значением: %d", before - len(df))
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Скважины и холостой ход
    # ------------------------------------------------------------------ #
    def wells(self, only: Iterable[str] | None = None) -> Iterator[WellSeries]:
        """Итерируется по скважинам, отдавая очищенный ряд и уровень холостого хода."""
        wanted = {str(w) for w in only} if only else None
        for well, group in self.df.groupby(COL_WELL, sort=True):
            if wanted and well not in wanted:
                continue

            values = clean_values(group[COL_VALUE].to_numpy(), self.signal)
            series = pd.Series(values, index=pd.DatetimeIndex(group[COL_DATE]), name=well)
            series = series.dropna()
            if series.empty:
                log.warning("Скважина %s: после очистки не осталось данных", well)
                continue

            field = str(group[COL_FIELD].iloc[0]) if COL_FIELD in group else ""
            yield WellSeries(well, field, series, estimate_idle_power(series.to_numpy(), self.signal))

    def idle_report(self, only: Iterable[str] | None = None) -> pd.DataFrame:
        """Таблица холостого хода по каждому двигателю — задача №1 из README."""
        rows = []
        for w in self.wells(only):
            rows.append(
                {
                    "Мест-е": w.field,
                    "Скв.": w.well,
                    "холостой_ход_кВт": round(w.level.idle, 2),
                    "амплитуда_кВт": round(w.level.scale, 2),
                    "порог_кВт": round(w.level.threshold, 2),
                    "отсчётов": w.level.n_samples,
                    "метод": w.level.method,
                    "надёжно": w.level.reliable,
                }
            )
        return pd.DataFrame(rows).sort_values("холостой_ход_кВт", ascending=False, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Отрисовка
    # ------------------------------------------------------------------ #
    def render(
        self,
        out_dir: Path | str = cfg_mod.PLOTS_DIR,
        only: Iterable[str] | None = None,
        limit_per_well: int | None = None,
        image_size: tuple[int, int] = (640, 480),
        dpi: int = 100,
        markers: bool = False,
    ) -> int:
        """Строит графики окон и раскладывает их по папкам ``Скв_<номер>``.

        Размер картинки одинаков для всех окон. В текущем датасете он
        гуляет (640x480, 660x499, 432x288), причём почти все 660x499 —
        это класс ``good``: сеть может выучить размер холста вместо формы
        сигнала. Одинаковый холст убирает эту утечку.

        Фигура создаётся один раз и переиспользуется: старый код на каждой
        итерации рисовал поверх глобального состояния pyplot и накапливал
        память.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(image_size[0] / dpi, image_size[1] / dpi), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()

        total = 0
        try:
            for well in self.wells(only):
                well_dir = out_dir / f"Скв_{well.well}"
                well_dir.mkdir(parents=True, exist_ok=True)
                written = 0

                for window in iter_time_windows(well.series, self.window):
                    if limit_per_well is not None and written >= limit_per_well:
                        break
                    name = f"{well.well}_{window.start:%Y-%m-%d_%H.%M}.png"
                    self._draw(ax, window, well.level, markers)
                    fig.savefig(well_dir / name, dpi=dpi, pad_inches=0)
                    written += 1

                total += written
                log.info(
                    "Скв. %s: окон %d, холостой ход %.2f кВт (%s)",
                    well.well, written, well.level.idle, well.level.method,
                )
        finally:
            plt.close(fig)

        return total

    @staticmethod
    def _draw(ax: plt.Axes, window: Window, level: IdleLevel, markers: bool = False) -> None:
        """Рисует одно окно. Ось Y общая для всех графиков всех скважин."""
        y = normalize(window.values, level)
        x = np.arange(y.size)

        ax.clear()
        ax.set_axis_off()
        # Границы фиксированы: 0 — холостой ход, 1 — типовая нагрузка.
        # Раньше каждое окно тянулось min-max по себе, и плоская «полка»
        # холостого хода на картинке выглядела как нормальная работа.
        ax.set_xlim(-0.5, y.size - 0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.plot(x, y, linewidth=1.5, color="#1f77b4")
        if markers:
            ax.plot(x, y, "o", markersize=3, color="#d62728")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = PowerDataset()
    report = dataset.idle_report()
    print(report.to_string(index=False))
    print(f"\nВсего построено графиков: {dataset.render()}")


if __name__ == "__main__":
    main()
