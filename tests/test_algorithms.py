"""Тесты алгоритмов обработки сигнала.

Запуск: ``python tests/test_algorithms.py`` или ``pytest tests``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms import (
    clean_values,
    estimate_idle_power,
    hampel_mask,
    iter_time_windows,
    normalize,
)
from config import SignalConfig, WindowConfig


def test_hampel_finds_single_spike() -> None:
    values = np.full(40, 5.0)
    values[17] = -32.0
    mask = hampel_mask(values, window=7, k=4.0)
    assert mask[17]
    assert mask.sum() == 1


def test_hampel_keeps_oscillation_peaks() -> None:
    """Фильтр не должен принимать вершины нормальных колебаний за выбросы."""
    rng = np.random.default_rng(4)
    signal = 5.0 + 3.0 * np.sin(np.arange(200) / 4.0) + rng.normal(0, 0.05, 200)
    assert hampel_mask(signal).sum() == 0

    with_spike = signal.copy()
    with_spike[100] = -32.0
    assert np.flatnonzero(hampel_mask(with_spike)).tolist() == [100]


def test_clean_values_negative_strategies() -> None:
    values = np.array([2.0, -1.0, 2.0, -3.0, 2.0])
    assert clean_values(values, SignalConfig(negatives="clip"))[1] == 0.0
    assert clean_values(values, SignalConfig(negatives="abs"))[3] == 3.0
    assert np.isnan(clean_values(values, SignalConfig(negatives="drop"))[1])
    assert clean_values(values, SignalConfig(negatives="keep"))[1] == -1.0


def test_idle_power_finds_lower_mode() -> None:
    rng = np.random.default_rng(0)
    idle = rng.normal(2.0, 0.1, 400)      # холостой ход
    load = rng.normal(12.0, 1.0, 600)     # работа под нагрузкой
    level = estimate_idle_power(np.concatenate([idle, load]))
    assert level.method == "otsu"
    assert abs(level.idle - 2.0) < 0.3
    assert 2.0 < level.threshold < 12.0


def test_idle_power_differs_between_motors() -> None:
    """Задача №1: у каждого двигателя свой холостой ход."""
    rng = np.random.default_rng(1)
    a = np.concatenate([rng.normal(1.0, 0.1, 300), rng.normal(9.0, 0.8, 300)])
    b = np.concatenate([rng.normal(15.0, 0.2, 300), rng.normal(24.0, 0.8, 300)])
    assert estimate_idle_power(b).idle - estimate_idle_power(a).idle > 10


def test_normalize_is_comparable_across_wells() -> None:
    """Один и тот же режим у разных двигателей даёт одинаковую картинку."""
    rng = np.random.default_rng(2)
    a = np.concatenate([rng.normal(1.0, 0.05, 300), rng.normal(9.0, 0.5, 300)])
    b = a + 14.0
    la, lb = estimate_idle_power(a), estimate_idle_power(b)
    assert abs(float(np.mean(normalize(a[:50], la))) - float(np.mean(normalize(b[:50], lb)))) < 0.1


def test_normalize_keeps_idle_flat() -> None:
    """Плоский холостой ход остаётся внизу, а не растягивается на всю картинку."""
    rng = np.random.default_rng(3)
    level = estimate_idle_power(np.concatenate([rng.normal(2.0, 0.05, 300), rng.normal(10.0, 0.5, 300)]))
    flat = normalize(np.full(12, 2.0), level)
    assert np.allclose(flat, 0.0, atol=0.05)


def test_normalize_survives_constant_window() -> None:
    """Старый min-max падал с делением на ноль на константном окне."""
    level = estimate_idle_power(np.full(200, 3.0))
    assert np.all(np.isfinite(normalize(np.full(12, 3.0), level)))


def _series(freq: str, hours: int) -> pd.Series:
    index = pd.date_range("2021-10-12", periods=int(hours * 3600 / pd.Timedelta(freq).total_seconds()), freq=freq)
    return pd.Series(np.sin(np.arange(index.size) / 3.0) + 5.0, index=index)


def test_window_duration_is_independent_of_sampling_rate() -> None:
    """Задача №3: окно задано временем, а не числом отсчётов."""
    cfg = WindowConfig(window="60min", resample="5min", points=12, min_coverage=0.0)
    for freq in ("30s", "5min", "20min"):
        windows = list(iter_time_windows(_series(freq, 6), cfg))
        assert windows, freq
        assert all(w.values.size == 12 for w in windows)
        for w in windows:
            assert w.end - w.start == pd.Timedelta("55min")


def test_gaps_are_dropped_by_coverage() -> None:
    cfg = WindowConfig(window="60min", resample="5min", points=12, min_coverage=0.9)
    series = _series("5min", 6)
    gapped = pd.concat([series.iloc[:12], series.iloc[60:]])  # дыра в 4 часа
    starts = [w.start for w in iter_time_windows(gapped, cfg)]
    assert pd.Timestamp("2021-10-12 01:00") not in starts


def test_empty_series_yields_nothing() -> None:
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))
    assert list(iter_time_windows(empty)) == []


if __name__ == "__main__":
    failed = 0
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            try:
                func()
                print(f"ok   {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{'провалено: ' + str(failed) if failed else 'все тесты прошли'}")
    sys.exit(1 if failed else 0)
