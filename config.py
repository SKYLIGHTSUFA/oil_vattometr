"""Единая конфигурация проекта: пути и параметры алгоритмов.

Все модули берут значения отсюда, чтобы в коде не оставалось
захардкоженных путей вида ``C:\\Users\\User\\...``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES_DIR = ROOT / "files"
CLASSIFICATOR_DIR = ROOT / "classificator"
DATASET_DIR = CLASSIFICATOR_DIR / "dataset"
PLOTS_DIR = ROOT / "plots"

#: Исходная выгрузка активной мощности по скважинам.
SOURCE_XLSX = FILES_DIR / "Выявление обрыва ремней СК для СППР.xlsx"
#: Обученная модель классификатора (good / bad).
DEFAULT_WEIGHTS = FILES_DIR / "best.pt"

COL_FIELD = "Мест-е"
COL_WELL = "Скв."
COL_DATE = "Дата"
COL_VALUE = "Значение"

CLASS_NAMES = ("bad", "good")


@dataclass(frozen=True)
class WindowConfig:
    """Параметры нарезки временного ряда на окна.

    Раньше ряд резался на куски по 12 отсчётов. Период опроса счётчиков
    гуляет от 15 с до 45 мин, поэтому одно и то же «окно из 12 точек»
    покрывало от 3 минут до 9 часов. Теперь окно задаётся во времени.
    """

    #: Длительность окна.
    window: str = "60min"
    #: Шаг между соседними окнами (``None`` — без перекрытия, шаг = window).
    step: str | None = None
    #: Шаг равномерной сетки, на которую ресемплится ряд внутри окна.
    resample: str = "5min"
    #: Сколько точек рисуется в окне (после ресемплинга).
    points: int = 12
    #: Минимальная доля реальных (не интерполированных) отсчётов в окне.
    min_coverage: float = 0.6


@dataclass(frozen=True)
class SignalConfig:
    """Параметры очистки сигнала и оценки холостого хода."""

    #: Стратегия для отрицательных значений: keep | clip | abs | drop.
    negatives: str = "clip"
    #: Порог Хампеля в MAD: выброс, если |x - median| > k * MAD.
    hampel_k: float = 4.0
    #: Ширина окна фильтра Хампеля в отсчётах.
    hampel_window: int = 7
    #: Нижняя граница масштаба выбросов, кВт (разрешение счётчика).
    hampel_abs_floor: float = 0.05
    #: Нижняя граница масштаба как доля общего разброса ряда.
    hampel_rel_floor: float = 0.25
    #: Запасной квантиль для холостого хода, если Оцу не разделил выборку.
    idle_quantile: float = 0.05
    #: Минимум точек по скважине, чтобы оценка холостого хода имела смысл.
    min_samples_for_idle: int = 60


@dataclass(frozen=True)
class TrainConfig:
    """Параметры обучения YOLO-классификатора."""

    model: str = "yolov8n-cls.pt"
    epochs: int = 60
    imgsz: int = 128
    batch: int = 32
    patience: int = 15
    seed: int = 0
    project: Path = field(default=CLASSIFICATOR_DIR / "runs")
    name: str = "belt-cls"
