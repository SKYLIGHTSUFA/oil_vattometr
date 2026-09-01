"""Инференс классификатора «оборван ремень / норма».

Обёртка над YOLOv8-cls: батчевый предикт, отсечка по уверенности и
подсчёт метрик на размеченной папке.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import config as cfg_mod
from config import CLASS_NAMES

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Prediction:
    """Результат по одной картинке."""

    path: Path
    label: str
    confidence: float
    probs: tuple[float, ...]

    @property
    def is_broken(self) -> bool:
        return self.label == "bad"


class BeltPredictor:
    """Классификатор обрыва ремня станка-качалки."""

    def __init__(self, weights: Path | str = cfg_mod.DEFAULT_WEIGHTS, min_confidence: float = 0.0) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"Не найдены веса модели: {self.weights}")
        self.min_confidence = min_confidence

        from ultralytics import YOLO  # импорт здесь: тяжёлый и нужен не всем командам

        self.model = YOLO(str(self.weights))
        names = self.model.names
        self.names = tuple(names[i] for i in sorted(names)) if isinstance(names, dict) else tuple(names)
        if set(self.names) != set(CLASS_NAMES):
            log.warning("Классы модели %s не совпадают с ожидаемыми %s", self.names, CLASS_NAMES)

    # ------------------------------------------------------------------ #
    @staticmethod
    def collect_images(source: Path | str) -> list[Path]:
        """Собирает картинки из файла или рекурсивно из папки."""
        source = Path(source)
        if source.is_file():
            return [source]
        if not source.is_dir():
            raise FileNotFoundError(f"Нет такого пути: {source}")
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)

    def predict(self, source: Path | str, batch_size: int = 32) -> list[Prediction]:
        """Предсказание по файлу или папке.

        Картинки идут пачками, а не по одной: старый скрипт вызывал
        ``model.predict`` в цикле на каждый файл, из-за чего почти всё
        время уходило на накладные расходы, а не на сеть.
        """
        images = self.collect_images(source)
        if not images:
            log.warning("В %s не найдено картинок", source)
            return []

        out: list[Prediction] = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            results = self.model.predict(source=[str(p) for p in batch], verbose=False)
            for path, result in zip(batch, results):
                probs = result.probs
                # top1 / top1conf вместо ручного conf.index(max(conf)) — тот
                # вариант молча ломался при повторе максимума в списке.
                idx = int(probs.top1)
                out.append(
                    Prediction(
                        path=path,
                        label=self.names[idx],
                        confidence=float(probs.top1conf),
                        probs=tuple(float(v) for v in probs.data.tolist()),
                    )
                )
        return out

    def broken_belts(self, source: Path | str, batch_size: int = 32) -> list[Prediction]:
        """Только уверенные срабатывания по классу ``bad``."""
        return [p for p in self.predict(source, batch_size) if p.is_broken and p.confidence >= self.min_confidence]

    # ------------------------------------------------------------------ #
    def evaluate(self, dataset_dir: Path | str, batch_size: int = 32) -> dict[str, object]:
        """Считает метрики на папке вида ``<split>/<класс>/*.png``.

        Классы несбалансированы (good примерно в пять раз больше, чем bad),
        поэтому одной accuracy мало — возвращаем precision/recall/F1 по bad
        и матрицу ошибок.
        """
        dataset_dir = Path(dataset_dir)
        class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir()) if dataset_dir.is_dir() else []
        if not class_dirs:
            raise FileNotFoundError(f"Нет папок классов в {dataset_dir}")

        y_true: list[str] = []
        y_pred: list[str] = []
        for class_dir in class_dirs:
            preds = self.predict(class_dir, batch_size)
            y_true.extend([class_dir.name] * len(preds))
            y_pred.extend(p.label for p in preds)

        if not y_true:
            raise ValueError(f"В {dataset_dir} не нашлось картинок")

        labels = sorted(self.names)
        matrix = confusion_matrix(y_true, y_pred, labels)
        report = {
            "n": len(y_true),
            "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
            "labels": labels,
            "confusion": matrix,
        }
        report.update({f"{name}_{k}": v for name in labels for k, v in _prf(y_true, y_pred, name).items()})
        return report


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Матрица ошибок: строки — истина, столбцы — предсказание."""
    position = {name: i for i, name in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in position and p in position:
            matrix[position[t], position[p]] += 1
    return matrix


def _prf(y_true: Sequence[str], y_pred: Sequence[str], positive: str) -> dict[str, float]:
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def format_report(report: dict[str, object], names: Iterable[str]) -> str:
    """Человекочитаемый вывод метрик."""
    lines = [f"Картинок: {report['n']}, accuracy: {report['accuracy']:.3f}", "", "Матрица ошибок (строки — истина):"]
    labels = list(report["labels"])  # type: ignore[arg-type]
    matrix = report["confusion"]  # type: ignore[index]
    lines.append("            " + "".join(f"{name:>10}" for name in labels))
    for i, name in enumerate(labels):
        lines.append(f"{name:>12}" + "".join(f"{v:>10}" for v in matrix[i]))
    lines.append("")
    for name in names:
        lines.append(
            f"{name:>6}: precision {report[f'{name}_precision']:.3f}  "
            f"recall {report[f'{name}_recall']:.3f}  f1 {report[f'{name}_f1']:.3f}"
        )
    return "\n".join(lines)
