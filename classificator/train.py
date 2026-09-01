"""Обучение классификатора good/bad на графиках мощности (YOLOv8-cls).

Раньше файл был пустым, а обучение запускалось руками из консоли —
поэтому параметры аугментаций нигде не были зафиксированы и результат
не воспроизводился.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg_mod
from config import TrainConfig


def count_images(dataset_dir: Path) -> dict[str, dict[str, int]]:
    """Сколько картинок в каждом сплите и классе — сразу видно перекос классов."""
    stats: dict[str, dict[str, int]] = {}
    for split in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        stats[split.name] = {
            cls.name: sum(1 for f in cls.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
            for cls in sorted(p for p in split.iterdir() if p.is_dir())
        }
    return stats


def train(dataset_dir: Path, cfg: TrainConfig) -> Path:
    """Запускает обучение и возвращает путь к лучшим весам."""
    from ultralytics import YOLO

    for split, classes in count_images(dataset_dir).items():
        print(f"{split}: {classes}")

    model = YOLO(cfg.model)
    results = model.train(
        data=str(dataset_dir),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        patience=cfg.patience,   # ранняя остановка: датасет маленький, легко переобучиться
        seed=cfg.seed,
        deterministic=True,
        project=str(cfg.project),
        name=cfg.name,
        exist_ok=True,
        # --- аугментации под графики, а не под фотографии ---
        flipud=0.0,      # переворот по вертикали меняет смысл: холостой ход стал бы нагрузкой
        fliplr=0.5,      # обращение времени форму колебаний не ломает
        degrees=0.0,     # поворот увёл бы график с осей
        translate=0.05,
        scale=0.1,
        shear=0.0,
        erasing=0.0,     # дефолтные 0.4 стирают часть линии — на разреженном графике это потеря сигнала
        hsv_h=0.0,       # цвет несёт смысл: чёрная линия и красные точки
        hsv_s=0.0,
        hsv_v=0.2,
        auto_augment=None,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nЛучшие веса: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение классификатора обрыва ремня")
    parser.add_argument("--dataset", type=Path, default=cfg_mod.DATASET_DIR, help="папка датасета")
    parser.add_argument("--model", default=TrainConfig.model, help="стартовые веса")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--imgsz", type=int, default=TrainConfig.imgsz)
    parser.add_argument("--batch", type=int, default=TrainConfig.batch)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        raise SystemExit(f"Нет папки датасета: {args.dataset}")

    train(
        args.dataset,
        TrainConfig(model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch),
    )


if __name__ == "__main__":
    main()
