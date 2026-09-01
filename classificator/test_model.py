"""Проверка обученной модели на размеченном сплите.

Было: жёстко зашитый путь ``C:\\Users\\User\\...``, предикт по одной
картинке в цикле, ``cv2.imshow`` + ``waitKey(0)`` на каждом кадре (скрипт
невозможно прогнать без оператора у монитора) и импорт customtkinter,
который здесь вообще не использовался. Стало: параметры из командной
строки, батчевый предикт и метрики.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg_mod
from Class_Predict import BeltPredictor, format_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики классификатора на размеченной папке")
    parser.add_argument("--weights", type=Path, default=cfg_mod.DEFAULT_WEIGHTS)
    parser.add_argument("--split", default="valid", help="train | valid | test")
    parser.add_argument("--dataset", type=Path, default=cfg_mod.DATASET_DIR)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--errors", action="store_true", help="показать список ошибок")
    args = parser.parse_args()

    split_dir = args.dataset / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"Нет сплита: {split_dir}")

    predictor = BeltPredictor(args.weights)
    report = predictor.evaluate(split_dir, args.batch)
    print(format_report(report, predictor.names))

    if args.errors:
        print("\nОшибки:")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for pred in predictor.predict(class_dir, args.batch):
                if pred.label != class_dir.name:
                    print(f"  {pred.path.name}: истина {class_dir.name}, модель {pred.label} ({pred.confidence:.2f})")


if __name__ == "__main__":
    main()
