"""Точка входа: одна команда на каждый шаг конвейера.

    python main.py idle                     # холостой ход по каждому двигателю
    python main.py dataset --limit 5        # выгрузка -> картинки
    python main.py train                    # обучение классификатора
    python main.py eval --split valid       # метрики модели
    python main.py predict plots/Скв_1101   # инференс по папке
    python main.py gui                      # окно оператора
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import config as cfg_mod
from config import SignalConfig, WindowConfig


def cmd_idle(args: argparse.Namespace) -> None:
    from Class_dataset import PowerDataset

    dataset = PowerDataset(args.source, SignalConfig(negatives=args.negatives))
    report = dataset.idle_report(args.wells)
    print(report.to_string(index=False))
    if args.csv:
        report.to_csv(args.csv, index=False, encoding="utf-8-sig")
        print(f"\nСохранено: {args.csv}")


def cmd_dataset(args: argparse.Namespace) -> None:
    from Class_dataset import PowerDataset

    dataset = PowerDataset(
        args.source,
        SignalConfig(negatives=args.negatives),
        WindowConfig(window=args.window, step=args.step, resample=args.resample, points=args.points),
    )
    total = dataset.render(
        args.out,
        args.wells,
        limit_per_well=args.limit,
        image_size=(args.width, args.height),
        markers=args.markers,
    )
    print(f"Построено графиков: {total} -> {args.out}")


def cmd_train(args: argparse.Namespace) -> None:
    from classificator.train import train
    from config import TrainConfig

    train(args.dataset, TrainConfig(epochs=args.epochs))


def cmd_eval(args: argparse.Namespace) -> None:
    from Class_Predict import BeltPredictor, format_report

    predictor = BeltPredictor(args.weights)
    print(format_report(predictor.evaluate(args.dataset / args.split), predictor.names))


def cmd_predict(args: argparse.Namespace) -> None:
    from Class_Predict import BeltPredictor

    predictor = BeltPredictor(args.weights, args.min_confidence)
    predictions = predictor.predict(args.source)
    for pred in predictions:
        print(f"{pred.label:>5}  {pred.confidence:6.1%}  {pred.path}")
    broken = sum(p.is_broken and p.confidence >= args.min_confidence for p in predictions)
    print(f"\nВсего {len(predictions)}, подозрений на обрыв: {broken}")


def cmd_gui(args: argparse.Namespace) -> None:
    from Class_start import BeltApp

    BeltApp(args.weights).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Выявление обрыва ремней станков-качалок")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", type=Path, default=cfg_mod.SOURCE_XLSX, help="выгрузка Excel")
        p.add_argument("--wells", nargs="*", help="номера скважин (по умолчанию все)")
        p.add_argument("--negatives", default=SignalConfig.negatives, choices=["keep", "clip", "abs", "drop"])

    p_idle = sub.add_parser("idle", help="холостой ход по каждому двигателю")
    add_source(p_idle)
    p_idle.add_argument("--csv", type=Path, help="куда сохранить таблицу")
    p_idle.set_defaults(func=cmd_idle)

    p_data = sub.add_parser("dataset", help="построить графики из выгрузки")
    add_source(p_data)
    p_data.add_argument("--out", type=Path, default=cfg_mod.PLOTS_DIR)
    p_data.add_argument("--window", default=WindowConfig.window, help="длительность окна, например 60min")
    p_data.add_argument("--step", default=WindowConfig.step, help="шаг окна (по умолчанию без перекрытия)")
    p_data.add_argument("--resample", default=WindowConfig.resample, help="шаг сетки внутри окна")
    p_data.add_argument("--points", type=int, default=WindowConfig.points, help="точек на графике")
    p_data.add_argument("--limit", type=int, help="не больше N графиков на скважину")
    p_data.add_argument("--width", type=int, default=640, help="ширина картинки, px")
    p_data.add_argument("--height", type=int, default=480, help="высота картинки, px")
    p_data.add_argument("--markers", action="store_true", help="рисовать точки отсчётов")
    p_data.set_defaults(func=cmd_dataset)

    p_train = sub.add_parser("train", help="обучить классификатор")
    p_train.add_argument("--dataset", type=Path, default=cfg_mod.DATASET_DIR)
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="метрики модели на сплите")
    p_eval.add_argument("--dataset", type=Path, default=cfg_mod.DATASET_DIR)
    p_eval.add_argument("--split", default="valid")
    p_eval.add_argument("--weights", type=Path, default=cfg_mod.DEFAULT_WEIGHTS)
    p_eval.set_defaults(func=cmd_eval)

    p_pred = sub.add_parser("predict", help="классифицировать картинку или папку")
    p_pred.add_argument("source", type=Path)
    p_pred.add_argument("--weights", type=Path, default=cfg_mod.DEFAULT_WEIGHTS)
    p_pred.add_argument("--min-confidence", type=float, default=0.5)
    p_pred.set_defaults(func=cmd_predict)

    p_gui = sub.add_parser("gui", help="окно оператора")
    p_gui.add_argument("--weights", type=Path, default=cfg_mod.DEFAULT_WEIGHTS)
    p_gui.set_defaults(func=cmd_gui)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
