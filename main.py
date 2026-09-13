"""Command-line entry point for preprocessing, training, and evaluation."""

from __future__ import annotations

import argparse

from data.california_housing import load_data
from src.common.preprocessing.preprocess import split_data
from src.evaluation.metrics import evaluate_model
from src.training.hyperparameter_tuning.tuner import choose_parameters
from src.training.unified_trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="California housing classifier.")
    parser.add_argument(
        "--pipeline",
        choices=("preprocess", "train", "train-evaluate"),
        default="train-evaluate",
        help="Pipeline to run. Preprocessing always runs first.",
    )
    parser.add_argument(
        "--model",
        choices=("logistic", "xgboost"),
        default="logistic",
        help="Classifier to train.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1/4: Loading California housing data...", flush=True)
    X, y, feature_names = load_data()
    print(
        f"Step 1/4 complete: loaded {len(y)} samples and "
        f"{len(feature_names)} features.",
        flush=True,
    )

    print("Step 2/4: Creating train, tuning, and holdout splits...", flush=True)
    splits = split_data(X, y, random_state=args.random_state)
    print(
        f"Step 2/4 complete: train={len(splits.y_train)}, "
        f"tuning={len(splits.y_tuning)}, holdout={len(splits.y_holdout)}.",
        flush=True,
    )

    if args.pipeline == "preprocess":
        print("Preprocessing pipeline complete.", flush=True)
        return

    print(
        f"Step 3/4: Tuning and training the {args.model} model...",
        flush=True,
    )
    parameters = choose_parameters(args.model, splits)
    trained_model = train_model(args.model, splits, **parameters)
    print(
        f"Step 3/4 complete: trained {args.model} with parameters "
        f"{parameters}.",
        flush=True,
    )

    if args.pipeline == "train-evaluate":
        print("Step 4/4: Evaluating on the holdout partition...", flush=True)
        print("Holdout metrics:", flush=True)
        for name, value in evaluate_model(trained_model, splits).items():
            print(f"  {name}: {value:.4f}", flush=True)
        print("Step 4/4 complete: evaluation finished.", flush=True)


if __name__ == "__main__":
    main()
