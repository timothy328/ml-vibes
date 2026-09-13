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

    # Step 1 complete: load the California housing data in memory and create
    # binary above/below-median labels.
    X, y, feature_names = load_data()

    # Step 2 complete: create deterministic training, tuning, and holdout
    # partitions. The holdout data remains untouched until evaluation.
    splits = split_data(X, y, random_state=args.random_state)
    print(
        f"Preprocessed {len(y)} samples with {len(feature_names)} features. "
        f"Splits: train={len(splits.y_train)}, "
        f"tuning={len(splits.y_tuning)}, holdout={len(splits.y_holdout)}."
    )

    if args.pipeline == "preprocess":
        return

    # Step 3 complete: select simple hyperparameters on the tuning partition
    # and fit the requested classifier on the training partition.
    parameters = choose_parameters(args.model, splits)
    trained_model = train_model(args.model, splits, **parameters)
    print(f"Trained {args.model} model with parameters: {parameters}.")

    if args.pipeline == "train-evaluate":
        # Step 4 complete: evaluate the fitted model once on the holdout
        # partition and report binary classification metrics.
        print("Holdout metrics:")
        for name, value in evaluate_model(trained_model, splits).items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
