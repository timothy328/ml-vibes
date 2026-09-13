"""Evaluation pipeline for holdout binary classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.common.preprocessing.preprocess import DataSplits, split_data
from src.training.unified_trainer import TrainedModel, train_model


def evaluate_model(
    trained_model: TrainedModel, splits: DataSplits
) -> dict[str, float]:
    """Evaluate a fitted model on the untouched holdout partition."""
    X_holdout = trained_model.preprocessor.transform(splits.X_holdout)
    predictions = trained_model.model.predict(X_holdout)
    probabilities = trained_model.model.predict_proba(X_holdout)[:, 1]
    return {
        "accuracy": float(accuracy_score(splits.y_holdout, predictions)),
        "precision": float(
            precision_score(splits.y_holdout, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(splits.y_holdout, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(splits.y_holdout, probabilities)),
    }


def evaluate_without_training_pipeline(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    """Train and evaluate independently when main.py did not train first."""
    splits = split_data(X, y, random_state=random_state)
    trained_model = train_model(model_name, splits)
    return evaluate_model(trained_model, splits)
