"""Small, readable hyperparameter search using the tuning split."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score

from src.common.preprocessing.preprocess import DataSplits, FeaturePreprocessor
from src.training.unified_trainer import build_model


def choose_parameters(model_name: str, splits: DataSplits) -> dict[str, Any]:
    """Choose the best simple candidate based on tuning accuracy."""
    candidates: list[dict[str, Any]]
    if model_name == "logistic":
        candidates = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]
    elif model_name == "xgboost":
        candidates = [
            {"n_estimators": 100},
            {"n_estimators": 150},
            {"n_estimators": 250},
        ]
    else:
        raise ValueError(f"Unsupported model '{model_name}'.")

    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(splits.X_train)
    X_tuning = preprocessor.transform(splits.X_tuning)
    best_parameters = candidates[0]
    best_score = -1.0

    for parameters in candidates:
        model = build_model(model_name, **parameters)
        model.fit(X_train, splits.y_train)
        score = accuracy_score(splits.y_tuning, model.predict(X_tuning))
        if score > best_score:
            best_score = score
            best_parameters = parameters
    return best_parameters
