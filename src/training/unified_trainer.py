"""Simple training interface for the supported model types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import ClassifierMixin

from src.common.preprocessing.preprocess import (
    DataSplits,
    FeaturePreprocessor,
    split_data,
)
from src.training.models.glm import build_model as build_logistic_model
from src.training.models.xgboost_model import build_model as build_xgboost_model


@dataclass
class TrainedModel:
    """A fitted classifier and the preprocessor fitted on training data."""

    model_name: str
    model: ClassifierMixin
    preprocessor: FeaturePreprocessor


def build_model(model_name: str, **parameters: object) -> ClassifierMixin:
    """Dispatch model creation by name."""
    normalized_name = model_name.lower()
    if normalized_name in {"logistic", "logistic_regression", "glm"}:
        return build_logistic_model(**parameters)
    if normalized_name in {"xgboost", "xgb"}:
        return build_xgboost_model(**parameters)
    raise ValueError(f"Unsupported model '{model_name}'.")


def train_model(
    model_name: str, splits: DataSplits, **parameters: object
) -> TrainedModel:
    """Fit preprocessing on train data and fit the selected model."""
    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(splits.X_train)
    model = build_model(model_name, **parameters)
    model.fit(X_train, splits.y_train)
    return TrainedModel(model_name, model, preprocessor)


def train_from_raw_data(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    **parameters: object,
) -> tuple[TrainedModel, DataSplits]:
    """Split raw data and train one selected model."""
    splits = split_data(X, y, random_state=random_state)
    return train_model(model_name, splits, **parameters), splits
