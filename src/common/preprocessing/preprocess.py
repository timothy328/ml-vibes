"""Reusable preprocessing and deterministic dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DataSplits:
    """Features and labels for training, tuning, and final holdout evaluation."""

    X_train: np.ndarray
    X_tuning: np.ndarray
    X_holdout: np.ndarray
    y_train: np.ndarray
    y_tuning: np.ndarray
    y_holdout: np.ndarray


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Impute missing values and standardize numeric features."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FeaturePreprocessor":
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.transform(X)

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        return self.pipeline.fit_transform(X, y)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    train_size: float = 0.6,
    tuning_size: float = 0.2,
) -> DataSplits:
    """Split data into train, tuning, and untouched holdout partitions."""
    if not 0 < train_size < 1 or not 0 < tuning_size < 1:
        raise ValueError("train_size and tuning_size must be between 0 and 1.")
    if train_size + tuning_size >= 1:
        raise ValueError("train_size + tuning_size must be less than 1.")

    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y
    )
    remaining_tuning_share = tuning_size / (1 - train_size)
    X_tuning, X_holdout, y_tuning, y_holdout = train_test_split(
        X_remaining,
        y_remaining,
        train_size=remaining_tuning_share,
        random_state=random_state,
        stratify=y_remaining,
    )
    return DataSplits(
        X_train, X_tuning, X_holdout, y_train, y_tuning, y_holdout
    )
