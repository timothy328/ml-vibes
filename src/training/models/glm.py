"""Logistic regression model for binary classification."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_model(**parameters: object) -> LogisticRegression:
    """Create a configurable logistic regression classifier."""
    defaults: dict[str, object] = {"max_iter": 1000, "random_state": 42}
    defaults.update(parameters)
    return LogisticRegression(**defaults)
