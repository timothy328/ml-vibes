"""XGBoost model for binary classification."""

from __future__ import annotations

from typing import Any


def build_model(**parameters: object) -> Any:
    """Create an XGBoost classifier with conservative defaults."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the xgboost model. Run `uv sync` first."
        ) from exc

    defaults: dict[str, object] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 1,
    }
    defaults.update(parameters)
    return XGBClassifier(**defaults)
