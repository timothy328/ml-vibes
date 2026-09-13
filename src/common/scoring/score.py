"""Scoring helpers shared by the API and other inference entry points."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.training.unified_trainer import TrainedModel


def score_one(trained_model: TrainedModel, features: list[float]) -> dict[str, Any]:
    """Return a binary prediction and probability for one feature vector."""
    values = np.asarray([features], dtype=float)
    transformed = trained_model.preprocessor.transform(values)
    probability = float(trained_model.model.predict_proba(transformed)[0, 1])
    return {"prediction": int(probability >= 0.5), "probability": probability}
