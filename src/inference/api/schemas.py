"""Request and response schemas for the FastAPI scoring service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="Eight California housing features in dataset order.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
