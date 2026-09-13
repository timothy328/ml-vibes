"""FastAPI application for local binary classification scoring."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.common.scoring.score import score_one
from src.inference.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="ML Vibes California Housing Classifier", version="0.1.0")
_trained_model = None


@app.get("/health")
def health() -> dict[str, str]:
    """Return service availability."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score one feature vector using the loaded model."""
    if _trained_model is None:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Run the training pipeline first.",
        )
    return PredictionResponse(**score_one(_trained_model, request.features))
