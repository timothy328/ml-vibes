import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from fastapi import HTTPException

from src.inference.api import app as api_module
from src.inference.api.schemas import PredictionRequest


def test_health_endpoint_returns_ok():
    assert api_module.health() == {"status": "ok"}


def test_predict_requires_a_loaded_model():
    with pytest.raises(HTTPException) as error:
        api_module.predict(PredictionRequest(features=[0.0] * 8))

    assert error.value.status_code == 503
