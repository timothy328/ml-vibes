# Runtime-only image for the FastAPI inference service.
# Training, notebooks, tests, and raw data are intentionally excluded.
FROM python:3.14.5-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# Install only packages needed to start the API and run the sklearn-backed
# scoring path. The model artifact can be mounted or added separately later.
RUN uv pip install --system \
    "certifi>=2025.1.31" \
    "fastapi>=0.120.0" \
    "numpy>=2.3.0" \
    "pydantic>=2.12.0" \
    "scikit-learn>=1.7.0" \
    "uvicorn[standard]>=0.34.0"

COPY src/__init__.py /app/src/__init__.py
COPY src/common /app/src/common
COPY src/inference /app/src/inference
COPY src/training/__init__.py /app/src/training/__init__.py
COPY src/training/unified_trainer.py /app/src/training/unified_trainer.py
COPY src/training/models /app/src/training/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.inference.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
