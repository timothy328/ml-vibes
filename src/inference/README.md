# Inference pipelines

The inference package contains API and batch entry points that can share the
common scoring helper.

The FastAPI application in `api/app.py` exposes:

- `GET /health` for a basic service check
- `POST /predict` for one eight-feature prediction request

The API currently requires a trained model to be loaded by the application
startup/runtime integration. Until that integration is added, `/predict`
returns HTTP 503 rather than silently producing a result.
