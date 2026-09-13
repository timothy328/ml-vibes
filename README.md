# ml-vibes

An extensible machine-learning classification project scaffold. The repository is organized to support model development, evaluation, API and batch inference, experimentation, deployment, and governance.

## Project layout

- `main.py` - application entry point
- `src/common/` - shared preprocessing and scoring utilities
- `src/training/` - model implementations, training orchestration, and tuning
- `src/evaluation/` - metrics, explainability, and evaluation logging
- `src/inference/` - FastAPI and batch scoring components
- `src/iteration/` - model-development workflow and research-loop documentation
- `src/ab_testing/` - champion/challenger configuration and traffic routing
- `notebooks/` - exploratory and consolidated model demonstrations
- `tests/` - unit tests
- `data/`, `model_registry/`, and `monitoring/` - data, model artifacts, and observability resources
- `deployment/`, `.github/workflows/`, and `governance/` - deployment and operational configuration
- `Dockerfile`, `docker-compose.yml`, and `.env.example` - planned local/container deployment support

## Getting started

This repository currently contains the project structure and blank implementation placeholders. As components are implemented, use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies:

```bash
uv sync
uv run python main.py
```

Run the test suite with:

```bash
uv run pytest
```

## Local API deployment

The repository includes placeholders and local tooling for a future FastAPI
scoring service:

```bash
cp .env.example .env
make install
make api
```

The service is planned to expose interactive documentation at
`http://127.0.0.1:8000/docs`. Container configuration will be completed after
the API dependencies and application routes are implemented.

## Development notes

Keep reusable logic under `src/`, place experiments in `notebooks/`, and add focused tests under `tests/`. Avoid committing secrets, local environments, generated model artifacts, or production data.
