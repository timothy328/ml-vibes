# ml-vibes

An extensible machine-learning classification project scaffold. The repository is organized to support model development, evaluation, API and batch inference, experimentation, deployment, and governance.

> **Experimentation note:** This repository was coded using GPT 5.6 Luna for
> experimentation purposes. Review and validate generated code before using it
> with production data or production services.

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
- `Dockerfile`, `docker-compose.yml`, and `.env.example` - local/container deployment support

## Getting started

Use [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies:

```bash
# Use Python 3.14.5 for the project.
uv python install 3.14.5

# Create the project virtual environment.
uv venv --python 3.14.5

# Activate it for interactive development.
source .venv/bin/activate

# Install the project and development dependencies.
uv sync
uv run python main.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Run the application with `uv run` (or activate `.venv` first). Running
`python main.py` directly can use a different system interpreter that does not
have the project dependencies, including `numpy` and `certifi`. If you want to
use the active interpreter directly, install the pinned requirements into that
same environment first:

```bash
uv pip install --python "$(which python)" -r requirements.txt
python main.py
```

The default command runs preprocessing, logistic-regression training with
simple tuning, and holdout evaluation:

```bash
uv run python main.py
```

Other pipeline examples:

```bash
uv run python main.py --pipeline preprocess
uv run python main.py --pipeline train --model xgboost
uv run python main.py --pipeline train-evaluate --model logistic --random-state 7
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
