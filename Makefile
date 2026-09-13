.PHONY: install test api

install:
	uv sync

test:
	uv run pytest

api:
	uv run uvicorn src.inference.api.app:app --host $${API_HOST:-127.0.0.1} --port $${API_PORT:-8000}

