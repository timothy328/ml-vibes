# Local API

The scoring service is a FastAPI application exposed from
`src.inference.api.app`.

## Local development

1. Copy `.env.example` to `.env` and adjust the host, port, and model settings.
2. Install dependencies with `make install`.
3. Start the service with `make api`.
4. Use FastAPI's interactive documentation at `http://127.0.0.1:8000/docs`
   once the application routes have been implemented.

The API currently provides a health check and a versioned prediction contract.
Model loading still needs to be connected to a persisted model artifact before
`/predict` can serve production predictions. Keep request and response
contracts in `src/inference/api/schemas.py`.
