# Local API

The scoring service is planned as a FastAPI application exposed from
`src.inference.api.app`.

## Local development

1. Copy `.env.example` to `.env` and adjust the host, port, and model settings.
2. Install dependencies with `make install`.
3. Start the service with `make api`.
4. Use FastAPI's interactive documentation at `http://127.0.0.1:8000/docs`
   once the application routes have been implemented.

The API should eventually provide health/readiness checks, model metadata, and
a versioned prediction endpoint. Keep request and response contracts in
`src/inference/api/schemas.py`.
