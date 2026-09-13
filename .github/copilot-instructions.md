# Copilot instructions

## Repository context

`ml-vibes` is an experimental California housing binary-classification
repository. Use `uv` for dependency management and execution.

## Code requirements

- Keep the implementation simple and readable.
- Preserve the documented eight-feature input schema and `y_binary` target.
- Keep preprocessing fitted on training data only.
- Keep tuning and holdout data separate; never tune on the holdout partition.
- Do not write the sklearn dataset to CSV.
- Use FastAPI for API work and keep shared scoring logic under
  `src/common/scoring/`.
- Add focused pytest coverage for behavior changed.

## Validation

When dependencies are installed, run:

```bash
uv run pytest
uv run python main.py
```

For retraining or deployment work, review the relevant skills under
`.copilot/skills/` and do not promote a candidate model without validation.

