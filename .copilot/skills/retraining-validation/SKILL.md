---
name: retraining-validation
description: Validate a retrained candidate before promoting it to the API champion.
---

# Retraining validation

Confirm the candidate uses the documented features and `y_binary` target,
training-only preprocessing, a separate tuning partition, and an untouched
holdout. Run tests and `uv run python main.py`, compare candidate and champion
metrics on the same protocol, and exercise valid and invalid API inputs.

Record the candidate version, data range, parameters, metrics, and promotion
decision. Keep the champion active if validation fails.

