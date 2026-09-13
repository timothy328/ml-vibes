# Instructions for Claude

## Project purpose

This is an experimental California housing binary-classification project. The
target is `y_binary`: `1` when the original house value is above its dataset
median and `0` otherwise.

## Development conventions

- Use `uv` for dependency management and command execution.
- Keep the eight documented feature names and their order unchanged.
- Keep raw data in memory; do not add CSV exports as part of the example
  workflow.
- Preserve the train/tuning/holdout split and never tune on the holdout set.
- Prefer simple, readable sklearn-style components over clever abstractions.
- Add or update focused tests for preprocessing, training, evaluation, and API
  behavior when changing those areas.
- Run `uv run pytest` and the default `uv run python main.py` before reporting
  completion when dependencies are available.

## Retraining safety

Use the skills in `.claude/skills/` for data checks, monitoring, routine
retraining, and candidate validation. Never replace a champion model before a
candidate has passed the documented validation checks.

