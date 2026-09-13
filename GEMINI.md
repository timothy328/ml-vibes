# Instructions for Gemini

This repository is an experimental California housing binary classifier.
Implementations should remain simple, explicit, and easy to test.

## Important project facts

- Manage dependencies and commands with `uv`, not ad hoc pip installs.
- The model consumes eight numeric California housing features in the order
  documented in `data/README.md`.
- The target is `y_binary`, derived from the median of the original continuous
  house-value target.
- Preprocessing must be fitted on training data only.
- Use separate training, tuning, and holdout partitions.
- Do not write the downloaded dataset to CSV.

Before completing model changes, run the unit tests and the default entry point.
For retraining work, check `.gemini/skills/` and preserve the current champion
until the candidate has passed validation.

