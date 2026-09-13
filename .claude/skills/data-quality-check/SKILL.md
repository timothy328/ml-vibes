---
name: data-quality-check
description: Check refreshed California housing data before preprocessing or retraining.
---

# Data quality check

Use this skill before scheduled retraining or when a new data extract arrives.

## Checks

- Exactly eight feature columns in the order documented in `data/README.md`
- Numeric, finite values after parsing
- Reasonable missing-value rate for every feature
- Non-empty target values with both binary classes after median conversion
- No accidental target leakage in feature columns
- No unexpected duplicate or malformed records
- Stable row counts and summary statistics relative to the previous extract

Keep raw and derived data in approved storage. This repository's example loader
works in memory and must not create a CSV as a side effect.

If a check fails, stop retraining, report the failing rule and observed value,
and do not promote a new model.
