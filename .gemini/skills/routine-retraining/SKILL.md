---
name: routine-retraining
description: Safely repeat training for the California housing classification model.
---

# Routine retraining

Validate new data, preserve the current champion, and run the deterministic
training/tuning/holdout workflow. Tune only on tuning data, evaluate the
candidate on holdout, compare it with the champion using the configured
metrics, and record the data version, parameters, metrics, and decision.
Promote only after all checks pass.

