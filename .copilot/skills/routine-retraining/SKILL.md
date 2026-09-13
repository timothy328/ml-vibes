---
name: routine-retraining
description: Run a repeatable retraining workflow for the California housing classifier.
---

# Routine retraining

1. Validate the refreshed data against `data/README.md`.
2. Confirm the feature order, target derivation, row count, and label balance.
3. Preserve the current champion model and its metrics.
4. Run the configured train/tuning/holdout workflow.
5. Tune only on the tuning partition and evaluate the candidate on holdout.
6. Compare candidate and champion on accuracy, precision, recall, and ROC AUC.
7. Record the data version, code revision, parameters, metrics, and decision.
8. Promote only after the candidate passes validation.

Never overwrite the champion on a failed check.

