---
name: routine-retraining
description: Run a safe, repeatable model retraining workflow for the California housing classifier.
---

# Routine retraining

Use this skill when new labeled housing data is available or when a scheduled
retraining job is due.

## Workflow

1. Confirm the data source, extraction date, row count, feature names, target
   definition, and label coverage.
2. Run the preprocessing and schema checks without writing raw data to CSV.
3. Preserve the existing champion model and record its version and metrics.
4. Split the refreshed data into training, tuning, and holdout partitions with
   the configured random seed.
5. Tune only on the tuning partition, then evaluate the candidate once on the
   holdout partition.
6. Compare candidate and champion on accuracy, precision, recall, and ROC AUC.
7. Promote the candidate only if it passes the configured quality thresholds
   and does not introduce a material regression.
8. Record the data version, code revision, parameters, metrics, and promotion
   decision in the model registry.

Never overwrite the champion before the candidate has passed validation.

