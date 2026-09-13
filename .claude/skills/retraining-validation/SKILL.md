---
name: retraining-validation
description: Validate a retrained candidate before it is promoted to the API champion.
---

# Retraining validation

Use this skill after training a candidate model and before deployment.

## Required validation

1. Confirm the candidate uses the documented eight features and `y_binary`
   target.
2. Confirm preprocessing is fitted on training data only.
3. Confirm tuning data is not used as the final holdout.
4. Run unit tests and the default pipeline.
5. Evaluate the candidate and champion on the same holdout protocol.
6. Check accuracy, precision, recall, and ROC AUC against acceptance
   thresholds.
7. Exercise the scoring function with valid and invalid feature vectors.
8. Verify the model can be loaded by the local FastAPI service.

Document the candidate version, data range, parameters, metrics, test result,
and promotion decision. If validation fails, keep the current champion active
and report the failure instead of silently falling back.

