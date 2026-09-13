---
name: model-monitoring
description: Check production model health, input drift, prediction behavior, and retraining signals.
---

# Model monitoring

Use this skill for scheduled or on-demand checks of the deployed classifier.

## Checks

- API health and latency, including `/health` and prediction error rates
- Feature schema: eight numeric fields in the configured order
- Missing values, unexpected ranges, and invalid binary outputs
- Feature distribution drift against the training reference data
- Prediction class balance and positive-class probability distribution
- Delayed target availability and recent holdout-style performance

## Retraining triggers

Recommend retraining when any agreed threshold is exceeded, such as sustained
feature drift, a meaningful drop in ROC AUC or recall, a change in class
balance, or a materially different data-generating process. A single noisy
observation is not enough; confirm the signal over an appropriate time window.

Report the observation window, baseline, measured value, threshold, and
recommended action.

