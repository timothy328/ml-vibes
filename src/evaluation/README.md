# Evaluation pipeline

`metrics.py` evaluates a fitted model on the holdout partition using accuracy,
precision, recall, and ROC AUC.

The `evaluate_without_training_pipeline` helper can load raw arrays, create
the same deterministic splits, train a selected model, and evaluate it. This
keeps evaluation usable even when `main.py` was not used for training first.
