# Training pipeline

`unified_trainer.py` is the single training interface. It dispatches to either
the logistic regression model or the XGBoost model, fits preprocessing on the
training partition, and returns a `TrainedModel` bundle.

`hyperparameter_tuning/tuner.py` performs a small readable candidate search
using only the tuning partition. `models/` contains the model builders.

The normal sequence is:

1. Split raw data into training, tuning, and holdout partitions.
2. Select parameters using the tuning partition.
3. Fit the selected model on the training partition.
4. Evaluate once on the untouched holdout partition.
