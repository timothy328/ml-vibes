# Preprocessing pipeline

`preprocess.py` provides two pieces:

1. `FeaturePreprocessor` imputes missing values with the median and scales
   numeric features with `StandardScaler`.
2. `split_data` creates deterministic 60% training, 20% tuning, and 20%
   holdout partitions using stratification.

The preprocessor is fitted only on training data. The tuning partition is
available for simple model selection, while the holdout partition is reserved
for final evaluation.
