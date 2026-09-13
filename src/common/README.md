# Common pipelines

This directory contains reusable components shared by training and inference.

- `preprocessing/` loads no data itself; it imputes missing numeric values,
  standardizes features, and creates deterministic train/tuning/holdout splits.
- `scoring/` applies a fitted preprocessor and classifier to one feature vector,
  returning a `0`/`1` prediction and its positive-class probability.

These components are intentionally small so the API and offline workflows use
the same feature transformations.
