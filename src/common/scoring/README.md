# Scoring pipeline

`score.py` is the shared scoring boundary for API and batch inference. It
accepts a fitted model bundle and one list of eight California housing
features, applies the fitted preprocessing steps, and returns:

- `prediction`: binary class `0` or `1`
- `probability`: positive-class probability from `0.0` to `1.0`
