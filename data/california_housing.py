"""Load the California housing data without writing the dataset to disk."""

from __future__ import annotations

import ssl
from typing import Any
from urllib.error import URLError

import numpy as np
from sklearn.datasets import fetch_california_housing


def load_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return features, binary labels, and the feature names."""
    try:
        dataset: Any = fetch_california_housing(as_frame=False)
    except URLError:
        # Some local Python installations do not trust the certificate chain
        # used by the sklearn dataset host. Retry with certifi's CA bundle.
        try:
            import certifi
        except ImportError as exc:
            raise RuntimeError(
                "certifi is required for the dataset download retry. "
                "Run `uv sync` and then use `uv run python main.py`."
            ) from exc
        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = lambda: ssl.create_default_context(
            cafile=certifi.where()
        )
        try:
            dataset = fetch_california_housing(as_frame=False)
        finally:
            ssl._create_default_https_context = original_context
    X = np.asarray(dataset.data)
    y = np.asarray(dataset.target)
    y_binary = (y > np.median(y)).astype(int)
    feature_names = list(dataset.feature_names)
    return X, y_binary, feature_names
