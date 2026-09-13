from types import SimpleNamespace

import numpy as np

from data.california_housing import load_data


def test_load_data_returns_eight_features_and_binary_labels(monkeypatch):
    raw_target = np.array([1.0, 2.0, 3.0, 4.0])
    raw_features = np.arange(32, dtype=float).reshape(4, 8)
    dataset = SimpleNamespace(
        data=raw_features,
        target=raw_target,
        feature_names=[f"feature_{i}" for i in range(8)],
    )
    monkeypatch.setattr(
        "data.california_housing.fetch_california_housing",
        lambda as_frame=False: dataset,
    )

    X, y, feature_names = load_data()

    assert X.shape == (4, 8)
    assert y.tolist() == [0, 0, 1, 1]
    assert set(y.tolist()) == {0, 1}
    assert len(feature_names) == 8

