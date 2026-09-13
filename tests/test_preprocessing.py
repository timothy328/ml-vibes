import numpy as np
import pytest

from src.common.preprocessing.preprocess import FeaturePreprocessor, split_data


def test_split_data_creates_train_tuning_and_holdout_partitions():
    X = np.arange(800, dtype=float).reshape(100, 8)
    y = np.array([0, 1] * 50)

    splits = split_data(X, y, random_state=42)

    assert [len(part) for part in (splits.y_train, splits.y_tuning, splits.y_holdout)] == [
        60,
        20,
        20,
    ]
    assert set(np.concatenate((splits.y_train, splits.y_tuning, splits.y_holdout))) == {
        0,
        1,
    }


def test_preprocessor_imputes_and_scales_features():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [np.nan, 30.0]])
    preprocessor = FeaturePreprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape == X.shape
    assert np.isfinite(transformed).all()
    assert np.allclose(transformed.mean(axis=0), 0.0)


def test_split_data_rejects_invalid_partition_sizes():
    X = np.ones((10, 2))
    y = np.array([0, 1] * 5)

    with pytest.raises(ValueError):
        split_data(X, y, train_size=0.8, tuning_size=0.3)

