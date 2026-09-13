import numpy as np

from src.common.preprocessing.preprocess import split_data
from src.common.scoring.score import score_one
from src.evaluation.metrics import (
    evaluate_model,
    evaluate_without_training_pipeline,
)
from src.training.hyperparameter_tuning.tuner import choose_parameters
from src.training.unified_trainer import build_model, train_model


def make_dataset(n_samples=120):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_model_dispatch_supports_logistic_regression():
    model = build_model("logistic")

    assert model.__class__.__name__ == "LogisticRegression"


def test_training_tuning_evaluation_and_scoring_work_together():
    X, y = make_dataset()
    splits = split_data(X, y)
    parameters = choose_parameters("logistic", splits)
    trained_model = train_model("logistic", splits, **parameters)

    metrics = evaluate_model(trained_model, splits)
    result = score_one(trained_model, X[0].tolist())

    assert set(metrics) == {"accuracy", "precision", "recall", "roc_auc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert result["prediction"] in {0, 1}
    assert 0.0 <= result["probability"] <= 1.0


def test_evaluation_can_run_without_main_training_pipeline():
    X, y = make_dataset()

    metrics = evaluate_without_training_pipeline("logistic", X, y)

    assert metrics["roc_auc"] >= 0.5

