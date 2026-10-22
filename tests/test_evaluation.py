"""Tests for held-out model evaluation."""

import numpy as np

from src.evaluation import evaluate_model


def test_evaluation_uses_test_set_and_returns_all_metrics(trained_model, split, dataset):
    _, _, _, target_names = dataset
    metrics = evaluate_model(trained_model, split.X_test, split.y_test, target_names)

    assert set(("Accuracy", "Precision", "Recall", "F1-score")) <= metrics.keys()
    assert metrics["metric_average"] == "macro"
    assert 0.0 <= metrics["Accuracy"] <= 1.0
    assert 0.0 <= metrics["Precision"] <= 1.0
    assert 0.0 <= metrics["Recall"] <= 1.0
    assert 0.0 <= metrics["F1-score"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert set(target_names) <= metrics["classification_report"].keys()

    expected_predictions = trained_model.predict(split.X_test)
    assert np.trace(metrics["confusion_matrix"]) == np.sum(expected_predictions == split.y_test.to_numpy())


def test_evaluation_changes_when_test_labels_change(trained_model, split, dataset):
    _, _, _, target_names = dataset
    original = evaluate_model(trained_model, split.X_test, split.y_test, target_names)
    flipped = evaluate_model(trained_model, split.X_test, 1 - split.y_test, target_names)

    assert original["Accuracy"] != flipped["Accuracy"]
