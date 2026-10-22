"""Tests for model training and persistence."""

import numpy as np

from src.model import load_saved_model, save_model


def test_random_forest_predicts_and_returns_probabilities(trained_model, split):
    predictions = trained_model.predict(split.X_test)
    probabilities = trained_model.predict_proba(split.X_test)

    assert predictions.shape == (114,)
    assert probabilities.shape == (114, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert trained_model.classes_.tolist() == [0, 1]


def test_model_can_be_saved_and_loaded(trained_model, split, tmp_path):
    model_path = tmp_path / "model.joblib"
    save_model(trained_model, model_path)
    loaded_model = load_saved_model(model_path)

    assert loaded_model.classes_.tolist() == trained_model.classes_.tolist()
    assert np.array_equal(loaded_model.predict(split.X_test), trained_model.predict(split.X_test))
