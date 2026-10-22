"""Focused tests for SHAP, LIME, permutation importance, PDP, and ICE."""

import matplotlib
import numpy as np
from sklearn.inspection import PartialDependenceDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.explainability import (
    build_lime_explanation,
    calculate_permutation_importance,
    calculate_shap_values,
    create_shap_explanation,
    get_shap_feature_importance,
)


def test_global_shap_values_have_expected_dimensions(trained_model, split, dataset):
    _, _, feature_names, _ = dataset
    shap_values = calculate_shap_values(trained_model, split.X_test)
    importance = get_shap_feature_importance(shap_values, feature_names)

    assert shap_values["values"].shape == (114, 30)
    assert shap_values["base_values"].shape == (114,)
    assert shap_values["class_label"] == 1
    assert importance.shape == (30, 2)
    assert set(importance["Feature"]) == set(feature_names)


def test_local_shap_additivity_for_both_classes(trained_model, split):
    predictions = trained_model.predict(split.X_test)
    indices_by_class = {class_label: int(np.flatnonzero(predictions == class_label)[0]) for class_label in trained_model.classes_}

    for class_label, row_index in indices_by_class.items():
        row = split.X_test.iloc[[row_index]]
        shap_values = calculate_shap_values(trained_model, row, class_label=class_label)
        class_index = list(trained_model.classes_).index(class_label)
        reconstructed = shap_values["base_values"][0] + shap_values["values"][0].sum()
        probability = trained_model.predict_proba(row)[0, class_index]

        assert shap_values["values"].shape == (1, 30)
        assert shap_values["class_label"] == class_label
        assert np.isclose(reconstructed, probability, atol=1e-6)
        figure = create_shap_explanation(shap_values, row, 0)
        plt.close(figure)


def test_lime_explains_multiple_observations(trained_model, split, dataset):
    _, _, feature_names, target_names = dataset
    explanations = []
    for row_index in (0, 1):
        row = split.X_test.iloc[[row_index]]
        predicted_class = trained_model.predict(row)[0]
        explanation = build_lime_explanation(
            trained_model,
            split.X_train,
            row.iloc[0].to_numpy(),
            feature_names,
            predicted_class,
            target_names,
        )
        explanations.append(explanation)

    assert all(len(explanation) == 10 for explanation in explanations)
    assert all(isinstance(feature, str) and isinstance(weight, float) for explanation in explanations for feature, weight in explanation)
    assert explanations[0] != explanations[1]


def test_permutation_importance_uses_test_schema(trained_model, split):
    importance = calculate_permutation_importance(trained_model, split.X_test, split.y_test)

    assert importance.shape == (30, 3)
    assert set(importance["Feature"]) == set(split.X_test.columns)
    assert importance["ImportanceStd"].ge(0).all()
    assert importance["Importance"].notna().all()


def test_pdp_and_ice_render_for_valid_feature(trained_model, split):
    for kind in ("average", "individual"):
        figure, axis = plt.subplots()
        PartialDependenceDisplay.from_estimator(
            trained_model,
            split.X_test,
            [0],
            kind=kind,
            subsample=50 if kind == "individual" else None,
            response_method="predict_proba",
            ax=axis,
        )
        assert figure.axes
        plt.close(figure)
