"""SHAP, LIME, and permutation-based model explanations."""

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def _select_shap_class(raw_values, expected_value, class_index: int, row_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize SHAP list and Explanation outputs across supported versions."""
    if isinstance(raw_values, list):
        values = np.asarray(raw_values[class_index])
        base_values = np.asarray(expected_value)
    else:
        values = np.asarray(raw_values.values if hasattr(raw_values, "values") else raw_values)
        base_values = np.asarray(raw_values.base_values if hasattr(raw_values, "base_values") else expected_value)
        if values.ndim == 3:
            values = values[:, :, class_index]
        if base_values.ndim == 2:
            base_values = base_values[:, class_index]
        elif base_values.ndim == 1 and base_values.size != row_count:
            base_values = np.asarray([base_values[class_index]])
    base_values = np.atleast_1d(base_values)
    if base_values.size == 1:
        base_values = np.repeat(base_values.item(), row_count)
    return values, base_values


def calculate_shap_values(model, data: pd.DataFrame, class_label=None) -> dict:
    explainer = shap.TreeExplainer(model)
    classes = np.asarray(model.classes_)
    if class_label is None:
        class_index = 1 if len(classes) > 1 else 0
        class_label = classes[class_index]
    else:
        matching_indices = np.flatnonzero(classes == class_label)
        if len(matching_indices) != 1:
            raise ValueError(f"Class label {class_label!r} is not present in model.classes_.")
        class_index = int(matching_indices[0])
    raw_values = explainer(data)
    values, base_values = _select_shap_class(raw_values, explainer.expected_value, class_index, len(data))
    return {
        "values": values,
        "base_values": base_values,
        "class_index": class_index,
        "class_label": class_label,
        "output_name": "predicted probability",
    }


def get_shap_feature_importance(shap_values: dict, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Feature": feature_names, "Importance": np.abs(shap_values["values"]).mean(axis=0)}).sort_values("Importance", ascending=False)


def create_shap_explanation(shap_values: dict, row: pd.DataFrame, row_position: int):
    values = shap_values["values"][row_position]
    base_value = float(shap_values["base_values"][row_position])
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=row.iloc[0].to_numpy(),
        feature_names=row.columns.tolist(),
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    import matplotlib.pyplot as plt

    return plt.gcf()


def calculate_permutation_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    return pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": result.importances_mean,
            "ImportanceStd": result.importances_std,
        }
    ).sort_values("Importance", ascending=False)


def build_lime_explanation(
    model,
    X_train: pd.DataFrame,
    row: np.ndarray,
    feature_names: list[str],
    predicted_class,
    target_names: list[str],
):
    from lime.lime_tabular import LimeTabularExplainer

    class_index = int(np.flatnonzero(np.asarray(model.classes_) == predicted_class)[0])
    explainer = LimeTabularExplainer(
        X_train.to_numpy(),
        feature_names=feature_names,
        class_names=target_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    def predict_proba(values):
        return model.predict_proba(pd.DataFrame(values, columns=feature_names))

    explanation = explainer.explain_instance(row, predict_proba, num_features=10, labels=[class_index])
    return explanation.as_list(label=class_index)