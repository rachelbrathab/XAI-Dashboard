"""Streamlit entry point for the user-driven XAI Model Explainability Dashboard."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.inspection import PartialDependenceDisplay

from src.data_loader import prepare_uploaded_dataset, split_dataset
from src.evaluation import evaluate_model
from src.explainability import (
    build_lime_explanation,
    calculate_permutation_importance,
    calculate_shap_values,
    create_shap_explanation,
    get_shap_feature_importance,
)
from src.model import train_model
from utils.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_lime_contributions,
    plot_shap_summary,
    plot_target_distribution,
)

st.set_page_config(page_title="XAI Model Explainability Dashboard", page_icon="🔍", layout="wide")


def home_page() -> None:
    """Minimal landing page centred on uploading a dataset."""
    st.title("XAI Model Explainability Dashboard")
    st.caption("Upload a tabular dataset to train a classifier and explore why it makes predictions.")
    st.markdown("### Start with your dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], help="Use a CSV with rows as observations, columns as features, and one target column to predict.")
    if uploaded is None:
        st.info("Upload a CSV to begin. You will choose the target column before training the model.")
        with st.expander("Dataset requirements"):
            st.markdown(
                "- CSV format, with a header row.\n"
                "- One target column containing at least two classes.\n"
                "- At least one numeric feature column.\n"
                "- At least 10 complete rows; each target class needs at least two rows.\n"
                "- Missing/infinite values in selected features or target are removed before training.\n"
                "- Non-numeric feature columns are excluded for this version; the target may be text or numeric."
            )
        return

    if uploaded.size > 50 * 1024 * 1024:
        st.error("This upload is larger than 50 MB. Please upload a smaller CSV file.")
        return
    try:
        raw = pd.read_csv(uploaded)
        raw.columns = [str(column) if str(column).strip() else f"column_{index + 1}" for index, column in enumerate(raw.columns)]
        if raw.columns.duplicated().any():
            st.error("Column names must be unique. Please rename duplicate columns in the CSV and upload it again.")
            return
    except Exception as error:
        st.error(f"Could not read this CSV file: {error}")
        return
    if raw.empty or len(raw.columns) < 2:
        st.error("The CSV must contain at least one row and at least two columns (features plus target).")
        return
    if len(raw) > 100_000 or len(raw.columns) > 200:
        st.error("For this interactive demo, please limit uploads to 100,000 rows and 200 columns.")
        return

    file_signature = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if st.session_state.get("uploaded_signature") != file_signature:
        st.session_state.pop("analysis_bundle", None)
        st.session_state["uploaded_signature"] = file_signature
    st.caption(f"File: {uploaded.name} · {len(raw):,} rows · {len(raw.columns)} columns")
    st.markdown("#### Preview")
    st.dataframe(raw.head(8), use_container_width=True, hide_index=True)
    target_column = st.selectbox("Choose the target column to predict", list(raw.columns), index=len(raw.columns) - 1)
    if st.button("Analyze dataset", type="primary", use_container_width=False):
        with st.spinner("Preparing data, training the model, and calculating evaluation metrics…"):
            try:
                X, y, feature_names, target_names, dropped_columns = prepare_uploaded_dataset(raw, target_column)
                split = split_dataset(X, y)
                model = train_model(split.X_train, split.y_train)
                metrics = evaluate_model(model, split.X_test, split.y_test, target_names)
                st.session_state["analysis_bundle"] = {
                    "file_name": uploaded.name,
                    "signature": file_signature,
                    "target_column": target_column,
                    "X": X,
                    "y": y,
                    "feature_names": feature_names,
                    "target_names": target_names,
                    "dropped_columns": dropped_columns,
                    "split": split,
                    "model": model,
                    "metrics": metrics,
                }
                st.success("Dataset analyzed successfully. Opening your dashboard…")
                st.rerun()
            except Exception as error:
                st.error(f"Could not analyze this dataset: {error}")
    bundle = st.session_state.get("analysis_bundle")
    if bundle and bundle.get("signature") == file_signature:
        st.success(f"Current analysis: {bundle['file_name']} · target: {bundle['target_column']}")
        if st.button("Clear analysis and upload another dataset"):
            st.session_state.pop("analysis_bundle", None)
            st.rerun()


def metric_cards(metrics: dict) -> None:
    columns = st.columns(4)
    for column, name in zip(columns, ("Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)")):
        key = {"Precision (macro)": "Precision", "Recall (macro)": "Recall", "F1-score (macro)": "F1-score"}.get(name, name)
        column.metric(name, f"{metrics[key]:.3f}")


def dataset_page(bundle: dict) -> None:
    st.title("Dataset Explorer")
    X, y = bundle["X"], bundle["y"]
    columns = st.columns(4)
    columns[0].metric("Usable rows", f"{len(X):,}")
    columns[1].metric("Numeric features", len(bundle["feature_names"]))
    columns[2].metric("Target classes", len(bundle["target_names"]))
    columns[3].metric("Missing values after cleaning", int(X.isna().sum().sum()))
    st.caption(f"Source: {bundle['file_name']} · Target: {bundle['target_column']}")
    if bundle["dropped_columns"]:
        st.warning("Non-numeric predictor columns excluded: " + ", ".join(bundle["dropped_columns"]))
    st.markdown("### Data preview")
    preview = X.head(10).copy()
    preview[bundle["target_column"]] = y.head(10).map(lambda label: bundle["target_names"][int(label)])
    st.dataframe(preview, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        st.markdown("### Target distribution")
        st.plotly_chart(plot_target_distribution(y, bundle["target_names"]), use_container_width=True)
    with right:
        st.markdown("### Feature statistics")
        st.dataframe(X.describe().T, use_container_width=True)
    st.markdown("### Included numeric features")
    st.dataframe(pd.DataFrame({"Feature": bundle["feature_names"]}), use_container_width=True, hide_index=True)


def performance_page(bundle: dict) -> None:
    st.title("Model Performance")
    split, metrics = bundle["split"], bundle["metrics"]
    st.caption(f"Random Forest · 200 trees · stratified 80/20 split · random_state=42 · {len(split.X_train):,} train / {len(split.X_test):,} test rows")
    metric_cards(metrics)
    left, right = st.columns(2)
    with left:
        st.markdown("### Confusion matrix")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"], metrics["target_names"]), clear_figure=True)
    with right:
        st.markdown("### Classification report")
        st.dataframe(pd.DataFrame(metrics["classification_report"]).T, use_container_width=True)
    st.info("Metrics are calculated on the held-out test set. They describe predictive performance, not the reasons behind individual predictions.")


@st.cache_data(show_spinner="Calculating global explanations…")
def global_explanations(_model, X_test: pd.DataFrame, y_test: pd.Series, signature: str):
    del signature
    class_label = _model.classes_[1] if len(_model.classes_) > 1 else _model.classes_[0]
    shap_values = calculate_shap_values(_model, X_test, class_label=class_label)
    return shap_values, get_shap_feature_importance(shap_values, X_test.columns.tolist()), calculate_permutation_importance(_model, X_test, y_test)


def global_page(bundle: dict) -> None:
    st.title("Global Explainability")
    st.write("These views summarize which features influence the model across the held-out test observations.")
    model, split = bundle["model"], bundle["split"]
    signature = f"{bundle['signature']}:{len(split.X_test)}:{repr(model.get_params())}"
    shap_values, shap_importance, permutation = global_explanations(model, split.X_test, split.y_test, signature)
    class_index = int(shap_values["class_index"])
    class_name = bundle["target_names"][class_index]
    left, right = st.columns(2)
    with left:
        st.markdown("### SHAP summary")
        st.pyplot(plot_shap_summary(shap_values, split.X_test), clear_figure=True)
    with right:
        st.markdown("### SHAP feature importance")
        st.pyplot(plot_feature_importance(shap_importance, "Mean absolute SHAP value"), clear_figure=True)
    st.caption(f"SHAP contribution values explain the model output for class **{class_name}**. These are model attributions, not causal effects.")
    st.markdown("### Permutation importance")
    st.pyplot(plot_feature_importance(permutation, "Mean accuracy decrease"), clear_figure=True)
    st.caption("Features are shuffled in the held-out test set; larger accuracy decreases indicate greater model reliance. Error bars show variation across repeats.")
    st.markdown("### Most influential features")
    st.dataframe(shap_importance.head(10), use_container_width=True, hide_index=True)


def local_page(bundle: dict) -> None:
    st.title("Local Explainability")
    st.write("Select a held-out observation to inspect the features contributing to its predicted class.")
    model, split = bundle["model"], bundle["split"]
    row_index = st.selectbox("Test observation", range(len(split.X_test)), format_func=lambda value: f"Test row {value}")
    row = split.X_test.iloc[[row_index]]
    predicted_class = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    class_index = int(list(model.classes_).index(predicted_class))
    st.metric("Predicted class", bundle["target_names"][class_index])
    st.markdown("#### Class probabilities")
    st.dataframe(
        pd.DataFrame({"Class": bundle["target_names"], "Predicted probability": probabilities}).sort_values("Predicted probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Selected feature values"):
        st.dataframe(row.T.rename(columns={row.index[0]: "Value"}), use_container_width=True)
    shap_values = calculate_shap_values(model, row, class_label=predicted_class)
    st.markdown("### SHAP waterfall")
    st.pyplot(create_shap_explanation(shap_values, row, 0), clear_figure=True)
    st.caption(f"Each bar represents a feature's contribution to the selected class output ({bundle['target_names'][class_index]}).")
    st.markdown("### LIME explanation")
    try:
        lime_items = build_lime_explanation(model, split.X_train, row.iloc[0].to_numpy(), bundle["feature_names"], predicted_class, bundle["target_names"])
        st.pyplot(plot_lime_contributions(lime_items), clear_figure=True)
        st.dataframe(pd.DataFrame(lime_items, columns=["Feature", "Contribution"]), use_container_width=True, hide_index=True)
    except ImportError:
        st.warning("LIME is unavailable. Install dependencies from requirements.txt.")
    contributions = shap_values["values"][0]
    strongest = pd.Series(abs(contributions), index=bundle["feature_names"]).nlargest(min(3, len(bundle["feature_names"]))).index.tolist()
    st.info("Largest-magnitude SHAP contributions for this prediction: " + ", ".join(strongest))


def feature_behaviour_page(bundle: dict) -> None:
    st.title("Feature Behaviour")
    st.write("PDP shows an average model response as a feature varies; ICE shows individual response curves.")
    model, X_test = bundle["model"], bundle["split"].X_test
    feature_names = bundle["feature_names"]
    selected_feature = st.selectbox("Feature", feature_names)
    class_options = bundle["target_names"] if len(bundle["target_names"]) > 2 else [bundle["target_names"][1]]
    class_name = st.selectbox("Class probability to inspect", class_options)
    class_index = bundle["target_names"].index(class_name)
    feature_index = feature_names.index(selected_feature)
    left, right = st.columns(2)
    with left:
        st.markdown("### Partial dependence")
        figure, axis = plt.subplots(figsize=(7, 4.5))
        kwargs = {"target": class_index} if len(bundle["target_names"]) > 2 else {}
        PartialDependenceDisplay.from_estimator(model, X_test, [feature_index], ax=axis, response_method="predict_proba", **kwargs)
        axis.set_title(f"Average response: P({class_name})")
        axis.set_ylabel(f"Predicted probability of {class_name}")
        st.pyplot(figure, clear_figure=True)
    with right:
        st.markdown("### ICE curves")
        figure, axis = plt.subplots(figsize=(7, 4.5))
        kwargs = {"target": class_index} if len(bundle["target_names"]) > 2 else {}
        PartialDependenceDisplay.from_estimator(model, X_test, [feature_index], kind="individual", subsample=min(50, len(X_test)), ax=axis, response_method="predict_proba", **kwargs)
        axis.set_title(f"Individual curves: P({class_name})")
        axis.set_ylabel(f"Predicted probability of {class_name}")
        st.pyplot(figure, clear_figure=True)
    st.info("These plots describe model behaviour, not causality. With correlated features, varying one feature independently can create unrealistic combinations.")


def comparison_page() -> None:
    st.title("XAI Comparison")
    comparison = pd.DataFrame([
        ["SHAP", "Yes", "Yes", "Yes", "Yes"],
        ["LIME", "No", "Yes", "Yes", "No"],
        ["Permutation importance", "Yes", "No", "Yes", "No"],
        ["Partial dependence", "Yes", "No", "No", "Yes"],
        ["ICE", "No", "Individual curves", "No", "Yes"],
    ], columns=["Method", "Global", "Local attribution / individual behaviour", "Feature importance", "Feature-effect analysis"])
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.write("SHAP attributes a model output to features, LIME approximates a local decision, permutation importance measures reliance, PDP averages feature effects, and ICE reveals variation between observations. Each method answers a different question.")


def main() -> None:
    bundle = st.session_state.get("analysis_bundle")
    pages = {"Home / Upload": home_page}
    if bundle:
        pages.update({
            "Dataset Explorer": lambda: dataset_page(bundle),
            "Model Performance": lambda: performance_page(bundle),
            "Global Explainability": lambda: global_page(bundle),
            "Local Explainability": lambda: local_page(bundle),
            "Feature Behaviour": lambda: feature_behaviour_page(bundle),
            "XAI Comparison": comparison_page,
        })
    st.sidebar.title("XAI Dashboard")
    selected_page = st.sidebar.radio("Navigate", list(pages))
    if bundle:
        st.sidebar.caption(f"Dataset: {bundle['file_name']}")
        if st.sidebar.button("Upload a different dataset"):
            st.session_state.pop("analysis_bundle", None)
            st.rerun()
    pages[selected_page]()
    st.sidebar.divider()
    st.sidebar.caption("Educational use only · Not a diagnostic tool")


if __name__ == "__main__":
    main()
