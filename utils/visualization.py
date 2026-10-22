"""Reusable Matplotlib and Plotly visualizations."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def plot_target_distribution(target: pd.Series, target_names: list[str]):
    counts = target.value_counts().rename_axis("class").reset_index(name="count")
    counts["class"] = counts["class"].map(dict(enumerate(target_names)))
    return px.bar(counts, x="class", y="count", title="Target class counts", labels={"class": "Target class", "count": "Observations"})


def plot_confusion_matrix(matrix, target_names):
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis, xticklabels=target_names, yticklabels=target_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    return figure


def plot_feature_importance(data: pd.DataFrame, x_label: str):
    figure, axis = plt.subplots(figsize=(8, 6))
    plot_data = data.head(12).sort_values("Importance")
    error = plot_data["ImportanceStd"] if "ImportanceStd" in plot_data else None
    axis.barh(plot_data["Feature"], plot_data["Importance"], xerr=error, color="#0f766e", capsize=3)
    axis.set_xlabel(x_label)
    axis.set_ylabel("")
    figure.tight_layout()
    return figure


def plot_shap_summary(shap_values: dict, data: pd.DataFrame):
    import shap

    shap.summary_plot(shap_values["values"], data, show=False, max_display=12, plot_size=(8, 6))
    return plt.gcf()


def plot_lime_contributions(items):
    labels, values = zip(*items)
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = ["#b91c1c" if value < 0 else "#0f766e" for value in values]
    axis.barh(list(labels)[::-1], list(values)[::-1], color=colors[::-1])
    axis.set_xlabel("LIME contribution")
    figure.tight_layout()
    return figure