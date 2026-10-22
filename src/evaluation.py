"""Evaluation metrics for the trained classifier."""

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model: ClassifierMixin, X_test: pd.DataFrame, y_test: pd.Series, target_names: list[str]) -> dict:
    predictions = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions, target_names=target_names, output_dict=True, zero_division=0),
        "target_names": target_names,
        "metric_average": "macro",
    }