"""Dataset loading and reproducible train/test preparation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass(frozen=True)
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.copy()
    features.columns = [str(name) for name in features.columns]
    return features, target, [str(name) for name in dataset.feature_names], [str(name) for name in dataset.target_names]


def split_dataset(features: pd.DataFrame, target: pd.Series) -> DatasetSplit:
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)

def prepare_uploaded_dataset(
    dataframe: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    """Validate a user-uploaded table and prepare numeric predictors for classification.

    Returns X, encoded y, feature names, human-readable class names, and dropped
    non-numeric predictor column names. Missing rows are removed only from the
    selected target and numeric predictors.
    """
    if dataframe.empty:
        raise ValueError("The uploaded CSV contains no rows.")
    if len(dataframe.columns) < 2:
        raise ValueError("The CSV must contain at least one feature column and one target column.")
    if target_column not in dataframe.columns:
        raise ValueError("Select a target column that exists in the uploaded dataset.")

    frame = dataframe.copy()
    frame.columns = [str(column) if str(column).strip() else f"column_{i + 1}" for i, column in enumerate(frame.columns)]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Column names must be unique. Duplicates found: {duplicates}")
    if target_column not in frame.columns:
        raise ValueError("The selected target column name is invalid after cleaning column names.")

    target = frame[target_column]
    numeric_candidates = frame.drop(columns=[target_column]).select_dtypes(include=["number", "bool"]).copy()
    dropped = [str(col) for col in frame.columns if col != target_column and col not in numeric_candidates.columns]
    if numeric_candidates.shape[1] == 0:
        raise ValueError("No numeric feature columns were found. Keep at least one numeric predictor besides the target.")

    clean = numeric_candidates.copy()
    clean[target_column] = target
    clean = clean.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=list(numeric_candidates.columns) + [target_column])
    if clean.empty:
        raise ValueError("No complete rows remain after removing missing or infinite values.")

    y_raw = clean[target_column].astype(str)
    class_counts = y_raw.value_counts()
    if len(class_counts) < 2:
        raise ValueError("The target column must contain at least two classes for classification.")
    if class_counts.min() < 2:
        raise ValueError("Each target class must contain at least two rows so a stratified train/test split is possible.")
    if len(clean) < 10:
        raise ValueError("Please upload at least 10 complete rows for a meaningful train/test split.")

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y_raw)
    X = clean[numeric_candidates.columns].astype(float).reset_index(drop=True)
    y = pd.Series(encoded, name=target_column, dtype=int)
    feature_names = [str(name) for name in X.columns]
    class_names = [str(name) for name in encoder.classes_]
    return X, y, feature_names, class_names, dropped
