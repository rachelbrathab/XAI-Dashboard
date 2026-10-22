"""Shared fixtures for the dashboard test suite."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_dataset, split_dataset
from src.model import train_model


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def split(dataset):
    features, target, _, _ = dataset
    return split_dataset(features, target)


@pytest.fixture(scope="session")
def trained_model(split):
    return train_model(split.X_train, split.y_train)
