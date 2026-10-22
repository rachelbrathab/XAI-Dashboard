"""Tests for dataset loading and splitting."""


def test_dataset_shape_and_schema(dataset):
    features, target, feature_names, target_names = dataset

    assert features.shape == (569, 30)
    assert len(feature_names) == 30
    assert all(type(name) is str for name in feature_names)
    assert target.shape == (569,)
    assert target.notna().all()
    assert len(target_names) == 2
    assert all(type(name) is str for name in target_names)


def test_stratified_split_is_reproducible(dataset):
    from src.data_loader import split_dataset

    features, target, _, _ = dataset
    first = split_dataset(features, target)
    second = split_dataset(features, target)

    assert first.X_train.shape == (455, 30)
    assert first.X_test.shape == (114, 30)
    assert first.y_train.shape == (455,)
    assert first.y_test.shape == (114,)
    assert first.X_train.equals(second.X_train)
    assert first.X_test.equals(second.X_test)
    assert first.y_train.equals(second.y_train)
    assert first.y_test.equals(second.y_test)


def test_prepare_uploaded_dataset_supports_numeric_features_and_text_target():
    import pandas as pd
    from src.data_loader import prepare_uploaded_dataset

    frame = pd.DataFrame({
        "measurement_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "measurement_b": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2],
        "group": ["low", "high"] * 6,
        "notes": ["x"] * 12,
    })
    X, y, feature_names, target_names, dropped = prepare_uploaded_dataset(frame, "group")
    assert X.shape == (12, 2)
    assert feature_names == ["measurement_a", "measurement_b"]
    assert target_names == ["high", "low"]
    assert set(y.unique()) == {0, 1}
    assert dropped == ["notes"]


def test_prepare_uploaded_dataset_rejects_single_class_target():
    import pandas as pd
    import pytest
    from src.data_loader import prepare_uploaded_dataset

    frame = pd.DataFrame({"feature": range(10), "target": ["only"] * 10})
    with pytest.raises(ValueError, match="at least two classes"):
        prepare_uploaded_dataset(frame, "target")
