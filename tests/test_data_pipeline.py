import numpy as np
import pytest

from src.data_pipeline import get_splits, load_data, preprocess_data


def test_load_data_shape(sample_csv):
    X, y = load_data(sample_csv)
    assert X.shape == (300, 30)
    assert len(y) == 300


def test_load_data_fraud_present(sample_csv):
    _, y = load_data(sample_csv)
    assert y.sum() > 0, "Fixture must contain at least one fraud case"


def test_get_splits_sizes(sample_df):
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]
    X_train, X_test, y_train, y_test = get_splits(X, y, test_size=0.2)
    assert len(X_train) + len(X_test) == 300
    assert len(y_train) + len(y_test) == 300


def test_get_splits_stratified(sample_df):
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]
    X_train, X_test, y_train, y_test = get_splits(X, y, test_size=0.2)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    assert abs(train_rate - test_rate) < 0.02, "Stratification should keep fraud rates close"


def test_preprocess_no_leakage(sample_df):
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]
    X_train, X_test, y_train, _ = get_splits(X, y)
    X_train_s, X_test_s, scaler = preprocess_data(X_train, X_test)
    assert X_train_s.shape == X_train.shape
    assert X_test_s.shape == X_test.shape
    # Scaler mean must be fitted on train only — test mean won't be exactly 0
    assert not np.allclose(X_test_s.mean(axis=0), 0, atol=1e-6)
