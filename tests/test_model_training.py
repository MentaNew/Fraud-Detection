import numpy as np

from src.data_pipeline import get_splits
from src.model_training import compute_optimal_threshold, train_baseline


def test_compute_optimal_threshold_range():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.7])
    t = compute_optimal_threshold(y_true, y_proba)
    assert 0.0 <= t <= 1.0


def test_compute_optimal_threshold_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = compute_optimal_threshold(y_true, y_proba)
    # Any threshold between 0.3 and 0.7 is optimal; just verify range
    assert 0.3 <= t <= 0.7


def test_train_baseline_returns_valid_auc(sample_df, tmp_path):
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]
    X_train, X_test, y_train, y_test = get_splits(X, y)
    save_path = tmp_path / "baseline.pkl"
    _, auc = train_baseline(X_train, y_train, X_test, y_test, save_path=str(save_path))
    assert 0.0 <= auc <= 1.0


def test_train_baseline_saves_artifact(sample_df, tmp_path):
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]
    X_train, X_test, y_train, y_test = get_splits(X, y)
    save_path = tmp_path / "baseline.pkl"
    train_baseline(X_train, y_train, X_test, y_test, save_path=str(save_path))
    assert save_path.exists()
