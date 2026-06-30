"""Shared pytest fixtures using synthetic data.

The real creditcard.csv is gitignored and not available in CI.
All fixtures build a small in-memory dataset that mirrors its structure
(30 numeric features, binary Class label) so every test runs without the
original file.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_FEATURES = 30
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
N_SAMPLES = 300
N_FRAUD = 15  # 5% — higher than real data so CV folds always have positive cases


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_SAMPLES, N_FEATURES))
    df = pd.DataFrame(data, columns=FEATURE_NAMES)
    labels = np.zeros(N_SAMPLES, dtype=int)
    labels[:N_FRAUD] = 1
    rng.shuffle(labels)
    df["Class"] = labels
    return df


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory, sample_df) -> Path:
    p = tmp_path_factory.mktemp("data") / "creditcard.csv"
    sample_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def trained_pipeline(sample_df) -> Pipeline:
    """Tiny but real LightGBM pipeline trained on synthetic data."""
    X = sample_df.drop("Class", axis=1)
    y = sample_df["Class"]
    model = lgb.LGBMClassifier(
        n_estimators=20,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X, y)
    return pipe
