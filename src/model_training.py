import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CHECKPOINTS_DIR

logger = logging.getLogger(__name__)


def compute_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the classification threshold that maximises F1 on the positive class."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve yields n+1 precision/recall values but only n thresholds
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[np.argmax(f1)])


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str | Path = CHECKPOINTS_DIR / "baseline.pkl",
) -> tuple[Pipeline, float]:
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    threshold = compute_optimal_threshold(np.array(y_test), proba)
    preds = (proba >= threshold).astype(int)

    logger.info("ROC-AUC: %.4f | Optimal F1 threshold: %.4f", auc, threshold)
    logger.info("\n%s", classification_report(y_test, preds))

    joblib.dump(pipe, save_path)
    logger.info("Baseline model saved to %s", save_path)
    return pipe, auc
