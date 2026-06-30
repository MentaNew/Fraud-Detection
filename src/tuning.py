import json
import logging
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CHECKPOINTS_DIR, DATA_PATH, METADATA_PATH
from src.data_pipeline import load_data
from src.model_training import compute_optimal_threshold
from src.utils import setup_mlflow

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def make_pipeline(trial: optuna.Trial) -> Pipeline:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary",
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**params)
    return Pipeline([("scaler", StandardScaler()), ("model", clf)])


def objective(trial: optuna.Trial, X, y) -> float:
    pipe = make_pipeline(trial)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1, error_score="raise")
    return float(np.mean(scores))


def tune(
    n_trials: int = 40,
    data_path: Path | str = DATA_PATH,
    save_path: Path | str = CHECKPOINTS_DIR / "best_tuned_model.pkl",
) -> tuple[float, str]:
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    setup_mlflow()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=n_trials)

    logger.info("Best CV AUC : %.4f", study.best_value)
    logger.info("Best params : %s", study.best_params)

    best_pipe = make_pipeline(optuna.trial.FixedTrial(study.best_params))
    best_pipe.fit(X_train, y_train)

    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    threshold = compute_optimal_threshold(np.array(y_test), y_proba)

    logger.info("Test AUC    : %.4f | Optimal threshold : %.4f", test_auc, threshold)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_pipe, save_path)
    logger.info("Saved pipeline to: %s", save_path)

    metadata = {
        "optimal_threshold": threshold,
        "test_auc": test_auc,
        "best_params": study.best_params,
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", METADATA_PATH)

    with mlflow.start_run(run_name="optuna_best_retrain"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("optimal_threshold", threshold)
        mlflow.sklearn.log_model(best_pipe, artifact_path="model")

    return test_auc, str(save_path)


if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    tune()
