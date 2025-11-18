# src/tuning.py
import os
import joblib
import optuna
import lightgbm as lgb
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

from src.data_pipeline import load_data
from src.utils import setup_mlflow
import mlflow
import mlflow.sklearn


def make_pipeline(trial):
    """Build a Pipeline(scaler + LGBM) with trial-suggested hyperparams."""
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
    }
    clf = lgb.LGBMClassifier(**params)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    return pipe


def objective(trial, X, y):
    """Optuna objective: CV AUC of pipeline."""
    pipe = make_pipeline(trial)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # cross_val_score calls fit/predict_proba on each fold with the pipeline (scales inside CV → no leakage)
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        error_score="raise",
    )
    return float(np.mean(scores))


def tune(n_trials=40, data_path = "/Users/elmehdielkasmi/Desktop/ML_DL_projects/fraud_detection/creditcard.csv", save_path="checkpoints/best_tuned_model.pkl"):
    # 1) Load once
    X, y = load_data(data_path)

    # 2) Hold-out test split (NOT used during CV tuning)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) Optuna study
    setup_mlflow()
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, X_train, y_train), n_trials=n_trials)

    print("✅ Best CV AUC:", study.best_value)
    print("✅ Best Params:", study.best_params)

    # 4) Retrain best pipeline on full training split and evaluate on held-out test
    # Rebuild pipeline with best params
    trial = optuna.trial.FixedTrial(study.best_params)
    best_pipe = make_pipeline(trial)
    best_pipe.fit(X_train, y_train)

    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    print(f"🎯 Test AUC with best pipeline: {test_auc:.4f}")

    # 5) Save and log
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_pipe, save_path)
    print(f"💾 Saved pipeline to: {save_path}")

    with mlflow.start_run(run_name="optuna_best_retrain"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.sklearn.log_model(best_pipe, artifact_path="model")  # pipeline logging

    return test_auc, save_path


if __name__ == "__main__":
    tune(n_trials=40,
         data_path="/Users/elmehdielkasmi/Desktop/ML_DL_projects/fraud_detection/creditcard.csv",
         save_path="checkpoints/best_tuned_model.pkl")