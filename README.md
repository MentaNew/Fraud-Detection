# Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-inference%20API-009688?logo=fastapi)
![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20tuning-blueviolet)
![SHAP](https://img.shields.io/badge/SHAP-explainability-orange)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ED?logo=docker)
![CI](https://img.shields.io/github/actions/workflow/status/MentaNew/fraud-detection/ci.yml?label=CI&logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end ML system for detecting fraudulent credit card transactions — from raw data through Bayesian hyperparameter search to a containerised REST API with per-prediction SHAP explanations and full MLflow experiment tracking.

---

## Results

> Tuned model — 40 Optuna trials optimising **PR-AUC** (correct metric for imbalanced data), evaluated on a held-out 20% test set.

| Metric | Score |
|---|---|
| ROC-AUC | **0.9780** |
| PR-AUC (Avg. Precision) | **0.8842** |
| Precision — fraud class | **0.9326** |
| Recall — fraud class | **0.8469** |
| F1 — fraud class | **0.8877** |
| Accuracy | 0.9996 |
| Optimal threshold | 0.7665 |

**Confusion matrix (56,962 test transactions, 98 fraud):**

|  | Predicted legit | Predicted fraud |
|---|---|---|
| **Actual legit** | 56,858 (TN) | 6 (FP) |
| **Actual fraud** | 15 (FN) | 83 (TP) |

> Full scores and best hyperparameters are persisted in [`results.json`](results.json).

---

## Architecture

```
creditcard.csv
      │
      ▼
data_pipeline.py          load · stratified split · StandardScaler (fit on train only)
      │
      ├──► train.py        baseline training entry point
      │         │
      │         ▼
      │   model_training.py   LightGBM Pipeline + compute_optimal_threshold()
      │
      └──► tuning.py       Optuna HPO — 40 trials · 3-fold CV · PR-AUC objective
                │           saves best_tuned_model.pkl + metadata.json
                └──► mlflow tracks every trial

checkpoints/best_tuned_model.pkl
checkpoints/metadata.json  ← optimal threshold, AUC, best params
      │
      ▼
explainer.py → SHAPExplainer    TreeExplainer on LightGBM; exact Shapley values
      │
      ▼
api.py (FastAPI)
  ├── POST /predict              single transaction → fraud_proba + fraud_pred
  ├── POST /predict_batch        batch scoring in one call
  ├── POST /explain              score + top-10 signed SHAP contributions
  └── GET  /health               liveness probe
      │
      ▼
docker-compose.yml → containerised API (checkpoints mounted as volume)
```

---

## Project Structure

```
fraud_detection/
├── src/
│   ├── config.py          → Path constants (ROOT, DATA_PATH, CHECKPOINTS_DIR, METADATA_PATH)
│   ├── data_pipeline.py   → load_data · get_splits · preprocess_data
│   ├── model_training.py  → train_baseline · compute_optimal_threshold
│   ├── tuning.py          → Optuna HPO (PR-AUC objective) + MLflow logging
│   ├── explainer.py       → SHAPExplainer (SHAP TreeExplainer wrapper)
│   ├── api.py             → FastAPI: /predict · /predict_batch · /explain · /health
│   └── utils.py           → setup_logging · setup_mlflow
├── tests/
│   ├── conftest.py        → synthetic fixtures (no real CSV needed in CI)
│   ├── test_api.py        → all 4 endpoints · input validation
│   ├── test_data_pipeline.py
│   └── test_model_training.py
├── notebooks/
│   ├── 01_eda.ipynb       → class imbalance · feature distributions · correlations
│   └── 02_results.ipynb   → confusion matrix · ROC · PR curve · SHAP plots
├── .github/workflows/
│   └── ci.yml             → lint (ruff) + pytest on every push/PR
├── checkpoints/           → model artifacts (gitignored)
├── data/                  → dataset (gitignored — see below)
├── mlruns/                → MLflow experiment logs (gitignored)
├── results.json           → final scores + best hyperparameters
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml         → ruff + pytest config
├── .pre-commit-config.yaml
├── train.py               → baseline training entry point
├── Makefile
├── requirements.txt
└── requirements-dev.txt
```

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/MentaNew/fraud-detection.git
cd fraud-detection

conda create -n fraud python=3.10
conda activate fraud

make install          # runtime deps
make install-dev      # + pytest, ruff, pre-commit
```

### 2. Download the dataset

Download [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle and place it at:

```
data/creditcard.csv
```

---

## Usage

### Train baseline model

```bash
make train
```

### Run Optuna hyperparameter tuning

```bash
make tune
```

Runs 40 Optuna trials with 3-fold stratified CV optimising **PR-AUC** (not ROC-AUC — chosen because it focuses on the minority fraud class on a 500:1 imbalanced dataset). Logs all runs to MLflow and saves:
- `checkpoints/best_tuned_model.pkl` — sklearn Pipeline (scaler → LightGBM)
- `checkpoints/metadata.json` — optimal threshold, AUC, best hyperparameters

### Run tests

```bash
make test
```

17 tests covering the data pipeline, model training, and all API endpoints. Tests use synthetic data — no `creditcard.csv` needed.

### Inspect experiments with MLflow

```bash
make mlflow
# open http://127.0.0.1:5000
```

### Start the inference API

```bash
make api
# or: docker compose up (after make tune)
```

**Single prediction:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36,
                    0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.02, 0.40,
                    0.25, -0.02, 0.28, -0.11, 0.07, -0.37, -0.23, -0.42, -0.12, 149.62]}'
```

```json
{"fraud_proba": 0.000043, "fraud_pred": 0, "threshold_used": 0.7665}
```

**Batch prediction:**

```bash
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [{"features": [...]}, {"features": [...]}]}'
```

**Explanation (SHAP):**

```bash
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

```json
{
  "fraud_proba": 0.991,
  "fraud_pred": 1,
  "threshold_used": 0.7665,
  "top_contributions": [
    {"feature": "V14", "shap_value": -2.31},
    {"feature": "V4",  "shap_value":  1.87},
    {"feature": "V12", "shap_value": -1.54}
  ]
}
```

Auto-generated OpenAPI docs available at `http://localhost:8000/docs`.

### Run with Docker

```bash
make tune            # train first — artifacts go to ./checkpoints/
make docker-build
make docker-run      # mounts ./checkpoints into the container
```

---

## Best Hyperparameters

Found by Optuna after 40 trials (best CV PR-AUC: **0.8570**):

| Parameter | Value |
|---|---|
| `n_estimators` | 791 |
| `learning_rate` | 0.0399 |
| `num_leaves` | 185 |
| `max_depth` | 10 |
| `min_child_samples` | 137 |
| `subsample` | 0.858 |
| `colsample_bytree` | 0.706 |

---

## Tech Stack

| Tool | Role |
|---|---|
| **LightGBM** | Gradient boosting classifier; `class_weight="balanced"` handles 500:1 imbalance |
| **Optuna** | Bayesian HPO (TPE sampler); optimises PR-AUC not ROC-AUC |
| **scikit-learn Pipeline** | Scaler → model in one artifact; prevents data leakage in CV |
| **SHAP TreeExplainer** | Exact per-prediction feature contributions; required for regulated-domain deployment |
| **MLflow** | Tracks params, metrics, and model artifact for every tuning trial |
| **FastAPI + Pydantic** | Typed REST API; validates feature count; auto-generates `/docs` |
| **Docker + Compose** | Containerised API; model artifacts mounted as volume (image is model-agnostic) |
| **pytest + httpx** | 17 tests on synthetic data — CI runs without the real dataset |
| **ruff** | Lint + format; enforced via pre-commit hook and CI |
| **GitHub Actions** | Lint + test gate on every push and PR |

---

## Author

**El Mehdi EL KASMI**
