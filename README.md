# Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-brightgreen)
![MLflow](https://img.shields.io/badge/MLflow-experiment%20tracking-blue?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-inference%20API-009688?logo=fastapi)
![Optuna](https://img.shields.io/badge/Optuna-hyperparameter%20tuning-blueviolet)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end ML pipeline for detecting fraudulent credit card transactions — from raw data through hyperparameter tuning to a production-ready REST API, with full MLflow experiment tracking.

---

## Results

| Metric | Score |
|---|---|
| ROC-AUC | **0.9646** |
| Precision (fraud) | 0.83 |
| Recall (fraud) | 0.86 |
| Overall accuracy | 0.999 |

---

## Project Structure

```
fraud_detection/
├── src/
│   ├── config.py            → Project root & path constants
│   ├── data_pipeline.py     → Data loading and preprocessing
│   ├── model_training.py    → LightGBM baseline training
│   ├── tuning.py            → Optuna hyperparameter optimization
│   ├── api.py               → FastAPI real-time inference server
│   ├── utils.py             → MLflow setup helpers
│   └── __init__.py
├── checkpoints/             → Saved model artifacts (gitignored)
├── data/                    → Dataset (gitignored — see below)
├── mlruns/                  → MLflow experiment logs (gitignored)
├── train.py                 → Entry point: train baseline model
├── Makefile                 → Common commands
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/MentaNew/fraud-detection.git
cd fraud-detection

conda create -n fraud python=3.10
conda activate fraud

make install
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
# or: python train.py --data data/creditcard.csv
```

### Run Optuna hyperparameter tuning

```bash
make tune
# or: python -m src.tuning
```

Runs 40 Optuna trials with 3-fold stratified CV, logs all runs to MLflow, and saves the best pipeline to `checkpoints/best_tuned_model.pkl`.

### Inspect experiments with MLflow

```bash
mlflow ui
# open http://127.0.0.1:5000
```

### Start the inference API

```bash
make api
# or: uvicorn src.api:app --reload
```

**Predict endpoint:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.5, ...]}'
```

```json
{"fraud_proba": 0.999999, "fraud_pred": 1}
```

---

## Tech Stack

- **LightGBM** — gradient boosting classifier with `class_weight="balanced"` for imbalanced data
- **Optuna** — Bayesian hyperparameter search with pruning
- **Scikit-learn** — `Pipeline` (scaler → model) for leak-free CV and deployment
- **MLflow** — parameter and metric logging across all tuning runs
- **FastAPI + Pydantic** — typed REST API with auto-generated OpenAPI docs at `/docs`

---

## Author

**El Mehdi EL KASMI**
