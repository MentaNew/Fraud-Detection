Fraud Detection System (End-to-End ML + MLOps)

Overview

This project implements a complete machine learning pipeline for detecting fraudulent credit card transactions.

It integrates data preprocessing, model training, hyperparameter tuning, experiment tracking, and real-time inference 

through an API — following modern MLOps best practices.

⸻

Features
	•	Data Pipeline: Efficient preprocessing and train/test splitting from the raw dataset.

	•	Model Training: LightGBM classifier with imbalanced learning (class_weight='balanced').

	•	Hyperparameter Tuning: Optuna integration with automatic MLflow experiment tracking.

	•	Scalable Pipeline: Scikit-learn Pipeline combines preprocessing + model for reproducibility.

	•	Model Serving: FastAPI backend for real-time prediction requests.

	•	Monitoring: ROC-AUC score evaluation and MLflow metric logging.

⸻

## Project Structure

```
fraud_detection/
│
├── src/
│   ├── data_pipeline.py             → Data loading and preprocessing
│   ├── model_training.py            → Baseline LightGBM training
│   ├── tuning.py                    → Optuna hyperparameter optimization
│   ├── api.py                       → FastAPI app for real-time inference
│   ├── utils.py                     → MLflow setup and helper functions
│   └── __init__.py
│
├── checkpoints/
│   └── best_tuned_model.pkl         → Serialized pipeline (scaler + model)
│
├── train.py                         → Retrain final model with tuned params
├── requirements.txt
└── README.md
```

⸻

Installation

# Clone repo
git clone https://github.com/MentaNew/fraud-detection.git
cd fraud-detection

# Create env
conda create -n fraud python=3.10
conda activate fraud

# Install dependencies
pip install -r requirements.txt


⸻

🧠 Training & Tuning

1. Train baseline

python train.py

2. Run Optuna tuning

python -m src.tuning

This performs random search over key LightGBM parameters and logs results to MLflow.

3. Launch MLflow UI

mlflow ui

View experiments at http://127.0.0.1:5000￼.

⸻

🧪 API Deployment

Start the server

uvicorn src.api:app --reload

Test the endpoint

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.5, ...]}'

Example output

{"fraud_proba": 0.9999999989, "fraud_pred": 1}

✅ Fraud example correctly predicted
✅ Normal transaction returns near-zero probability

⸻

📊 Results

ROC-AUC (final) : 0.9646

Precision (fraud): 0.83

Recall (fraud) :	0.86

Accuracy (overall)	: 0.999
---- 

Tech Stack

	•	Python 3.10

	•	LightGBM

	•	Optuna

	•	Scikit-learn

	•	FastAPI

	•	MLflow

	•	Joblib

⸻

Key Takeaways

	•	Built a production-ready pipeline combining ML and MLOps principles.

	•	Automated tuning + experiment tracking via Optuna and MLflow.

	•	Deployed model as a real-time prediction API.

	•	Achieved 99.9% accuracy and AUC ≈ 0.96 on real-world imbalanced data.

⸻

👤 Author

El Mehdi EL KASMI
ML & Data Science — Mines Paris / HEC Paris