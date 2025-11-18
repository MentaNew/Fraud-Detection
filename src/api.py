from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# load artifacts
model = joblib.load("checkpoints/best_tuned_model.pkl")

app = FastAPI(title="Fraud Detection API")

class Transaction(BaseModel):
    features: list  # length must match model input dim (after scaling step)

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(tx: Transaction):
    X = np.array(tx.features, dtype=float).reshape(1, -1)
    proba = model.predict_proba(X)[:,1].item()
    pred = int(proba > 0.5)
    return {"fraud_proba": proba, "fraud_pred": pred}