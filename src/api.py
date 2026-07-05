import json
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import CHECKPOINTS_DIR, METADATA_PATH
from src.explainer import SHAPExplainer

logger = logging.getLogger(__name__)

_MODEL_PATH = CHECKPOINTS_DIR / "best_tuned_model.pkl"
N_FEATURES = 30

_resources: dict = {}


## Context manager lifespan (given to instantiate app) that handles expensive startup/shutdown
# resources (DB connections, ML models, thread pools, etc.)
@asynccontextmanager
async def lifespan(app: FastAPI):
    if _MODEL_PATH.exists():
        _resources["model"] = joblib.load(_MODEL_PATH)
        _resources["threshold"] = 0.5
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                meta = json.load(f)
            _resources["threshold"] = meta.get("optimal_threshold", 0.5)
        _resources["explainer"] = SHAPExplainer(_resources["model"])
        logger.info("Model loaded — threshold: %.4f", _resources["threshold"])
    else:
        logger.warning("Model not found at %s — endpoints will return 503", _MODEL_PATH)
        _resources["model"] = None
        _resources["threshold"] = 0.5
        _resources["explainer"] = None
    yield
    _resources.clear()


app = FastAPI(
    title="Fraud Detection API",
    version="2.0.0",
    description="Real-time fraud scoring with LightGBM + SHAP explanations.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependency providers — override in tests via app.dependency_overrides
# ---------------------------------------------------------------------------


def get_model():
    model = _resources.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def get_threshold() -> float:
    return _resources.get("threshold", 0.5)


def get_explainer() -> SHAPExplainer:
    explainer = _resources.get("explainer")
    if explainer is None:
        raise HTTPException(status_code=503, detail="Explainer not available")
    return explainer


ModelDep = Annotated[object, Depends(get_model)]
ThresholdDep = Annotated[float, Depends(get_threshold)]
ExplainerDep = Annotated[SHAPExplainer, Depends(get_explainer)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Transaction(BaseModel):
    features: list[float] = Field(
        ..., description="Exactly 30 transaction features: Time, V1–V28, Amount"
    )

    @field_validator("features")
    @classmethod
    def check_feature_count(cls, v: list[float]) -> list[float]:
        if len(v) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {len(v)}")
        return v


class Prediction(BaseModel):
    fraud_proba: float
    fraud_pred: int
    threshold_used: float


class BatchRequest(BaseModel):
    transactions: list[Transaction]


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class Explanation(BaseModel):
    fraud_proba: float
    fraud_pred: int
    threshold_used: float
    top_contributions: list[FeatureContribution]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "model": "LightGBM fraud classifier",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    ready = _resources.get("model") is not None
    return {"status": "healthy" if ready else "degraded", "model_loaded": ready}


@app.post("/predict", response_model=Prediction, summary="Score a single transaction")
def predict(tx: Transaction, model: ModelDep, threshold: ThresholdDep) -> Prediction:
    X = np.array(tx.features, dtype=float).reshape(1, -1)
    proba = float(model.predict_proba(X)[0, 1])
    return Prediction(
        fraud_proba=round(proba, 6),
        fraud_pred=int(proba >= threshold),
        threshold_used=round(threshold, 6),
    )


@app.post(
    "/predict_batch",
    response_model=list[Prediction],
    summary="Score multiple transactions in one call",
)
def predict_batch(req: BatchRequest, model: ModelDep, threshold: ThresholdDep) -> list[Prediction]:
    X = np.array([tx.features for tx in req.transactions], dtype=float)
    probas = model.predict_proba(X)[:, 1]
    return [
        Prediction(
            fraud_proba=round(float(p), 6),
            fraud_pred=int(p >= threshold),
            threshold_used=round(threshold, 6),
        )
        for p in probas
    ]


@app.post(
    "/explain",
    response_model=Explanation,
    summary="Score a transaction and return top SHAP feature contributions",
)
def explain(
    tx: Transaction,
    model: ModelDep,
    threshold: ThresholdDep,
    explainer: ExplainerDep,
) -> Explanation:
    X = np.array(tx.features, dtype=float).reshape(1, -1)
    proba = float(model.predict_proba(X)[0, 1])
    contributions = explainer.explain(X, top_k=10)[0]
    return Explanation(
        fraud_proba=round(proba, 6),
        fraud_pred=int(proba >= threshold),
        threshold_used=round(threshold, 6),
        top_contributions=[FeatureContribution(**c) for c in contributions],
    )
