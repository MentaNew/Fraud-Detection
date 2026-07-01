"""API tests use FastAPI dependency overrides to inject a pre-trained tiny
pipeline instead of loading from disk — so no model file is needed on CI.
"""
import pytest
from fastapi.testclient import TestClient

from src.explainer import SHAPExplainer

VALID_FEATURES = [0.0] * 30


@pytest.fixture(scope="module")
def client(trained_pipeline):
    from src.api import app, get_explainer, get_model, get_threshold

    explainer = SHAPExplainer(trained_pipeline)
    app.dependency_overrides[get_model] = lambda: trained_pipeline
    app.dependency_overrides[get_threshold] = lambda: 0.5
    app.dependency_overrides[get_explainer] = lambda: explainer

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# --- /health ---

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


# --- /predict ---

def test_predict_valid(client):
    resp = client.post("/predict", json={"features": VALID_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_proba" in body
    assert "fraud_pred" in body
    assert "threshold_used" in body
    assert 0.0 <= body["fraud_proba"] <= 1.0
    assert body["fraud_pred"] in (0, 1)


def test_predict_wrong_feature_count_rejected(client):
    resp = client.post("/predict", json={"features": [0.0] * 10})
    assert resp.status_code == 422


def test_predict_empty_features_rejected(client):
    resp = client.post("/predict", json={"features": []})
    assert resp.status_code == 422


# --- /predict_batch ---

def test_predict_batch_returns_list(client):
    payload = {
        "transactions": [
            {"features": VALID_FEATURES},
            {"features": [1.0] * 30},
        ]
    }
    resp = client.post("/predict_batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2


def test_predict_batch_invalid_transaction_rejected(client):
    payload = {"transactions": [{"features": [0.0] * 5}]}
    resp = client.post("/predict_batch", json=payload)
    assert resp.status_code == 422


# --- /explain ---

def test_explain_valid(client):
    resp = client.post("/explain", json={"features": VALID_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_proba" in body
    assert "top_contributions" in body
    contribs = body["top_contributions"]
    assert len(contribs) > 0
    assert "feature" in contribs[0]
    assert "shap_value" in contribs[0]


def test_explain_wrong_feature_count_rejected(client):
    resp = client.post("/explain", json={"features": [0.0] * 15})
    assert resp.status_code == 422
