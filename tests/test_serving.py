"""Packaging, promotion gate, drift maths, scoring service and API (small synthetic data, CPU, temporary MLflow)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudml import monitor
from fraudml.api import create_app
from fraudml.online import HistoryStore
from fraudml.package import build, should_promote
from fraudml.serving import FraudService
from tests.test_fraudml import transactions


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    d = tmp_path_factory.mktemp("fraud")
    path = d / "transactions.csv"
    transactions(4000, seed=1).to_csv(path, index=False)
    uri = f"sqlite:///{(d / 'mlflow.db').as_posix()}"
    meta = build(str(path), d / "bundle", valid_start="2024-07-01", test_start="2024-09-01", tracking_uri=uri, device="cpu")
    return d / "bundle", meta


def txn(**kw):
    base = {"transaction_id": 1, "user_id": 7, "account_age_days": 400, "avg_amount_user": 80.0, "amount": 95.0,
            "country": "FR", "bin_country": "FR", "channel": "web", "merchant_category": "travel", "promo_used": 0,
            "avs_match": 1, "cvv_result": 1, "three_ds_flag": 1, "transaction_time": "2024-10-01T12:00:00Z",
            "shipping_distance_km": 12.0}
    return {**base, **kw}


def test_bundle_contents_and_registry(bundle):
    path, meta = bundle
    assert (path / "model.joblib").exists()
    saved = json.loads((path / "metadata.json").read_text())
    assert "prior_transactions" not in saved["numeric_features"]  # drifting feature excluded from production
    assert 0 < saved["thresholds"]["cost_optimal"] < 1 and saved["registry"]["version"] == "1"
    assert saved["registry"]["promoted_to_champion"] and "max_score_psi" in saved["stability"]


def test_promotion_gate():
    stable = {"test_pr_auc": 0.85, "max_score_psi": 0.01}
    assert should_promote(stable, None)[0]
    assert not should_promote({"test_pr_auc": 0.95, "max_score_psi": 0.4}, None)[0]  # drifting challenger
    assert should_promote(stable, {"test_pr_auc": 0.86, "max_score_psi": 0.44})[0]  # drifting champion
    assert should_promote(stable, {"test_pr_auc": 0.853, "max_score_psi": 0.02})[0]  # within tolerance
    assert not should_promote(stable, {"test_pr_auc": 0.90, "max_score_psi": 0.02})[0]  # clearly worse


def test_psi():
    assert monitor.psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert monitor.psi([0.5, 0.5], [0.9, 0.1]) > 0.25
    edges = list(np.linspace(0, 1, 11))
    same = monitor.numeric_shares(np.random.default_rng(0).random(10_000), edges)
    assert monitor.psi([0.1] * 10, same) < 0.01
    assert monitor.categorical_psi({"a": 0.5, "b": 0.5}, __import__("pandas").Series(["a"] * 90 + ["c"] * 10)) > 0.25


def test_service_scores_explains_and_updates_history(bundle):
    store = HistoryStore()
    svc = FraudService(bundle[0], store)
    risky = svc.score([txn(avs_match=0, cvv_result=0, three_ds_flag=0, shipping_distance_km=2500.0)])[0]
    safe = svc.score([txn(transaction_id=2)])[0]
    assert risky["fraud_probability"] > safe["fraud_probability"]
    assert risky["decision"] in {"review", "approve"} and len(risky["reasons"]) == 3
    assert {r["effect"] for r in risky["reasons"]} <= {"raises risk", "lowers risk"}
    assert store.get(7).n == 2
    svc.score([txn(transaction_id=3)], update_history=False)
    assert store.get(7).n == 2


@pytest.fixture
def client(bundle):
    return TestClient(create_app(FraudService(bundle[0], HistoryStore())))


def test_api_score_batch_model_health(client):
    r = client.post("/score", json=txn())
    assert r.status_code == 200 and 0 <= r.json()["fraud_probability"] <= 1
    b = client.post("/score/batch", json={"transactions": [txn(transaction_id=i) for i in range(5)]})
    assert b.status_code == 200 and len(b.json()["results"]) == 5 and b.json()["results"][0]["reasons"] is None
    m = client.get("/model").json()
    assert m["model"] == "fraud-xgboost" and "cost_optimal" in m["thresholds"]
    h = client.get("/health").json()
    assert h["scored"] == 6 and h["model_loaded"]


@pytest.mark.parametrize("bad", [
    {"amount": -5}, {"channel": "phone"}, {"country": "France"}, {"avs_match": 2},
    {"transaction_time": "2024-10-01T12:00:00"}, {"unexpected_field": 1}, {"merchant_category": "cars"},
])
def test_api_rejects_invalid_transactions(client, bad):
    r = client.post("/score", json=txn(**bad))
    assert r.status_code == 422


def test_api_rejects_oversized_batch(client):
    r = client.post("/score/batch", json={"transactions": [txn(transaction_id=i) for i in range(1001)]})
    assert r.status_code == 422


def test_api_without_bundle_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_BUNDLE", str(tmp_path / "missing"))
    assert TestClient(create_app()).post("/score", json=txn()).status_code == 503
