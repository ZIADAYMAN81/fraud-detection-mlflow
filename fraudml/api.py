"""Fraud scoring API.

    FRAUD_BUNDLE=models/fraud_xgboost FRAUD_HISTORY_DB=history.db uvicorn fraudml.api:app --port 8000

POST /score         one transaction  -> probability, decision (approve / review), top reasons
POST /score/batch   up to 1,000 transactions
GET  /model         model version, thresholds, test metrics, library versions
GET  /health
Every request is validated against the schema below; invalid input gets HTTP 422 with the reason.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .online import HistoryStore
from .serving import FraudService

log = logging.getLogger("fraudml")
MAX_BATCH = 1000


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(ge=0)
    user_id: int = Field(ge=0)
    account_age_days: int = Field(ge=0, le=36500)
    avg_amount_user: float = Field(ge=0)
    amount: float = Field(gt=0, le=1_000_000)
    country: str = Field(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2, e.g. FR")
    bin_country: str = Field(pattern=r"^[A-Z]{2}$", description="country that issued the card")
    channel: Literal["web", "app"]
    merchant_category: Literal["travel", "fashion", "electronics", "grocery", "gaming"]
    promo_used: Literal[0, 1]
    avs_match: Literal[0, 1]
    cvv_result: Literal[0, 1]
    three_ds_flag: Literal[0, 1]
    transaction_time: datetime
    shipping_distance_km: float = Field(ge=0, le=25_000)

    @field_validator("transaction_time")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("transaction_time needs a timezone, e.g. 2024-09-01T12:00:00Z")
        return v


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH)


def create_app(service: FraudService | None = None) -> FastAPI:
    app = FastAPI(title="Fraud scoring", version="1.0.0")
    state = {"service": service, "started": time.time(), "scored": 0, "flagged": 0}

    def svc() -> FraudService:
        if state["service"] is None:
            bundle = os.environ.get("FRAUD_BUNDLE", "models/fraud_xgboost")
            if not os.path.isfile(os.path.join(bundle, "model.joblib")):
                raise HTTPException(503, f"model bundle not found at {bundle}; set FRAUD_BUNDLE")
            state["service"] = FraudService(bundle, HistoryStore(os.environ.get("FRAUD_HISTORY_DB", ":memory:")))
        return state["service"]

    def run(txns: list[Transaction], update_history: bool, explain: bool) -> list[dict]:
        rows = [t.model_dump() for t in txns]
        res = svc().score(rows, update_history=update_history, explain=explain)
        state["scored"] += len(res)
        state["flagged"] += sum(r["decision"] == "review" for r in res)
        for r in res:
            log.info("scored transaction=%s probability=%.4f decision=%s", r["transaction_id"], r["fraud_probability"], r["decision"])
        return res

    @app.post("/score")
    def score(txn: Transaction, update_history: bool = True, explain: bool = True):
        return run([txn], update_history, explain)[0]

    @app.post("/score/batch")
    def score_batch(batch: Batch, update_history: bool = True, explain: bool = False):
        return {"results": run(batch.transactions, update_history, explain)}

    @app.get("/model")
    def model():
        return svc().info()

    @app.get("/health")
    def health():
        s = state["service"]
        return {"status": "ok", "model_loaded": s is not None, "uptime_seconds": round(time.time() - state["started"], 1),
                "scored": state["scored"], "flag_rate": round(state["flagged"] / state["scored"], 4) if state["scored"] else None}

    return app


app = create_app()
