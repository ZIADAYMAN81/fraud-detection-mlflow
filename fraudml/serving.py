"""Scoring service: online features -> fraud probability -> decision, with per-transaction reasons."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .online import HistoryStore, OnlineFeatures
from .package import load_bundle

# Human-readable names for the reasons returned with each decision.
READABLE = {
    "shipping_distance_km": "shipping distance", "security_checks_failed": "failed security checks",
    "avs_match": "address verification", "cvv_result": "CVV check", "three_ds_flag": "3-D Secure",
    "account_age_days": "account age", "country_mismatch": "card country differs from purchase country",
    "amount": "amount", "log_amount": "amount", "amount_to_profile_avg": "amount vs customer's usual amount",
    "amount_to_prior_mean": "amount vs customer's previous purchases", "prior_transactions": "number of previous purchases",
    "prior_mean_amount": "average previous purchase", "hours_since_previous": "time since previous purchase",
    "prior_country_changes": "previous country changes", "promo_used": "promotion used", "channel": "channel",
    "merchant_category": "merchant category", "country": "country", "hour": "time of day", "is_night": "night-time",
    "weekday": "day of week", "avg_amount_user": "customer's usual amount",
}


def _source_column(name: str) -> str:
    """'num__amount' -> 'amount', 'cat__country_FR' -> 'country', 'num__missingindicator_x' -> 'x'."""
    kind, rest = name.split("__", 1)
    if kind == "cat":
        for col in ("merchant_category", "channel", "country"):
            if rest.startswith(col + "_"):
                return col
    return rest.replace("missingindicator_", "")


class FraudService:
    def __init__(self, bundle: str | Path, store: HistoryStore | None = None):
        self.pipe, self.meta = load_bundle(bundle)
        self.online = OnlineFeatures(store or HistoryStore())
        self.columns = self.meta["numeric_features"] + self.meta["categorical_features"]
        names = self.pipe.named_steps["prep"].get_feature_names_out()
        self.sources = [_source_column(n) for n in names]
        self.threshold = float(self.meta["thresholds"][self.meta["default_threshold"]])

    def _frame(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)[self.columns]

    def reasons(self, X: pd.DataFrame, top: int = 3) -> list[list[dict]]:
        """Per-transaction SHAP contributions from XGBoost (log-odds), summed back to the original columns."""
        import xgboost as xgb

        Z = self.pipe.named_steps["prep"].transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else Z
        contribs = self.pipe.named_steps["model"].get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)[:, :-1]
        out = []
        for row in contribs:
            agg = defaultdict(float)
            for src, v in zip(self.sources, row):
                agg[src] += float(v)
            ranked = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top]
            out.append([{"feature": k, "description": READABLE.get(k, k), "effect": "raises risk" if v > 0 else "lowers risk",
                         "contribution": round(v, 3)} for k, v in ranked])
        return out

    def score(self, txns: list[dict], update_history: bool = True, explain: bool = True) -> list[dict]:
        t0 = time.perf_counter()
        feats = self.online.features_batch(txns, update=update_history)
        X = self._frame(feats)
        probs = self.pipe.predict_proba(X)[:, 1]
        reasons = self.reasons(X) if explain else [None] * len(txns)
        ms = (time.perf_counter() - t0) * 1000 / max(len(txns), 1)
        return [{"transaction_id": t.get("transaction_id"), "fraud_probability": round(float(p), 6),
                 "decision": "review" if p >= self.threshold else "approve", "threshold": round(self.threshold, 6),
                 "reasons": r, "model_version": self.meta.get("registry", {}).get("version"),
                 "latency_ms": round(ms, 2)} for t, p, r in zip(txns, probs, reasons)]

    def info(self) -> dict:
        m = self.meta
        return {"model": m["model_name"], "version": m.get("registry", {}).get("version"), "created": m["created"],
                "train_period": m["train_period"], "test_period": m["test_period"], "thresholds": m["thresholds"],
                "default_threshold": m["default_threshold"], "test_metrics": m["test_metrics"][m["default_threshold"]],
                "libraries": m["libraries"], "customers_in_history": self.online.store.users()}
