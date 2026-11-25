"""Online feature computation for one transaction at a time.

Training builds the behavioural features with pandas over the whole history (`features.add_features`). A live
API sees one transaction at a time, so it keeps a small per-customer state instead:

    transactions so far, their total amount, time and country of the last one, country changes so far

`compute` returns exactly the features `add_features` would give the same transaction, and `observe` updates the
state afterwards. `tests/test_online.py` checks the two paths agree row by row on real-shaped data, which is the
usual place where a model that worked offline silently degrades in production (training/serving skew).
"""
from __future__ import annotations

import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data import TIME
from .features import CATEGORICAL, NUMERIC


@dataclass
class UserState:
    n: int = 0
    amount_sum: float = 0.0
    last_time: float | None = None  # POSIX seconds
    last_country: str | None = None
    country_changes: int = 0


class HistoryStore:
    """Per-customer history in SQLite (a file for persistence, or ":memory:"). Thread-safe."""

    def __init__(self, path: str = ":memory:"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("""CREATE TABLE IF NOT EXISTS user_history (
            user_id INTEGER PRIMARY KEY, n INTEGER NOT NULL, amount_sum REAL NOT NULL,
            last_time REAL, last_country TEXT, country_changes INTEGER NOT NULL)""")
        self.conn.commit()

    def get(self, user_id: int) -> UserState:
        row = self.conn.execute("SELECT n, amount_sum, last_time, last_country, country_changes FROM user_history "
                                "WHERE user_id = ?", (int(user_id),)).fetchone()
        return UserState(*row) if row else UserState()

    def put(self, user_id: int, s: UserState) -> None:
        self.conn.execute("INSERT OR REPLACE INTO user_history VALUES (?, ?, ?, ?, ?, ?)",
                          (int(user_id), s.n, s.amount_sum, s.last_time, s.last_country, s.country_changes))

    def users(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM user_history").fetchone()[0]

    def bootstrap(self, history: pd.DataFrame) -> int:
        """Load the state implied by past transactions (sorted by time), e.g. everything before go-live."""
        with self.lock:
            for _, r in history.sort_values([TIME, "transaction_id"]).iterrows():
                self.put(r["user_id"], advance(self.get(r["user_id"]), r))
            self.conn.commit()
        return len(history)


def _seconds(t) -> float:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    return ts.timestamp()


def advance(s: UserState, txn) -> UserState:
    """State after observing `txn` (mirrors the cumulative sums and shifts in `add_features`)."""
    changes = s.country_changes + (1 if s.n > 0 and txn["country"] != s.last_country else 0)
    return UserState(n=s.n + 1, amount_sum=s.amount_sum + float(txn["amount"]), last_time=_seconds(txn[TIME]),
                     last_country=txn["country"], country_changes=changes)


def compute(txn: dict, s: UserState) -> dict:
    """All model inputs for one transaction, given the customer's state before it."""
    t = pd.Timestamp(txn[TIME])
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    amount, profile_avg = float(txn["amount"]), float(txn["avg_amount_user"])
    prior_mean = s.amount_sum / s.n if s.n else math.nan
    f = {
        "account_age_days": float(txn["account_age_days"]), "avg_amount_user": profile_avg, "amount": amount,
        "log_amount": float(np.log1p(amount)), "amount_to_profile_avg": amount / profile_avg if profile_avg else math.nan,
        "shipping_distance_km": float(txn["shipping_distance_km"]), "promo_used": int(txn["promo_used"]),
        "avs_match": int(txn["avs_match"]), "cvv_result": int(txn["cvv_result"]), "three_ds_flag": int(txn["three_ds_flag"]),
        "security_checks_failed": 3 - int(txn["avs_match"]) - int(txn["cvv_result"]) - int(txn["three_ds_flag"]),
        "country_mismatch": int(txn["country"] != txn["bin_country"]),
        "hour": t.hour, "is_night": int(0 <= t.hour <= 5), "weekday": t.weekday(),
        "prior_transactions": s.n, "prior_mean_amount": prior_mean,
        "amount_to_prior_mean": amount / prior_mean if s.n else math.nan,
        "hours_since_previous": (t.timestamp() - s.last_time) / 3600 if s.last_time is not None else math.nan,
        "prior_country_changes": s.country_changes,
        "channel": txn["channel"], "merchant_category": txn["merchant_category"], "country": txn["country"],
    }
    return {k: f[k] for k in NUMERIC + CATEGORICAL}


class OnlineFeatures:
    """compute() then observe(), under one lock so concurrent requests for a customer stay consistent."""

    def __init__(self, store: HistoryStore):
        self.store = store

    def features(self, txn: dict, update: bool = True) -> dict:
        return self.features_batch([txn], update)[0]

    def features_batch(self, txns: list[dict], update: bool = True) -> list[dict]:
        """Features for transactions in arrival order. One lock and one database commit for the whole batch:
        committing after every transaction made a 300,000-transaction replay disk-bound."""
        out = []
        with self.store.lock:
            for txn in txns:
                s = self.store.get(txn["user_id"])
                out.append(compute(txn, s))
                if update:
                    self.store.put(txn["user_id"], advance(s, txn))
            if update:
                self.store.conn.commit()
        return out


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
