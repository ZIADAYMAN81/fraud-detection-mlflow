"""Feature engineering. Every behavioural feature uses only transactions that happened *before* the current one."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TIME

NUMERIC = [
    "account_age_days", "avg_amount_user", "amount", "log_amount", "amount_to_profile_avg", "shipping_distance_km",
    "promo_used", "avs_match", "cvv_result", "three_ds_flag", "security_checks_failed", "country_mismatch",
    "hour", "is_night", "weekday", "prior_transactions", "prior_mean_amount", "amount_to_prior_mean",
    "hours_since_previous", "prior_country_changes",
]
CATEGORICAL = ["channel", "merchant_category", "country"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with engineered columns. Rows must be sorted by time (load_transactions does this)."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["amount_to_profile_avg"] = out["amount"] / out["avg_amount_user"].replace(0, np.nan)
    out["security_checks_failed"] = 3 - out[["avs_match", "cvv_result", "three_ds_flag"]].sum(axis=1)
    out["country_mismatch"] = (out["country"] != out["bin_country"]).astype(int)
    out["hour"] = out[TIME].dt.hour
    out["is_night"] = out["hour"].between(0, 5).astype(int)
    out["weekday"] = out[TIME].dt.weekday

    g = out.groupby("user_id", sort=False)
    out["prior_transactions"] = g.cumcount()
    prior_sum = g["amount"].cumsum() - out["amount"]
    out["prior_mean_amount"] = prior_sum / out["prior_transactions"].replace(0, np.nan)
    out["amount_to_prior_mean"] = out["amount"] / out["prior_mean_amount"]
    previous_time = g[TIME].shift(1)
    out["hours_since_previous"] = (out[TIME] - previous_time).dt.total_seconds() / 3600
    changed = (out["country"] != g["country"].shift(1)) & previous_time.notna()
    out["prior_country_changes"] = changed.astype(int).groupby(out["user_id"]).cumsum() - changed.astype(int)
    # total_transactions_user is deliberately not used: it counts the user's transactions over the whole
    # period, including the future, so it leaks information a live system would not have.
    return out
