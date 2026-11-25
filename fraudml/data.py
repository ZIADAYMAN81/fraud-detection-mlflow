"""Loading the two datasets and the time-based split."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "is_fraud"
TIME = "transaction_time"
TRANSACTION_COLUMNS = [
    "transaction_id", "user_id", "account_age_days", "total_transactions_user", "avg_amount_user", "amount",
    "country", "bin_country", "channel", "merchant_category", "promo_used", "avs_match", "cvv_result",
    "three_ds_flag", TIME, "shipping_distance_km", TARGET,
]

DIABETES_TARGET = "Outcome"
# In the Pima data a zero in these columns means "not measured", not a real value.
DIABETES_ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in TRANSACTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"transactions file is missing columns: {missing}")
    df[TIME] = pd.to_datetime(df[TIME], utc=True)
    return df.sort_values([TIME, "transaction_id"]).reset_index(drop=True)


def load_diabetes(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[DIABETES_ZERO_AS_MISSING] = df[DIABETES_ZERO_AS_MISSING].replace(0, np.nan)
    return df


def time_split(df: pd.DataFrame, valid_start: str, test_start: str):
    """Train on the past, tune on the next period, test on the most recent one (no shuffling across time)."""
    t = df[TIME]
    valid_start, test_start = pd.Timestamp(valid_start, tz="UTC"), pd.Timestamp(test_start, tz="UTC")
    if not valid_start < test_start:
        raise ValueError("validation must start before the test period")
    train = df[t < valid_start]
    valid = df[(t >= valid_start) & (t < test_start)]
    test = df[t >= test_start]
    return train, valid, test
