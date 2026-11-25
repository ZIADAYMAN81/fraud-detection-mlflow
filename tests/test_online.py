"""The online features must equal the offline training features, row by row (no training/serving skew)."""
import numpy as np
import pandas as pd
import pytest

from fraudml import data, features
from fraudml.online import HistoryStore, OnlineFeatures, UserState, advance, compute
from tests.test_fraudml import transactions


@pytest.fixture
def df(tmp_path):
    path = tmp_path / "t.csv"
    raw = transactions(2000, seed=3)
    # Force awkward cases: same user twice in the same second, a zero profile average, country flips.
    raw.loc[5, ["user_id", "transaction_time"]] = [raw.loc[4, "user_id"], raw.loc[4, "transaction_time"]]
    raw.loc[7, "avg_amount_user"] = 0.0
    raw.to_csv(path, index=False)
    return data.load_transactions(path)


def test_online_equals_offline_for_every_row(df):
    offline = features.add_features(df)[features.NUMERIC + features.CATEGORICAL]
    online = OnlineFeatures(HistoryStore())
    rows = [online.features(r) for r in df.to_dict("records")]
    got = pd.DataFrame(rows, index=df.index)[features.NUMERIC + features.CATEGORICAL]
    for col in features.NUMERIC:
        np.testing.assert_allclose(got[col].astype(float), offline[col].astype(float), rtol=1e-9, atol=1e-9,
                                   equal_nan=True, err_msg=col)
    for col in features.CATEGORICAL:
        assert (got[col] == offline[col]).all(), col


def test_bootstrap_then_serve_matches_offline(df, tmp_path):
    """Go-live in the middle of the history: load the past into the store, then score the rest live."""
    cut = len(df) // 2
    offline = features.add_features(df).iloc[cut:]
    store = HistoryStore(str(tmp_path / "history.db"))
    store.bootstrap(df.iloc[:cut])
    online = OnlineFeatures(store)
    got = pd.DataFrame([online.features(r) for r in df.iloc[cut:].to_dict("records")], index=offline.index)
    np.testing.assert_allclose(got["prior_transactions"], offline["prior_transactions"])
    np.testing.assert_allclose(got["prior_mean_amount"], offline["prior_mean_amount"], equal_nan=True)
    np.testing.assert_allclose(got["prior_country_changes"], offline["prior_country_changes"])
    # The file-backed store survives a restart.
    assert HistoryStore(str(tmp_path / "history.db")).users() == store.users() > 0


def test_scoring_without_update_leaves_state_unchanged():
    store = HistoryStore()
    online = OnlineFeatures(store)
    txn = {"user_id": 1, "transaction_time": "2024-05-01T10:00:00Z", "amount": 50.0, "avg_amount_user": 40.0,
           "account_age_days": 100, "shipping_distance_km": 5.0, "promo_used": 0, "avs_match": 1, "cvv_result": 1,
           "three_ds_flag": 1, "country": "FR", "bin_country": "FR", "channel": "web", "merchant_category": "travel"}
    online.features(txn, update=False)
    assert store.get(1) == UserState()
    online.features(txn)
    assert store.get(1).n == 1


def test_state_transitions():
    s = UserState()
    first = {"amount": 10.0, "transaction_time": "2024-01-01T00:00:00Z", "country": "FR"}
    s = advance(s, first)
    s = advance(s, {**first, "amount": 30.0, "country": "US"})
    s = advance(s, {**first, "amount": 20.0, "country": "US"})
    assert (s.n, s.amount_sum, s.country_changes, s.last_country) == (3, 60.0, 1, "US")
    f = compute({**first, "user_id": 1, "avg_amount_user": 20.0, "account_age_days": 1, "shipping_distance_km": 1,
                 "promo_used": 0, "avs_match": 1, "cvv_result": 0, "three_ds_flag": 0, "bin_country": "FR",
                 "channel": "app", "merchant_category": "gaming", "amount": 40.0}, s)
    assert f["prior_mean_amount"] == 20.0 and f["amount_to_prior_mean"] == 2.0 and f["security_checks_failed"] == 2
