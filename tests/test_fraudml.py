import json

import numpy as np
import pandas as pd
import pytest

from fraudml import data, evaluate, features
from fraudml.__main__ import main


def transactions(n=3000, seed=0):
    """Synthetic data with the real schema: fraud is more likely with failed checks and long shipping distances."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 300 * 24 * 3600, n)), unit="s")
    avs, cvv = rng.integers(0, 2, n), rng.integers(0, 2, n)
    distance = rng.exponential(300, n)
    logit = -5 + 1.5 * (1 - avs) + 1.5 * (1 - cvv) + distance / 400
    df = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1), "user_id": rng.integers(1, 200, n),
        "account_age_days": rng.integers(1, 2000, n), "total_transactions_user": 10,
        "avg_amount_user": rng.uniform(20, 300, n), "amount": rng.exponential(150, n),
        "country": rng.choice(["FR", "US", "DE"], n), "bin_country": rng.choice(["FR", "US", "DE"], n),
        "channel": rng.choice(["web", "app"], n), "merchant_category": rng.choice(["travel", "gaming"], n),
        "promo_used": rng.integers(0, 2, n), "avs_match": avs, "cvv_result": cvv,
        "three_ds_flag": rng.integers(0, 2, n), "transaction_time": times.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shipping_distance_km": distance, "is_fraud": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
    })
    return df


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "transactions.csv"
    transactions().to_csv(path, index=False)
    return path


def test_load_sorts_by_time_and_validates(csv, tmp_path):
    df = data.load_transactions(csv)
    assert df[data.TIME].is_monotonic_increasing
    bad = tmp_path / "bad.csv"
    transactions(10).drop(columns=["amount"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="amount"):
        data.load_transactions(bad)


def test_behaviour_features_use_only_the_past():
    t = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 12:00", "2024-01-02 12:00"], utc=True)
    df = pd.DataFrame({"user_id": [7, 7, 7], "amount": [100.0, 300.0, 50.0], "avg_amount_user": [100.0] * 3,
                       "country": ["FR", "US", "US"], "bin_country": ["FR", "FR", "US"],
                       "avs_match": [1, 0, 1], "cvv_result": [1, 0, 1], "three_ds_flag": [1, 1, 0],
                       data.TIME: t})
    f = features.add_features(df)
    assert f["prior_transactions"].tolist() == [0, 1, 2]
    assert np.isnan(f.loc[0, "prior_mean_amount"])
    assert f.loc[1, "prior_mean_amount"] == 100 and f.loc[2, "prior_mean_amount"] == 200
    assert f.loc[1, "amount_to_prior_mean"] == 3
    assert np.isnan(f.loc[0, "hours_since_previous"]) and f.loc[1, "hours_since_previous"] == 2
    assert f["prior_country_changes"].tolist() == [0, 0, 1]
    assert f["country_mismatch"].tolist() == [0, 1, 0]
    assert f["security_checks_failed"].tolist() == [0, 2, 1]


def test_features_do_not_change_when_future_rows_are_removed(csv):
    df = data.load_transactions(csv)
    full = features.add_features(df).iloc[:1000]
    cut = features.add_features(df.iloc[:1000])
    pd.testing.assert_frame_equal(full[features.NUMERIC], cut[features.NUMERIC])


def test_time_split_is_ordered_and_disjoint(csv):
    df = data.load_transactions(csv)
    train, valid, test = data.time_split(df, "2024-07-01", "2024-09-01")
    assert len(train) + len(valid) + len(test) == len(df)
    assert train[data.TIME].max() < valid[data.TIME].min() <= valid[data.TIME].max() < test[data.TIME].min()
    with pytest.raises(ValueError):
        data.time_split(df, "2024-09-01", "2024-07-01")


def test_diabetes_zeros_become_missing(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"Pregnancies": [0, 2], "Glucose": [0, 120], "BloodPressure": [70, 0], "SkinThickness": [0, 20],
                  "Insulin": [0, 80], "BMI": [30.1, 0], "DiabetesPedigreeFunction": [0.5, 0.2], "Age": [30, 40],
                  "Outcome": [0, 1]}).to_csv(path, index=False)
    d = data.load_diabetes(path)
    assert d["Pregnancies"].tolist() == [0, 2]  # a real zero stays a zero
    assert d["Glucose"].isna().tolist() == [True, False] and d["BMI"].isna().tolist() == [False, True]


def test_threshold_and_cost_helpers():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    assert evaluate.best_f1_threshold(y, s) == pytest.approx(0.8)
    amounts = np.array([10, 10, 10, 100, 200])
    assert evaluate.cost(y, s >= 0.8, amounts, review_cost=5) == 10
    assert evaluate.cost(y, s >= 0.95, amounts, review_cost=5) == 300
    assert evaluate.best_cost_threshold(y, s, amounts, review_cost=5) <= 0.8
    m = evaluate.metrics(y, s, 0.8, amounts)
    assert m["precision"] == 1 and m["recall"] == 1 and m["fraud_value_caught_share"] == 1


def test_recall_at_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(100)]
    s = np.r_[rng.random(1000), rng.random(100) + 1]  # every fraud scores above every legitimate one
    assert evaluate.recall_at_fpr(y, s, 0.01) == 1.0


def test_end_to_end_with_mlflow(csv, tmp_path):
    out = tmp_path / "results"
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    summary = main(["transactions", "--data", str(csv), "--valid-start", "2024-07-01", "--test-start",
                    "2024-09-01", "--device", "cpu", "--only", "Logistic Regression (class-weighted)", "XGBoost",
                    "--out", str(out), "--tracking-uri", uri])
    saved = json.loads((out / "transactions_results.json").read_text())
    assert {r["model"] for r in saved["results"]} == {"Logistic Regression (class-weighted)", "XGBoost"}
    assert all(r["roc_auc"] > 0.7 for r in summary["results"])  # the planted signal is found
    assert (out / "precision_recall_curves.png").exists() and (out / "shap_importance.png").exists()

    import mlflow

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["fraud-detection"])
    assert len(runs) == 2 and runs["metrics.pr_auc"].notna().all()
