"""Fraud model monitoring dashboard.

    python -m fraudml.package && python -m fraudml.monitor      # produce the files this page reads
    streamlit run dashboard/app.py

Reads the model bundle metadata, the drift report and the row-level scores for the months after training.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:  # pandas 3 stores text as Arrow large strings, which older Streamlit versions cannot render
    pd.set_option("future.infer_string", False)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "models/fraud_xgboost/metadata.json"
DRIFT = ROOT / "results/monitoring/drift_report.json"
SCORED = ROOT / "data/scored_after_training.csv"

st.set_page_config(page_title="Fraud model monitoring", layout="wide")
missing = [p.name for p in (META, DRIFT, SCORED) if not p.exists()]
if missing:
    st.error(f"Missing {missing}. Run `python -m fraudml.package` and `python -m fraudml.monitor` first.")
    st.stop()

meta = json.loads(META.read_text())
drift = json.loads(DRIFT.read_text())
df = pd.read_csv(SCORED, parse_dates=["transaction_time"])
df["month"] = df["transaction_time"].dt.strftime("%Y-%m")
default_t = float(meta["thresholds"][meta["default_threshold"]])
review_cost = float(meta["review_cost"])

st.title("Fraud model monitoring")
reg = meta.get("registry", {})
st.caption(f"Model {meta['model_name']} version {reg.get('version', '?')} · trained on {meta['train_period'][0]} to "
           f"{meta['train_period'][1]} · promotion decision: {reg.get('reason', 'n/a')}")

# ---- decision threshold --------------------------------------------------------------------------------
st.subheader("Decision threshold")
t = st.slider("Flag transactions for review when the fraud probability is at least", 0.01, 0.99, round(default_t, 2), 0.01,
              help=f"Deployed value {default_t:.3f}: the threshold that minimised cost on the validation month "
                   f"(missed fraud costs its amount, each review costs ${review_cost:.0f}).")
flag = df["score"] >= t
fraud = df["is_fraud"] == 1
missed = df.loc[fraud & ~flag, "amount"].sum()
cost = missed + review_cost * flag.sum()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fraud caught (recall)", f"{(flag & fraud).sum() / fraud.sum():.1%}")
c2.metric("Alerts that are fraud (precision)", f"{(flag & fraud).sum() / max(flag.sum(), 1):.1%}")
c3.metric("Transactions flagged", f"{flag.mean():.2%}")
c4.metric("Cost: missed fraud + reviews", f"${cost:,.0f}")
c5.metric("Cost with no model", f"${df.loc[fraud, 'amount'].sum():,.0f}")

grid = np.round(np.linspace(0.02, 0.98, 49), 2)
curve = pd.DataFrame({"threshold": grid,
                      "cost": [df.loc[fraud & (df.score < g), "amount"].sum() + review_cost * (df.score >= g).sum() for g in grid]})
fig = px.line(curve, x="threshold", y="cost", title="Total cost by threshold (all months after training)")
fig.add_vline(x=default_t, line_dash="dash", annotation_text="deployed", annotation_position="top left")
fig.add_vline(x=t, line_dash="dot", annotation_text="selected", annotation_position="bottom right")
st.plotly_chart(fig, use_container_width=True)

# ---- performance and drift by month --------------------------------------------------------------------
st.subheader("Performance and drift by month")
months = pd.DataFrame([{"month": m, "transactions": r["transactions"], "fraud rate": r["fraud_rate"], "PR-AUC": r["pr_auc"],
                        "recall": r["recall"], "precision": r["precision"], "flag rate": r["flag_rate"],
                        "score PSI": r["score_psi"], "score drift": r["score_status"], "max feature PSI": r["max_feature_psi"]}
                       for m, r in drift["months"].items()])
st.dataframe(months, use_container_width=True, hide_index=True)
st.caption("Population Stability Index against the training period: below 0.10 stable, 0.10–0.25 moderate, above 0.25 significant.")
psi = pd.DataFrame({m: r["feature_psi"] for m, r in drift["months"].items()})
st.plotly_chart(px.imshow(psi, text_auto=".3f", aspect="auto", color_continuous_scale="Blues", zmin=0, zmax=0.25,
                          title="Feature drift (PSI) by month"), use_container_width=True)

# ---- risk breakdown and highest-risk transactions ---------------------------------------------------------
left, right = st.columns(2)
by_cat = df.groupby("merchant_category").agg(flag_rate=("score", lambda s: (s >= t).mean()), fraud_rate=("is_fraud", "mean")).reset_index()
bars = go.Figure([go.Bar(name="flag rate", x=by_cat["merchant_category"], y=by_cat["flag_rate"]),
                  go.Bar(name="actual fraud rate", x=by_cat["merchant_category"], y=by_cat["fraud_rate"])])
bars.update_layout(barmode="group", title="Flag rate vs actual fraud rate by merchant category", yaxis_tickformat=".0%")
left.plotly_chart(bars, use_container_width=True)
hist = go.Figure([go.Histogram(x=df.loc[df.is_fraud == k, "score"], name=name, nbinsx=50, opacity=0.6)
                  for k, name in ((0, "legitimate"), (1, "fraud"))])
hist.update_layout(barmode="overlay", title="Score distribution (log scale)", yaxis_type="log", xaxis_title="fraud probability")
right.plotly_chart(hist, use_container_width=True)
st.subheader("Highest-risk transactions")
st.dataframe(df.sort_values("score", ascending=False).head(25)[["transaction_id", "transaction_time", "amount", "merchant_category",
             "channel", "country", "shipping_distance_km", "security_checks_failed", "score", "is_fraud"]],
             use_container_width=True, hide_index=True)
