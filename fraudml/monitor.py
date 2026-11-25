"""Drift and performance monitoring against the training period.

    python -m fraudml.monitor --bundle models/fraud_xgboost --data data/transactions.csv --out results/monitoring

For every month after training it reports:
  * Population Stability Index (PSI) of each feature and of the model's scores against the training distribution
    (bin edges come from the bundle, so monitoring uses exactly the reference the model was trained on);
  * the model's PR-AUC, recall and precision at the deployed threshold, and the flag rate, once labels exist.
Common rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift (investigate), > 0.25 significant shift (act).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features

EPS = 1e-4
STABLE, MODERATE = 0.10, 0.25


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """sum((a - e) * ln(a / e)) over bins, with empty bins floored at a small share."""
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_shares(values: np.ndarray, edges: list) -> np.ndarray:
    v = values[~np.isnan(values)]
    counts = np.histogram(np.clip(v, edges[0], edges[-1]), np.asarray(edges))[0]
    return counts / max(counts.sum(), 1)


def categorical_psi(reference: dict, values: pd.Series) -> float:
    cats = sorted(set(reference) | set(values.unique()))
    cur = values.value_counts(normalize=True)
    return psi([reference.get(c, 0.0) for c in cats], [float(cur.get(c, 0.0)) for c in cats])


def status(value: float) -> str:
    return "stable" if value < STABLE else ("moderate" if value < MODERATE else "significant")


def monitor(df: pd.DataFrame, meta: dict, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score

    prof = meta["reference_profile"]
    after = df[data.TIME] > pd.Timestamp(meta["train_period"][1], tz="UTC") + pd.Timedelta(days=1)
    months = df.loc[after, data.TIME].dt.strftime("%Y-%m")
    report = {"reference_period": meta["train_period"], "threshold": threshold, "months": {}}
    for month in sorted(months.unique()):
        idx = months.index[months == month]
        part, s = df.loc[idx], scores[df.index.get_indexer(idx)]
        feat_psi = {c: round(psi(p["shares"], numeric_shares(part[c].astype(float).to_numpy(), p["edges"])), 4)
                    for c, p in prof["numeric"].items()}
        feat_psi.update({c: round(categorical_psi(ref, part[c]), 4) for c, ref in prof["categorical"].items()})
        score_psi = psi(meta["reference_score_profile"]["shares"], numeric_shares(s, meta["reference_score_profile"]["edges"]))
        y = part[data.TARGET].to_numpy()
        flagged = s >= threshold
        worst = sorted(feat_psi.items(), key=lambda kv: -kv[1])[:5]
        report["months"][month] = {
            "transactions": int(len(part)), "fraud_rate": round(float(y.mean()), 4),
            "score_psi": round(score_psi, 4), "score_status": status(score_psi),
            "max_feature_psi": worst[0][1], "features_needing_attention": [c for c, v in feat_psi.items() if v >= STABLE],
            "top_shifted_features": dict(worst), "feature_psi": feat_psi,
            "pr_auc": round(float(average_precision_score(y, s)), 4) if 0 < y.sum() < len(y) else None,
            "recall": round(float(flagged[y == 1].mean()), 4) if y.sum() else None,
            "precision": round(float(y[flagged].mean()), 4) if flagged.any() else None,
            "flag_rate": round(float(flagged.mean()), 4)}
    return report


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = list(report["months"])
    feats = list(next(iter(report["months"].values()))["feature_psi"])
    grid = np.array([[report["months"][m]["feature_psi"][f] for m in months] for f in feats])
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(0.25, grid.max()), aspect="auto")
    ax.set_xticks(range(len(months)), months)
    ax.set_yticks(range(len(feats)), feats, fontsize=8)
    for i in range(len(feats)):
        for j in range(len(months)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > 0.15 else "black")
    ax.set_title("Feature drift vs training (PSI)\n< 0.10 stable, 0.10-0.25 moderate, > 0.25 significant", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    from .package import load_bundle

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", default="models/fraud_xgboost")
    p.add_argument("--data", default="data/transactions.csv")
    p.add_argument("--out", default="results/monitoring")
    p.add_argument("--scored", default="data/scored_after_training.csv",
                   help="row-level scores for the dashboard (derived from the dataset, so kept out of git)")
    args = p.parse_args(argv)
    pipe, meta = load_bundle(args.bundle)
    df = features.add_features(data.load_transactions(args.data))
    scores = pipe.predict_proba(df[meta["numeric_features"] + meta["categorical_features"]])[:, 1]
    threshold = meta["thresholds"][meta["default_threshold"]]
    report = monitor(df, meta, scores, threshold)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=1))
    plot(report, out / "feature_psi.png")
    after = df[data.TIME] > pd.Timestamp(meta["train_period"][1], tz="UTC") + pd.Timedelta(days=1)
    scored = df.loc[after, ["transaction_id", data.TIME, "amount", "merchant_category", "channel", "country",
                            "shipping_distance_km", "security_checks_failed", data.TARGET]].copy()
    scored["score"] = scores[after.to_numpy()]
    Path(args.scored).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scored, index=False)
    for m, r in report["months"].items():
        print(f"{m}: score PSI {r['score_psi']:.3f} ({r['score_status']}), max feature PSI {r['max_feature_psi']:.3f}, "
              f"PR-AUC {r['pr_auc']}, recall {r['recall']}, precision {r['precision']}, flag rate {r['flag_rate']}")
    return report


if __name__ == "__main__":
    main()
