"""Command line.

    python -m fraudml transactions --data data/transactions.csv
    python -m fraudml diabetes --data data/diabetes.csv
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate  # noqa: E402

from . import data, evaluate, features, models  # noqa: E402


def setup_mlflow(uri: str, experiment: str) -> None:
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)


def run_transactions(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_mlflow(args.tracking_uri, "fraud-detection")

    df = features.add_features(data.load_transactions(args.data))
    train, valid, test = data.time_split(df, args.valid_start, args.test_start)
    X = {k: part[features.NUMERIC + features.CATEGORICAL] for k, part in
         {"train": train, "valid": valid, "test": test}.items()}
    y = {"train": train[data.TARGET], "valid": valid[data.TARGET], "test": test[data.TARGET]}
    pos_weight = float((y["train"] == 0).sum() / max((y["train"] == 1).sum(), 1))
    device = "cuda" if (args.device == "auto" and models.gpu_available()) or args.device == "cuda" else "cpu"
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}  fraud rate "
          f"{y['train'].mean():.2%}  device {device}")

    results, curves, fitted = [], {}, {}
    for name, pipe in models.candidates(features.NUMERIC, features.CATEGORICAL, pos_weight, device).items():
        if args.only and name not in args.only:
            continue
        t0 = time.time()
        with mlflow.start_run(run_name=name):
            pipe.fit(X["train"], y["train"])
            fit_seconds = time.time() - t0
            s_valid = pipe.predict_proba(X["valid"])[:, 1]
            s_test = pipe.predict_proba(X["test"])[:, 1]
            f1_t = evaluate.best_f1_threshold(y["valid"], s_valid)
            cost_t = evaluate.best_cost_threshold(y["valid"], s_valid, valid["amount"], args.review_cost)
            m = evaluate.metrics(y["test"], s_test, f1_t, test["amount"], args.review_cost)
            m_cost = evaluate.metrics(y["test"], s_test, cost_t, test["amount"], args.review_cost)
            m.update({"cost_optimal_" + k: v for k, v in m_cost.items()
                      if k in ("threshold", "precision", "recall", "cost", "fraud_value_caught_share", "flag_rate")})
            m["fit_seconds"] = round(fit_seconds, 1)
            mlflow.log_params({"model": name, "device": device if "XGBoost" in name else "cpu",
                               "train_rows": len(train), "valid_start": args.valid_start,
                               "test_start": args.test_start, "review_cost": args.review_cost})
            mlflow.log_metrics(m)
            if args.log_models:
                mlflow.sklearn.log_model(pipe, name="model")
        results.append({"model": name, **m})
        curves[name] = precision_recall_curve(y["test"], s_test)
        fitted[name] = pipe
        print(f"{name:40s} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  F1 {m['f1']:.3f}  "
              f"({fit_seconds:.0f}s)")

    table = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    best = table.iloc[0]["model"]
    plot_pr_curves(curves, y["test"].mean(), out / "precision_recall_curves.png")
    shap_top = explain(fitted["XGBoost"], X["test"], out / "shap_importance.png") if "XGBoost" in fitted else None
    summary = {
        "split": {"train": [str(train[data.TIME].min().date()), str(train[data.TIME].max().date())],
                  "validation": [str(valid[data.TIME].min().date()), str(valid[data.TIME].max().date())],
                  "test": [str(test[data.TIME].min().date()), str(test[data.TIME].max().date())],
                  "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
                  "test_fraud_rate": round(float(y["test"].mean()), 4)},
        "device": device,
        "review_cost": args.review_cost,
        "results": table.to_dict(orient="records"),
        "best_model": best,
        "shap_top_features": shap_top,
    }
    (out / "transactions_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_pr_curves(curves, base_rate, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (p, r, _) in curves.items():
        ax.plot(r, p, label=name, linewidth=2 if "XGBoost" in name else 1.2)
    ax.axhline(base_rate, color="grey", linestyle=":", label=f"Random ({base_rate:.1%} fraud)")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Test period: precision-recall curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def explain(pipe, X_test, path, sample: int = 5000) -> list:
    """SHAP values for the tree model on a sample of test transactions."""
    import shap

    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xs = X_test.sample(min(sample, len(X_test)), random_state=0)
    Z = prep.transform(Xs)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    model.set_params(device="cpu")
    values = shap.TreeExplainer(model).shap_values(Z)
    importance = pd.Series(np.abs(values).mean(axis=0), index=names).sort_values(ascending=False)
    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.index, top.values, color="#2f6fb0")
    ax.set_xlabel("Mean |SHAP value| (impact on the fraud score)")
    ax.set_title("What drives the XGBoost fraud score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [{"feature": k, "mean_abs_shap": round(float(v), 4)} for k, v in importance.head(10).items()]


def run_diabetes(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_mlflow(args.tracking_uri, "diabetes")
    df = data.load_diabetes(args.data)
    X, y = df.drop(columns=[data.DIABETES_TARGET]), df[data.DIABETES_TARGET]
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    rows = []
    numeric = list(X.columns)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced",
                                                n_jobs=-1, random_state=0),
        "XGBoost": models.xgboost(pos_weight, "cpu", 0).set_params(n_estimators=300, max_depth=3, learning_rate=0.05),
    }
    for name, model in candidates.items():
        pipe = Pipeline([("prep", models.preprocessor(numeric, [])), ("model", model)])
        res = cross_validate(pipe, X, y, cv=cv, scoring={"roc_auc": "roc_auc", "f1": "f1", "recall": "recall"})
        row = {"model": name}
        for k in ("roc_auc", "f1", "recall"):
            vals = res[f"test_{k}"]
            row[k], row[k + "_std"] = round(float(vals.mean()), 4), round(float(vals.std()), 4)
        rows.append(row)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "cv": "5-fold x 5 repeats"})
            mlflow.log_metrics({k: v for k, v in row.items() if k != "model"})
        print(f"{name:22s} ROC-AUC {row['roc_auc']:.3f} ± {row['roc_auc_std']:.3f}   F1 {row['f1']:.3f}")
    summary = {"rows": len(df), "positive_share": round(float(y.mean()), 4),
               "missing_after_zero_fix": df.isna().sum()[df.isna().sum() > 0].to_dict(), "results": rows}
    (out / "diabetes_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Fraud detection with MLflow tracking")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("transactions", help="train and evaluate the fraud models")
    t.add_argument("--data", default="data/transactions.csv")
    t.add_argument("--valid-start", default="2024-08-01")
    t.add_argument("--test-start", default="2024-09-01")
    t.add_argument("--review-cost", type=float, default=5.0, help="cost of reviewing one flagged transaction")
    t.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    t.add_argument("--only", nargs="*", help="train only these model names")
    t.add_argument("--log-models", action="store_true", help="also store each fitted pipeline in MLflow")
    d = sub.add_parser("diabetes", help="the second, small dataset")
    d.add_argument("--data", default="data/diabetes.csv")
    for s in (t, d):
        s.add_argument("--out", default="results")
        s.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = p.parse_args(argv)
    return run_transactions(args) if args.command == "transactions" else run_diabetes(args)


if __name__ == "__main__":
    main()
