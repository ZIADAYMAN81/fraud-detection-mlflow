"""Train the production model, register it in MLflow and export a serving bundle.

    python -m fraudml.package --data data/transactions.csv --out models/fraud_xgboost

Protocol (same as the evaluation in the README): train on January-July, choose decision thresholds on August,
report September-October. The model version is registered as "fraud-xgboost"; it becomes the "champion" alias
only if its test PR-AUC beats the current champion's (champion/challenger). The bundle written to `--out`
holds the fitted pipeline, decision thresholds, metrics, library versions and the reference distributions that
drift monitoring compares live traffic against.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data, evaluate, features, monitor

MODEL_NAME = "fraud-xgboost"
BUNDLE_MODEL = "model.joblib"
BUNDLE_META = "metadata.json"


def reference_profile(X: pd.DataFrame, bins: int = 10) -> dict:
    """Quantile bin edges and shares per numeric feature, category shares per categorical feature."""
    prof = {"numeric": {}, "categorical": {}}
    for c in [c for c in X.columns if c not in features.CATEGORICAL]:
        v = X[c].astype(float).dropna().to_numpy()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        counts = np.histogram(np.clip(v, edges[0], edges[-1]), edges)[0]
        prof["numeric"][c] = {"edges": edges.tolist(), "shares": (counts / counts.sum()).tolist(),
                              "missing_share": float(X[c].isna().mean())}
    for c in [c for c in X.columns if c in features.CATEGORICAL]:
        prof["categorical"][c] = X[c].value_counts(normalize=True).round(6).to_dict()
    return prof


def library_versions() -> dict:
    import sklearn
    import xgboost

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


def build(data_path: str, out: Path, valid_start="2024-08-01", test_start="2024-09-01", review_cost=5.0,
          tracking_uri="sqlite:///mlflow.db", device="auto", register=True) -> dict:
    from . import models  # training-only dependencies (imbalanced-learn) stay out of the serving image

    df = features.add_features(data.load_transactions(data_path))
    train, valid, test = data.time_split(df, valid_start, test_start)
    numeric = features.PRODUCTION_NUMERIC
    cols = numeric + features.CATEGORICAL
    pos_weight = float((train[data.TARGET] == 0).sum() / (train[data.TARGET] == 1).sum())
    dev = "cuda" if device == "auto" and models.gpu_available() else ("cpu" if device == "auto" else device)
    pipe = models.candidates(numeric, features.CATEGORICAL, pos_weight, dev)["XGBoost"]
    t0 = time.time()
    pipe.fit(train[cols], train[data.TARGET])
    fit_seconds = round(time.time() - t0, 1)
    pipe.named_steps["model"].set_params(device="cpu")  # serve on CPU regardless of where it was trained

    s_valid = pipe.predict_proba(valid[cols])[:, 1]
    s_test = pipe.predict_proba(test[cols])[:, 1]
    thresholds = {"cost_optimal": evaluate.best_cost_threshold(valid[data.TARGET], s_valid, valid["amount"], review_cost),
                  "f1_optimal": evaluate.best_f1_threshold(valid[data.TARGET], s_valid)}
    metrics = {name: evaluate.metrics(test[data.TARGET], s_test, t, test["amount"], review_cost)
               for name, t in thresholds.items()}

    meta = {"model_name": MODEL_NAME, "created": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "train_period": [str(train[data.TIME].min().date()), str(train[data.TIME].max().date())],
            "threshold_period": [str(valid[data.TIME].min().date()), str(valid[data.TIME].max().date())],
            "test_period": [str(test[data.TIME].min().date()), str(test[data.TIME].max().date())],
            "rows": {"train": len(train), "validation": len(valid), "test": len(test)},
            "numeric_features": numeric, "categorical_features": features.CATEGORICAL,
            "thresholds": {k: round(v, 6) for k, v in thresholds.items()}, "default_threshold": "cost_optimal",
            "review_cost": review_cost, "test_metrics": metrics, "trained_on": dev, "fit_seconds": fit_seconds,
            "libraries": library_versions(), "reference_profile": reference_profile(train[cols]),
            "reference_score_profile": score_profile(pipe.predict_proba(train[cols])[:, 1])}
    meta["stability"] = score_stability(pipe, meta["reference_score_profile"], pd.concat([valid, test]), cols)

    if register:
        meta["registry"] = register_model(pipe, train[cols].head(5), meta, tracking_uri)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / BUNDLE_MODEL)
    (out / BUNDLE_META).write_text(json.dumps(meta, indent=1, default=str))
    return meta


def score_stability(pipe, reference: dict, later: pd.DataFrame, cols: list) -> dict:
    """PSI of the model's scores in each month after training, against the training-period scores."""
    by_month = {}
    for month, part in later.groupby(later[data.TIME].dt.strftime("%Y-%m")):
        s = pipe.predict_proba(part[cols])[:, 1]
        by_month[month] = round(monitor.psi(reference["shares"], monitor.numeric_shares(s, reference["edges"])), 4)
    return {"score_psi_by_month": by_month, "max_score_psi": max(by_month.values())}


# Promotion gate: a challenger replaces the champion only if its scores are not drifting significantly and its
# PR-AUC is at most this much below the champion's. Comparing PR-AUC alone would have kept a drifting model.
PR_AUC_TOLERANCE = 0.005
MAX_SCORE_PSI = monitor.MODERATE


def should_promote(challenger: dict, champion: dict | None) -> tuple[bool, str]:
    """challenger / champion: {"test_pr_auc": float, "max_score_psi": float}."""
    if challenger["max_score_psi"] >= MAX_SCORE_PSI:
        return False, f"challenger scores drift significantly (PSI {challenger['max_score_psi']:.3f})"
    if champion is None:
        return True, "no current champion"
    if champion.get("max_score_psi", float("inf")) >= MAX_SCORE_PSI:
        return True, f"champion scores drift significantly (PSI {champion.get('max_score_psi')}); challenger is stable"
    if challenger["test_pr_auc"] >= champion["test_pr_auc"] - PR_AUC_TOLERANCE:
        return True, "stable and PR-AUC within tolerance of the champion"
    return False, "PR-AUC more than the tolerance below the champion"


def score_profile(scores: np.ndarray, bins: int = 10) -> dict:
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)))
    counts = np.histogram(np.clip(scores, edges[0], edges[-1]), edges)[0]
    return {"edges": edges.tolist(), "shares": (counts / counts.sum()).tolist()}


def use_experiment(tracking_uri: str, name: str) -> None:
    """Select an MLflow experiment; with a SQLite store, artifacts go next to the database file instead of the
    current working directory."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(name) is None and tracking_uri.startswith("sqlite:///"):
        db = Path(tracking_uri[len("sqlite:///"):]).resolve()
        mlflow.create_experiment(name, artifact_location=(db.parent / "mlartifacts").as_uri())
    mlflow.set_experiment(name)


def register_model(pipe, example: pd.DataFrame, meta: dict, tracking_uri: str) -> dict:
    """Log the run, register a new model version, and promote it to "champion" only if it beats the current one."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    use_experiment(tracking_uri, "fraud-detection-production")
    test_pr_auc = meta["test_metrics"]["cost_optimal"]["pr_auc"]
    with mlflow.start_run(run_name="production-xgboost") as run:
        mlflow.log_params({"train_period": "/".join(meta["train_period"]), "test_period": "/".join(meta["test_period"]),
                           "threshold_cost_optimal": meta["thresholds"]["cost_optimal"], "review_cost": meta["review_cost"]})
        mlflow.log_metrics({f"test_{k}": v for k, v in meta["test_metrics"]["cost_optimal"].items()})
        info = mlflow.sklearn.log_model(pipe, name="model", signature=infer_signature(example, pipe.predict_proba(example)),
                                        input_example=example, registered_model_name=MODEL_NAME)
    client = MlflowClient()
    version = str(info.registered_model_version)
    mine = {"test_pr_auc": test_pr_auc, "max_score_psi": meta["stability"]["max_score_psi"]}
    for k, v in mine.items():
        client.set_model_version_tag(MODEL_NAME, version, k, str(v))
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champ = {k: float(champion.tags[k]) for k in ("test_pr_auc", "max_score_psi") if k in champion.tags}
    except Exception:
        champion, champ = None, None
    promoted, reason = should_promote(mine, champ)
    client.set_registered_model_alias(MODEL_NAME, "champion" if promoted else "challenger", version)
    client.set_model_version_tag(MODEL_NAME, version, "promotion_decision", reason)
    return {"run_id": run.info.run_id, "version": version, "promoted_to_champion": promoted, "reason": reason,
            "previous_champion": None if champion is None else {"version": champion.version, **champ}}


def load_bundle(path: str | Path):
    """Returns (pipeline, metadata). Warns loudly if library versions differ from the ones it was trained with."""
    import warnings

    path = Path(path)
    meta = json.loads((path / BUNDLE_META).read_text())
    now = library_versions()
    diff = {k: (v, now.get(k)) for k, v in meta["libraries"].items() if k in ("scikit-learn", "xgboost") and now.get(k) != v}
    if diff:
        warnings.warn(f"model trained with different library versions (trained, now): {diff}")
    return joblib.load(path / BUNDLE_MODEL), meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/transactions.csv")
    p.add_argument("--out", default="models/fraud_xgboost")
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--no-register", action="store_true")
    args = p.parse_args(argv)
    meta = build(args.data, Path(args.out), tracking_uri=args.tracking_uri, device=args.device, register=not args.no_register)
    print(json.dumps({k: meta[k] for k in ("thresholds", "test_metrics", "stability", "trained_on", "fit_seconds", "registry")
                      if k in meta}, indent=1, default=str))


if __name__ == "__main__":
    main()
