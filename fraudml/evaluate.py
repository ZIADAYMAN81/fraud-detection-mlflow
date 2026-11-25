"""Metrics for imbalanced problems, threshold selection and the money view."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)


def best_f1_threshold(y_true, scores) -> float:
    """Threshold that maximises F1. Chosen on the validation period, then frozen for the test period."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def cost(y_true, flagged, amounts, review_cost: float) -> float:
    """Missed fraud costs the transaction amount; every flagged transaction costs one manual review."""
    y_true, flagged, amounts = map(np.asarray, (y_true, flagged, amounts))
    return float(amounts[(y_true == 1) & ~flagged].sum() + review_cost * flagged.sum())


def best_cost_threshold(y_true, scores, amounts, review_cost: float) -> float:
    candidates = np.unique(np.quantile(scores, np.linspace(0.5, 0.9999, 400)))
    costs = [cost(y_true, scores >= t, amounts, review_cost) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> float:
    """Share of fraud caught while flagging at most `max_fpr` of legitimate transactions."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    threshold = np.quantile(scores[y_true == 0], 1 - max_fpr)
    return float((scores[y_true == 1] > threshold).mean())


def metrics(y_true, scores, threshold: float, amounts=None, review_cost: float = 5.0) -> dict:
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    out = {
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged),
        "f1": f1_score(y_true, flagged),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "flag_rate": float(flagged.mean()),
        "threshold": float(threshold),
    }
    if amounts is not None:
        amounts = np.asarray(amounts)
        fraud_value = amounts[y_true == 1].sum()
        out["fraud_value_caught_share"] = float(amounts[(y_true == 1) & flagged].sum() / fraud_value) if fraud_value else 0.0
        out["cost"] = cost(y_true, flagged, amounts, review_cost)
        out["cost_without_model"] = float(fraud_value)
    return {k: round(float(v), 4) for k, v in out.items()}
