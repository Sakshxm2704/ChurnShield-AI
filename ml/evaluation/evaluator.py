"""
ml/evaluation/evaluator.py
--------------------------
Model evaluation utilities.

Computes Accuracy, Precision, Recall, F1, ROC-AUC for every model and
generates a structured report dictionary suitable for JSON serialisation
and dashboard display.

Public API
----------
- ``evaluate_model(clf, X_test, y_test, name)``  → metrics dict
- ``evaluate_all_models(models, X_test, y_test)``→ {name: metrics}
- ``print_evaluation_report(eval_results)``      → console pretty-print
- ``get_roc_curve_data(clf, X_test, y_test)``    → (fpr, tpr, auc)
- ``get_confusion_matrix(clf, X_test, y_test)``  → np.ndarray (2×2)
- ``get_feature_importance(clf, feature_names)`` → sorted list of (feat, importance)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── Single model evaluation ────────────────────────────────────────────────────

def evaluate_model(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str = "model",
    threshold: float = 0.50,
) -> dict[str, Any]:
    """
    Evaluate *clf* against *X_test* / *y_test* and return a metrics dict.

    Parameters
    ----------
    clf       : fitted sklearn-compatible classifier
    X_test    : feature matrix
    y_test    : true binary labels
    name      : display name for the model
    threshold : probability threshold for class-1 prediction

    Returns
    -------
    dict with keys: model_name, accuracy, precision, recall, f1, roc_auc,
                    threshold, support_0, support_1, class_report
    """
    # Probability scores
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)

    y_pred = (y_prob >= threshold).astype(int)

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    auc       = roc_auc_score(y_test, y_prob)
    report    = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "model_name": name,
        "accuracy":   round(acc,       4),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "roc_auc":    round(auc,       4),
        "threshold":  threshold,
        "support_0":  int(np.sum(y_test == 0)),
        "support_1":  int(np.sum(y_test == 1)),
        "class_report": report,
    }

    logger.info(
        "[%s]  ACC=%.3f  PRE=%.3f  REC=%.3f  F1=%.3f  AUC=%.4f",
        name, acc, precision, recall, f1, auc,
    )
    return metrics


# ── Evaluate all models ────────────────────────────────────────────────────────

def evaluate_all_models(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
    threshold: float = 0.50,
) -> dict[str, dict]:
    """
    Run ``evaluate_model`` for every model in *models* dict.

    Returns
    -------
    dict mapping model_name → metrics dict
    """
    results: dict[str, dict] = {}
    for name, clf in models.items():
        metrics = evaluate_model(clf, X_test, y_test, name=name, threshold=threshold)
        if feature_names:
            fi = get_feature_importance(clf, feature_names, top_n=20)
            metrics["feature_importance"] = fi
        results[name] = metrics
    return results


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_evaluation_report(eval_results: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in eval_results.items():
        print(
            f"{name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>10.4f}"
        )
    print("=" * len(header) + "\n")


# ── ROC curve data ─────────────────────────────────────────────────────────────

def get_roc_curve_data(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (fpr, tpr, auc) for plotting an ROC curve.
    """
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = clf.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    return fpr, tpr, round(auc, 4)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def get_confusion_matrix(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.50,
) -> np.ndarray:
    """Return a 2×2 confusion matrix as a NumPy array."""
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
    return confusion_matrix(y_test, y_pred)


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(
    clf: Any,
    feature_names: list[str],
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Extract and rank feature importances from a fitted model.

    Supports tree-based models (feature_importances_) and linear models
    (coef_).  Returns a sorted list of dicts: [{feature, importance}, ...].
    """
    importances: np.ndarray | None = None

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0]) if clf.coef_.ndim > 1 else np.abs(clf.coef_)

    if importances is None or len(importances) != len(feature_names):
        return []

    # Normalise
    total = importances.sum()
    if total > 0:
        importances = importances / total

    paired = sorted(
        zip(feature_names, importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {"feature": feat, "importance": round(imp, 6)}
        for feat, imp in paired[:top_n]
    ]
