"""
ml/models/trainer.py
--------------------
Trains Logistic Regression, Random Forest, and XGBoost (or GradientBoosting
as fallback) on the preprocessed feature matrix and selects the best model
by ROC-AUC score.

Public API
----------
- ``train_all_models(X_train, y_train)``         → dict of fitted models
- ``select_best_model(models, X_val, y_val)``    → (name, model, metrics_dict)
- ``run_training_pipeline(...)``                 → full end-to-end run
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml.config import (
    LR_PARAMS,
    MODEL_DIR,
    MODEL_FILENAMES,
    MODEL_METADATA_FILENAME,
    RF_PARAMS,
    XGB_PARAMS,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# ── Try to import XGBoost; fall back to GradientBoosting ─────────────────────
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
    logger.info("XGBoost available.")
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — using GradientBoostingClassifier as substitute.")


def _build_models() -> dict[str, Any]:
    """Instantiate all classifiers with configured hyperparameters."""
    xgb_params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}

    if _XGB_AVAILABLE:
        boosting_model = XGBClassifier(**xgb_params)
        boosting_key   = "xgboost"
    else:
        # GradientBoosting shares most hyperparameters; filter unsupported ones
        gb_keys = {"n_estimators", "max_depth", "learning_rate", "subsample", "random_state"}
        gb_params = {k: v for k, v in xgb_params.items() if k in gb_keys}
        boosting_model = GradientBoostingClassifier(**gb_params)
        boosting_key   = "xgboost"   # keep the same dict key for downstream code

    return {
        "logistic_regression": LogisticRegression(**LR_PARAMS),
        "random_forest":       RandomForestClassifier(**RF_PARAMS),
        boosting_key:          boosting_model,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """
    Fit all models on (X_train, y_train).

    Returns
    -------
    dict mapping model_name → fitted estimator
    """
    models = _build_models()
    trained: dict[str, Any] = {}

    for name, clf in models.items():
        logger.info("Training %s …", name)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        logger.info("  ✓ %s trained in %.1f s", name, elapsed)
        trained[name] = clf

    return trained


# ── Model selection ────────────────────────────────────────────────────────────

def select_best_model(
    models: dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[str, Any, dict[str, float]]:
    """
    Evaluate all models on the validation set and return the best one by AUC.

    Returns
    -------
    (best_name, best_model, {model_name: auc_score})
    """
    scores: dict[str, float] = {}

    for name, clf in models.items():
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_val)[:, 1]
        else:
            y_prob = clf.decision_function(X_val)
        auc = roc_auc_score(y_val, y_prob)
        scores[name] = round(auc, 6)
        logger.info("  %s  AUC = %.4f", name, auc)

    best_name  = max(scores, key=lambda k: scores[k])
    best_model = models[best_name]
    logger.info("Best model: %s (AUC %.4f)", best_name, scores[best_name])
    return best_name, best_model, scores


# ── Persistence ────────────────────────────────────────────────────────────────

def save_models(
    models: dict[str, Any],
    best_name: str,
    metadata: dict[str, Any],
) -> None:
    """Persist every model and the best-model alias plus a JSON metadata file."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        filename = MODEL_FILENAMES.get(name, f"{name}.joblib")
        path = MODEL_DIR / filename
        joblib.dump(clf, path)
        logger.info("Saved %s → %s", name, path)

    # Best-model alias
    best_path = MODEL_DIR / MODEL_FILENAMES["best_model"]
    joblib.dump(models[best_name], best_path)
    logger.info("Best model alias saved → %s", best_path)

    # Metadata JSON
    meta_path = MODEL_DIR / MODEL_METADATA_FILENAME
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info("Metadata saved → %s", meta_path)


def load_model(name: str = "best_model") -> Any:
    """Load a previously saved model by name key."""
    filename = MODEL_FILENAMES.get(name, f"{name}.joblib")
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run the training pipeline first.")
    return joblib.load(path)


# ── Full training pipeline ─────────────────────────────────────────────────────

def run_training_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    save: bool = True,
) -> tuple[dict[str, Any], str, Any, dict]:
    """
    Train all models, select the best, optionally persist, return results.

    Returns
    -------
    models, best_name, best_model, metadata_dict
    """
    from ml.evaluation.evaluator import evaluate_all_models  # deferred to avoid circular

    # Train
    models = train_all_models(X_train, y_train)

    # Select
    best_name, best_model, auc_scores = select_best_model(models, X_test, y_test)

    # Evaluate all
    eval_results = evaluate_all_models(models, X_test, y_test, feature_names)

    metadata = {
        "best_model":     best_name,
        "auc_scores":     auc_scores,
        "feature_count":  len(feature_names),
        "train_size":     int(len(y_train)),
        "test_size":      int(len(y_test)),
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test":  float(y_test.mean()),
        "evaluation":     eval_results,
        "xgboost_available": _XGB_AVAILABLE,
    }

    if save:
        save_models(models, best_name, metadata)

    return models, best_name, best_model, metadata
