"""
ml/evaluation/explainer.py
--------------------------
SHAP-based model explainability for the Churn Intelligence Platform.

Provides three levels of explanation:
1. Global feature importance  — which features drive churn across all customers
2. Prediction explanation     — why a specific prediction was made
3. Customer-level explanation — individual SHAP waterfall breakdown

Falls back to sklearn's built-in feature_importances_ / coef_ when SHAP
is not installed, so the pipeline never fails in restricted environments.

Public API
----------
- ``ChurnExplainer``                           (class)
  - ``.fit(model, X_background, feature_names)``
  - ``.global_importance(X, top_n)``    → list[{feature, importance, direction}]
  - ``.explain_prediction(X_row)``      → list[{feature, value, shap_value, impact}]
  - ``.explain_customer(customer_dict)``→ dict summary
- ``build_explainer(model, X_train, feature_names)`` → ChurnExplainer
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Try to import SHAP ────────────────────────────────────────────────────────
try:
    import shap
    _SHAP_AVAILABLE = True
    logger.info("SHAP available.")
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning("SHAP not installed — using fallback importance method.")


class ChurnExplainer:
    """
    Wraps a fitted model with SHAP (or fallback) explainability.

    Parameters
    ----------
    model         : fitted sklearn-compatible classifier
    feature_names : list of feature names matching the model's input
    """

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model         = model
        self.feature_names = feature_names
        self._explainer    = None
        self._background   = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_background: np.ndarray, n_background: int = 100) -> "ChurnExplainer":
        """
        Create a SHAP explainer using a sample of the training data as background.

        Parameters
        ----------
        X_background : training feature matrix (used to build SHAP background)
        n_background : number of background samples (more = slower but more accurate)
        """
        if not _SHAP_AVAILABLE:
            logger.info("Skipping SHAP fit — using built-in importances.")
            return self

        # Subsample background for speed
        n = min(n_background, len(X_background))
        idx = np.random.default_rng(42).choice(len(X_background), size=n, replace=False)
        self._background = X_background[idx]

        # Choose the best explainer for the model type
        model_type = type(self.model).__name__.lower()

        try:
            if any(t in model_type for t in ("xgb", "gradient", "forest", "tree")):
                self._explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer for %s.", type(self.model).__name__)
            else:
                self._explainer = shap.LinearExplainer(self.model, self._background)
                logger.info("Using LinearExplainer for %s.", type(self.model).__name__)
        except Exception as exc:
            logger.warning("SHAP explainer init failed (%s); using KernelExplainer.", exc)
            predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            self._explainer = shap.KernelExplainer(predict_fn, self._background)

        return self

    # ── Global importance ─────────────────────────────────────────────────────

    def global_importance(
        self,
        X: np.ndarray,
        top_n: int = 15,
    ) -> list[dict[str, Any]]:
        """
        Compute mean |SHAP value| across *X* and return top-*n* features.

        Returns
        -------
        list of dicts: {feature, importance, direction}
        direction = "increases_churn" | "decreases_churn"
        """
        if _SHAP_AVAILABLE and self._explainer is not None:
            return self._shap_global_importance(X, top_n)
        return self._fallback_global_importance(top_n)

    def _shap_global_importance(self, X: np.ndarray, top_n: int) -> list[dict]:
        try:
            shap_values = self._explainer.shap_values(X)
            # For binary classifiers, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            mean_abs   = np.abs(shap_values).mean(axis=0)
            mean_signed = shap_values.mean(axis=0)

            indices = np.argsort(mean_abs)[::-1][:top_n]
            return [
                {
                    "feature":   self.feature_names[i],
                    "importance": round(float(mean_abs[i]), 6),
                    "direction":  "increases_churn" if mean_signed[i] > 0 else "decreases_churn",
                }
                for i in indices
            ]
        except Exception as exc:
            logger.warning("SHAP global importance failed: %s. Falling back.", exc)
            return self._fallback_global_importance(top_n)

    def _fallback_global_importance(self, top_n: int) -> list[dict]:
        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            imps = np.abs(coef[0] if coef.ndim > 1 else coef)
        else:
            return []

        total = imps.sum()
        if total > 0:
            imps = imps / total

        indices = np.argsort(imps)[::-1][:top_n]
        return [
            {
                "feature":    self.feature_names[i],
                "importance": round(float(imps[i]), 6),
                "direction":  "increases_churn",   # direction unknown without SHAP
            }
            for i in indices
        ]

    # ── Single-row explanation ─────────────────────────────────────────────────

    def explain_prediction(
        self,
        X_row: np.ndarray,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Explain a single prediction row.

        Returns
        -------
        list of dicts: {feature, shap_value, impact}
        sorted by |shap_value| descending.
        """
        if _SHAP_AVAILABLE and self._explainer is not None:
            return self._shap_explain_row(X_row, top_n)
        return self._fallback_explain_row(X_row, top_n)

    def _shap_explain_row(self, X_row: np.ndarray, top_n: int) -> list[dict]:
        try:
            if X_row.ndim == 1:
                X_row = X_row.reshape(1, -1)
            shap_values = self._explainer.shap_values(X_row)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            row_shap = shap_values[0]

            indices = np.argsort(np.abs(row_shap))[::-1][:top_n]
            return [
                {
                    "feature":    self.feature_names[i],
                    "feature_value": float(X_row[0, i]),
                    "shap_value": round(float(row_shap[i]), 6),
                    "impact":     "positive" if row_shap[i] > 0 else "negative",
                }
                for i in indices
            ]
        except Exception as exc:
            logger.warning("SHAP row explanation failed: %s. Falling back.", exc)
            return self._fallback_explain_row(X_row, top_n)

    def _fallback_explain_row(self, X_row: np.ndarray, top_n: int) -> list[dict]:
        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
        else:
            return []

        if X_row.ndim == 2:
            X_row = X_row[0]

        indices = np.argsort(imps)[::-1][:top_n]
        return [
            {
                "feature":       self.feature_names[i],
                "feature_value": float(X_row[i]),
                "shap_value":    round(float(imps[i]), 6),
                "impact":        "positive",
            }
            for i in indices
        ]

    # ── Customer-level explanation ─────────────────────────────────────────────

    def explain_customer(
        self,
        customer_dict: dict[str, Any],
        X_transformed: np.ndarray,
        churn_probability: float,
    ) -> dict[str, Any]:
        """
        Generate a human-readable explanation for a customer prediction.

        Parameters
        ----------
        customer_dict    : raw customer record
        X_transformed    : preprocessed feature row (shape [1, n_features])
        churn_probability: already computed probability

        Returns
        -------
        dict with top_risk_factors and explanation_text
        """
        row_explanations = self.explain_prediction(X_transformed, top_n=5)

        # Build narrative
        positive_factors = [e for e in row_explanations if e["impact"] == "positive"]
        negative_factors = [e for e in row_explanations if e["impact"] == "negative"]

        risk_factor_names = [e["feature"] for e in positive_factors[:3]]
        narrative = (
            f"This customer has a {churn_probability:.0%} probability of churning. "
        )
        if risk_factor_names:
            narrative += (
                f"The primary drivers are: {', '.join(risk_factor_names)}. "
            )
        if negative_factors:
            protective = [e["feature"] for e in negative_factors[:2]]
            narrative += f"Protective factors include: {', '.join(protective)}."

        return {
            "churn_probability":  churn_probability,
            "top_risk_factors":   row_explanations,
            "positive_drivers":   positive_factors,
            "negative_drivers":   negative_factors,
            "explanation_text":   narrative,
            "shap_available":     _SHAP_AVAILABLE and self._explainer is not None,
        }


    # ── Compatibility aliases ─────────────────────────────────────────────────

    def explain_instance(self, x: np.ndarray, top_n: int = 15) -> list[dict]:
        """Alias: explain a single 1-D feature vector (backward-compatible)."""
        flat = x.flatten()
        result = self.explain_prediction(flat)
        # explain_prediction returns a list of dicts directly
        if isinstance(result, list):
            return result[:top_n]
        # or a dict with top_risk_factors
        return result.get("top_risk_factors", [])[:top_n]

    def get_global_importance(self, X: np.ndarray, top_n: int = 15) -> list[dict]:
        """Alias for global_importance (backward-compatible)."""
        return self.global_importance(X, top_n=top_n)


# ── Factory ────────────────────────────────────────────────────────────────────

def build_explainer(
    model: Any,
    X_train: np.ndarray,
    feature_names: list[str],
    n_background: int = 100,
) -> ChurnExplainer:
    """Create and fit a ChurnExplainer."""
    exp = ChurnExplainer(model, feature_names)
    exp.fit(X_train, n_background=n_background)
    return exp
