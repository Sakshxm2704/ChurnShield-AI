"""
ml/models/predictor.py
----------------------
Inference pipeline for the trained churn model.

Responsibilities
----------------
1. Load the saved model + preprocessor + feature names from disk.
2. Accept raw customer records (dict or DataFrame).
3. Run feature engineering + preprocessing.
4. Return:
   - churn_probability  float [0, 1]
   - churn_label        str  "Churn" | "No Churn"
   - risk_score         int  [0, 100]
   - risk_category      str  "High" | "Medium" | "Low"

Public API
----------
- ``ChurnPredictor``                         (class, load-once, predict-many)
- ``predict_single(customer_dict)``          convenience wrapper
- ``predict_batch(df)``                      bulk inference on DataFrame
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml.config import (
    FEATURE_NAMES_FILENAME,
    MODEL_DIR,
    MODEL_FILENAMES,
    PREPROCESSOR_FILENAME,
    RISK_THRESHOLDS,
    TARGET_COL,
)
from ml.features.preprocessing import clean_data, _binary_encode
from ml.features.feature_engineering import engineer_features

logger = logging.getLogger(__name__)


# ── Risk scoring helpers ───────────────────────────────────────────────────────

def probability_to_risk_score(probability: float) -> int:
    """
    Convert a churn probability [0, 1] to a risk score [0, 100].
    Uses a non-linear curve that emphasises the high-probability tail.
    """
    # Sigmoid-like stretch: amplifies differences in the 0.4–0.8 range
    adjusted = 1 / (1 + np.exp(-10 * (probability - 0.5)))
    return int(round(adjusted * 100))


def probability_to_risk_category(probability: float) -> str:
    """
    Map a churn probability to a risk category label.

    Thresholds (from ml/config.py):
    - High   : probability >= 0.65
    - Medium : 0.35 <= probability < 0.65
    - Low    : probability < 0.35
    """
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    elif probability >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def probability_to_churn_label(probability: float, threshold: float = 0.50) -> str:
    """Return 'Churn' or 'No Churn' based on a probability threshold."""
    return "Churn" if probability >= threshold else "No Churn"


# ── Predictor class ────────────────────────────────────────────────────────────

class ChurnPredictor:
    """
    Load-once, predict-many inference wrapper.

    Parameters
    ----------
    model_name : str
        Key in ``MODEL_FILENAMES``.  Defaults to "best_model".

    Usage
    -----
    ::

        predictor = ChurnPredictor()
        result = predictor.predict(customer_dict)
        # result = {
        #   "churn_probability": 0.73,
        #   "churn_label":       "Churn",
        #   "risk_score":        84,
        #   "risk_category":     "High",
        # }
    """

    def __init__(self, model_name: str = "best_model") -> None:
        self.model_name = model_name
        self._model       = None
        self._preprocessor = None
        self._feature_names: list[str] = []
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def load(self) -> "ChurnPredictor":
        """Load model + preprocessor from disk.  Idempotent."""
        if self._loaded:
            return self

        model_path = MODEL_DIR / MODEL_FILENAMES.get(self.model_name, f"{self.model_name}.joblib")
        prep_path  = MODEL_DIR / PREPROCESSOR_FILENAME
        feat_path  = MODEL_DIR / FEATURE_NAMES_FILENAME

        for p in (model_path, prep_path, feat_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Artefact not found: {p}\n"
                    "Run: python -m ml.pipeline first."
                )

        self._model        = joblib.load(model_path)
        self._preprocessor = joblib.load(prep_path)
        self._feature_names = joblib.load(feat_path)
        self._loaded        = True
        logger.info("ChurnPredictor loaded model '%s'.", self.model_name)
        return self

    @property
    def model(self):
        if not self._loaded:
            self.load()
        return self._model

    @property
    def preprocessor(self):
        if not self._loaded:
            self.load()
        return self._preprocessor

    # ── Core predict ─────────────────────────────────────────────────────────

    # API snake_case -> ML CamelCase column name mapping
    _FIELD_MAP = {
        "monthly_charges":   "MonthlyCharges",
        "total_charges":     "TotalCharges",
        "inactive_days":     "InactiveDays",
        "subscription_type": "SubscriptionType",
        "senior_citizen":    "SeniorCitizen",
        "phone_service":     "PhoneService",
        "multiple_lines":    "MultipleLines",
        "internet_service":  "InternetService",
        "online_security":   "OnlineSecurity",
        "online_backup":     "OnlineBackup",
        "device_protection": "DeviceProtection",
        "tech_support":      "TechSupport",
        "streaming_tv":      "StreamingTV",
        "streaming_movies":  "StreamingMovies",
        "paperless_billing": "PaperlessBilling",
        "payment_method":    "PaymentMethod",
        "contract":          "Contract",
    }

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """Apply cleaning, feature engineering, and preprocessing to a DataFrame."""
        df = df.copy()

        # Rename API snake_case fields to original ML CamelCase column names
        rename = {k: v for k, v in self._FIELD_MAP.items()
                  if k in df.columns and v not in df.columns}
        if rename:
            df = df.rename(columns=rename)

        # Drop target if accidentally included
        df = df.drop(columns=[TARGET_COL], errors="ignore")

        # Ensure TotalCharges is string (preprocessing expects that)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].astype(str)

        # Inject optional columns with sensible defaults so the fitted
        # ColumnTransformer does not raise "columns are missing" errors
        _DEFAULTS = {
            "SeniorCitizen":   0,
            "Partner":         "No",
            "Dependents":      "No",
            "PhoneService":    "Yes",
            "MultipleLines":   "No",
            "InternetService": "DSL",
            "OnlineSecurity":  "No",
            "OnlineBackup":    "No",
            "DeviceProtection":"No",
            "TechSupport":     "No",
            "StreamingTV":     "No",
            "StreamingMovies": "No",
            "PaperlessBilling":"No",
            "gender":          "Male",
            "InactiveDays":    0,
            "SubscriptionType":"Standard",
            "TotalCharges":    "0",
        }
        for col, default in _DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        # Clean
        df = clean_data(df)

        # Feature engineering
        df = engineer_features(df)

        # Binary encode
        df = _binary_encode(df)

        # Transform
        return self.preprocessor.transform(df)

    def predict(
        self,
        customer: dict[str, Any],
        threshold: float = 0.50,
    ) -> dict[str, Any]:
        """
        Predict churn for a single customer record.

        Parameters
        ----------
        customer  : dict with raw feature values (same schema as training data)
        threshold : probability threshold for binary label

        Returns
        -------
        dict with churn_probability, churn_label, risk_score, risk_category
        """
        df = pd.DataFrame([customer])
        X  = self._prepare(df)

        prob = float(self.model.predict_proba(X)[0, 1])
        return {
            "churn_probability": round(prob, 4),
            "churn_label":       probability_to_churn_label(prob, threshold),
            "risk_score":        probability_to_risk_score(prob),
            "risk_category":     probability_to_risk_category(prob),
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Predict churn for a batch of customers.

        Returns the input DataFrame with four new columns appended:
        churn_probability, churn_label, risk_score, risk_category.
        """
        X     = self._prepare(df.copy())
        probs = self.model.predict_proba(X)[:, 1]

        out = df.copy()
        out["churn_probability"] = np.round(probs, 4)
        out["churn_label"]       = [probability_to_churn_label(p, threshold) for p in probs]
        out["risk_score"]        = [probability_to_risk_score(p) for p in probs]
        out["risk_category"]     = [probability_to_risk_category(p) for p in probs]
        return out

    def predict_proba_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Return raw probability array shape (n, 2) for SHAP / external use."""
        X = self._prepare(df.copy())
        return self.model.predict_proba(X)


# ── Convenience wrappers ──────────────────────────────────────────────────────

_default_predictor: ChurnPredictor | None = None


def get_predictor(model_name: str = "best_model") -> ChurnPredictor:
    """Return a lazily-loaded singleton predictor."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = ChurnPredictor(model_name).load()
    return _default_predictor


def predict_single(customer: dict[str, Any]) -> dict[str, Any]:
    """Predict churn for one customer dict (uses cached predictor)."""
    return get_predictor().predict(customer)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Predict churn for a batch DataFrame (uses cached predictor)."""
    return get_predictor().predict_batch(df)
