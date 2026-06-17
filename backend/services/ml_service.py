"""
backend/services/ml_service.py
--------------------------------
Unified ML service: loads all trained artefacts once and exposes
clean methods for prediction, batch inference, segmentation,
recommendations, revenue estimation, and SHAP explanations.

All heavy imports are deferred to the first call to keep startup fast.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once, reused forever) ─────────────────────
_lock     = threading.Lock()
_loaded   = False
_predictor   = None
_segmenter   = None
_explainer   = None
_rec_engine  = None
_rev_estimator = None


def _load_all() -> None:
    """Load every ML artefact (called once on first request)."""
    global _loaded, _predictor, _segmenter, _explainer, _rec_engine, _rev_estimator

    with _lock:
        if _loaded:
            return

        import sys
        sys.path.insert(0, str(settings.ML_MODEL_DIR.parent.parent.parent))

        from ml.models.predictor import ChurnPredictor
        from analytics.segmentation import CustomerSegmenter
        from analytics.revenue_estimator import RevenueEstimator
        from services.retention.recommendation_engine import RetentionRecommendationEngine

        logger.info("Loading ML artefacts from %s …", settings.ML_MODEL_DIR)

        _predictor = ChurnPredictor("best_model").load()

        segmenter_path = settings.ML_MODEL_DIR / "customer_segmenter.joblib"
        _segmenter = CustomerSegmenter.load(segmenter_path) if segmenter_path.exists() else None

        explainer_path = settings.ML_MODEL_DIR / "explainer.joblib"
        _explainer = joblib.load(explainer_path) if explainer_path.exists() else None

        _rec_engine    = RetentionRecommendationEngine()
        _rev_estimator = RevenueEstimator()

        _loaded = True
        logger.info("ML artefacts loaded successfully.")


def get_predictor():
    _load_all()
    return _predictor


def get_model_metadata() -> dict:
    """Return saved training metadata JSON."""
    meta_path = settings.ML_MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            return json.load(fh)
    return {}


# ── Single prediction ─────────────────────────────────────────────────────────

def _normalize_keys(data: dict) -> dict:
    """
    snake_case API keys ko CamelCase ML keys mein convert karo.
    Recommendation engine aur Revenue estimator CamelCase expect karte hain.
    """
    mapping = {
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
    normalized = dict(data)
    for snake, camel in mapping.items():
        if snake in normalized and camel not in normalized:
            normalized[camel] = normalized[snake]
    return normalized


def predict_customer(customer_data: dict, include_shap: bool = True) -> dict:
    """
    Run full prediction pipeline for one customer record.

    Returns
    -------
    dict with keys:
      prediction, risk_score, risk_category, churn_label,
      recommendations, revenue_risk, shap_explanation
    """
    _load_all()
    pred = _predictor.predict(customer_data)
    prob  = pred["churn_probability"]
    score = pred["risk_score"]
    cat   = pred["risk_category"]
    label = pred["churn_label"]

    # Segment (best-effort)
    segment = None
    if _segmenter:
        try:
            from ml.features.feature_engineering import engineer_features
            df = pd.DataFrame([customer_data])
            df = engineer_features(df)
            df["churn_probability"] = prob
            seg_df = _segmenter.predict(df)
            segment = seg_df.iloc[0].get("segment", None)
        except Exception as e:
            logger.warning("Segmentation failed: %s", e)

    # Recommendations
    recs = []
    if _rec_engine:
        try:
            recs = _rec_engine.recommend(_normalize_keys(customer_data), prob, segment)
            recs = [r.to_dict() for r in recs]
        except Exception as e:
            logger.warning("Recommendations failed: %s", e)

    # Revenue risk
    rev_risk = {}
    if _rev_estimator:
        try:
            risk = _rev_estimator.estimate_customer(
                _normalize_keys(customer_data), prob, recs, risk_category=cat)
            rev_risk = risk.to_dict()
        except Exception as e:
            logger.warning("Revenue estimation failed: %s", e)

    # SHAP explanation
    shap_exp = []
    if include_shap and _explainer:
        try:
            X = _predictor._prepare(pd.DataFrame([customer_data]))
            shap_exp = _explainer.explain_prediction(X, top_n=10)
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)

    return {
        "churn_probability":  prob,
        "risk_score":         score,
        "risk_category":      cat,
        "churn_label":        label,
        "segment":            segment,
        "recommendations":    recs,
        "revenue_risk":       rev_risk,
        "shap_explanation":   shap_exp,
        "model_used":         "best_model",
    }


# ── Batch prediction ──────────────────────────────────────────────────────────

def predict_batch_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run batch prediction on a DataFrame.
    Returns the DataFrame with prediction columns appended.
    """
    _load_all()
    df_out = _predictor.predict_batch(df.copy())

    # Recommendations (top action only for batch)
    if _rec_engine:
        top_actions = []
        for _, row in df_out.iterrows():
            record = row.to_dict()
            prob   = float(record.get("churn_probability", 0))
            try:
                recs = _rec_engine.recommend(record, prob)
                top_actions.append(recs[0].action if recs else "No action required.")
            except Exception:
                top_actions.append("No action required.")
        df_out["top_action"] = top_actions

    return df_out


def predict_from_csv(csv_bytes: bytes) -> dict:
    """
    Parse a CSV upload, run batch prediction, return summary + CSV string.

    Returns
    -------
    dict with keys: total, high_risk, medium_risk, low_risk,
                    avg_churn_probability, csv_output (str)
    """
    _load_all()
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    if len(df) == 0:
        raise ValueError("CSV file is empty.")
    if len(df) > 10_000:
        raise ValueError("CSV file exceeds 10,000 row limit.")

    df_pred = predict_batch_df(df)

    # Build output CSV
    out_buf = io.StringIO()
    df_pred.to_csv(out_buf, index=False)
    csv_str = out_buf.getvalue()

    probs = df_pred["churn_probability"].astype(float)
    cats  = df_pred["risk_category"]

    return {
        "total":                 len(df_pred),
        "high_risk":             int((cats == "High").sum()),
        "medium_risk":           int((cats == "Medium").sum()),
        "low_risk":              int((cats == "Low").sum()),
        "avg_churn_probability": round(float(probs.mean()), 4),
        "csv_output":            csv_str,
        "rows":                  df_pred[["churn_probability","risk_score",
                                          "risk_category","churn_label"]].head(100).to_dict("records"),
    }


# ── Global feature importance ─────────────────────────────────────────────────

def get_feature_importance(top_n: int = 15) -> list[dict]:
    """Return global SHAP or fallback feature importance."""
    _load_all()
    if _explainer:
        try:
            scored_path = settings.ML_REPORTS_DIR / "scored_customers.csv"
            if scored_path.exists():
                df = pd.read_csv(scored_path).head(500)
                X  = _predictor._prepare(df)
                return _explainer.global_importance(X, top_n=top_n)
        except Exception as e:
            logger.warning("Global importance failed: %s", e)

    # Fallback: use model coefficients / feature_importances_
    meta = get_model_metadata()
    best = meta.get("best_model", "")
    eval_data = meta.get("evaluation", {}).get(best, {})
    return eval_data.get("feature_importance", [])[:top_n]


# ── Portfolio analytics ───────────────────────────────────────────────────────

def get_portfolio_stats() -> dict:
    """Return portfolio-level churn and revenue stats from the scored dataset."""
    _load_all()
    scored_path = settings.ML_REPORTS_DIR / "scored_customers.csv"
    if not scored_path.exists():
        return {}

    df = pd.read_csv(scored_path)
    if "churn_probability" not in df.columns:
        return {}

    probs   = df["churn_probability"].astype(float)
    charges = df.get("MonthlyCharges", pd.Series(dtype=float)).astype(float) \
        if "MonthlyCharges" in df.columns else pd.Series(dtype=float)

    stats: dict[str, Any] = {
        "total_customers":       len(df),
        "high_risk":             int((probs >= 0.65).sum()),
        "medium_risk":           int(((probs >= 0.35) & (probs < 0.65)).sum()),
        "low_risk":              int((probs < 0.35).sum()),
        "avg_churn_probability": round(float(probs.mean()), 4),
        "churn_rate":            round(float((df["churn_label"] == "Churn").mean())
                                       if "churn_label" in df.columns else probs.mean(), 4),
    }

    if len(charges) > 0:
        stats["total_monthly_revenue"]  = round(float(charges.sum()), 2)
        stats["expected_monthly_loss"]  = round(float((charges * probs).sum()), 2)
        stats["expected_annual_loss"]   = round(stats["expected_monthly_loss"] * 12, 2)

    if "segment" in df.columns:
        stats["segment_distribution"] = df["segment"].value_counts().to_dict()

    return stats
