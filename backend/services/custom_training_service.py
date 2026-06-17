"""
backend/services/custom_training_service.py
--------------------------------------------
Custom model retraining service.

User apna CSV upload karta hai → model us data pe train hota hai →
ab woh industry ke liye accurate predictions milti hain.

Supported CSV formats:
- Koi bhi columns ho sakti hain
- Ek "churn" ya "Churn" column hona chahiye (0/1 ya Yes/No)
- Minimum 100 rows required

Public API:
-----------
- train_custom_model(csv_content, industry, target_col)  → TrainingResult
- get_training_status()                                   → dict
- list_custom_models()                                    → list
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Training status tracker ───────────────────────────────────────────────────
_training_status = {
    "is_training": False,
    "progress": 0,
    "message": "No training in progress",
    "last_trained": None,
    "last_model": None,
    "error": None,
}
_status_lock = threading.Lock()


# ── Main training function ────────────────────────────────────────────────────

def train_custom_model(
    csv_content: bytes,
    industry: str = "general",
    target_col: str = "churn",
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    User ke CSV se custom model train karo.

    Parameters
    ----------
    csv_content : CSV file ka binary content
    industry    : industry name (for saving/display)
    target_col  : churn column ka naam (default: "churn")
    model_name  : save karne ke liye naam (default: industry_timestamp)

    Returns
    -------
    dict with training results: accuracy, features used, model path, etc.
    """
    _update_status(is_training=True, progress=0, message="Reading CSV file...")

    try:
        # Step 1: CSV padhho
        df = _read_csv(csv_content)
        _update_status(progress=10, message=f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Step 2: Target column dhundho
        target = _find_target_column(df, target_col)
        _update_status(progress=20, message=f"Target column found: '{target}'")

        # Step 3: Data clean karo
        df, feature_cols = _auto_preprocess(df, target)
        _update_status(progress=35, message=f"Preprocessed: {len(feature_cols)} features ready")

        # Step 4: Train/test split
        from sklearn.model_selection import train_test_split
        X = df[feature_cols].values
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        _update_status(progress=45, message="Train/test split done")

        # Step 5: Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        _update_status(progress=55, message="Feature scaling done")

        # Step 6: Train 3 models
        _update_status(progress=60, message="Training Logistic Regression...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
            "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        results = {}
        best_model_name = None
        best_auc = 0

        for i, (name, model) in enumerate(models.items()):
            _update_status(
                progress=60 + i * 10,
                message=f"Training {name.replace('_', ' ').title()}..."
            )
            model.fit(X_train_scaled, y_train)
            y_pred  = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            auc = roc_auc_score(y_test, y_proba)
            results[name] = {
                "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
                "f1":        round(float(f1_score(y_test, y_pred)), 4),
                "roc_auc":   round(float(auc), 4),
            }

            if auc > best_auc:
                best_auc = auc
                best_model_name = name

        _update_status(progress=90, message=f"Best model: {best_model_name} (AUC: {best_auc:.3f})")

        # Step 7: Save best model
        save_name = model_name or f"{industry}_{int(time.time())}"
        save_dir  = settings.ML_MODEL_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save as custom model (don't overwrite best_model.joblib — keep original!)
        model_path     = save_dir / f"custom_{save_name}.joblib"
        scaler_path    = save_dir / f"custom_{save_name}_scaler.joblib"
        feat_path      = save_dir / f"custom_{save_name}_features.joblib"
        meta_path      = save_dir / f"custom_{save_name}_meta.json"

        joblib.dump(models[best_model_name], model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_cols, feat_path)

        meta = {
            "industry":        industry,
            "model_name":      save_name,
            "best_model":      best_model_name,
            "best_auc":        round(best_auc, 4),
            "feature_columns": list(feature_cols),
            "target_column":   target,
            "n_samples":       len(df),
            "n_features":      len(feature_cols),
            "churn_rate":      round(float(y.mean()), 4),
            "model_results":   results,
            "trained_at":      datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        _update_status(
            is_training=False, progress=100,
            message=f"Training complete! AUC: {best_auc:.3f}",
            last_trained=datetime.now(timezone.utc).isoformat(),
            last_model=save_name,
        )

        logger.info(
            "Custom model trained: industry=%s model=%s auc=%.3f features=%d samples=%d",
            industry, best_model_name, best_auc, len(feature_cols), len(df)
        )

        return {
            "success":         True,
            "model_name":      save_name,
            "industry":        industry,
            "best_model":      best_model_name,
            "best_auc":        round(best_auc, 4),
            "n_samples":       len(df),
            "n_features":      len(feature_cols),
            "churn_rate":      round(float(y.mean()), 4),
            "feature_columns": list(feature_cols),
            "model_results":   results,
            "model_path":      str(model_path),
        }

    except Exception as e:
        logger.error("Custom training failed: %s", e, exc_info=True)
        _update_status(is_training=False, progress=0, message="Training failed", error=str(e))
        raise


# ── Prediction with custom model ──────────────────────────────────────────────

def predict_with_custom_model(
    customer_data: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """
    Custom trained model se prediction karo.

    Parameters
    ----------
    customer_data : customer feature dict
    model_name    : train_custom_model se mila hua model_name

    Returns
    -------
    Same format as ML model prediction
    """
    save_dir = settings.ML_MODEL_DIR

    model_path  = save_dir / f"custom_{model_name}.joblib"
    scaler_path = save_dir / f"custom_{model_name}_scaler.joblib"
    feat_path   = save_dir / f"custom_{model_name}_features.joblib"
    meta_path   = save_dir / f"custom_{model_name}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Custom model '{model_name}' not found.")

    model         = joblib.load(model_path)
    scaler        = joblib.load(scaler_path)
    feature_cols  = joblib.load(feat_path)

    with open(meta_path) as f:
        meta = json.load(f)

    # Build feature vector
    feature_vector = []
    for col in feature_cols:
        val = customer_data.get(col, 0)
        try:
            feature_vector.append(float(val))
        except (TypeError, ValueError):
            feature_vector.append(0.0)

    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob       = float(model.predict_proba(X_scaled)[0][1])
    risk_score = round(prob * 100)

    if prob >= 0.65:
        risk_cat = "High"
    elif prob >= 0.35:
        risk_cat = "Medium"
    else:
        risk_cat = "Low"

    monthly = float(customer_data.get("monthly_charges", 50))

    return {
        "churn_probability": round(prob, 4),
        "risk_score":        risk_score,
        "risk_category":     risk_cat,
        "churn_label":       "Churn" if prob >= 0.5 else "No Churn",
        "segment":           "Risky" if prob >= 0.65 else "Loyal",
        "model_used":        f"custom/{model_name}",
        "industry":          meta.get("industry", "custom"),
        "source":            "custom_model",
        "recommendations":   [],
        "revenue_risk": {
            "expected_monthly_loss": round(prob * monthly, 2),
            "expected_annual_loss":  round(prob * monthly * 12, 2),
            "ltv_at_risk":           round(prob * monthly * 24 * 0.65, 2),
        },
    }


# ── List custom models ─────────────────────────────────────────────────────────

def list_custom_models() -> list[dict]:
    """Saare trained custom models ki list return karo."""
    save_dir = settings.ML_MODEL_DIR
    models = []

    for meta_file in save_dir.glob("custom_*_meta.json"):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            models.append({
                "model_name":  meta["model_name"],
                "industry":    meta["industry"],
                "best_model":  meta["best_model"],
                "best_auc":    meta["best_auc"],
                "n_samples":   meta["n_samples"],
                "n_features":  meta["n_features"],
                "churn_rate":  meta["churn_rate"],
                "trained_at":  meta["trained_at"],
                "features":    meta["feature_columns"][:5],  # First 5 only
            })
        except Exception as e:
            logger.warning("Could not read model meta %s: %s", meta_file, e)

    return sorted(models, key=lambda x: x["trained_at"], reverse=True)


# ── Training status ────────────────────────────────────────────────────────────

def get_training_status() -> dict:
    """Current training status return karo."""
    with _status_lock:
        return dict(_training_status)


def _update_status(**kwargs) -> None:
    """Training status update karo (thread-safe)."""
    with _status_lock:
        _training_status.update(kwargs)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _read_csv(content: bytes) -> pd.DataFrame:
    """CSV bytes se DataFrame banao."""
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        # Try with different encoding
        df = pd.read_csv(io.BytesIO(content), encoding="latin-1")

    if len(df) < 50:
        raise ValueError(f"CSV mein sirf {len(df)} rows hain. Minimum 50 rows chahiye.")
    if len(df.columns) < 2:
        raise ValueError("CSV mein kam se kam 2 columns hone chahiye.")

    return df


def _find_target_column(df: pd.DataFrame, hint: str = "churn") -> str:
    """Churn/target column dhundho — case-insensitive."""
    # Direct match
    for col in df.columns:
        if col.lower() == hint.lower():
            return col

    # Partial match
    for col in df.columns:
        if hint.lower() in col.lower():
            return col

    # Common names
    common = ["churn", "churned", "is_churn", "target", "label",
              "cancelled", "left", "attrition", "exited"]
    for name in common:
        for col in df.columns:
            if col.lower() == name:
                return col

    raise ValueError(
        f"Target column '{hint}' not found in CSV. "
        f"Available columns: {list(df.columns)}. "
        f"Please rename your churn column to 'churn'."
    )


def _auto_preprocess(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Automatic preprocessing — kisi bhi CSV ke liye kaam karta hai.

    1. Target column ko 0/1 mein convert karo
    2. Missing values fill karo
    3. Categorical columns encode karo
    4. Feature list return karo
    """
    df = df.copy()

    # Convert target to binary
    y = df[target_col]
    if y.dtype == object or y.dtype.name == "category":
        # "Yes"/"No", "True"/"False", etc.
        positive_vals = {"yes", "true", "1", "churn", "churned", "left", "cancelled"}
        df[target_col] = y.astype(str).str.lower().isin(positive_vals).astype(int)
    else:
        df[target_col] = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # Drop ID-like columns (too many unique values = not useful)
    cols_to_drop = [target_col]
    for col in df.columns:
        if col == target_col:
            continue
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95:  # Likely ID column
            cols_to_drop.append(col)
            logger.debug("Dropping ID-like column: %s", col)

    feature_df = df.drop(columns=cols_to_drop, errors="ignore")

    # Handle missing values
    for col in feature_df.columns:
        if feature_df[col].dtype in [np.float64, np.int64, float, int]:
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())
        else:
            feature_df[col] = feature_df[col].fillna(feature_df[col].mode()[0] if len(feature_df[col].mode()) > 0 else "unknown")

    # Encode categorical columns
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)

    # Convert all to numeric
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0)

    feature_cols = list(feature_df.columns)
    result_df = feature_df.copy()
    result_df[target_col] = df[target_col]

    return result_df, feature_cols
