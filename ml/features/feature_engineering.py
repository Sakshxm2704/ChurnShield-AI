"""
ml/features/feature_engineering.py
------------------------------------
Derived feature creation for the Telco Churn dataset.

All transformations are pure functions that accept a DataFrame and return
an enriched copy.  They can be applied to both training and inference data.

Engineered features
-------------------
charge_per_tenure       Avg monthly spend per active month (value density)
service_count           Number of active add-on services (0-8)
is_high_value           Binary flag: monthly charges > 70
contract_risk_score     Ordinal risk encoding of contract type (0-2)
payment_risk_score      Ordinal risk encoding of payment method (0-3)
engagement_score        Composite engagement (inverse of inactivity + service count)
tenure_band             Ordinal band: New(0) / Growing(1) / Established(2) / Loyal(3)
charge_band             Ordinal band: Low(0) / Mid(1) / High(2) / Premium(3)
total_charge_ratio      TotalCharges / (tenure × MonthlyCharges) — billing completeness
has_streaming           Any streaming service subscribed (binary)
has_security            Online security or tech support (binary)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Contract risk mapping (higher = more likely to churn) ─────────────────────
_CONTRACT_RISK = {
    "Month-to-month": 2,
    "One year":       1,
    "Two year":       0,
}

# ── Payment method risk mapping ───────────────────────────────────────────────
_PAYMENT_RISK = {
    "Electronic check":             3,
    "Mailed check":                 2,
    "Bank transfer (automatic)":    0,
    "Credit card (automatic)":      0,
}

# ── Add-on service columns (each counts +1 when "Yes") ────────────────────────
_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_charge_per_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """charge_per_tenure = MonthlyCharges / (tenure + 1)."""
    df = df.copy()
    # When tenure is 0, charge_per_tenure is undefined — return 0
    denom = df["tenure"].clip(lower=0)
    df["charge_per_tenure"] = df["MonthlyCharges"].where(denom > 0, 0) / denom.clip(lower=1)
    df.loc[denom == 0, "charge_per_tenure"] = 0
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """service_count = number of active add-on services."""
    df = df.copy()
    available = [c for c in _SERVICE_COLS if c in df.columns]
    df["service_count"] = (df[available] == "Yes").sum(axis=1)
    return df


def add_is_high_value(df: pd.DataFrame) -> pd.DataFrame:
    """is_high_value = 1 if MonthlyCharges > 70, else 0."""
    df = df.copy()
    df["is_high_value"] = (df["MonthlyCharges"] > 70).astype(int)
    return df


def add_contract_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """contract_risk_score: Month-to-month=2, One year=1, Two year=0."""
    df = df.copy()
    df["contract_risk_score"] = df["Contract"].map(_CONTRACT_RISK).fillna(1).astype(int)
    return df


def add_payment_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """payment_risk_score: Electronic check=3, Mailed=2, auto=0."""
    df = df.copy()
    df["payment_risk_score"] = df["PaymentMethod"].map(_PAYMENT_RISK).fillna(1).astype(int)
    return df


def add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    engagement_score = service_count * 10 - InactiveDays * 0.5
    Clamped to [0, 100].
    """
    df = df.copy()
    svc = df["service_count"] if "service_count" in df.columns else 0
    inactive = df.get("InactiveDays", pd.Series(0, index=df.index))
    df["engagement_score"] = np.clip(svc * 10 - inactive * 0.5, 0, 100)
    return df


def add_tenure_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    tenure_band:
    0  = New        (≤ 6 months)
    1  = Growing    (7–24 months)
    2  = Established(25–48 months)
    3  = Loyal      (> 48 months)
    """
    df = df.copy()
    bins   = [-1, 6, 24, 48, float("inf")]
    labels = [0, 1, 2, 3]
    df["tenure_band"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(int)
    return df


def add_charge_band(df: pd.DataFrame) -> pd.DataFrame:
    """
    charge_band:
    0 = Low     (≤ 35)
    1 = Mid     (36–65)
    2 = High    (66–85)
    3 = Premium (> 85)
    """
    df = df.copy()
    bins   = [-1, 35, 65, 85, float("inf")]
    labels = [0, 1, 2, 3]
    df["charge_band"] = pd.cut(df["MonthlyCharges"], bins=bins, labels=labels).astype(int)
    return df


def add_total_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    total_charge_ratio = TotalCharges / (tenure * MonthlyCharges).
    Values near 1.0 indicate consistent billing history.
    """
    df = df.copy()
    # TotalCharges may be a string column in raw data — coerce to numeric
    tc_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    expected = df["tenure"] * df["MonthlyCharges"]
    df["total_charge_ratio"] = np.where(
        expected > 0,
        (tc_numeric / expected).clip(0, 2),
        1.0,
    )
    return df


def add_has_streaming(df: pd.DataFrame) -> pd.DataFrame:
    """has_streaming = 1 if StreamingTV or StreamingMovies is 'Yes'."""
    df = df.copy()
    streaming_cols = [c for c in ["StreamingTV", "StreamingMovies"] if c in df.columns]
    df["has_streaming"] = (df[streaming_cols] == "Yes").any(axis=1).astype(int)
    return df


def add_has_security(df: pd.DataFrame) -> pd.DataFrame:
    """has_security = 1 if OnlineSecurity or TechSupport is 'Yes'."""
    df = df.copy()
    sec_cols = [c for c in ["OnlineSecurity", "TechSupport"] if c in df.columns]
    df["has_security"] = (df[sec_cols] == "Yes").any(axis=1).astype(int)
    return df


# ── Master transformer ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to *df* and return an enriched copy.
    Safe to call on both training and inference DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (but not yet preprocessed) DataFrame.

    Returns
    -------
    pd.DataFrame
        Original columns + all engineered features.
    """
    df = add_charge_per_tenure(df)
    df = add_service_count(df)
    df = add_is_high_value(df)
    df = add_contract_risk_score(df)
    df = add_payment_risk_score(df)
    df = add_engagement_score(df)
    df = add_tenure_band(df)
    df = add_charge_band(df)
    df = add_total_charge_ratio(df)
    df = add_has_streaming(df)
    df = add_has_security(df)

    new_cols = [
        "charge_per_tenure", "service_count", "is_high_value",
        "contract_risk_score", "payment_risk_score", "engagement_score",
        "tenure_band", "charge_band", "total_charge_ratio",
        "has_streaming", "has_security",
    ]
    logger.debug("Engineered %d new features: %s", len(new_cols), new_cols)
    return df


def get_engineered_feature_names() -> list[str]:
    """Return the list of feature names added by ``engineer_features``."""
    return [
        "charge_per_tenure", "service_count", "is_high_value",
        "contract_risk_score", "payment_risk_score", "engagement_score",
        "tenure_band", "charge_band", "total_charge_ratio",
        "has_streaming", "has_security",
    ]
