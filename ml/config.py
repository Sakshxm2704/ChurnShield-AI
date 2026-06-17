"""
ml/config.py
------------
Central configuration for the ML pipeline.
All feature lists, hyperparameters, thresholds, and paths live here
so that every module reads from a single source of truth.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATA_RAW      = PROJECT_ROOT / "data" / "raw"
DATA_PROC     = PROJECT_ROOT / "data" / "processed"
MODEL_DIR     = PROJECT_ROOT / "ml" / "models" / "saved"
REPORTS_DIR   = PROJECT_ROOT / "reports"

for _d in (DATA_RAW, DATA_PROC, MODEL_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw dataset ────────────────────────────────────────────────────────────────
RAW_DATASET_PATH  = DATA_RAW / "telco_churn.csv"
PROC_DATASET_PATH = DATA_PROC / "telco_churn_processed.csv"

# ── Target & ID columns ────────────────────────────────────────────────────────
TARGET_COL  = "Churn"
ID_COL      = "customerID"

# ── Feature groups (raw column names before encoding) ─────────────────────────
NUMERIC_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "InactiveDays",
]

BINARY_COLS = [
    # Yes/No columns that map to 1/0
    "gender",          # Female=1
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

ORDINAL_COLS: list[str] = []   # none in this dataset

NOMINAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
    "SubscriptionType",
]

# ── Engineered feature names (created by feature_engineering.py) ───────────────
ENGINEERED_FEATURES = [
    "charge_per_tenure",
    "service_count",
    "is_high_value",
    "contract_risk_score",
    "payment_risk_score",
    "engagement_score",
    "tenure_band",
    "charge_band",
]

# ── Model artefact file names ──────────────────────────────────────────────────
MODEL_FILENAMES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest":       "random_forest.joblib",
    "xgboost":             "xgboost.joblib",
    "best_model":          "best_model.joblib",
}
PREPROCESSOR_FILENAME = "preprocessor.joblib"
FEATURE_NAMES_FILENAME = "feature_names.joblib"
MODEL_METADATA_FILENAME = "model_metadata.json"
SCALER_FILENAME = "scaler.joblib"

# ── Risk thresholds ────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "high":   0.65,   # churn_probability >= 0.65 → High Risk
    "medium": 0.35,   # 0.35 <= churn_probability < 0.65 → Medium Risk
    # below 0.35 → Low Risk
}

# ── Segmentation (KMeans) ──────────────────────────────────────────────────────
N_CLUSTERS          = 4
SEGMENT_LABELS      = ["Loyal", "Risky", "Premium", "Inactive"]
KMEANS_FEATURES     = ["tenure", "MonthlyCharges", "InactiveDays", "churn_probability"]

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
CV_FOLDS       = 5

# ── Model hyperparameters ──────────────────────────────────────────────────────
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Retention recommendation rules ────────────────────────────────────────────
RECOMMENDATION_RULES = [
    {
        "name": "high_monthly_charges",
        "condition": lambda r: r.get("MonthlyCharges", 0) > 70,
        "action": "Offer a 15 % loyalty discount on the next 3 billing cycles.",
        "savings_multiplier": 0.15,
    },
    {
        "name": "month_to_month_contract",
        "condition": lambda r: r.get("Contract", "") == "Month-to-month",
        "action": "Incentivise upgrade to an annual contract with a free month.",
        "savings_multiplier": 0.10,
    },
    {
        "name": "high_inactivity",
        "condition": lambda r: r.get("InactiveDays", 0) > 45,
        "action": "Send personalised re-engagement email with product highlights.",
        "savings_multiplier": 0.08,
    },
    {
        "name": "low_tenure",
        "condition": lambda r: r.get("tenure", 99) <= 6,
        "action": "Assign a dedicated onboarding specialist for the first 90 days.",
        "savings_multiplier": 0.12,
    },
    {
        "name": "electronic_check",
        "condition": lambda r: "electronic check" in r.get("PaymentMethod", "").lower(),
        "action": "Offer a 5 % auto-pay discount for switching to bank transfer.",
        "savings_multiplier": 0.05,
    },
    {
        "name": "fiber_no_security",
        "condition": lambda r: (
            r.get("InternetService", "") == "Fiber optic"
            and r.get("OnlineSecurity", "") == "No"
        ),
        "action": "Offer 3 months of free Online Security add-on.",
        "savings_multiplier": 0.07,
    },
    {
        "name": "senior_citizen",
        "condition": lambda r: r.get("SeniorCitizen", 0) == 1,
        "action": "Offer senior-tier support package with priority call routing.",
        "savings_multiplier": 0.09,
    },
    {
        "name": "no_tech_support",
        "condition": lambda r: r.get("TechSupport", "") == "No",
        "action": "Provide complimentary 24/7 tech-support trial for 30 days.",
        "savings_multiplier": 0.06,
    },
]
