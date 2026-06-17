"""
data/generate_dataset.py
------------------------
Generates a realistic synthetic Telco Customer Churn dataset that mirrors
the structure of the IBM Telco churn benchmark.

Usage
-----
    python -m data.generate_dataset                      # 7 000 rows
    python -m data.generate_dataset --rows 20000         # custom size
    python -m data.generate_dataset --seed 99 --rows 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent / "raw"
DEFAULT_ROWS = 7_000
DEFAULT_SEED = 42


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_telco_dataset(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Return a DataFrame with the same schema as the IBM Telco Churn dataset.

    Churn probability is derived from realistic business rules:
    - month-to-month contract → higher churn
    - electronic check payment → higher churn
    - low tenure → higher churn
    - high monthly charges + low tenure → very high churn
    - many inactivity days → higher churn
    """
    rng = _rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────
    gender          = rng.choice(["Male", "Female"], n_rows)
    senior_citizen  = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner         = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents      = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])

    # ── Contract & billing ────────────────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_rows, p=[0.55, 0.24, 0.21]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_rows, p=[0.34, 0.23, 0.22, 0.21]
    )
    paperless_billing = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])

    # ── Tenure (months) — skewed by contract type ──────────────────────────
    tenure = np.where(
        contract == "Month-to-month",
        rng.integers(1, 36, n_rows),
        np.where(
            contract == "One year",
            rng.integers(12, 60, n_rows),
            rng.integers(24, 72, n_rows),
        )
    )

    # ── Service subscriptions ──────────────────────────────────────────────
    phone_service       = rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    multiple_lines      = np.where(
        phone_service == "Yes",
        rng.choice(["Yes", "No"], n_rows, p=[0.42, 0.58]),
        "No phone service"
    )
    internet_service    = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22])
    online_security     = np.where(internet_service != "No",
                                   rng.choice(["Yes", "No"], n_rows, p=[0.29, 0.71]), "No internet service")
    online_backup       = np.where(internet_service != "No",
                                   rng.choice(["Yes", "No"], n_rows, p=[0.34, 0.66]), "No internet service")
    device_protection   = np.where(internet_service != "No",
                                   rng.choice(["Yes", "No"], n_rows, p=[0.34, 0.66]), "No internet service")
    tech_support        = np.where(internet_service != "No",
                                   rng.choice(["Yes", "No"], n_rows, p=[0.29, 0.71]), "No internet service")
    streaming_tv        = np.where(internet_service != "No",
                                   rng.choice(["Yes", "No"], n_rows, p=[0.38, 0.62]), "No internet service")
    streaming_movies    = np.where(internet_service != "No",
                                   rng.choice(["Yes", "No"], n_rows, p=[0.39, 0.61]), "No internet service")

    # ── Monthly charges (business-rule derived) ────────────────────────────
    base_charge = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
    addon_charge = (
        (multiple_lines == "Yes").astype(int) * 10 +
        (online_security == "Yes").astype(int) * 5 +
        (online_backup == "Yes").astype(int) * 5 +
        (device_protection == "Yes").astype(int) * 5 +
        (tech_support == "Yes").astype(int) * 5 +
        (streaming_tv == "Yes").astype(int) * 8 +
        (streaming_movies == "Yes").astype(int) * 8
    )
    monthly_charges = base_charge + addon_charge + rng.uniform(-3, 5, n_rows)
    monthly_charges = np.round(np.clip(monthly_charges, 18, 120), 2)

    total_charges = np.round(monthly_charges * tenure + rng.uniform(-10, 50, n_rows), 2)
    total_charges = np.clip(total_charges, 0, None)

    # ── Inactivity days (platform-specific feature) ────────────────────────
    inactive_days = np.where(
        contract == "Month-to-month",
        rng.integers(0, 90, n_rows),
        rng.integers(0, 30, n_rows)
    )

    # ── Subscription type ─────────────────────────────────────────────────
    subscription_type = np.where(
        monthly_charges > 80, "Premium",
        np.where(monthly_charges > 50, "Standard", "Basic")
    )

    # ── Churn label (business-rule probability) ───────────────────────────
    churn_score = (
        (contract == "Month-to-month").astype(float) * 0.35 +
        (payment_method == "Electronic check").astype(float) * 0.15 +
        (internet_service == "Fiber optic").astype(float) * 0.10 +
        (paperless_billing == "Yes").astype(float) * 0.05 +
        (1 / (tenure + 1)) * 0.20 +
        (inactive_days / 90) * 0.15
    )
    churn_noise = rng.uniform(-0.10, 0.10, n_rows)
    churn_prob  = np.clip(churn_score + churn_noise, 0.02, 0.95)
    churn       = (rng.uniform(0, 1, n_rows) < churn_prob).astype(int)

    # ── Inject ~3 % missing values in TotalCharges (mirrors real dataset) ─
    missing_mask = rng.random(n_rows) < 0.03
    total_charges_str = total_charges.astype(str)
    total_charges_str = np.where(missing_mask, "", total_charges_str)

    # ── Inject ~1 % duplicate rows ────────────────────────────────────────
    df = pd.DataFrame({
        "customerID":        [f"CUST-{str(i).zfill(6)}" for i in range(1, n_rows + 1)],
        "gender":            gender,
        "SeniorCitizen":     senior_citizen,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone_service,
        "MultipleLines":     multiple_lines,
        "InternetService":   internet_service,
        "OnlineSecurity":    online_security,
        "OnlineBackup":      online_backup,
        "DeviceProtection":  device_protection,
        "TechSupport":       tech_support,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_movies,
        "Contract":          contract,
        "PaperlessBilling":  paperless_billing,
        "PaymentMethod":     payment_method,
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges_str,
        "InactiveDays":      inactive_days,
        "SubscriptionType":  subscription_type,
        "Churn":             churn,
    })

    # Duplicate ~1 % of rows
    n_dups = max(1, int(n_rows * 0.01))
    dup_idx = rng.choice(df.index, size=n_dups, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Telco churn dataset.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "telco_churn.csv"

    print(f"Generating {args.rows:,} rows (seed={args.seed}) …")
    df = generate_telco_dataset(args.rows, args.seed)
    df.to_csv(out_path, index=False)

    churn_rate = df["Churn"].mean() * 100
    print(f"✓ Saved → {out_path}  ({len(df):,} rows, churn rate {churn_rate:.1f} %)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
