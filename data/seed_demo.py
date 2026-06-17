"""
data/seed_demo.py
------------------
Seeds the platform database with realistic demo data.

Creates:
  - 1 admin + 2 analyst users
  - 50 sample customers (varied risk profiles)
  - Predictions for all customers
  - Retention recommendations
  - Alert log entries for high-risk customers
  - 7 days of monitoring metrics

Usage
-----
    python -m data.seed_demo                    # seed everything
    python -m data.seed_demo --reset            # drop + reseed
    python -m data.seed_demo --customers 100    # custom count
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("seed_demo")

from backend.core.database import get_db, init_db
from backend.auth.password import hash_password

# ── Seed constants ────────────────────────────────────────────────────────────
random.seed(42)

CONTRACTS    = ["Month-to-month", "One year",   "Two year"]
PAYMENTS     = ["Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"]
INTERNET     = ["DSL", "Fiber optic", "No"]
SUBSCRIPTIONS= ["Basic", "Standard", "Premium"]
GENDERS      = ["Male", "Female"]
YES_NO       = ["Yes", "No"]

# Risk profile weights: (contract_idx, payment_idx, internet_idx, tenure_range, charges_range, inactive_range)
RISK_PROFILES = {
    "high":   {"contract": 0, "payment": 0, "internet": 1, "tenure": (1, 12),  "charges": (70, 100), "inactive": (40, 90)},
    "medium": {"contract": 1, "payment": 2, "internet": 0, "tenure": (12, 36), "charges": (45, 75),  "inactive": (15, 45)},
    "low":    {"contract": 2, "payment": 3, "internet": 0, "tenure": (30, 72), "charges": (25, 55),  "inactive": (0, 15)},
}

DEMO_USERS = [
    {"name": "Alex Morgan",    "email": "admin@demo.churn.io",    "password": "Admin1234!", "role": "admin"},
    {"name": "Sarah Chen",     "email": "analyst@demo.churn.io",  "password": "Demo1234!",  "role": "analyst"},
    {"name": "James Wilson",   "email": "viewer@demo.churn.io",   "password": "Demo1234!",  "role": "viewer"},
]

RETENTION_ACTIONS = [
    ("month_to_month_upgrade",          "Offer annual contract upgrade with 1 free month.",          85.0,  2),
    ("high_monthly_charges_discount",   "Apply 15% loyalty discount for next 3 billing cycles.",     120.0, 2),
    ("high_inactivity_reengagement",    "Send personalised re-engagement campaign with feature reel.",65.0,  1),
    ("low_tenure_onboarding",           "Assign dedicated Customer Success Manager for 90 days.",     75.0,  1),
    ("electronic_check_autopay",        "Offer 5% auto-pay discount to switch from e-check.",         40.0,  3),
    ("fiber_security_bundle",           "Offer 3 months free Online Security for Fiber customers.",   55.0,  3),
    ("no_tech_support_trial",           "Provide 30-day complimentary Tech Support trial.",           45.0,  4),
    ("senior_citizen_priority_support", "Enrol in Senior Care tier with priority support.",           70.0,  2),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rand_customer(profile_key: str) -> dict:
    p       = RISK_PROFILES[profile_key]
    tenure  = random.randint(*p["tenure"])
    charges = round(random.uniform(*p["charges"]), 2)
    return {
        "gender":            random.choice(GENDERS),
        "senior_citizen":    1 if random.random() < 0.16 else 0,
        "partner":           random.choice(YES_NO),
        "dependents":        random.choice(YES_NO),
        "tenure":            tenure,
        "phone_service":     random.choice(YES_NO),
        "multiple_lines":    random.choice(YES_NO),
        "internet_service":  INTERNET[p["internet"]],
        "online_security":   "No" if profile_key == "high" else random.choice(YES_NO),
        "online_backup":     random.choice(YES_NO),
        "device_protection": random.choice(YES_NO),
        "tech_support":      "No" if profile_key == "high" else random.choice(YES_NO),
        "streaming_tv":      random.choice(YES_NO),
        "streaming_movies":  random.choice(YES_NO),
        "contract":     CONTRACTS[p["contract"]],
        "payment_method":    PAYMENTS[p["payment"]],
        "paperless_billing": "Yes" if profile_key in ("high", "medium") else random.choice(YES_NO),
        "monthly_charges":   charges,
        "total_charges":     round(charges * tenure, 2),
        "inactive_days":     random.randint(*p["inactive"]),
        "subscription_type": SUBSCRIPTIONS[{"high": 2, "medium": 1, "low": 0}[profile_key]],
    }


def _churn_prob_for_profile(profile_key: str, customer: dict) -> float:
    """Generate a realistic churn probability based on profile."""
    base = {"high": 0.72, "medium": 0.42, "low": 0.12}[profile_key]
    noise = random.gauss(0, 0.06)
    return max(0.02, min(0.97, round(base + noise, 4)))


def _risk_category(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Seeding functions ─────────────────────────────────────────────────────────

def seed_users(conn) -> list[int]:
    """Seed demo users. Returns list of user IDs."""
    ids = []
    for u in DEMO_USERS:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (u["email"],)).fetchone()
        if existing:
            ids.append(existing[0])
            log.info("  User exists: %s", u["email"])
            continue
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?,?)",
            (u["name"], u["email"], hash_password(u["password"]), u["role"]),
        )
        ids.append(cur.lastrowid)
        log.info("  + User: %s (%s)", u["email"], u["role"])
    return ids


def seed_customers_and_predictions(conn, n: int = 50) -> list[int]:
    """Seed n customers with predictions and recommendations."""
    # Mix: 30% high, 35% medium, 35% low
    n_high   = max(1, int(n * 0.30))
    n_medium = max(1, int(n * 0.35))
    n_low    = n - n_high - n_medium

    profile_queue = (
        ["high"]   * n_high +
        ["medium"] * n_medium +
        ["low"]    * n_low
    )
    random.shuffle(profile_queue)

    customer_ids = []
    for i, profile in enumerate(profile_queue):
        cust = _rand_customer(profile)
        days_ago = random.randint(0, 180)

        # Insert customer
        cur = conn.execute(
            """INSERT INTO customers
               (gender, senior_citizen, partner, dependents, tenure,
                phone_service, multiple_lines, internet_service, online_security,
                online_backup, device_protection, tech_support, streaming_tv,
                streaming_movies, contract, payment_method, paperless_billing,
                monthly_charges, total_charges, inactive_days, subscription_type,
                created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                cust["gender"], cust["senior_citizen"], cust["partner"],
                cust["dependents"], cust["tenure"], cust["phone_service"],
                cust["multiple_lines"], cust["internet_service"],
                cust["online_security"], cust["online_backup"],
                cust["device_protection"], cust["tech_support"],
                cust["streaming_tv"], cust["streaming_movies"],
                cust["contract"], cust["payment_method"],
                cust["paperless_billing"], cust["monthly_charges"],
                cust["total_charges"], cust["inactive_days"],
                cust["subscription_type"], _days_ago(days_ago),
            ),
        )
        cid = cur.lastrowid
        customer_ids.append(cid)

        # Insert prediction
        prob       = _churn_prob_for_profile(profile, cust)
        risk_cat   = _risk_category(prob)
        risk_score = min(100, max(0, int(prob * 100 + random.randint(-5, 5))))
        churn_label= "Churn" if prob >= 0.5 else "No Churn"

        # Simple SHAP mock
        shap_vals = json.dumps([
            {"feature": "Contract_Two year",              "shap_value": round(-0.45 if cust["contract"]=="Two year" else 0.45, 3)},
            {"feature": "MonthlyCharges",                 "shap_value": round((cust["monthly_charges"] - 64) / 100, 3)},
            {"feature": "tenure",                         "shap_value": round(-(cust["tenure"] - 24) / 120, 3)},
            {"feature": "PaymentMethod_Electronic check", "shap_value": 0.312 if cust["payment_method"]=="Electronic check" else -0.1},
            {"feature": "InactiveDays",                   "shap_value": round(cust["inactive_days"] / 200, 3)},
        ])

        pcur = conn.execute(
            """INSERT INTO predictions
               (customer_id, churn_probability, risk_score, prediction_label,
                churn_label, model_used, shap_values, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                cid, prob, risk_score, risk_cat, churn_label,
                "logistic_regression", shap_vals, _days_ago(days_ago - random.randint(0, 2)),
            ),
        )
        pid = pcur.lastrowid

        # Retention recommendations for high/medium risk
        if profile in ("high", "medium"):
            # Pick 2-3 relevant actions
            n_recs = 3 if profile == "high" else 2
            actions = random.sample(RETENTION_ACTIONS, k=min(n_recs, len(RETENTION_ACTIONS)))
            for rule_name, action, savings, priority in actions:
                weighted_savings = round(savings * (0.5 + prob * 0.5) * cust["monthly_charges"] / 65, 2)
                conn.execute(
                    """INSERT INTO recommendations
                       (customer_id, recommended_action, estimated_savings,
                        priority, is_completed, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        cid, action, weighted_savings, priority,
                        1 if random.random() < 0.15 else 0,
                        _days_ago(days_ago - random.randint(0, 1)),
                    ),
                )

        # Alert log for high-risk customers
        if risk_cat == "High" and prob >= 0.65:
            conn.execute(
                """INSERT INTO alert_log
                   (customer_id, prediction_id, alert_type, risk_category,
                    churn_probability, email_sent, email_status, message)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    cid, pid, "high_risk", "High", prob,
                    1 if random.random() < 0.7 else 0,
                    "sent" if random.random() < 0.7 else "pending",
                    f"Customer #{cid} flagged with {prob*100:.1f}% churn probability. "
                    f"Monthly charges: ${cust['monthly_charges']:.2f}. "
                    f"Inactive: {cust['inactive_days']} days.",
                ),
            )
        elif profile == "medium" and cust["inactive_days"] >= 30:
            conn.execute(
                """INSERT INTO alert_log
                   (customer_id, prediction_id, alert_type, risk_category,
                    churn_probability, email_sent, email_status, message)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    cid, pid, "medium_risk", "Medium", prob,
                    0, "pending",
                    f"Customer #{cid} at medium risk ({prob*100:.1f}%) with {cust['inactive_days']} inactive days.",
                ),
            )

    return customer_ids


def seed_monitoring_metrics(conn) -> None:
    """Seed 7 days of daily monitoring metrics."""
    for d in range(7, 0, -1):
        ts = _days_ago(d)
        day_preds  = random.randint(45, 180)
        day_high   = int(day_preds * random.uniform(0.25, 0.35))
        day_medium = int(day_preds * random.uniform(0.30, 0.40))
        day_api    = day_preds * random.randint(4, 8)
        day_users  = random.randint(3, 12)

        for metric, value, label in [
            ("prediction_count",   day_preds,  "daily"),
            ("high_risk_count",    day_high,   "daily"),
            ("medium_risk_count",  day_medium, "daily"),
            ("api_call_count",     day_api,    "daily"),
            ("active_users",       day_users,  "daily"),
            ("avg_latency_ms",     round(random.uniform(80, 220), 1), "daily"),
        ]:
            conn.execute(
                "INSERT INTO monitoring_metrics (metric_name, metric_value, metric_label, created_at) VALUES (?,?,?,?)",
                (metric, value, label, ts),
            )

    log.info("  + 7 days of monitoring metrics")


def seed_api_logs(conn) -> None:
    """Seed 100 sample API access log entries."""
    endpoints = [
        ("/api/v1/predict",            "POST", 200),
        ("/api/v1/customers",          "GET",  200),
        ("/api/v1/analytics",          "GET",  200),
        ("/api/v1/segments",           "GET",  200),
        ("/api/v1/recommendations",    "GET",  200),
        ("/api/v1/analytics/revenue",  "GET",  200),
        ("/api/v1/batch_predict",      "POST", 200),
        ("/api/v1/auth/login",         "POST", 200),
        ("/api/v1/predictions",        "GET",  200),
        ("/api/v1/monitoring",         "GET",  200),
    ]
    for i in range(100):
        path, method, status = random.choice(endpoints)
        if random.random() < 0.03:
            status = 500
        days_ago_val = random.randint(0, 7)
        ts = _days_ago(days_ago_val)
        conn.execute(
            """INSERT INTO api_logs
               (method, path, status_code, user_id, duration_ms, ip_address, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                method, path, status,
                random.randint(1, 3),
                round(random.uniform(50, 400), 1),
                f"192.168.1.{random.randint(1, 50)}",
                ts,
            ),
        )
    log.info("  + 100 API log entries")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_seed(n_customers: int = 50, reset: bool = False) -> None:
    log.info("=" * 60)
    log.info("Churn Intelligence Platform — Demo Data Seeder")
    log.info("=" * 60)

    init_db()

    with get_db() as conn:
        if reset:
            log.warning("⚠  Resetting demo data tables...")
            for table in ["alert_log", "recommendations", "predictions",
                          "customers", "monitoring_metrics", "api_logs", "users"]:
                conn.execute(f"DELETE FROM {table}")
            log.info("Tables cleared.")

        log.info("Seeding users...")
        user_ids = seed_users(conn)
        log.info("  %d users ready.", len(user_ids))

        log.info("Seeding %d customers + predictions...", n_customers)
        cust_ids = seed_customers_and_predictions(conn, n_customers)
        log.info("  %d customers created.", len(cust_ids))

        log.info("Seeding monitoring metrics...")
        seed_monitoring_metrics(conn)

        log.info("Seeding API logs...")
        seed_api_logs(conn)

    # Summary
    with get_db() as conn:
        counts = {}
        for t in ["users", "customers", "predictions", "recommendations",
                  "alert_log", "api_logs", "monitoring_metrics"]:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

    log.info("")
    log.info("─" * 40)
    log.info("Seed complete!")
    for table, count in counts.items():
        log.info("  %-25s %d rows", table, count)
    log.info("─" * 40)
    log.info("")
    log.info("Demo login credentials:")
    for u in DEMO_USERS:
        log.info("  %-35s %s", u["email"], u["password"])
    log.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the Churn Intelligence Platform with demo data.")
    parser.add_argument("--reset",     action="store_true", help="Clear existing data before seeding.")
    parser.add_argument("--customers", type=int, default=50, help="Number of demo customers (default: 50).")
    args = parser.parse_args()
    run_seed(n_customers=args.customers, reset=args.reset)
