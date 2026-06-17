"""
tests/conftest.py
------------------
Shared helpers, fixtures, and test data for the entire suite.
"""
from __future__ import annotations
import io, json, os, sys, tempfile, threading, unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Set test env BEFORE any imports ──────────────────────────────────────────
os.environ["APP_ENV"]   = "testing"
os.environ["DEBUG"]     = "false"
os.environ["SECRET_KEY"]= "test-secret-key-at-least-32-chars-long!!"

# ── Shared payloads ───────────────────────────────────────────────────────────
CUSTOMER_PAYLOAD: dict[str, Any] = {
    "tenure": 6, "monthly_charges": 89.99,
    "contract": "Month-to-month", "payment_method": "Electronic check",
    "inactive_days": 45, "subscription_type": "Standard",
    "internet_service": "Fiber optic", "online_security": "No",
    "tech_support": "No", "save_customer": False, "include_shap": False,
}
HIGH_RISK_PAYLOAD  = {**CUSTOMER_PAYLOAD, "tenure": 2, "monthly_charges": 95.0,
                      "inactive_days": 75, "contract": "Month-to-month",
                      "payment_method": "Electronic check"}
LOW_RISK_PAYLOAD   = {**CUSTOMER_PAYLOAD, "tenure": 60, "monthly_charges": 25.0,
                      "inactive_days": 2, "contract": "Two year",
                      "payment_method": "Bank transfer (automatic)"}
MEDIUM_RISK_PAYLOAD= {**CUSTOMER_PAYLOAD, "tenure": 18, "monthly_charges": 55.0,
                      "inactive_days": 28, "contract": "One year",
                      "payment_method": "Credit card (automatic)"}
SIGNUP_PAYLOAD     = {"name": "Test Admin", "email": "admin@tp.io",
                      "password": "TestPass1234!", "role": "admin"}

SAMPLE_CUSTOMER_RECORD: dict[str, Any] = {
    "customerID": "T001", "tenure": 6, "MonthlyCharges": 89.99, "TotalCharges": 539.94,
    "Contract": "Month-to-month", "PaymentMethod": "Electronic check",
    "InactiveDays": 45, "SubscriptionType": "Standard",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "TechSupport": "No",
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "PaperlessBilling": "Yes",
}

# ── DB isolation helper ───────────────────────────────────────────────────────
class IsolatedDBMixin:
    """Mixin that gives each test its own temp SQLite file, avoiding shared state."""
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        # Patch the DB path at module level (thread-local conn will be recreated)
        from backend.core import database as db_mod
        from backend.core import config as cfg_mod
        self._orig_url = cfg_mod.settings.DATABASE_URL
        cfg_mod.settings.DATABASE_URL = f"sqlite:///{self._tmp.name}"
        # Reset thread-local connection so next get_db() opens fresh file
        if hasattr(db_mod._local, "conn") and db_mod._local.conn:
            try: db_mod._local.conn.close()
            except: pass
            db_mod._local.conn = None
        db_mod.init_db()

    def tearDown(self):
        from backend.core import database as db_mod
        from backend.core import config as cfg_mod
        if hasattr(db_mod._local, "conn") and db_mod._local.conn:
            try: db_mod._local.conn.close()
            except: pass
            db_mod._local.conn = None
        cfg_mod.settings.DATABASE_URL = self._orig_url
        try: os.unlink(self._tmp.name)
        except: pass

# ── Flask app fixture ─────────────────────────────────────────────────────────
_flask_app = None

def get_test_app():
    global _flask_app
    if _flask_app is None:
        from backend.app import create_app
        _flask_app = create_app()
        _flask_app.config["TESTING"] = True
    return _flask_app

def get_auth_token(client, role="admin") -> str:
    import random
    email = f"u{random.randint(100000,999999)}@test.io"
    client.post("/api/v1/auth/signup",
        data=json.dumps({"name": "Test User", "email": email,
                         "password": "Pass1234!", "role": role}),
        content_type="application/json")
    r = client.post("/api/v1/auth/login",
        data=json.dumps({"email": email, "password": "Pass1234!"}),
        content_type="application/json")
    d = r.get_json()
    return d["data"]["access_token"]

def auth_hdr(token): return {"Authorization": f"Bearer {token}"}

def make_csv_bytes():
    return (b"tenure,monthly_charges,contract,payment_method,inactive_days\n"
            b"3,89.99,Month-to-month,Electronic check,60\n"
            b"24,45.0,One year,Bank transfer (automatic),5\n"
            b"60,25.0,Two year,Credit card (automatic),2\n")

# ── Additional fixtures needed by test_unit.py ───────────────────────────────
import numpy as np

HIGH_RISK_CUSTOMER: dict = {
    "customerID": "HR001", "tenure": 2, "MonthlyCharges": 95.0,
    "TotalCharges": "190.0", "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "TechSupport": "No", "InactiveDays": 75,
    "SubscriptionType": "Premium", "gender": "Male", "SeniorCitizen": 0,
    "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "PaperlessBilling": "Yes",
    "Churn": 1,
}

LOW_RISK_CUSTOMER: dict = {
    "customerID": "LR001", "tenure": 60, "MonthlyCharges": 25.0,
    "TotalCharges": "1500.0", "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "TechSupport": "Yes", "InactiveDays": 2,
    "SubscriptionType": "Basic", "gender": "Female", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "Yes", "PhoneService": "Yes",
    "MultipleLines": "No", "OnlineBackup": "Yes", "DeviceProtection": "Yes",
    "StreamingTV": "No", "StreamingMovies": "No", "PaperlessBilling": "No",
    "Churn": 0,
}

VALID_CUSTOMER: dict = {**SAMPLE_CUSTOMER_RECORD}


# ── App factory helpers ───────────────────────────────────────────────────────
def make_app():
    """Create a test Flask app instance."""
    import os
    os.environ.setdefault("SECRET_KEY", "test-secret-key-at-least-32-chars-long!!")
    os.environ.setdefault("APP_ENV", "testing")
    from backend.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app


def get_auth_token(client, role: str = "admin", email: str = None) -> str:
    """Signup + login and return JWT access token."""
    import json, random
    if email is None:
        email = f"perf_{random.randint(100000, 999999)}@test.io"
    client.post(
        "/api/v1/auth/signup",
        data=json.dumps({"name": "Perf User", "email": email,
                         "password": "Pass1234!", "role": role}),
        content_type="application/json",
    )
    r = client.post(
        "/api/v1/auth/login",
        data=json.dumps({"email": email, "password": "Pass1234!"}),
        content_type="application/json",
    )
    data = r.get_json()
    return data["data"]["access_token"]
