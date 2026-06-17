"""
backend/models/schemas.py
--------------------------
Input validation and response serialisation helpers.
Uses plain Python dataclasses + validators (no Pydantic dependency required).

All validate_*() functions raise ValueError with a descriptive message on failure.
"""
from __future__ import annotations
import re
from typing import Any


# ── Validators ────────────────────────────────────────────────────────────────

def _require(data: dict, *keys: str) -> None:
    missing = [k for k in keys if data.get(k) is None and k not in data]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")


def _require_any(data: dict, aliases: list, field_name: str) -> None:
    """Raise ValueError if none of the alias keys are present and non-None."""
    if not any(k in data and data[k] is not None for k in aliases):
        raise ValueError(f"Missing required field: {field_name}")


def _enum(value: str, allowed: list[str], field: str) -> str:
    if value not in allowed:
        raise ValueError(f"'{field}' must be one of {allowed}, got '{value}'")
    return value


def _positive_float(value: Any, field: str) -> float:
    try:
        v = float(value)
        if v < 0:
            raise ValueError()
        return round(v, 4)
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a non-negative number, got '{value}'")


def _non_neg_int(value: Any, field: str) -> int:
    """Parse non-negative integer; negative values are clamped to 0."""
    try:
        v = int(value)
        return max(0, v)   # clamp negatives to 0
    except (TypeError, ValueError):
        return 0  # default gracefully


def _valid_email(email: str) -> str:
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise ValueError(f"Invalid email address: '{email}'")
    return email.lower().strip()


# ── Customer schema ───────────────────────────────────────────────────────────

VALID_CONTRACTS    = ["Month-to-month", "One year", "Two year"]
VALID_PAYMENTS     = ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"]
VALID_SUBSCRIPTIONS = ["Basic", "Standard", "Premium"]
VALID_INTERNET     = ["DSL", "Fiber optic", "No"]
VALID_YES_NO       = ["Yes", "No"]
VALID_YES_NO_NOSVC = ["Yes", "No", "No internet service", "No phone service"]


def validate_customer(data: dict) -> dict:
    """
    Validate and normalise a raw customer dict.
    Returns a cleaned dict, raises ValueError on invalid input.
    """
    # Support both API snake_case and ML CamelCase field names
    _require_any(data, ["monthly_charges", "MonthlyCharges"], "monthly_charges")
    _require_any(data, ["contract", "Contract"], "contract")
    _require_any(data, ["payment_method", "PaymentMethod"], "payment_method")
    if "tenure" not in data:
        raise ValueError("Missing required field: tenure")

    clean: dict[str, Any] = {}

    clean["tenure"]          = _non_neg_int(data.get("tenure", 0), "tenure")
    clean["monthly_charges"] = _positive_float(data.get("monthly_charges") or data.get("MonthlyCharges", 0), "monthly_charges")
    clean["inactive_days"]   = _non_neg_int(data.get("inactive_days", 0), "inactive_days")
    clean["senior_citizen"]  = 1 if str(data.get("senior_citizen", "0")) in ("1", "true", "True") else 0

    # Optional demographics
    clean["gender"]     = str(data.get("gender", "")).strip() or None
    clean["partner"]    = str(data.get("partner", "No")).strip()
    clean["dependents"] = str(data.get("dependents", "No")).strip()

    # TotalCharges
    tc = data.get("total_charges") or data.get("TotalCharges")
    clean["total_charges"] = float(tc) if tc not in (None, "", " ") else (
        clean["monthly_charges"] * clean["tenure"])

    # Services
    for col in ["phone_service", "multiple_lines", "paperless_billing"]:
        src = data.get(col) or data.get("".join(w.capitalize() for w in col.split("_")), "No")
        clean[col] = str(src).strip()

    for col in ["online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies"]:
        src = data.get(col) or data.get("".join(w.capitalize() for w in col.split("_")), "No")
        clean[col] = str(src).strip()

    clean["internet_service"] = str(data.get("internet_service") or
                                    data.get("InternetService", "DSL")).strip()

    clean["contract"] = _enum(
        str(data.get("contract") or data.get("Contract", "Month-to-month")).strip(),
        VALID_CONTRACTS, "contract")

    clean["payment_method"] = _enum(
        str(data.get("payment_method") or data.get("PaymentMethod", "Electronic check")).strip(),
        VALID_PAYMENTS, "payment_method")

    sub = str(data.get("subscription_type") or data.get("SubscriptionType", "Standard")).strip()
    clean["subscription_type"] = sub if sub in VALID_SUBSCRIPTIONS else "Standard"

    return clean


def validate_signup(data: dict) -> dict:
    """Validate user signup payload."""
    _require(data, "name", "email", "password")
    name     = str(data["name"]).strip()
    email    = _valid_email(str(data["email"]))
    password = str(data["password"])
    role     = str(data.get("role", "viewer")).lower()

    if len(name) < 2:
        raise ValueError("Name must be at least 2 characters.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if role not in ("admin", "analyst", "viewer", "retention"):
        role = "viewer"

    return {"name": name, "email": email, "password": password, "role": role}


def validate_login(data: dict) -> dict:
    """Validate login payload."""
    _require(data, "email", "password")
    return {
        "email":    _valid_email(str(data["email"])),
        "password": str(data["password"]),
    }


def validate_pagination(args: dict) -> tuple[int, int]:
    """Parse and clamp page + page_size from query args."""
    from backend.core.config import settings
    page      = max(1, int(args.get("page", 1)))
    page_size = min(settings.MAX_PAGE_SIZE,
                    max(1, int(args.get("page_size", settings.DEFAULT_PAGE_SIZE))))
    return page, page_size


# ── Standard response helpers ─────────────────────────────────────────────────

def ok(data: Any, message: str = "Success", status: int = 200) -> tuple[dict, int]:
    """Return a standardised success response."""
    return {"success": True, "message": message, "data": data}, status


def err(message: str, code: str = "ERROR", status: int = 400,
        details: Any = None) -> tuple[dict, int]:
    """Return a standardised error response."""
    body: dict = {"success": False, "error": message, "code": code}
    if details:
        body["details"] = details
    return body, status
