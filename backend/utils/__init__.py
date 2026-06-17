"""
backend/utils/__init__.py
--------------------------
Shared utility helpers for the backend layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_request_id() -> str:
    """Generate a short unique request identifier."""
    return str(uuid.uuid4())[:8]


def sanitize_string(value: str, max_len: int = 255) -> str:
    """Strip and truncate a string input."""
    return str(value).strip()[:max_len]


def flatten_shap(shap_list: list[dict]) -> dict[str, float]:
    """Convert SHAP explanation list to {feature: value} dict."""
    return {item["feature"]: item.get("shap_value", 0) for item in shap_list}


def mask_email(email: str) -> str:
    """Mask an email for safe logging: jane@example.com → j***@example.com"""
    parts = email.split("@")
    if len(parts) != 2:
        return "***"
    local = parts[0]
    masked = local[0] + "***" if len(local) > 1 else "***"
    return f"{masked}@{parts[1]}"


def pagination_meta(total: int, page: int, page_size: int) -> dict[str, Any]:
    """Build a pagination metadata dict."""
    pages = max(1, (total + page_size - 1) // page_size)
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     pages,
        "has_next":  page < pages,
        "has_prev":  page > 1,
    }
