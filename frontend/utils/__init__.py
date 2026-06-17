"""
frontend/utils/__init__.py
--------------------------
Frontend utility helpers for the Churn Intelligence Platform.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def call_api(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: dict | None = None,
    timeout: int = 30,
) -> tuple[int, dict | None]:
    """
    Make an HTTP request to the backend API.

    Returns (status_code, response_dict | None).
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(url, data=data, headers=headers, method=method.upper())

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None
    except Exception as exc:
        return 0, {"error": str(exc), "success": False}


def fmt_usd(value: float | None, compact: bool = True) -> str:
    """Format a number as USD."""
    if value is None:
        return "—"
    if compact:
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"${value/1_000:.0f}K"
    return f"${value:,.0f}"


def fmt_pct(value: float | None) -> str:
    """Format a fraction as percentage."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def risk_color(category: str) -> str:
    """Return a CSS/hex color for a risk category."""
    return {"High": "#fc8181", "Medium": "#f6ad55", "Low": "#48bb78"}.get(category, "#63b3ed")
