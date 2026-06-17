"""
backend/auth/dependencies.py
------------------------------
Flask auth decorators: require_auth, require_role, optional_auth.
Mirrors FastAPI Depends() pattern.
"""
from __future__ import annotations
import functools, logging
from flask import request, jsonify, g
from backend.auth.jwt_handler import verify_token
from backend.core.database import get_db, row_to_dict

logger = logging.getLogger(__name__)


def _extract_token() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.args.get("token")


def _get_user_from_token(token: str) -> dict | None:
    payload = verify_token(token)
    if not payload or payload.get("type") != "access":
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, name, email, role, is_active FROM users WHERE id = ?",
            (int(user_id),)).fetchone()
    user = row_to_dict(row)
    if not user or not user.get("is_active"):
        return None
    return user


def require_auth(f):
    """Require valid JWT. Injects current_user kwarg."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        token = _extract_token()
        if not token:
            return jsonify({"success": False, "error": "Authentication required.", "code": "MISSING_TOKEN"}), 401
        user = _get_user_from_token(token)
        if not user:
            return jsonify({"success": False, "error": "Token is invalid or expired.", "code": "INVALID_TOKEN"}), 401
        g.current_user = user
        return f(*args, current_user=user, **kwargs)
    return wrapper


def require_role(*allowed_roles: str):
    """Require auth + one of the allowed roles."""
    def decorator(f):
        @functools.wraps(f)
        @require_auth
        def wrapper(*args, current_user, **kwargs):
            if current_user["role"] not in allowed_roles:
                return jsonify({"success": False,
                    "error": f"Insufficient permissions. Required: {list(allowed_roles)}",
                    "code": "FORBIDDEN"}), 403
            return f(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def optional_auth(f):
    """Set g.current_user if valid token present, else None."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        token = _extract_token()
        g.current_user = _get_user_from_token(token) if token else None
        return f(*args, current_user=g.current_user, **kwargs)
    return wrapper
