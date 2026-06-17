"""
backend/routes/auth_routes.py
------------------------------
Authentication endpoints:
  POST /api/v1/auth/signup   — register a new user
  POST /api/v1/auth/login    — exchange credentials for JWT tokens
  POST /api/v1/auth/refresh  — get a new access token via refresh token
  GET  /api/v1/auth/me       — return authenticated user profile
  POST /api/v1/auth/logout   — client-side token discard (stateless)
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, g

from backend.auth.dependencies import require_auth
from backend.auth.jwt_handler  import (create_access_token, create_refresh_token,
                                        verify_token)
from backend.auth.password     import hash_password, verify_password
from backend.models.schemas    import validate_signup, validate_login, ok, err
from backend.services.db_service import create_user, get_user_by_email, get_user_by_id

logger = logging.getLogger(__name__)
bp     = Blueprint("auth", __name__, url_prefix="/api/v1/auth")


# ── POST /signup ──────────────────────────────────────────────────────────────
@bp.route("/signup", methods=["POST"])
def signup():
    """
    Register a new platform user.
    ---
    tags: [Auth]
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [name, email, password]
            properties:
              name:     {type: string, example: "Jane Doe"}
              email:    {type: string, format: email, example: "jane@example.com"}
              password: {type: string, minLength: 8, example: "Secure123!"}
              role:     {type: string, enum: [admin,analyst,viewer,retention], default: viewer}
    responses:
      201:
        description: User created successfully
      400:
        description: Validation error or email already registered
    """
    data = request.get_json(silent=True) or {}
    try:
        clean = validate_signup(data)
    except ValueError as e:
        return jsonify(err(str(e), "VALIDATION_ERROR")[0]), 400

    # Email uniqueness check
    if get_user_by_email(clean["email"]):
        return jsonify(err("Email is already registered.", "EMAIL_TAKEN")[0]), 400

    pw_hash = hash_password(clean["password"])
    user    = create_user(clean["name"], clean["email"], pw_hash, clean["role"])

    # Remove sensitive field
    user.pop("password_hash", None)

    logger.info("New user registered: %s  role=%s", user["email"], user["role"])
    return jsonify(ok(user, "Account created successfully.")[0]), 201


# ── POST /login ───────────────────────────────────────────────────────────────
@bp.route("/login", methods=["POST"])
def login():
    """
    Exchange email + password for JWT access and refresh tokens.
    ---
    tags: [Auth]
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [email, password]
            properties:
              email:    {type: string, format: email}
              password: {type: string}
    responses:
      200:
        description: Login successful, returns access_token + refresh_token
      401:
        description: Invalid credentials
    """
    data = request.get_json(silent=True) or {}
    try:
        clean = validate_login(data)
    except ValueError as e:
        return jsonify(err(str(e), "VALIDATION_ERROR")[0]), 400

    user = get_user_by_email(clean["email"])
    if not user or not verify_password(clean["password"], user["password_hash"]):
        return jsonify(err("Invalid email or password.", "INVALID_CREDENTIALS")[0]), 401

    if not user.get("is_active"):
        return jsonify(err("Account is deactivated. Contact an administrator.", "ACCOUNT_DISABLED")[0]), 403

    token_data = {"sub": str(user["id"]), "email": user["email"], "role": user["role"]}
    access_token  = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    user_out = {k: v for k, v in user.items() if k != "password_hash"}
    logger.info("User logged in: %s", user["email"])

    return jsonify(ok({
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "token_type":    "bearer",
        "expires_in":    3600,
        "user":          user_out,
    }, "Login successful.")[0]), 200


# ── POST /refresh ─────────────────────────────────────────────────────────────
@bp.route("/refresh", methods=["POST"])
def refresh():
    """
    Obtain a new access token using a valid refresh token.
    ---
    tags: [Auth]
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [refresh_token]
            properties:
              refresh_token: {type: string}
    responses:
      200:
        description: New access token
      401:
        description: Invalid or expired refresh token
    """
    data  = request.get_json(silent=True) or {}
    token = data.get("refresh_token", "").strip()
    if not token:
        return jsonify(err("refresh_token is required.", "MISSING_TOKEN")[0]), 400

    payload = verify_token(token)
    if not payload or payload.get("type") != "refresh":
        return jsonify(err("Refresh token is invalid or expired.", "INVALID_TOKEN")[0]), 401

    user = get_user_by_id(int(payload["sub"]))
    if not user or not user.get("is_active"):
        return jsonify(err("User not found or account disabled.", "INVALID_TOKEN")[0]), 401

    new_access = create_access_token(
        {"sub": str(user["id"]), "email": user["email"], "role": user["role"]})

    return jsonify(ok({
        "access_token": new_access,
        "token_type":   "bearer",
        "expires_in":   3600,
    }, "Token refreshed.")[0]), 200


# ── GET /me ───────────────────────────────────────────────────────────────────
@bp.route("/me", methods=["GET"])
@require_auth
def me(current_user):
    """
    Return the profile of the currently authenticated user.
    ---
    tags: [Auth]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Authenticated user profile
      401:
        description: Not authenticated
    """
    user = get_user_by_id(current_user["id"])
    if not user:
        return jsonify(err("User not found.", "NOT_FOUND")[0]), 404
    user.pop("password_hash", None)
    return jsonify(ok(user)[0]), 200


# ── POST /logout ──────────────────────────────────────────────────────────────
@bp.route("/logout", methods=["POST"])
@require_auth
def logout(current_user):
    """
    Logout endpoint — client must discard the token.
    Server-side: stateless JWT (no blacklist in this implementation).
    ---
    tags: [Auth]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Logged out (client should discard token)
    """
    logger.info("User logged out: %s", current_user["email"])
    return jsonify(ok(None, "Logged out successfully. Please discard your token.")[0]), 200
