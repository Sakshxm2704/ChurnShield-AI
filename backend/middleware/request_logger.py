"""
backend/middleware/request_logger.py
--------------------------------------
Flask before/after request hooks for:
  - Request timing (X-Process-Time-Ms header)
  - Structured API access logging (database + file)
  - CORS headers injection
  - Global error catching
"""
from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, request, jsonify

from backend.core.config import settings

access_log  = logging.getLogger("api.access")
error_log   = logging.getLogger("api.error")


def register_middleware(app: Flask) -> None:
    """Register all middleware hooks on *app*."""

    # ── CORS ──────────────────────────────────────────────────────────
    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "*")
        # In production restrict to configured origins
        allowed = settings.CORS_ORIGINS
        if "*" in allowed or origin in allowed:
            response.headers["Access-Control-Allow-Origin"]      = origin
        response.headers["Access-Control-Allow-Methods"]          = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
        response.headers["Access-Control-Allow-Headers"]          = (
            "Authorization,Content-Type,Accept,X-Request-ID")
        response.headers["Access-Control-Allow-Credentials"]      = "true"
        response.headers["Access-Control-Max-Age"]                = "3600"
        return response

    @app.before_request
    def handle_options():
        if request.method == "OPTIONS":
            from flask import make_response
            resp = make_response("", 204)
            return resp

    # ── Timing ────────────────────────────────────────────────────────
    @app.before_request
    def start_timer():
        g.t_start     = time.perf_counter()
        g.request_id  = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

    @app.after_request
    def log_request(response):
        duration_ms = (time.perf_counter() - g.get("t_start", time.perf_counter())) * 1000
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        response.headers["X-Request-ID"]      = g.get("request_id", "")

        user_id = getattr(g, "current_user", {}).get("id") if hasattr(g, "current_user") else None

        access_log.info(
            "%s %s → %d  (%.0fms)",
            request.method, request.path, response.status_code, duration_ms,
            extra={
                "method":      request.method,
                "path":        request.path,
                "status":      response.status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id":  g.get("request_id", ""),
                "user_id":     user_id,
            },
        )

        # Async DB log (best-effort — don't fail the request if this errors)
        _persist_api_log(request.method, request.path, response.status_code,
                         user_id, duration_ms,
                         request.remote_addr or "",
                         request.user_agent.string if request.user_agent else "")
        # Record to in-memory monitoring counters
        try:
            from backend.services.monitoring import monitor
            monitor.record_api_call(
                path=request.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=user_id,
            )
        except Exception:
            pass
        return response

    # ── 404 & 405 handlers ─────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"success": False, "error": "Endpoint not found.", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"success": False, "error": "Method not allowed.", "code": "METHOD_NOT_ALLOWED"}), 405

    # ── Global exception handler ──────────────────────────────────────
    @app.errorhandler(Exception)
    def handle_exception(e):
        error_log.exception("Unhandled exception on %s %s", request.method, request.path)
        if settings.DEBUG:
            return jsonify({
                "success": False, "error": str(e),
                "code":    "INTERNAL_ERROR", "type": type(e).__name__,
            }), 500
        return jsonify({
            "success": False,
            "error":   "An internal server error occurred.",
            "code":    "INTERNAL_ERROR",
        }), 500


def _persist_api_log(method, path, status, user_id, duration_ms, ip, agent):
    """Write API call to database without raising (fire-and-forget)."""
    try:
        from backend.services.db_service import log_api_call
        log_api_call(method, path, status, user_id, duration_ms, ip, agent)
    except Exception:
        pass   # Never let logging crash the request
