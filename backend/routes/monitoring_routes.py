"""
backend/routes/monitoring_routes.py
-------------------------------------
Monitoring and alerting endpoints:

  GET  /api/v1/monitoring            — real-time platform dashboard metrics
  GET  /api/v1/monitoring/alerts     — recent alert log (paginated)
  GET  /api/v1/monitoring/alerts/stats — alert statistics
  GET  /api/v1/monitoring/cache      — cache hit/miss statistics
  POST /api/v1/monitoring/cache/clear — flush the in-memory cache (admin only)
  GET  /api/v1/monitoring/health     — detailed system health check
"""
from __future__ import annotations
import logging
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request

from backend.auth.dependencies  import require_auth, require_role
from backend.core.cache         import cache
from backend.core.config        import settings
from backend.models.schemas     import ok, err, validate_pagination
from backend.services.monitoring import monitor

logger = logging.getLogger(__name__)
bp = Blueprint("monitoring", __name__, url_prefix="/api/v1/monitoring")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── GET /monitoring ───────────────────────────────────────────────────────────
@bp.get("")
@require_auth
def dashboard_metrics(current_user):
    """
    Full real-time monitoring dashboard.
    ---
    tags: [Monitoring]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Platform metrics (predictions, API calls, churn trends, active users)
    """
    metrics = monitor.get_dashboard_metrics()
    return jsonify(ok(metrics, "Monitoring metrics retrieved.")[0]), 200


# ── GET /monitoring/realtime ──────────────────────────────────────────────────
@bp.get("/realtime")
@require_auth
def realtime_stats(current_user):
    """Lightweight stats for polling — returns predictions/API calls in last 15 min."""
    stats = monitor.get_realtime_stats()
    return jsonify(ok(stats)[0]), 200


# ── GET /monitoring/alerts ────────────────────────────────────────────────────
@bp.get("/alerts")
@require_auth
def list_alerts(current_user):
    """
    Return recent alerts from the alert_log table.
    ---
    tags: [Monitoring]
    parameters:
      - name: limit
        in: query
        schema: {type: integer, default: 50}
      - name: risk
        in: query
        schema: {type: string, enum: [High, Medium]}
    """
    try:
        from services.alerts.alert_service import alert_service
        limit = min(int(request.args.get("limit", 50)), 200)
        risk  = request.args.get("risk")
        alerts = alert_service.get_recent_alerts(limit=limit, risk_filter=risk)
        return jsonify(ok({"alerts": alerts, "count": len(alerts)})[0]), 200
    except Exception as exc:
        logger.exception("list_alerts failed")
        return jsonify(err(str(exc))[0]), 500


# ── GET /monitoring/alerts/stats ──────────────────────────────────────────────
@bp.get("/alerts/stats")
@require_auth
def alert_stats(current_user):
    """Return aggregated alert statistics."""
    try:
        from services.alerts.alert_service import alert_service
        stats = alert_service.get_alert_stats()
        return jsonify(ok(stats)[0]), 200
    except Exception as exc:
        return jsonify(err(str(exc))[0]), 500


# ── GET /monitoring/cache ─────────────────────────────────────────────────────
@bp.get("/cache")
@require_auth
def cache_stats(current_user):
    """Return in-memory cache statistics (size, hit rate, key count)."""
    return jsonify(ok(cache.stats, "Cache statistics.")[0]), 200


# ── POST /monitoring/cache/clear ──────────────────────────────────────────────
@bp.post("/cache/clear")
@require_auth
@require_role("admin")
def cache_clear(current_user):
    """Flush the entire in-memory cache. Admin only."""
    before = cache.stats["size"]
    cache.clear()
    return jsonify(ok({"cleared_keys": before}, "Cache cleared.")[0]), 200


# ── GET /monitoring/health ────────────────────────────────────────────────────
@bp.get("/health")
def detailed_health():
    """
    Detailed system health check (unauthenticated — for load balancers).
    Returns DB status, model status, and Python version.
    """
    checks: dict = {}

    # Database
    try:
        from backend.core.database import get_db
        with get_db() as conn:
            conn.execute("SELECT 1")
        checks["database"] = {"status": "ok"}
    except Exception as exc:
        checks["database"] = {"status": "error", "detail": str(exc)}

    # ML model
    try:
        from backend.services.ml_service import get_model_metadata
        meta = get_model_metadata()
        checks["ml_model"] = {
            "status": "ok",
            "best_model": meta.get("best_model", "unknown"),
        }
    except Exception as exc:
        checks["ml_model"] = {"status": "error", "detail": str(exc)}

    # Disk
    try:
        import shutil
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        checks["disk"] = {
            "status": "ok",
            "free_gb": round(free / 1e9, 1),
            "used_pct": round(used / total * 100, 1),
        }
    except Exception:
        checks["disk"] = {"status": "unknown"}

    overall = "healthy" if all(v.get("status") == "ok" for v in checks.values()) else "degraded"
    status_code = 200 if overall == "healthy" else 503

    payload = {
        "status":      overall,
        "version":     settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "python":      sys.version.split()[0],
        "platform":    platform.system(),
        "uptime":      monitor.get_realtime_stats(),
        "checks":      checks,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }
    return jsonify(ok(payload)[0]), status_code
