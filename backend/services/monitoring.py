"""
backend/services/monitoring.py
--------------------------------
Platform monitoring service.

Collects and aggregates real-time metrics:
  - Prediction counts and latencies
  - API call volumes per endpoint
  - Active user counts (rolling 15-min window)
  - Churn trend data (rolling 7-day window)
  - Alert dispatch statistics

All metrics are written to:
  1. In-memory counters (fast, reset on restart)
  2. SQLite monitoring_metrics table (persistent, queryable)

Usage
-----
::
    from backend.services.monitoring import monitor

    # Record a completed prediction
    monitor.record_prediction(risk_category="High", latency_ms=142.3)

    # Record an API call
    monitor.record_api_call("/api/v1/predict", status_code=200, duration_ms=145)

    # Get dashboard metrics
    stats = monitor.get_dashboard_metrics()
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from backend.core.database import get_db

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Rolling window counter
# ══════════════════════════════════════════════════════════════════════════════

class RollingCounter:
    """Thread-safe deque of (timestamp, value) pairs with automatic expiry."""

    def __init__(self, window_seconds: int = 900) -> None:
        self._window = window_seconds
        self._data: deque[tuple[float, float]] = deque()
        self._lock  = threading.Lock()

    def add(self, value: float = 1.0) -> None:
        now = time.monotonic()
        with self._lock:
            self._data.append((now, value))
            self._trim(now)

    def sum(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            return sum(v for _, v in self._data)

    def count(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            return len(self._data)

    def avg(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            if not self._data:
                return 0.0
            return sum(v for _, v in self._data) / len(self._data)

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()


# ══════════════════════════════════════════════════════════════════════════════
# Monitoring Service
# ══════════════════════════════════════════════════════════════════════════════

class MonitoringService:
    """
    Central metrics collector for the Churn Intelligence Platform.

    Uses rolling windows for real-time metrics and SQLite for persistence.
    """

    def __init__(self) -> None:
        # Rolling 15-min windows
        self._predictions_window  = RollingCounter(900)
        self._api_calls_window    = RollingCounter(900)
        self._pred_latency        = RollingCounter(900)
        self._active_users: set[int]      = set()
        self._active_lock         = threading.Lock()

        # Cumulative in-memory counters
        self._total_predictions   = 0
        self._high_risk_count     = 0
        self._medium_risk_count   = 0
        self._low_risk_count      = 0
        self._total_api_calls     = 0
        self._error_count         = 0

        # Per-endpoint counters
        self._endpoint_counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

        # Startup time
        self._started_at = datetime.now(timezone.utc)

    # ── Record events ─────────────────────────────────────────────────────────

    def record_prediction(
        self,
        risk_category: str = "Low",
        latency_ms:    float = 0.0,
        user_id:       int | None = None,
    ) -> None:
        """Record a completed prediction."""
        with self._lock:
            self._total_predictions += 1
            if risk_category == "High":
                self._high_risk_count += 1
            elif risk_category == "Medium":
                self._medium_risk_count += 1
            else:
                self._low_risk_count += 1

        self._predictions_window.add(1)
        if latency_ms > 0:
            self._pred_latency.add(latency_ms)

        if user_id:
            with self._active_lock:
                self._active_users.add(user_id)

        # Persist to DB async (fire-and-forget)
        self._persist_async("prediction_count", 1, risk_category)

    def record_api_call(
        self,
        path:        str,
        status_code: int,
        duration_ms: float = 0.0,
        user_id:     int | None = None,
    ) -> None:
        """Record an API request."""
        with self._lock:
            self._total_api_calls += 1
            self._endpoint_counts[path] += 1
            if status_code >= 500:
                self._error_count += 1

        self._api_calls_window.add(1)

        if user_id:
            with self._active_lock:
                self._active_users.add(user_id)

    def record_error(self, path: str, error: str) -> None:
        """Record an application error."""
        with self._lock:
            self._error_count += 1
        logger.warning("Monitored error on %s: %s", path, error)

    # ── Query metrics ─────────────────────────────────────────────────────────

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Return all metrics for the monitoring dashboard."""
        with self._lock:
            total_preds  = self._total_predictions
            high         = self._high_risk_count
            medium       = self._medium_risk_count
            low          = self._low_risk_count
            total_api    = self._total_api_calls
            error_cnt    = self._error_count
            top_endpoints = sorted(
                self._endpoint_counts.items(), key=lambda x: -x[1]
            )[:8]

        with self._active_lock:
            active_users = len(self._active_users)

        # Rolling window metrics
        preds_15m     = self._predictions_window.count()
        api_15m       = self._api_calls_window.count()
        avg_latency   = round(self._pred_latency.avg(), 1)

        # DB-derived trends
        churn_trend   = self._get_churn_trend()
        api_trend     = self._get_api_trend()

        uptime_secs   = (datetime.now(timezone.utc) - self._started_at).total_seconds()

        return {
            "uptime_seconds":       int(uptime_secs),
            "uptime_human":         _fmt_uptime(uptime_secs),
            "total_predictions":    total_preds,
            "predictions_15min":    preds_15m,
            "high_risk_total":      high,
            "medium_risk_total":    medium,
            "low_risk_total":       low,
            "total_api_calls":      total_api,
            "api_calls_15min":      api_15m,
            "error_count":          error_cnt,
            "error_rate_pct":       round(error_cnt / max(total_api, 1) * 100, 2),
            "active_users":         active_users,
            "avg_prediction_ms":    avg_latency,
            "top_endpoints":        [{"path": p, "calls": c} for p, c in top_endpoints],
            "churn_trend":          churn_trend,
            "api_trend":            api_trend,
            "timestamp":            datetime.now(timezone.utc).isoformat(),
        }

    def get_realtime_stats(self) -> dict[str, Any]:
        """Lightweight stats for the topbar status indicator."""
        return {
            "predictions_15min": self._predictions_window.count(),
            "api_calls_15min":   self._api_calls_window.count(),
            "active_users":      len(self._active_users),
            "avg_latency_ms":    round(self._pred_latency.avg(), 1),
        }

    # ── DB queries ─────────────────────────────────────────────────────────────

    def _get_churn_trend(self) -> list[dict]:
        """Return daily prediction counts for last 7 days."""
        try:
            with get_db() as conn:
                rows = conn.execute("""
                    SELECT DATE(created_at) AS day,
                           COUNT(*)         AS total,
                           SUM(CASE WHEN prediction_label='High'   THEN 1 ELSE 0 END) AS high,
                           SUM(CASE WHEN prediction_label='Medium' THEN 1 ELSE 0 END) AS medium,
                           AVG(churn_probability) AS avg_prob
                    FROM   predictions
                    WHERE  created_at >= DATE('now', '-7 days')
                    GROUP  BY day
                    ORDER  BY day ASC
                """).fetchall()
            return [
                {
                    "date":     r[0],
                    "total":    r[1],
                    "high":     r[2],
                    "medium":   r[3],
                    "avg_prob": round(r[4] or 0, 4),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("churn_trend query failed: %s", exc)
            return []

    def _get_api_trend(self) -> list[dict]:
        """Return hourly API call counts for last 24 h."""
        try:
            with get_db() as conn:
                rows = conn.execute("""
                    SELECT strftime('%Y-%m-%dT%H:00:00Z', created_at) AS hour,
                           COUNT(*) AS calls,
                           AVG(duration_ms) AS avg_ms
                    FROM   api_logs
                    WHERE  created_at >= datetime('now', '-24 hours')
                    GROUP  BY hour
                    ORDER  BY hour ASC
                """).fetchall()
            return [
                {
                    "hour":    r[0],
                    "calls":   r[1],
                    "avg_ms":  round(r[2] or 0, 1),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.debug("api_trend query failed: %s", exc)
            return []

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist_async(self, metric_name: str, value: float, label: str = "") -> None:
        """Write a metric to the DB without blocking the caller."""
        def _write():
            try:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO monitoring_metrics (metric_name, metric_value, metric_label)"
                        " VALUES (?,?,?)",
                        (metric_name, value, label),
                    )
            except Exception:
                pass   # Monitoring must never break the main flow

        threading.Thread(target=_write, daemon=True).start()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_uptime(seconds: float) -> str:
    s = int(seconds)
    days, s  = divmod(s, 86400)
    hours, s = divmod(s,  3600)
    mins,  _ = divmod(s,    60)
    parts = []
    if days:  parts.append(f"{days}d")
    if hours: parts.append(f"{hours}h")
    parts.append(f"{mins}m")
    return " ".join(parts)


# ── Module-level singleton ────────────────────────────────────────────────────
monitor = MonitoringService()
