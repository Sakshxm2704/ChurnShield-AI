"""
services/alerts/alert_service.py
---------------------------------
Automated alert system for the Churn Intelligence Platform.

Responsibilities
----------------
1. Evaluate every completed prediction and determine if an alert is warranted.
2. Log alerts to the ``alert_log`` database table.
3. Dispatch email notifications via SMTP (async, non-blocking).
4. Expose alert query helpers used by the monitoring dashboard.

Alert Thresholds
----------------
  High    : churn_probability ≥ 0.65  → always alert
  Medium  : churn_probability ≥ 0.45  → alert if inactive_days ≥ 30

Public API
----------
- ``AlertService``
  - ``.evaluate_and_alert(prediction_result, customer_data, customer_id)``
  - ``.get_recent_alerts(limit, risk_filter)``
  - ``.get_alert_stats()``
- ``alert_service``   module-level singleton
"""

from __future__ import annotations

import logging
import smtplib
import threading
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from backend.core.config   import settings
from backend.core.database import get_db, row_to_dict, rows_to_dicts

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
ALERT_THRESHOLD_HIGH   = 0.65
ALERT_THRESHOLD_MEDIUM = 0.45
MEDIUM_INACTIVE_DAYS   = 30


# ══════════════════════════════════════════════════════════════════════════════
# Email renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_alert_email(
    customer_id: int,
    churn_probability: float,
    risk_category: str,
    top_recommendation: str,
    expected_annual_loss: float,
) -> tuple[str, str]:
    """Return (subject, html_body) for a retention alert email."""
    prob_pct = round(churn_probability * 100, 1)
    color    = {"High": "#fc8181", "Medium": "#f6ad55"}.get(risk_category, "#63b3ed")
    loss_str = f"${expected_annual_loss:,.0f}" if expected_annual_loss else "N/A"
    ts       = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")

    subject = f"⚠ {risk_category} Churn Risk Alert — Customer #{customer_id} ({prob_pct}% probability)"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body{{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#e2e8f0; margin:0; padding:0; }}
  .wrapper{{ max-width:600px; margin:0 auto; }}
  .header{{ background:linear-gradient(135deg,#1a1d27,#0f1117); padding:32px; border-bottom:2px solid {color}; }}
  .logo{{ font-size:24px; font-weight:700; color:#e2e8f0; }}
  .logo span{{ color:{color}; }}
  .badge{{ display:inline-block; background:{color}22; color:{color}; border:1px solid {color}44;
            padding:4px 14px; border-radius:20px; font-size:13px; font-weight:600; margin-top:8px; }}
  .content{{ background:#111720; padding:32px; }}
  .metric-row{{ display:flex; justify-content:space-between; background:#161e2a;
                border:1px solid rgba(255,255,255,0.06); border-radius:8px;
                padding:16px 20px; margin-bottom:12px; }}
  .metric-label{{ font-size:12px; color:#8fa3bf; text-transform:uppercase; letter-spacing:0.08em; }}
  .metric-value{{ font-size:20px; font-weight:700; color:{color}; }}
  .rec-box{{ background:#1a3a5c22; border:1px solid #63b3ed33; border-left:3px solid #63b3ed;
             border-radius:0 8px 8px 0; padding:16px 20px; margin-top:20px; }}
  .rec-label{{ font-size:11px; color:#63b3ed; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }}
  .rec-text{{ font-size:13px; color:#e2e8f0; line-height:1.6; }}
  .footer{{ background:#0d1117; padding:20px 32px; font-size:11px; color:#4a6580;
             text-align:center; border-top:1px solid rgba(255,255,255,0.06); }}
  .cta{{ display:inline-block; background:linear-gradient(135deg,#6366f1,#8b5cf6);
          color:white; text-decoration:none; padding:12px 24px; border-radius:8px;
          font-weight:600; font-size:13px; margin-top:20px; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="logo">🧠 Churn<span>Intelligence</span></div>
    <div class="badge">⚠ {risk_category} Risk Alert</div>
    <p style="color:#8fa3bf;font-size:13px;margin-top:12px;margin-bottom:0">
      Customer #{customer_id} has been flagged for immediate retention action.
    </p>
  </div>
  <div class="content">
    <div class="metric-row">
      <div>
        <div class="metric-label">Churn Probability</div>
        <div class="metric-value">{prob_pct}%</div>
      </div>
      <div style="text-align:right">
        <div class="metric-label">Annual Revenue at Risk</div>
        <div class="metric-value" style="color:#fc8181">{loss_str}</div>
      </div>
    </div>
    <div class="metric-row">
      <div>
        <div class="metric-label">Risk Category</div>
        <div class="metric-value">{risk_category}</div>
      </div>
      <div style="text-align:right">
        <div class="metric-label">Alert Generated</div>
        <div class="metric-value" style="font-size:13px;color:#8fa3bf">{ts}</div>
      </div>
    </div>
    <div class="rec-box">
      <div class="rec-label">🎯 Top Recommended Action</div>
      <div class="rec-text">{top_recommendation}</div>
    </div>
    <div style="text-align:center">
      <a class="cta" href="{settings.FRONTEND_URL}/customers/{customer_id}">
        View Customer Profile →
      </a>
    </div>
  </div>
  <div class="footer">
    Churn Intelligence Platform &nbsp;|&nbsp; Automated Risk Alert &nbsp;|&nbsp;
    <a href="{settings.FRONTEND_URL}/admin" style="color:#63b3ed">Manage Alerts</a>
  </div>
</div>
</body>
</html>"""

    return subject, html


# ══════════════════════════════════════════════════════════════════════════════
# Alert Service
# ══════════════════════════════════════════════════════════════════════════════

class AlertService:
    """
    Centralised alert generation and dispatch service.

    All email sending happens on a background thread to avoid blocking
    the prediction request-response cycle.
    """

    def __init__(self) -> None:
        self._email_enabled = bool(settings.SMTP_USER and settings.SMTP_PASSWORD)
        if not self._email_enabled:
            logger.info(
                "Email alerts disabled (SMTP_USER/SMTP_PASSWORD not configured). "
                "Alerts will be logged to database only."
            )

    # ── Main entry point ──────────────────────────────────────────────────────

    def evaluate_and_alert(
        self,
        prediction_result: dict[str, Any],
        customer_data:    dict[str, Any],
        customer_id:      int | None = None,
        prediction_id:    int | None = None,
    ) -> dict[str, Any] | None:
        """
        Evaluate a prediction result and fire an alert if warranted.

        Parameters
        ----------
        prediction_result : output dict from ml_service.predict_customer()
        customer_data     : raw customer feature dict
        customer_id       : DB customer_id (None = anonymous prediction)
        prediction_id     : DB prediction_id (optional)

        Returns
        -------
        alert dict if alert was created, None otherwise
        """
        prob        = float(prediction_result.get("churn_probability", 0))
        risk_cat    = prediction_result.get("risk_category", "Low")
        inactive    = int(customer_data.get("inactive_days", 0))

        should_alert = (
            prob >= ALERT_THRESHOLD_HIGH
            or (prob >= ALERT_THRESHOLD_MEDIUM and inactive >= MEDIUM_INACTIVE_DAYS)
        )

        if not should_alert:
            return None

        # Build alert message
        recs        = prediction_result.get("recommendations", [])
        top_rec     = recs[0].get("action", "No specific action.") if recs else "No specific action."
        rev_risk    = prediction_result.get("revenue_risk", {})
        annual_loss = float(rev_risk.get("expected_annual_loss", 0))

        message = (
            f"Customer #{customer_id or 'N/A'} flagged with {prob*100:.1f}% churn probability. "
            f"Risk: {risk_cat}. Inactive: {inactive} days. "
            f"Annual revenue at risk: ${annual_loss:,.0f}."
        )

        # Log to DB
        alert_id = self._log_alert(
            customer_id=customer_id,
            prediction_id=prediction_id,
            alert_type="high_risk" if prob >= ALERT_THRESHOLD_HIGH else "medium_risk",
            risk_category=risk_cat,
            churn_probability=prob,
            message=message,
        )

        alert = {
            "alert_id":          alert_id,
            "customer_id":       customer_id,
            "risk_category":     risk_cat,
            "churn_probability": round(prob, 4),
            "message":           message,
            "email_sent":        False,
        }

        # Send email asynchronously (non-blocking)
        if self._email_enabled and alert_id:
            thread = threading.Thread(
                target=self._send_alert_email,
                args=(alert_id, customer_id or 0, prob, risk_cat, top_rec, annual_loss),
                daemon=True,
            )
            thread.start()
            alert["email_dispatched"] = True

        logger.info(
            "Alert created: customer=%s  risk=%s  prob=%.3f  alert_id=%s",
            customer_id, risk_cat, prob, alert_id,
        )
        return alert

    # ── Database operations ────────────────────────────────────────────────────

    def _log_alert(
        self,
        customer_id:        int | None,
        prediction_id:      int | None,
        alert_type:         str,
        risk_category:      str,
        churn_probability:  float,
        message:            str,
    ) -> int | None:
        """Insert an alert record into alert_log. Returns new alert id."""
        try:
            with get_db() as conn:
                cur = conn.execute(
                    """INSERT INTO alert_log
                       (customer_id, prediction_id, alert_type, risk_category,
                        churn_probability, message)
                       VALUES (?,?,?,?,?,?)""",
                    (customer_id, prediction_id, alert_type,
                     risk_category, churn_probability, message),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.warning("Failed to log alert: %s", exc)
            return None

    def _update_alert_email_status(self, alert_id: int, status: str, email: str) -> None:
        """Update email send status for an alert."""
        try:
            with get_db() as conn:
                conn.execute(
                    "UPDATE alert_log SET email_sent=1, email_address=?, email_status=? WHERE id=?",
                    (email, status, alert_id),
                )
        except Exception as exc:
            logger.warning("Failed to update alert email status: %s", exc)

    # ── Email dispatch ─────────────────────────────────────────────────────────

    def _send_alert_email(
        self,
        alert_id:           int,
        customer_id:        int,
        churn_probability:  float,
        risk_category:      str,
        top_recommendation: str,
        expected_annual_loss: float,
    ) -> None:
        """Send a retention alert email (runs in background thread)."""
        recipient = settings.ALERT_EMAIL_RECIPIENT or settings.SMTP_USER
        if not recipient:
            return

        subject, html = _render_alert_email(
            customer_id, churn_probability, risk_category,
            top_recommendation, expected_annual_loss,
        )

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = settings.SMTP_USER
            msg["To"]      = recipient
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_USER, recipient, msg.as_string())

            self._update_alert_email_status(alert_id, "sent", recipient)
            logger.info("Alert email sent → %s  (alert_id=%s)", recipient, alert_id)

        except Exception as exc:
            logger.error("Alert email failed (alert_id=%s): %s", alert_id, exc)
            self._update_alert_email_status(alert_id, "failed", recipient)

    # ── Query helpers ──────────────────────────────────────────────────────────

    def get_recent_alerts(
        self,
        limit: int = 50,
        risk_filter: str | None = None,
    ) -> list[dict]:
        """Return recent alerts from the database, newest first."""
        try:
            q = "SELECT * FROM alert_log"
            p: list = []
            if risk_filter:
                q += " WHERE risk_category = ?"
                p.append(risk_filter)
            q += " ORDER BY created_at DESC LIMIT ?"
            p.append(limit)
            with get_db() as conn:
                rows = conn.execute(q, p).fetchall()
            return rows_to_dicts(rows)
        except Exception as exc:
            logger.warning("get_recent_alerts failed: %s", exc)
            return []

    def get_alert_stats(self) -> dict[str, Any]:
        """Return aggregated alert statistics for the monitoring dashboard."""
        try:
            with get_db() as conn:
                total      = conn.execute("SELECT COUNT(*) FROM alert_log").fetchone()[0]
                today      = conn.execute(
                    "SELECT COUNT(*) FROM alert_log WHERE DATE(created_at)=DATE('now')"
                ).fetchone()[0]
                high_count = conn.execute(
                    "SELECT COUNT(*) FROM alert_log WHERE risk_category='High'"
                ).fetchone()[0]
                sent       = conn.execute(
                    "SELECT COUNT(*) FROM alert_log WHERE email_sent=1"
                ).fetchone()[0]
                last_alert = conn.execute(
                    "SELECT created_at FROM alert_log ORDER BY created_at DESC LIMIT 1"
                ).fetchone()

            return {
                "total_alerts":    total,
                "alerts_today":    today,
                "high_risk_alerts": high_count,
                "emails_sent":     sent,
                "last_alert_at":   last_alert[0] if last_alert else None,
            }
        except Exception as exc:
            logger.warning("get_alert_stats failed: %s", exc)
            return {}


# ── Module-level singleton ────────────────────────────────────────────────────
alert_service = AlertService()
