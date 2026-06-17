"""
backend/services/db_service.py
--------------------------------
Database service layer — all SQL queries in one place.
Routes never write SQL directly; they call these functions instead.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from backend.core.database import get_db, row_to_dict, rows_to_dicts, paginate

logger = logging.getLogger(__name__)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(name: str, email: str, password_hash: str, role: str = "viewer") -> dict:
    """Insert a new user. Returns the created row."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?,?)",
            (name, email, password_hash, role))
        user_id = cur.lastrowid
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return row_to_dict(row)


def get_user_by_email(email: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
    return row_to_dict(row)


def get_user_by_id(user_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return row_to_dict(row)


def update_user(user_id: int, **kwargs) -> dict | None:
    """Update allowed user fields. Returns updated row."""
    allowed = {"name", "role", "is_active"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return get_user_by_id(user_id)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [user_id]
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {set_clause}, updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE id=?",
                     values)
    return get_user_by_id(user_id)


def list_users(page: int = 1, page_size: int = 20) -> dict:
    return paginate("SELECT id,name,email,role,is_active,created_at FROM users ORDER BY created_at DESC",
                    (), page, page_size)


# ── Customers ─────────────────────────────────────────────────────────────────

_CUSTOMER_COLS = """customer_id, gender, senior_citizen, partner, dependents,
    tenure, phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection, tech_support,
    streaming_tv, streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges, total_charges, inactive_days,
    subscription_type, created_at"""


def create_customer(data: dict) -> dict:
    """Insert a customer record and return it."""
    cols = ["gender", "senior_citizen", "partner", "dependents", "tenure",
            "phone_service", "multiple_lines", "internet_service",
            "online_security", "online_backup", "device_protection", "tech_support",
            "streaming_tv", "streaming_movies", "contract", "paperless_billing",
            "payment_method", "monthly_charges", "total_charges",
            "inactive_days", "subscription_type"]
    values = [data.get(c) for c in cols]
    placeholders = ",".join("?" * len(cols))
    col_str = ",".join(cols)
    with get_db() as conn:
        cur = conn.execute(
            f"INSERT INTO customers ({col_str}) VALUES ({placeholders})", values)
        cid = cur.lastrowid
        row = conn.execute(f"SELECT {_CUSTOMER_COLS} FROM customers WHERE customer_id=?",
                           (cid,)).fetchone()
    return row_to_dict(row)


def get_customer(customer_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            f"SELECT {_CUSTOMER_COLS} FROM customers WHERE customer_id=?",
            (customer_id,)).fetchone()
    return row_to_dict(row)


def list_customers(page: int = 1, page_size: int = 20,
                   risk_filter: str | None = None) -> dict:
    """Return paginated customers, optionally filtered by risk category."""
    if risk_filter:
        # join with latest prediction
        q = f"""
            SELECT c.{_CUSTOMER_COLS.replace(chr(10),' ')},
                   p.prediction_label, p.churn_probability, p.risk_score
            FROM customers c
            LEFT JOIN (
                SELECT customer_id, prediction_label, churn_probability, risk_score,
                       MAX(created_at) as max_ts
                FROM predictions GROUP BY customer_id
            ) p ON c.customer_id = p.customer_id
            WHERE p.prediction_label = ?
            ORDER BY c.created_at DESC
        """
        return paginate(q, (risk_filter,), page, page_size)
    else:
        q = f"SELECT {_CUSTOMER_COLS} FROM customers ORDER BY created_at DESC"
        return paginate(q, (), page, page_size)


def update_customer(customer_id: int, data: dict) -> dict | None:
    allowed = {"tenure", "monthly_charges", "contract", "payment_method",
               "subscription_type", "inactive_days", "total_charges"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        return get_customer(customer_id)
    set_clause = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values()) + [customer_id]
    with get_db() as conn:
        conn.execute(
            f"UPDATE customers SET {set_clause}, updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE customer_id=?",
            values)
    return get_customer(customer_id)


# ── Predictions ───────────────────────────────────────────────────────────────

def save_prediction(customer_id: int | None, result: dict) -> int:
    """
    Persist a prediction result to the predictions table.
    Returns the new prediction_id.
    """
    shap_json = json.dumps(result.get("shap_explanation", [])) if result.get("shap_explanation") else None
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (customer_id, churn_probability, risk_score, prediction_label,
                churn_label, model_used, shap_values)
               VALUES (?,?,?,?,?,?,?)""",
            (customer_id,
             result["churn_probability"],
             result["risk_score"],
             result["risk_category"],
             result["churn_label"],
             result.get("model_used", "best_model"),
             shap_json))
    return cur.lastrowid


def get_prediction_history(customer_id: int) -> list[dict]:
    """Return all predictions for a customer, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT prediction_id, churn_probability, risk_score,
                      prediction_label, churn_label, model_used, created_at
               FROM predictions WHERE customer_id=?
               ORDER BY created_at DESC""",
            (customer_id,)).fetchall()
    return rows_to_dicts(rows)


def get_latest_prediction(customer_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM predictions WHERE customer_id=?
               ORDER BY created_at DESC LIMIT 1""",
            (customer_id,)).fetchone()
    return row_to_dict(row)


def list_predictions(page: int = 1, page_size: int = 20,
                     risk_label: str | None = None) -> dict:
    if risk_label:
        q = ("SELECT p.*, c.monthly_charges, c.contract, c.tenure "
             "FROM predictions p LEFT JOIN customers c USING(customer_id) "
             "WHERE p.prediction_label=? ORDER BY p.created_at DESC")
        return paginate(q, (risk_label,), page, page_size)
    q = ("SELECT p.*, c.monthly_charges, c.contract, c.tenure "
         "FROM predictions p LEFT JOIN customers c USING(customer_id) "
         "ORDER BY p.created_at DESC")
    return paginate(q, (), page, page_size)


# ── Recommendations ───────────────────────────────────────────────────────────

def save_recommendations(customer_id: int, recs: list[dict]) -> int:
    """Bulk-insert recommendations. Returns count saved."""
    if not recs:
        return 0
    with get_db() as conn:
        # Clear old recs for this customer
        conn.execute("DELETE FROM recommendations WHERE customer_id=?", (customer_id,))
        conn.executemany(
            """INSERT INTO recommendations
               (customer_id, rule_name, recommended_action, estimated_savings, priority)
               VALUES (?,?,?,?,?)""",
            [(customer_id,
              r.get("rule_name", ""),
              r["action"],
              r.get("estimated_savings"),
              r.get("priority", 1)) for r in recs])
    return len(recs)


def get_recommendations(customer_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM recommendations WHERE customer_id=?
               ORDER BY priority ASC, estimated_savings DESC""",
            (customer_id,)).fetchall()
    return rows_to_dicts(rows)


def list_all_recommendations(page: int = 1, page_size: int = 20) -> dict:
    q = ("SELECT r.*, c.monthly_charges, c.contract "
         "FROM recommendations r LEFT JOIN customers c USING(customer_id) "
         "ORDER BY r.priority ASC, r.created_at DESC")
    return paginate(q, (), page, page_size)


# ── User History ──────────────────────────────────────────────────────────────

def save_history(user_id: int, entry_type: str, title: str, summary: str | None = None,
                  risk_category: str | None = None, reference_id: int | None = None,
                  payload: dict | None = None, result: dict | None = None) -> int:
    """
    Persist one user-activity entry (prediction, batch, report, analysis...).
    Returns the new history_id. Never raises — callers treat history saving
    as best-effort so it can never break the underlying feature it records.
    """
    try:
        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO user_history
                   (user_id, entry_type, title, summary, risk_category,
                    reference_id, payload, result)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (user_id, entry_type, title, summary, risk_category, reference_id,
                 json.dumps(payload) if payload is not None else None,
                 json.dumps(result)  if result  is not None else None))
        return cur.lastrowid
    except Exception as e:
        logger.warning("Failed to save user history entry: %s", e)
        return 0


def list_history(user_id: int, page: int = 1, page_size: int = 20,
                  entry_type: str | None = None) -> dict:
    """Return a paginated list of a user's own history entries, newest first."""
    cols = ("history_id, user_id, entry_type, title, summary, risk_category, "
            "reference_id, created_at")
    if entry_type:
        q = (f"SELECT {cols} FROM user_history WHERE user_id=? AND entry_type=? "
             "ORDER BY created_at DESC")
        return paginate(q, (user_id, entry_type), page, page_size)
    q = f"SELECT {cols} FROM user_history WHERE user_id=? ORDER BY created_at DESC"
    return paginate(q, (user_id,), page, page_size)


def get_history_entry(user_id: int, history_id: int) -> dict | None:
    """Return one history entry (with full payload/result) scoped to its owner."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_history WHERE history_id=? AND user_id=?",
            (history_id, user_id)).fetchone()
    entry = row_to_dict(row)
    if not entry:
        return None
    for field in ("payload", "result"):
        if entry.get(field):
            try:
                entry[field] = json.loads(entry[field])
            except Exception:
                pass
    return entry


def delete_history_entry(user_id: int, history_id: int) -> bool:
    """Delete a history entry owned by user_id. Returns True if a row was removed."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM user_history WHERE history_id=? AND user_id=?",
            (history_id, user_id))
    return cur.rowcount > 0


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_analytics_summary() -> dict:
    """Aggregate metrics from predictions and customers tables."""
    with get_db() as conn:
        total_customers = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        # Risk distribution (latest prediction per customer)
        risk_dist = conn.execute("""
            SELECT prediction_label, COUNT(*) as cnt
            FROM (SELECT customer_id, prediction_label,
                  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) as rn
                  FROM predictions) sub
            WHERE rn=1 GROUP BY prediction_label
        """).fetchall()
        risk_dict = {row[0]: row[1] for row in risk_dist}

        # Recent prediction trend (last 7 days)
        trend = conn.execute("""
            SELECT DATE(created_at) as day, COUNT(*) as cnt,
                   AVG(churn_probability) as avg_prob
            FROM predictions
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY day ORDER BY day
        """).fetchall()

        # Top churn customers
        top_churn = conn.execute("""
            SELECT p.customer_id, p.churn_probability, p.risk_score,
                   p.prediction_label, c.monthly_charges, c.contract, c.tenure
            FROM predictions p
            JOIN customers c ON p.customer_id = c.customer_id
            WHERE p.created_at = (SELECT MAX(p2.created_at) FROM predictions p2
                                  WHERE p2.customer_id = p.customer_id)
            ORDER BY p.churn_probability DESC LIMIT 10
        """).fetchall()

        # Revenue at risk
        revenue_at_risk = conn.execute("""
            SELECT SUM(c.monthly_charges * p.churn_probability) * 12 as annual_loss
            FROM predictions p
            JOIN customers c ON p.customer_id = c.customer_id
            WHERE p.created_at = (SELECT MAX(p2.created_at) FROM predictions p2
                                  WHERE p2.customer_id = p.customer_id)
        """).fetchone()[0]

        # API call counts
        api_calls_today = conn.execute("""
            SELECT COUNT(*) FROM api_logs
            WHERE DATE(created_at) = DATE('now')
        """).fetchone()[0]

    return {
        "total_customers":    total_customers,
        "total_predictions":  total_predictions,
        "risk_distribution":  risk_dict,
        "high_risk":          risk_dict.get("High", 0),
        "medium_risk":        risk_dict.get("Medium", 0),
        "low_risk":           risk_dict.get("Low", 0),
        "prediction_trend":   [dict(r) for r in trend],
        "top_churn_customers": rows_to_dicts(top_churn),
        "expected_annual_revenue_loss": round(revenue_at_risk or 0, 2),
        "api_calls_today":    api_calls_today,
    }


def log_api_call(method: str, path: str, status_code: int,
                 user_id: int | None, duration_ms: float,
                 ip: str = "", agent: str = "", error: str = "") -> None:
    """Append an API call audit record."""
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO api_logs
                   (method, path, status_code, user_id, duration_ms, ip_address, user_agent, error_msg)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (method, path, status_code, user_id, duration_ms, ip, agent, error or None))
    except Exception as e:
        logger.error("Failed to log API call: %s", e)


# ── Segmentation ──────────────────────────────────────────────────────────────

def get_segments_summary() -> dict:
    """Return segment distribution and per-segment stats from predictions."""
    with get_db() as conn:
        # We store segment in a separate segmented_customers view (built on demand)
        # Fallback: derive rough segments from risk + tenure
        rows = conn.execute("""
            SELECT
                CASE
                    WHEN p.churn_probability >= 0.65 AND c.inactive_days > 30 THEN 'Inactive'
                    WHEN p.churn_probability >= 0.50 AND c.tenure <= 18 THEN 'Risky'
                    WHEN c.monthly_charges >= 70 AND p.churn_probability < 0.50 THEN 'Premium'
                    ELSE 'Loyal'
                END as segment,
                COUNT(*) as count,
                AVG(c.monthly_charges) as avg_charges,
                AVG(p.churn_probability) as avg_churn_prob,
                AVG(c.tenure) as avg_tenure
            FROM predictions p
            JOIN customers c ON p.customer_id = c.customer_id
            WHERE p.created_at = (SELECT MAX(p2.created_at) FROM predictions p2
                                  WHERE p2.customer_id = p.customer_id)
            GROUP BY segment
        """).fetchall()

    total = sum(r["count"] for r in rows) or 1
    return {
        "segments": [
            {
                "name":               dict(r)["segment"],
                "count":              dict(r)["count"],
                "pct_of_total":       round(dict(r)["count"] / total * 100, 1),
                "avg_monthly_charges": round(dict(r)["avg_charges"] or 0, 2),
                "avg_churn_probability": round(dict(r)["avg_churn_prob"] or 0, 4),
                "avg_tenure":         round(dict(r)["avg_tenure"] or 0, 1),
            }
            for r in rows
        ],
        "total_segmented": total,
    }
