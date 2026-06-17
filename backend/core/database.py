"""
backend/core/database.py
-------------------------
Thread-safe SQLite database layer.
Swap DATABASE_URL to postgresql:// + psycopg2 for production.
"""
from __future__ import annotations
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from backend.core.config import settings

logger = logging.getLogger(__name__)
_local = threading.local()


def _get_connection() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return _local.conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def row_to_dict(row) -> dict | None:
    return dict(row) if row else None


def rows_to_dicts(rows: list) -> list[dict]:
    return [dict(r) for r in rows]


def paginate(query: str, params: tuple, page: int, page_size: int) -> dict:
    with get_db() as conn:
        count_q = f"SELECT COUNT(*) FROM ({query})"
        total = conn.execute(count_q, params).fetchone()[0]
        offset = (page - 1) * page_size
        data_q = f"{query} LIMIT {page_size} OFFSET {offset}"
        rows = rows_to_dicts(conn.execute(data_q, params).fetchall())
    return {
        "items": rows, "total": total,
        "page": page, "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'viewer'
                          CHECK(role IN ('admin','analyst','viewer','retention')),
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_users_email      ON users(email);
CREATE INDEX IF NOT EXISTS ix_users_role       ON users(role);
CREATE INDEX IF NOT EXISTS ix_users_created_at ON users(created_at);

CREATE TABLE IF NOT EXISTS customers (
    customer_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    gender             TEXT,
    senior_citizen     INTEGER DEFAULT 0,
    partner            TEXT,
    dependents         TEXT,
    tenure             INTEGER NOT NULL DEFAULT 0,
    phone_service      TEXT,
    multiple_lines     TEXT,
    internet_service   TEXT,
    online_security    TEXT,
    online_backup      TEXT,
    device_protection  TEXT,
    tech_support       TEXT,
    streaming_tv       TEXT,
    streaming_movies   TEXT,
    contract           TEXT,
    paperless_billing  TEXT,
    payment_method     TEXT,
    monthly_charges    REAL NOT NULL,
    total_charges      REAL,
    inactive_days      INTEGER DEFAULT 0,
    subscription_type  TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_customers_contract        ON customers(contract);
CREATE INDEX IF NOT EXISTS ix_customers_monthly_charges ON customers(monthly_charges);
CREATE INDEX IF NOT EXISTS ix_customers_inactive_days   ON customers(inactive_days);
CREATE INDEX IF NOT EXISTS ix_customers_created_at      ON customers(created_at);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id        INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    churn_probability  REAL    NOT NULL,
    risk_score         INTEGER NOT NULL,
    prediction_label   TEXT    NOT NULL CHECK(prediction_label IN ('Low','Medium','High')),
    churn_label        TEXT    NOT NULL DEFAULT 'No Churn',
    model_used         TEXT    NOT NULL,
    shap_values        TEXT,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_predictions_customer_id ON predictions(customer_id);
CREATE INDEX IF NOT EXISTS ix_predictions_label       ON predictions(prediction_label);
CREATE INDEX IF NOT EXISTS ix_predictions_probability ON predictions(churn_probability);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at  ON predictions(created_at);

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id        INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    rule_name          TEXT,
    recommended_action TEXT    NOT NULL,
    estimated_savings  REAL,
    priority           INTEGER DEFAULT 1,
    is_completed       INTEGER DEFAULT 0,
    created_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_recommendations_customer_id ON recommendations(customer_id);
CREATE INDEX IF NOT EXISTS ix_recommendations_priority    ON recommendations(priority);

CREATE TABLE IF NOT EXISTS retention_logs (
    log_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id      INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    alert_sent       INTEGER DEFAULT 0,
    email_sent       INTEGER DEFAULT 0,
    response_status  TEXT DEFAULT 'pending'
                     CHECK(response_status IN ('pending','sent','opened','responded','ignored')),
    notes            TEXT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_retention_logs_customer_id ON retention_logs(customer_id);
CREATE INDEX IF NOT EXISTS ix_retention_logs_status      ON retention_logs(response_status);

CREATE TABLE IF NOT EXISTS analytics_logs (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    api_calls               INTEGER DEFAULT 0,
    predictions_generated   INTEGER DEFAULT 0,
    active_users            INTEGER DEFAULT 0,
    high_risk_customers     INTEGER DEFAULT 0,
    retention_actions_taken INTEGER DEFAULT 0,
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_analytics_logs_created_at ON analytics_logs(created_at);

CREATE TABLE IF NOT EXISTS api_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    method       TEXT NOT NULL,
    path         TEXT NOT NULL,
    status_code  INTEGER,
    user_id      INTEGER,
    duration_ms  REAL,
    ip_address   TEXT,
    user_agent   TEXT,
    error_msg    TEXT,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_api_logs_path       ON api_logs(path);
CREATE INDEX IF NOT EXISTS ix_api_logs_created_at ON api_logs(created_at);
CREATE INDEX IF NOT EXISTS ix_api_logs_user_id    ON api_logs(user_id);

CREATE TABLE IF NOT EXISTS alert_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    prediction_id   INTEGER REFERENCES predictions(prediction_id) ON DELETE SET NULL,
    alert_type      TEXT    NOT NULL DEFAULT 'high_risk',
    risk_category   TEXT    NOT NULL,
    churn_probability REAL  NOT NULL,
    email_sent      INTEGER DEFAULT 0,
    email_address   TEXT,
    email_status    TEXT    DEFAULT 'pending',
    message         TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_alert_log_customer_id  ON alert_log(customer_id);
CREATE INDEX IF NOT EXISTS ix_alert_log_alert_type   ON alert_log(alert_type);
CREATE INDEX IF NOT EXISTS ix_alert_log_created_at   ON alert_log(created_at);

CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name     TEXT    NOT NULL,
    metric_value    REAL    NOT NULL DEFAULT 0,
    metric_label    TEXT,
    window_start    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    window_end      TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_monitoring_name       ON monitoring_metrics(metric_name);
CREATE INDEX IF NOT EXISTS ix_monitoring_created_at ON monitoring_metrics(created_at);

CREATE TABLE IF NOT EXISTS cache_store (
    cache_key   TEXT    PRIMARY KEY,
    cache_value TEXT    NOT NULL,
    expires_at  TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

-- ── User activity history ──────────────────────────────────────────────────
-- Links every prediction / batch / report / analysis a user generates back to
-- that user, so they can revisit their own past results after logging back in.
CREATE TABLE IF NOT EXISTS user_history (
    history_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    entry_type     TEXT    NOT NULL CHECK(entry_type IN
                           ('single_prediction','batch_prediction','groq_prediction',
                            'groq_batch','report','analysis')),
    title          TEXT    NOT NULL,
    summary        TEXT,
    risk_category  TEXT,
    reference_id   INTEGER,
    payload        TEXT,
    result         TEXT,
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE INDEX IF NOT EXISTS ix_user_history_user_id    ON user_history(user_id);
CREATE INDEX IF NOT EXISTS ix_user_history_type        ON user_history(entry_type);
CREATE INDEX IF NOT EXISTS ix_user_history_created_at  ON user_history(created_at);
"""


def init_db() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    with get_db() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Database initialised.")
