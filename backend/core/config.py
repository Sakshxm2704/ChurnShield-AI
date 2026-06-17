"""
backend/core/config.py
-----------------------
Centralised application settings — single source of truth.

All values loaded from environment variables or the .env file.
Add new settings here; never use os.getenv() elsewhere in the codebase.
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_BASE_DIR / ".env")
except ImportError:
    _BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings:
    # ── Paths ─────────────────────────────────────────────────────────
    BASE_DIR: Path       = _BASE_DIR
    ML_MODEL_DIR: Path   = _BASE_DIR / "ml" / "models" / "saved"
    ML_REPORTS_DIR: Path = _BASE_DIR / "reports"
    ML_DATA_DIR: Path    = _BASE_DIR / "data"
    LOG_DIR: Path        = _BASE_DIR / "logs"

    # ── Application ───────────────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "ChurnShield AI")
    APP_ENV: str     = os.getenv("APP_ENV",     "development")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool      = os.getenv("DEBUG",       "true").lower() == "true"

    # ── Security ──────────────────────────────────────────────────────
    SECRET_KEY: str                  = os.getenv("SECRET_KEY", "dev-secret-change-in-prod-32plus-chars!!!")
    ALGORITHM: str                   = os.getenv("ALGORITHM",  "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    REFRESH_TOKEN_EXPIRE_DAYS: int   = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",   "7"))

    # ── Database ──────────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{_BASE_DIR / 'data' / 'churn_platform.db'}"
    )

    # ── API Server ────────────────────────────────────────────────────
    API_HOST: str    = os.getenv("API_HOST",    "0.0.0.0")
    API_PORT: int    = int(os.getenv("API_PORT",    "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD",  "true").lower() == "true"

    # ── Frontend ──────────────────────────────────────────────────────
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8501")

    # ── CORS ──────────────────────────────────────────────────────────
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8501", "http://localhost:8000", "*"]

    # ── Logging ───────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── SMTP / Email ──────────────────────────────────────────────────
    SMTP_HOST: str     = os.getenv("SMTP_HOST",     "smtp.gmail.com")
    SMTP_PORT: int     = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str     = os.getenv("SMTP_USER",     "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    EMAIL_FROM: str    = os.getenv("EMAIL_FROM",    "noreply@churnplatform.io")

    # ── Alerts ────────────────────────────────────────────────────────
    ALERT_EMAIL_RECIPIENT: str        = os.getenv("ALERT_EMAIL_RECIPIENT", "")
    ALERT_HIGH_RISK_THRESHOLD: float  = float(os.getenv("ALERT_HIGH_RISK_THRESHOLD",  "0.65"))
    ALERT_MEDIUM_RISK_THRESHOLD: float= float(os.getenv("ALERT_MEDIUM_RISK_THRESHOLD","0.45"))
    ALERT_INACTIVE_DAYS_TRIGGER: int  = int(os.getenv("ALERT_INACTIVE_DAYS_TRIGGER",  "30"))

    # ── Cache ─────────────────────────────────────────────────────────
    CACHE_TTL_ANALYTICS: int = int(os.getenv("CACHE_TTL_ANALYTICS", "300"))
    CACHE_TTL_SEGMENTS:  int = int(os.getenv("CACHE_TTL_SEGMENTS",  "600"))
    CACHE_TTL_ML_META:   int = int(os.getenv("CACHE_TTL_ML_META",   "1800"))
    CACHE_MAX_SIZE:      int = int(os.getenv("CACHE_MAX_SIZE",      "512"))

    # ── Pagination ────────────────────────────────────────────────────
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int     = 200

    # ── Groq AI ──────────────────────────────────────────────────────
    GROQ_API_KEY: str   = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_API_URL: str   = "https://api.groq.com/openai/v1/chat/completions"

    # ── Custom Training ───────────────────────────────────────────────
    ALLOW_CUSTOM_TRAINING: bool = os.getenv("ALLOW_CUSTOM_TRAINING", "true").lower() == "true"
    CUSTOM_DATA_DIR: Path       = _BASE_DIR / "data" / "custom"

    # ── ML thresholds ─────────────────────────────────────────────────
    ML_HIGH_RISK_THRESHOLD:   float = float(os.getenv("ML_HIGH_RISK_THRESHOLD",   "0.65"))
    ML_MEDIUM_RISK_THRESHOLD: float = float(os.getenv("ML_MEDIUM_RISK_THRESHOLD", "0.35"))

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def db_path(self) -> str:
        """Return the raw SQLite file path (strips sqlite:/// prefix)."""
        url = self.DATABASE_URL
        if url.startswith("sqlite:///"):
            return url[len("sqlite:///"):]
        return str(self.BASE_DIR / "data" / "churn_platform.db")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
