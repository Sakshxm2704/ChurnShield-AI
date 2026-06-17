"""
database/init_db.py
-------------------
Database initialisation script.

Usage
-----
Run once to create all tables::

    python -m database.init_db

Options
-------
--drop-all     Drop all existing tables before recreating (DANGEROUS in prod).
--seed         Insert minimal seed data after table creation.
--check        Only verify the database connection; do not create tables.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.core.config import settings
from backend.core.database import Base, async_engine

# Import all models so their metadata is registered on Base
from database.models import (  # noqa: F401
    AnalyticsLog,
    Customer,
    Prediction,
    Recommendation,
    RetentionLog,
    User,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("init_db")


# ── Helpers ───────────────────────────────────────────────────────────────────

async def check_connection() -> bool:
    """Return True if the database is reachable."""
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
        logger.info("✓ Connected to PostgreSQL: %s", version)
        return True
    except Exception as exc:
        logger.error("✗ Cannot connect to database: %s", exc)
        return False


async def drop_all_tables() -> None:
    """Drop every table managed by SQLAlchemy metadata (dev/test only)."""
    async with async_engine.begin() as conn:
        logger.warning("⚠  Dropping all tables …")
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("All tables dropped.")


async def create_all_tables() -> None:
    """Create all tables defined in the ORM metadata (idempotent)."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("✓ All tables created (or already exist).")


async def seed_data() -> None:
    """
    Insert a minimal set of seed records for local development.
    Safe to run multiple times — checks for existing records first.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from sqlalchemy import select
    import bcrypt

    session_factory = async_sessionmaker(async_engine, expire_on_commit=False)

    async with session_factory() as session:
        # ── Seed admin user ────────────────────────────────────────────────
        existing = await session.scalar(
            select(User).where(User.email == "admin@churnplatform.io")
        )
        if not existing:
            pw_hash = bcrypt.hashpw(b"admin1234", bcrypt.gensalt()).decode()
            admin = User(
                name="Platform Admin",
                email="admin@churnplatform.io",
                password_hash=pw_hash,
                role="admin",
                is_active=True,
            )
            session.add(admin)
            logger.info("  + Seeded admin user (admin@churnplatform.io / admin1234)")

        # ── Seed sample customers ──────────────────────────────────────────
        count = await session.scalar(
            text("SELECT COUNT(*) FROM customers")
        )
        if count == 0:
            sample_customers = [
                Customer(
                    gender="Male", tenure=24, monthly_charges=79.99,
                    contract_type="month-to-month", payment_method="electronic-check",
                    subscription_type="standard", inactive_days=45,
                ),
                Customer(
                    gender="Female", tenure=60, monthly_charges=49.99,
                    contract_type="two-year", payment_method="bank-transfer",
                    subscription_type="basic", inactive_days=5,
                ),
                Customer(
                    gender="Male", tenure=3, monthly_charges=99.99,
                    contract_type="month-to-month", payment_method="credit-card",
                    subscription_type="premium", inactive_days=72,
                ),
            ]
            session.add_all(sample_customers)
            logger.info("  + Seeded %d sample customers.", len(sample_customers))

        # ── Seed analytics baseline ────────────────────────────────────────
        al_count = await session.scalar(
            text("SELECT COUNT(*) FROM analytics_logs")
        )
        if al_count == 0:
            session.add(AnalyticsLog(
                api_calls=0, predictions_generated=0,
                active_users=0, high_risk_customers=0,
                retention_actions_taken=0,
            ))
            logger.info("  + Seeded initial analytics_logs row.")

        await session.commit()

    logger.info("✓ Seed data inserted.")


async def verify_schema() -> None:
    """Log every table and column visible in the public schema."""
    async with async_engine.connect() as conn:
        tables = await conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
        )
        rows = tables.fetchall()

    logger.info("Schema tables (%d):", len(rows))
    for (tbl,) in rows:
        logger.info("  · %s", tbl)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> int:
    logger.info("=" * 60)
    logger.info("Churn Intelligence Platform — DB Initialisation")
    logger.info("Target: %s / %s", settings.postgres_host, settings.postgres_db)
    logger.info("=" * 60)

    if not await check_connection():
        return 1

    if args.check:
        return 0

    if args.drop_all:
        if settings.is_production:
            logger.error("--drop-all is not allowed in production!")
            return 1
        await drop_all_tables()

    await create_all_tables()
    await verify_schema()

    if args.seed:
        await seed_data()

    logger.info("✓ Initialisation complete.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise the churn platform database.")
    parser.add_argument("--drop-all", action="store_true", help="Drop all tables first (dev only).")
    parser.add_argument("--seed",     action="store_true", help="Insert seed data.")
    parser.add_argument("--check",    action="store_true", help="Connection check only.")
    parsed = parser.parse_args()

    exit_code = asyncio.run(main(parsed))
    sys.exit(exit_code)
