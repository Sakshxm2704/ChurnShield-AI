"""
database/models.py
------------------
SQLAlchemy ORM models for the Churn Intelligence Platform.

Tables
------
- users              System users / admin accounts
- customers          Telecom customer records
- predictions        ML churn predictions per customer
- recommendations    AI-generated retention actions
- retention_logs     Outreach & alert audit trail
- analytics_logs     Daily platform usage telemetry

All timestamps are stored as UTC and auto-populated.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════

class UserRole(str, enum.Enum):
    ADMIN       = "admin"
    ANALYST     = "analyst"
    VIEWER      = "viewer"
    RETENTION   = "retention"


class ContractType(str, enum.Enum):
    MONTH_TO_MONTH = "month-to-month"
    ONE_YEAR       = "one-year"
    TWO_YEAR       = "two-year"


class PaymentMethod(str, enum.Enum):
    ELECTRONIC_CHECK  = "electronic-check"
    MAILED_CHECK      = "mailed-check"
    BANK_TRANSFER     = "bank-transfer"
    CREDIT_CARD       = "credit-card"


class SubscriptionType(str, enum.Enum):
    BASIC    = "basic"
    STANDARD = "standard"
    PREMIUM  = "premium"


class RiskLabel(str, enum.Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


class ResponseStatus(str, enum.Enum):
    PENDING   = "pending"
    SENT      = "sent"
    OPENED    = "opened"
    RESPONDED = "responded"
    IGNORED   = "ignored"


# ══════════════════════════════════════════════════════════════════════════════
# Mixin: auto-timestamps
# ══════════════════════════════════════════════════════════════════════════════

class TimestampMixin:
    """Adds ``created_at`` and ``updated_at`` to any model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="UTC timestamp when the row was inserted.",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="UTC timestamp of the last update.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# users
# ══════════════════════════════════════════════════════════════════════════════

class User(TimestampMixin, Base):
    """
    Platform system users (admins, analysts, retention agents).

    Constraints
    -----------
    - ``email`` must be unique across the table.
    - ``role`` is restricted to UserRole enum values.
    """

    __tablename__ = "users"

    __table_args__ = (
        UniqueConstraint("email", name="uq_users_email"),
        Index("ix_users_role", "role"),
        Index("ix_users_created_at", "created_at"),
        {"comment": "Platform system users with role-based access control."},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
        doc="Surrogate primary key.",
    )
    name: Mapped[str] = mapped_column(
        String(120), nullable=False,
        doc="Full display name of the user.",
    )
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True,
        doc="Unique login email address.",
    )
    password_hash: Mapped[str] = mapped_column(
        String(255), nullable=False,
        doc="Bcrypt-hashed password. Never store plain-text.",
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role_enum", create_type=True),
        nullable=False,
        default=UserRole.VIEWER,
        doc="Access role controlling API permissions.",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        doc="Soft-delete / disable flag.",
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role}>"


# ══════════════════════════════════════════════════════════════════════════════
# customers
# ══════════════════════════════════════════════════════════════════════════════

class Customer(TimestampMixin, Base):
    """
    Telecom customer subscription and demographic data.

    Relationships
    -------------
    - One → Many  :  predictions, recommendations, retention_logs
    """

    __tablename__ = "customers"

    __table_args__ = (
        Index("ix_customers_contract_type",   "contract_type"),
        Index("ix_customers_subscription",    "subscription_type"),
        Index("ix_customers_tenure",          "tenure"),
        Index("ix_customers_monthly_charges", "monthly_charges"),
        Index("ix_customers_inactive_days",   "inactive_days"),
        Index("ix_customers_created_at",      "created_at"),
        {"comment": "Core customer profile used as ML feature input."},
    )

    customer_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
        doc="Surrogate primary key.",
    )
    gender: Mapped[str | None] = mapped_column(
        String(10), nullable=True,
        doc="Customer gender (Male / Female / Other).",
    )
    tenure: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Number of months the customer has been with the company.",
    )
    monthly_charges: Mapped[float] = mapped_column(
        Numeric(10, 2), nullable=False,
        doc="Current monthly bill amount in USD.",
    )
    contract_type: Mapped[ContractType] = mapped_column(
        Enum(ContractType, name="contract_type_enum", create_type=True),
        nullable=False,
        doc="Billing contract length.",
    )
    payment_method: Mapped[PaymentMethod] = mapped_column(
        Enum(PaymentMethod, name="payment_method_enum", create_type=True),
        nullable=False,
        doc="How the customer pays their bill.",
    )
    subscription_type: Mapped[SubscriptionType] = mapped_column(
        Enum(SubscriptionType, name="subscription_type_enum", create_type=True),
        nullable=False,
        doc="Service tier the customer is subscribed to.",
    )
    inactive_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Days since last meaningful product interaction.",
    )

    # ── Relationships ──────────────────────────────────────────────────────
    predictions:    Mapped[list["Prediction"]]    = relationship(
        "Prediction",    back_populates="customer", cascade="all, delete-orphan",
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        "Recommendation", back_populates="customer", cascade="all, delete-orphan",
    )
    retention_logs: Mapped[list["RetentionLog"]] = relationship(
        "RetentionLog",  back_populates="customer", cascade="all, delete-orphan",
    )


# ══════════════════════════════════════════════════════════════════════════════
# predictions
# ══════════════════════════════════════════════════════════════════════════════

class Prediction(TimestampMixin, Base):
    """
    ML churn prediction results stored after each inference run.

    Constraints
    -----------
    - ``churn_probability`` ∈ [0.0, 1.0]  (enforced at application layer)
    - ``risk_score``         ∈ [0, 100]
    """

    __tablename__ = "predictions"

    __table_args__ = (
        Index("ix_predictions_customer_id",    "customer_id"),
        Index("ix_predictions_prediction_label","prediction_label"),
        Index("ix_predictions_risk_score",     "risk_score"),
        Index("ix_predictions_model_used",     "model_used"),
        Index("ix_predictions_created_at",     "created_at"),
        {"comment": "Churn probability scores generated by ML models."},
    )

    prediction_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    customer_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("customers.customer_id", ondelete="CASCADE", name="fk_predictions_customer"),
        nullable=False,
        doc="Reference to the scored customer.",
    )
    churn_probability: Mapped[float] = mapped_column(
        Float, nullable=False,
        doc="Model output probability in [0.0, 1.0].",
    )
    risk_score: Mapped[int] = mapped_column(
        Integer, nullable=False,
        doc="Normalised risk score in [0, 100] for display purposes.",
    )
    prediction_label: Mapped[RiskLabel] = mapped_column(
        Enum(RiskLabel, name="risk_label_enum", create_type=True),
        nullable=False,
        doc="Human-readable risk band: low / medium / high.",
    )
    model_used: Mapped[str] = mapped_column(
        String(80), nullable=False,
        doc="Identifier of the model version that produced this prediction.",
    )
    shap_values: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="JSON-serialised SHAP feature importance for explainability.",
    )

    # ── Relationship ───────────────────────────────────────────────────────
    customer: Mapped["Customer"] = relationship("Customer", back_populates="predictions")


# ══════════════════════════════════════════════════════════════════════════════
# recommendations
# ══════════════════════════════════════════════════════════════════════════════

class Recommendation(TimestampMixin, Base):
    """
    AI-generated retention actions surfaced for high-risk customers.
    """

    __tablename__ = "recommendations"

    __table_args__ = (
        Index("ix_recommendations_customer_id",  "customer_id"),
        Index("ix_recommendations_created_at",   "created_at"),
        {"comment": "Personalised retention offers and actions per customer."},
    )

    recommendation_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    customer_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("customers.customer_id", ondelete="CASCADE", name="fk_recommendations_customer"),
        nullable=False,
    )
    recommended_action: Mapped[str] = mapped_column(
        Text, nullable=False,
        doc="Description of the retention action to take.",
    )
    estimated_savings: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
        doc="Projected revenue saved if this action succeeds (USD).",
    )
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
        doc="Execution priority: 1 = highest.",
    )
    is_completed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        doc="Whether the action has been carried out.",
    )

    # ── Relationship ───────────────────────────────────────────────────────
    customer: Mapped["Customer"] = relationship("Customer", back_populates="recommendations")


# ══════════════════════════════════════════════════════════════════════════════
# retention_logs
# ══════════════════════════════════════════════════════════════════════════════

class RetentionLog(TimestampMixin, Base):
    """
    Audit trail for every outreach event triggered by the retention engine.
    """

    __tablename__ = "retention_logs"

    __table_args__ = (
        Index("ix_retention_logs_customer_id",     "customer_id"),
        Index("ix_retention_logs_response_status", "response_status"),
        Index("ix_retention_logs_created_at",      "created_at"),
        {"comment": "Outreach audit trail: alerts, emails, and customer responses."},
    )

    log_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    customer_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("customers.customer_id", ondelete="CASCADE", name="fk_retention_logs_customer"),
        nullable=False,
    )
    alert_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        doc="Whether an in-platform alert was dispatched.",
    )
    email_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False,
        doc="Whether a retention email was sent to the customer.",
    )
    response_status: Mapped[ResponseStatus] = mapped_column(
        Enum(ResponseStatus, name="response_status_enum", create_type=True),
        nullable=False,
        default=ResponseStatus.PENDING,
        doc="Current state of the customer's response to outreach.",
    )
    notes: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        doc="Optional agent notes about the interaction.",
    )

    # ── Relationship ───────────────────────────────────────────────────────
    customer: Mapped["Customer"] = relationship("Customer", back_populates="retention_logs")


# ══════════════════════════════════════════════════════════════════════════════
# analytics_logs
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticsLog(TimestampMixin, Base):
    """
    Daily platform usage telemetry (API calls, predictions, active users).
    One row per calendar day — inserted/upserted by a nightly job.
    """

    __tablename__ = "analytics_logs"

    __table_args__ = (
        Index("ix_analytics_logs_created_at", "created_at"),
        {"comment": "Aggregated daily platform usage metrics."},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True,
    )
    api_calls: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Total API calls recorded in the logging window.",
    )
    predictions_generated: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Number of churn predictions generated in the window.",
    )
    active_users: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Distinct platform users active during the window.",
    )
    high_risk_customers: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Customers flagged as high-risk in the window.",
    )
    retention_actions_taken: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        doc="Retention actions completed in the window.",
    )
