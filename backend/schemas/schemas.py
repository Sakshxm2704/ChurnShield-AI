"""
backend/schemas/schemas.py
--------------------------
Pydantic v2 schemas for request validation and response serialisation.
Mirrors the ORM models but keeps the API contract decoupled from the DB layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from database.models import (
    ContractType, PaymentMethod, ResponseStatus,
    RiskLabel, SubscriptionType, UserRole,
)


# ── Shared config ─────────────────────────────────────────────────────────────

class _Base(BaseModel):
    model_config = {"from_attributes": True}


# ══════════════════════════════════════════════════════════════════════════════
# User schemas
# ══════════════════════════════════════════════════════════════════════════════

class UserCreate(_Base):
    name: str        = Field(..., min_length=2, max_length=120)
    email: EmailStr
    password: str    = Field(..., min_length=8, max_length=128)
    role: UserRole   = UserRole.VIEWER


class UserRead(_Base):
    id:         int
    name:       str
    email:      str
    role:       UserRole
    is_active:  bool
    created_at: datetime


class UserUpdate(_Base):
    name:      Optional[str]      = None
    role:      Optional[UserRole] = None
    is_active: Optional[bool]     = None


# ══════════════════════════════════════════════════════════════════════════════
# Customer schemas
# ══════════════════════════════════════════════════════════════════════════════

class CustomerCreate(_Base):
    gender:            Optional[str]          = None
    tenure:            int                    = Field(..., ge=0)
    monthly_charges:   float                  = Field(..., gt=0)
    contract_type:     ContractType
    payment_method:    PaymentMethod
    subscription_type: SubscriptionType
    inactive_days:     int                    = Field(..., ge=0)


class CustomerRead(_Base):
    customer_id:       int
    gender:            Optional[str]
    tenure:            int
    monthly_charges:   float
    contract_type:     ContractType
    payment_method:    PaymentMethod
    subscription_type: SubscriptionType
    inactive_days:     int
    created_at:        datetime


class CustomerUpdate(_Base):
    tenure:            Optional[int]              = Field(None, ge=0)
    monthly_charges:   Optional[float]            = Field(None, gt=0)
    contract_type:     Optional[ContractType]     = None
    payment_method:    Optional[PaymentMethod]    = None
    subscription_type: Optional[SubscriptionType] = None
    inactive_days:     Optional[int]              = Field(None, ge=0)


# ══════════════════════════════════════════════════════════════════════════════
# Prediction schemas
# ══════════════════════════════════════════════════════════════════════════════

class PredictionCreate(_Base):
    customer_id:       int
    churn_probability: float     = Field(..., ge=0.0, le=1.0)
    risk_score:        int       = Field(..., ge=0, le=100)
    prediction_label:  RiskLabel
    model_used:        str       = Field(..., max_length=80)
    shap_values:       Optional[str] = None

    @field_validator("risk_score", mode="before")
    @classmethod
    def derive_risk_score(cls, v: int, info) -> int:
        """Auto-derive risk_score from probability if not supplied."""
        if v is None and "churn_probability" in (info.data or {}):
            return round(info.data["churn_probability"] * 100)
        return v


class PredictionRead(_Base):
    prediction_id:     int
    customer_id:       int
    churn_probability: float
    risk_score:        int
    prediction_label:  RiskLabel
    model_used:        str
    shap_values:       Optional[str]
    created_at:        datetime


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation schemas
# ══════════════════════════════════════════════════════════════════════════════

class RecommendationCreate(_Base):
    customer_id:        int
    recommended_action: str   = Field(..., min_length=5)
    estimated_savings:  Optional[float] = Field(None, ge=0)
    priority:           int   = Field(1, ge=1, le=5)


class RecommendationRead(_Base):
    recommendation_id:  int
    customer_id:        int
    recommended_action: str
    estimated_savings:  Optional[float]
    priority:           int
    is_completed:       bool
    created_at:         datetime


# ══════════════════════════════════════════════════════════════════════════════
# RetentionLog schemas
# ══════════════════════════════════════════════════════════════════════════════

class RetentionLogCreate(_Base):
    customer_id:     int
    alert_sent:      bool            = False
    email_sent:      bool            = False
    response_status: ResponseStatus  = ResponseStatus.PENDING
    notes:           Optional[str]   = None


class RetentionLogRead(_Base):
    log_id:          int
    customer_id:     int
    alert_sent:      bool
    email_sent:      bool
    response_status: ResponseStatus
    notes:           Optional[str]
    created_at:      datetime


# ══════════════════════════════════════════════════════════════════════════════
# AnalyticsLog schemas
# ══════════════════════════════════════════════════════════════════════════════

class AnalyticsLogCreate(_Base):
    api_calls:              int = Field(..., ge=0)
    predictions_generated:  int = Field(..., ge=0)
    active_users:           int = Field(..., ge=0)
    high_risk_customers:    int = Field(..., ge=0)
    retention_actions_taken: int = Field(..., ge=0)


class AnalyticsLogRead(_Base):
    id:                      int
    api_calls:               int
    predictions_generated:   int
    active_users:            int
    high_risk_customers:     int
    retention_actions_taken: int
    created_at:              datetime


# ── Auth / token schemas ──────────────────────────────────────────────────────

class Token(_Base):
    access_token: str
    token_type: str = "bearer"


class TokenData(_Base):
    user_id: Optional[int] = None
    email:   Optional[str] = None
    role:    Optional[UserRole] = None
