"""
analytics/revenue_estimator.py
-------------------------------
Revenue loss estimation for the Churn Intelligence Platform.

Computes three revenue risk metrics at both customer and portfolio level:

  expected_monthly_loss   Probability-weighted monthly revenue at risk
  expected_annual_loss    Annualised version of the above
  worst_case_loss         Revenue lost if ALL high-risk customers churn
  recoverable_revenue     Portion recoverable via recommended retention actions
  ltv_at_risk             Customer Lifetime Value at risk

Public API
----------
- ``RevenueEstimator``
  - ``.estimate_customer(record, churn_probability, recommendations)``
    → CustomerRevenueRisk
  - ``.estimate_portfolio(df)``
    → PortfolioRevenueRisk
- ``CustomerRevenueRisk``   (dataclass)
- ``PortfolioRevenueRisk``  (dataclass)
- ``estimate_revenue_risk(df)``   convenience wrapper
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Average months a customer stays after a successful retention intervention
_RETENTION_EXTENSION_MONTHS = 12

# Assumed average gross margin for revenue calculations
_GROSS_MARGIN = 0.65

# LTV discount rate (monthly, approximation of WACC / 12)
_MONTHLY_DISCOUNT_RATE = 0.01


# ── Customer-level risk ───────────────────────────────────────────────────────

@dataclass
class CustomerRevenueRisk:
    """Revenue risk metrics for a single customer."""

    customer_id:            Any
    monthly_charges:        float
    churn_probability:      float
    risk_category:          str

    # Risk metrics
    expected_monthly_loss:  float   # E[loss] = P(churn) × monthly_charges
    expected_annual_loss:   float
    worst_case_monthly:     float   # If customer definitely churns
    ltv_at_risk:            float   # LTV × churn_probability

    # Recovery metrics
    recoverable_revenue:    float   # Sum of all recommendation savings
    net_revenue_at_risk:    float   # expected_annual_loss − recoverable_revenue
    roi_of_intervention:    float   # recoverable / cost_to_retain (assumed 1 month charge)

    top_action:             str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id":           self.customer_id,
            "monthly_charges":       round(self.monthly_charges, 2),
            "churn_probability":     round(self.churn_probability, 4),
            "risk_category":         self.risk_category,
            "expected_monthly_loss": round(self.expected_monthly_loss, 2),
            "expected_annual_loss":  round(self.expected_annual_loss, 2),
            "worst_case_monthly":    round(self.worst_case_monthly, 2),
            "ltv_at_risk":           round(self.ltv_at_risk, 2),
            "recoverable_revenue":   round(self.recoverable_revenue, 2),
            "net_revenue_at_risk":   round(self.net_revenue_at_risk, 2),
            "roi_of_intervention":   round(self.roi_of_intervention, 2),
            "top_action":            self.top_action,
        }


# ── Portfolio-level risk ──────────────────────────────────────────────────────

@dataclass
class PortfolioRevenueRisk:
    """Aggregated revenue risk across the entire customer portfolio."""

    total_customers:            int
    high_risk_count:            int
    medium_risk_count:          int
    low_risk_count:             int

    total_monthly_revenue:      float
    expected_monthly_loss:      float
    expected_annual_loss:       float
    worst_case_annual_loss:     float

    recoverable_revenue_annual: float
    net_revenue_at_risk_annual: float

    avg_churn_probability:      float
    total_ltv_at_risk:          float

    # Revenue by segment (if segment column present)
    revenue_by_segment:         dict[str, float] = field(default_factory=dict)
    churn_rate_by_segment:      dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_customers":            self.total_customers,
            "high_risk_count":            self.high_risk_count,
            "medium_risk_count":          self.medium_risk_count,
            "low_risk_count":             self.low_risk_count,
            "total_monthly_revenue":      round(self.total_monthly_revenue, 2),
            "expected_monthly_loss":      round(self.expected_monthly_loss, 2),
            "expected_annual_loss":       round(self.expected_annual_loss, 2),
            "worst_case_annual_loss":     round(self.worst_case_annual_loss, 2),
            "recoverable_revenue_annual": round(self.recoverable_revenue_annual, 2),
            "net_revenue_at_risk_annual": round(self.net_revenue_at_risk_annual, 2),
            "avg_churn_probability":      round(self.avg_churn_probability, 4),
            "total_ltv_at_risk":          round(self.total_ltv_at_risk, 2),
            "revenue_by_segment":         {k: round(v, 2) for k, v in self.revenue_by_segment.items()},
            "churn_rate_by_segment":      {k: round(v, 4) for k, v in self.churn_rate_by_segment.items()},
        }


# ── LTV helper ────────────────────────────────────────────────────────────────

def estimate_ltv(
    monthly_charges: float,
    tenure_months: int,
    expected_remaining_months: int = 24,
    gross_margin: float = _GROSS_MARGIN,
    monthly_discount_rate: float = _MONTHLY_DISCOUNT_RATE,
) -> float:
    """
    Estimate Customer Lifetime Value (CLV) using a DCF approach.

    LTV = Σ (monthly_revenue × margin) / (1 + r)^t  for t in remaining months

    Parameters
    ----------
    monthly_charges          : current monthly bill
    tenure_months            : how long the customer has been active
    expected_remaining_months: expected future tenure
    gross_margin             : fraction of revenue that is gross profit
    monthly_discount_rate    : discount rate per month

    Returns
    -------
    float : estimated LTV in USD
    """
    monthly_profit = monthly_charges * gross_margin
    if monthly_discount_rate <= 0:
        return monthly_profit * expected_remaining_months

    # Present value of annuity
    ltv = monthly_profit * (
        (1 - (1 + monthly_discount_rate) ** -expected_remaining_months)
        / monthly_discount_rate
    )
    return round(float(ltv), 2)


# ── Revenue Estimator ─────────────────────────────────────────────────────────

class RevenueEstimator:
    """
    Compute revenue risk metrics at customer and portfolio level.

    Usage
    -----
    ::
        estimator = RevenueEstimator()
        portfolio = estimator.estimate_portfolio(df_with_predictions)
        print(portfolio.expected_annual_loss)
    """

    def __init__(
        self,
        gross_margin: float = _GROSS_MARGIN,
        retention_extension_months: int = _RETENTION_EXTENSION_MONTHS,
    ) -> None:
        self.gross_margin               = gross_margin
        self.retention_extension_months = retention_extension_months

    def estimate_customer(
        self,
        record: dict[str, Any],
        churn_probability: float,
        recommendations: list[dict[str, Any]] | None = None,
        risk_category: str = "Medium",
    ) -> CustomerRevenueRisk:
        """
        Compute revenue risk for a single customer.

        Parameters
        ----------
        record            : raw customer feature dict
        churn_probability : float in [0, 1]
        recommendations   : list of recommendation dicts (from recommendation engine)
        risk_category     : "High" | "Medium" | "Low"

        Returns
        -------
        CustomerRevenueRisk
        """
        monthly = float(record.get("MonthlyCharges", 0))
        tenure  = int(record.get("tenure", 12))
        cid     = record.get("customerID", record.get("customer_id", "unknown"))

        # Expected loss = E[revenue lost] = P(churn) × monthly_charges
        expected_monthly = monthly * churn_probability
        expected_annual  = expected_monthly * 12

        # Worst case (certain churn)
        worst_case = monthly

        # LTV at risk
        ltv      = estimate_ltv(monthly, tenure, gross_margin=self.gross_margin)
        ltv_risk = ltv * churn_probability

        # Recoverable revenue from all recommended actions (sum of savings × 12)
        recoverable = 0.0
        top_action  = "No action required."
        if recommendations:
            recoverable = sum(
                r.get("estimated_savings", 0) for r in recommendations
            ) * 12
            top_action = recommendations[0].get("action", "") if recommendations else ""

        # Net risk after recovery
        net_risk = max(0.0, expected_annual - recoverable)

        # ROI: recoverable / cost of intervention (1 month charge)
        intervention_cost = monthly
        roi = (recoverable / intervention_cost) if intervention_cost > 0 else 0.0

        return CustomerRevenueRisk(
            customer_id=cid,
            monthly_charges=monthly,
            churn_probability=churn_probability,
            risk_category=risk_category,
            expected_monthly_loss=expected_monthly,
            expected_annual_loss=expected_annual,
            worst_case_monthly=worst_case,
            ltv_at_risk=ltv_risk,
            recoverable_revenue=recoverable,
            net_revenue_at_risk=net_risk,
            roi_of_intervention=roi,
            top_action=top_action,
        )

    def estimate_portfolio(self, df: pd.DataFrame) -> PortfolioRevenueRisk:
        """
        Compute aggregated revenue risk across all customers in *df*.

        Expects columns: MonthlyCharges, churn_probability, risk_category.
        Optionally: segment (for segment-level breakdown).

        Returns
        -------
        PortfolioRevenueRisk
        """
        if "churn_probability" not in df.columns:
            raise ValueError("DataFrame must contain 'churn_probability' column.")
        if "MonthlyCharges" not in df.columns:
            raise ValueError("DataFrame must contain 'MonthlyCharges' column.")

        n = len(df)
        charges = df["MonthlyCharges"].astype(float)
        probs   = df["churn_probability"].astype(float)

        # Risk counts
        high_count   = int((probs >= 0.65).sum())
        medium_count = int(((probs >= 0.35) & (probs < 0.65)).sum())
        low_count    = int((probs < 0.35).sum())

        # Revenue metrics
        total_monthly_rev     = float(charges.sum())
        expected_monthly_loss = float((charges * probs).sum())
        expected_annual_loss  = expected_monthly_loss * 12

        # Worst case: all high-risk customers churn
        high_risk_mask    = probs >= 0.65
        worst_case_annual = float(charges[high_risk_mask].sum()) * 12

        # LTV at risk (simplified: use average 24-month remaining tenure)
        ltv_series  = charges.apply(lambda c: estimate_ltv(c, 12, gross_margin=self.gross_margin))
        total_ltv_risk = float((ltv_series * probs).sum())

        # Recoverable: approximate 10 % recovery rate × expected annual loss
        # (Overridden if recommendation data is available)
        recoverable = expected_annual_loss * 0.10
        if "estimated_savings" in df.columns:
            recoverable = float(df["estimated_savings"].fillna(0).sum()) * 12

        net_risk = max(0.0, expected_annual_loss - recoverable)

        # Segment breakdown
        revenue_by_segment:    dict[str, float] = {}
        churn_rate_by_segment: dict[str, float] = {}

        if "segment" in df.columns:
            for seg, grp in df.groupby("segment"):
                revenue_by_segment[str(seg)] = float(
                    (grp["MonthlyCharges"].astype(float) * grp["churn_probability"].astype(float)).sum() * 12
                )
                churn_rate_by_segment[str(seg)] = float(grp["churn_probability"].mean())

        return PortfolioRevenueRisk(
            total_customers=n,
            high_risk_count=high_count,
            medium_risk_count=medium_count,
            low_risk_count=low_count,
            total_monthly_revenue=total_monthly_rev,
            expected_monthly_loss=expected_monthly_loss,
            expected_annual_loss=expected_annual_loss,
            worst_case_annual_loss=worst_case_annual,
            recoverable_revenue_annual=recoverable,
            net_revenue_at_risk_annual=net_risk,
            avg_churn_probability=float(probs.mean()),
            total_ltv_at_risk=total_ltv_risk,
            revenue_by_segment=revenue_by_segment,
            churn_rate_by_segment=churn_rate_by_segment,
        )


# ── Convenience wrapper ───────────────────────────────────────────────────────

def estimate_revenue_risk(df: pd.DataFrame) -> PortfolioRevenueRisk:
    """
    Compute portfolio revenue risk for *df*.

    Parameters
    ----------
    df : DataFrame with churn_probability and MonthlyCharges columns.

    Returns
    -------
    PortfolioRevenueRisk
    """
    return RevenueEstimator().estimate_portfolio(df)
