"""
services/retention/recommendation_engine.py
--------------------------------------------
Rule-based retention recommendation engine.

For every high-risk or medium-risk customer the engine evaluates a ranked
set of business rules and returns a prioritised list of retention actions
together with estimated revenue savings.

Rule anatomy
------------
Each rule has:
  name               : unique identifier
  condition(record)  : callable returning bool — whether the rule fires
  action             : human-readable recommended action string
  savings_multiplier : fraction of MonthlyCharges saved if action succeeds
  priority           : lower number = higher priority (1 = urgent)
  segment_affinity   : optional list of segment labels this rule targets

Public API
----------
- ``RetentionRecommendationEngine``
  - ``.recommend(customer_record, churn_probability, segment)``
    → list[Recommendation]
  - ``.recommend_batch(df)``
    → df with recommendations column
- ``Recommendation``                      (dataclass)
- ``get_recommendations(customer_dict, churn_probability)``
  convenience function
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from ml.config import RISK_THRESHOLDS

logger = logging.getLogger(__name__)


# ── Recommendation dataclass ──────────────────────────────────────────────────

@dataclass
class Recommendation:
    """A single retention action for one customer."""

    rule_name:          str
    action:             str
    priority:           int
    estimated_savings:  float        # USD per month
    savings_multiplier: float
    triggered_by:       str          # human-readable reason
    segment_affinity:   list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name":         self.rule_name,
            "action":            self.action,
            "priority":          self.priority,
            "estimated_savings": round(self.estimated_savings, 2),
            "savings_multiplier": self.savings_multiplier,
            "triggered_by":      self.triggered_by,
        }


# ── Rule definitions ──────────────────────────────────────────────────────────

@dataclass
class Rule:
    """A retention rule with condition, action, and metadata."""

    name:               str
    condition:          Callable[[dict[str, Any]], bool]
    action:             str
    triggered_by:       str
    savings_multiplier: float
    priority:           int
    segment_affinity:   list[str] = field(default_factory=list)


# All rules ordered by priority (1 = most urgent)
_RULES: list[Rule] = [
    # ── Contract & tenure rules ──────────────────────────────────────────
    Rule(
        name="low_tenure_onboarding",
        condition=lambda r: r.get("tenure", 99) <= 6,
        action=(
            "Assign a dedicated Customer Success Manager for the first 90 days. "
            "Schedule a weekly check-in call and send a personalised onboarding kit."
        ),
        triggered_by="Low tenure (≤ 6 months) — new customers churn at 3× the rate.",
        savings_multiplier=0.15,
        priority=1,
        segment_affinity=["Risky"],
    ),
    Rule(
        name="month_to_month_upgrade",
        condition=lambda r: r.get("Contract", "") == "Month-to-month",
        action=(
            "Offer a free month when the customer upgrades to an annual contract. "
            "Highlight the 10 % effective discount vs month-to-month pricing."
        ),
        triggered_by="Month-to-month contract — highest churn risk tier.",
        savings_multiplier=0.12,
        priority=2,
        segment_affinity=["Risky", "Inactive"],
    ),
    # ── Charges & value rules ─────────────────────────────────────────────
    Rule(
        name="high_monthly_charges_discount",
        condition=lambda r: float(r.get("MonthlyCharges", 0)) > 70,
        action=(
            "Apply a 15 % loyalty discount for the next 3 billing cycles. "
            "Bundle with complimentary device protection to increase perceived value."
        ),
        triggered_by="High monthly charges (> $70) with elevated churn risk.",
        savings_multiplier=0.15,
        priority=2,
        segment_affinity=["Premium", "Risky"],
    ),
    Rule(
        name="mid_charges_retention_credit",
        condition=lambda r: 45 < float(r.get("MonthlyCharges", 0)) <= 70,
        action=(
            "Apply a $10 account credit for the next billing cycle. "
            "Pair with a personalised usage summary showing value delivered."
        ),
        triggered_by="Mid-range charges with month-to-month contract.",
        savings_multiplier=0.08,
        priority=4,
        segment_affinity=["Risky"],
    ),
    # ── Inactivity & engagement rules ────────────────────────────────────
    Rule(
        name="high_inactivity_reengagement",
        condition=lambda r: int(r.get("InactiveDays", 0)) > 45,
        action=(
            "Send a personalised re-engagement email with the top 3 features the "
            "customer has not yet used. Follow up with a push notification after 3 days."
        ),
        triggered_by="High inactivity (> 45 days) — disengaged customers churn 2× faster.",
        savings_multiplier=0.10,
        priority=1,
        segment_affinity=["Inactive"],
    ),
    Rule(
        name="moderate_inactivity_nudge",
        condition=lambda r: 20 <= int(r.get("InactiveDays", 0)) <= 45,
        action=(
            "Trigger an automated 'We miss you' campaign with a feature highlight reel. "
            "Offer a free 30-day premium trial of an unused add-on service."
        ),
        triggered_by="Moderate inactivity (20–45 days).",
        savings_multiplier=0.06,
        priority=3,
        segment_affinity=["Inactive", "Risky"],
    ),
    # ── Payment method rules ──────────────────────────────────────────────
    Rule(
        name="electronic_check_autopay_discount",
        condition=lambda r: "electronic check" in str(r.get("PaymentMethod", "")).lower(),
        action=(
            "Offer a 5 % auto-pay discount for switching from electronic cheque "
            "to bank transfer or credit card. Simplify the switch with a guided in-app flow."
        ),
        triggered_by="Electronic cheque payment — highest payment-method churn correlate.",
        savings_multiplier=0.05,
        priority=3,
        segment_affinity=["Risky"],
    ),
    # ── Service add-on rules ──────────────────────────────────────────────
    Rule(
        name="fiber_no_security_bundle",
        condition=lambda r: (
            r.get("InternetService", "") == "Fiber optic"
            and r.get("OnlineSecurity", "") == "No"
        ),
        action=(
            "Offer 3 months of free Online Security for Fiber optic customers. "
            "Highlight the value of combined Fiber + Security in the upgrade modal."
        ),
        triggered_by="Fiber optic without Online Security — high upgrade potential.",
        savings_multiplier=0.07,
        priority=3,
        segment_affinity=["Premium"],
    ),
    Rule(
        name="no_tech_support_trial",
        condition=lambda r: r.get("TechSupport", "") == "No",
        action=(
            "Provide a complimentary 30-day 24/7 Tech Support trial. "
            "Send a welcome email with the support hotline and first 5 use cases."
        ),
        triggered_by="No tech support subscription — high pain-point for churn.",
        savings_multiplier=0.06,
        priority=4,
        segment_affinity=["Risky"],
    ),
    # ── Demographic rules ─────────────────────────────────────────────────
    Rule(
        name="senior_citizen_priority_support",
        condition=lambda r: int(r.get("SeniorCitizen", 0)) == 1,
        action=(
            "Enrol in Senior Care tier: priority call routing, simplified billing PDF, "
            "and a dedicated account manager for all service queries."
        ),
        triggered_by="Senior citizen — elevated churn from service friction.",
        savings_multiplier=0.09,
        priority=2,
        segment_affinity=["Loyal", "Risky"],
    ),
    Rule(
        name="no_partner_loyalty_reward",
        condition=lambda r: r.get("Partner", "") == "No" and int(r.get("tenure", 0)) >= 12,
        action=(
            "Recognise long-term single-account loyalty with a 'Loyal Member' badge "
            "and a referral bonus: $20 credit for each successful referral."
        ),
        triggered_by="Long-tenure customer without a partner account.",
        savings_multiplier=0.05,
        priority=5,
        segment_affinity=["Loyal"],
    ),
    # ── Streaming / service bundle ────────────────────────────────────────
    Rule(
        name="no_streaming_bundle_offer",
        condition=lambda r: (
            r.get("StreamingTV", "") == "No"
            and r.get("StreamingMovies", "") == "No"
            and r.get("InternetService", "") != "No"
        ),
        action=(
            "Offer a 'Entertainment Bundle' trial: 2 months of Streaming TV + Movies "
            "at 50 % off. Personalise with genre preferences from browsing history."
        ),
        triggered_by="Internet customer with no streaming — high upsell and retention value.",
        savings_multiplier=0.08,
        priority=4,
        segment_affinity=["Premium", "Loyal"],
    ),
]


# ── Recommendation engine ─────────────────────────────────────────────────────

class RetentionRecommendationEngine:
    """
    Evaluate retention rules for a customer and return ranked recommendations.

    Usage
    -----
    ::
        engine = RetentionRecommendationEngine()
        recs   = engine.recommend(customer_dict, churn_probability=0.72)
        for r in recs:
            print(r.action, r.estimated_savings)
    """

    def __init__(
        self,
        rules: list[Rule] | None = None,
        max_recommendations: int = 5,
        min_churn_prob: float = 0.30,
    ) -> None:
        """
        Parameters
        ----------
        rules               : list of Rule objects (defaults to built-in _RULES)
        max_recommendations : cap on returned recommendations per customer
        min_churn_prob      : customers below this threshold get no recommendations
        """
        self.rules               = rules or _RULES
        self.max_recommendations = max_recommendations
        self.min_churn_prob      = min_churn_prob

    def recommend(
        self,
        customer_record: dict[str, Any],
        churn_probability: float,
        segment: str | None = None,
    ) -> list[Recommendation]:
        """
        Evaluate all rules for one customer and return ranked recommendations.

        Parameters
        ----------
        customer_record   : raw customer feature dict
        churn_probability : float in [0, 1]
        segment           : optional segment label for affinity boosting

        Returns
        -------
        list of Recommendation, sorted by priority then estimated_savings (desc)
        """
        if churn_probability < self.min_churn_prob:
            return []

        monthly_charges = float(customer_record.get("MonthlyCharges", 50))
        recommendations: list[Recommendation] = []

        for rule in self.rules:
            try:
                fired = rule.condition(customer_record)
            except Exception as exc:
                logger.warning("Rule '%s' condition raised: %s", rule.name, exc)
                fired = False

            if not fired:
                continue

            # Savings = multiplier × monthly_charges × churn_probability weight
            raw_savings = monthly_charges * rule.savings_multiplier
            # Scale by churn probability (higher risk = higher potential save)
            weighted_savings = raw_savings * (0.5 + churn_probability * 0.5)

            # Boost priority if rule matches customer's segment
            priority = rule.priority
            if segment and segment in rule.segment_affinity:
                priority = max(1, priority - 1)   # bump up by 1 tier

            recommendations.append(
                Recommendation(
                    rule_name=rule.name,
                    action=rule.action,
                    priority=priority,
                    estimated_savings=round(weighted_savings, 2),
                    savings_multiplier=rule.savings_multiplier,
                    triggered_by=rule.triggered_by,
                    segment_affinity=rule.segment_affinity,
                )
            )

        # Sort: priority ASC (1 = highest), then estimated_savings DESC
        recommendations.sort(key=lambda r: (r.priority, -r.estimated_savings))
        return recommendations[: self.max_recommendations]

    def recommend_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the recommendation engine to every row in *df*.

        Expects columns: churn_probability, MonthlyCharges, and optionally 'segment'.
        Appends a 'recommendations' column containing a list of Recommendation objects.

        Returns a copy of *df* with 'recommendations' and 'top_action' columns.
        """
        recs_list:  list[list[Recommendation]] = []
        top_actions: list[str] = []

        for _, row in df.iterrows():
            record = row.to_dict()
            prob   = float(record.get("churn_probability", 0))
            seg    = record.get("segment")
            recs   = self.recommend(record, prob, seg)
            recs_list.append(recs)
            top_actions.append(recs[0].action if recs else "No action required.")

        out = df.copy()
        out["recommendations"] = recs_list
        out["top_action"]      = top_actions
        return out


# ── Convenience function ──────────────────────────────────────────────────────

_engine: RetentionRecommendationEngine | None = None


def get_engine() -> RetentionRecommendationEngine:
    """Return a module-level singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = RetentionRecommendationEngine()
    return _engine


def get_recommendations(
    customer_dict: dict[str, Any],
    churn_probability: float,
    segment: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return serialisable recommendation dicts for one customer.

    Parameters
    ----------
    customer_dict     : raw feature dict
    churn_probability : float [0, 1]
    segment           : optional segment label

    Returns
    -------
    list of dicts (safe for JSON serialisation)
    """
    recs = get_engine().recommend(customer_dict, churn_probability, segment)
    return [r.to_dict() for r in recs]
