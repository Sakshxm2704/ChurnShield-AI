"""
analytics/whatif_simulator.py
------------------------------
What-If Scenario Simulator for the Churn Intelligence Platform.

Allows analysts and the Streamlit dashboard to answer questions like:
  "If we move this customer to an annual contract, how much does their
   churn probability drop?"
  "What happens to portfolio risk if we reduce all high-risk customers'
   monthly charges by 15 %?"

Public API
----------
- ``WhatIfSimulator``
  - ``.simulate_customer(base_record, scenario_changes, predictor)``
    → ScenarioResult
  - ``.simulate_portfolio(df, portfolio_scenario, predictor)``
    → PortfolioScenarioResult
  - ``.sensitivity_analysis(record, feature, values, predictor)``
    → list[ScenarioResult]
- ``ScenarioResult``                  (dataclass)
- ``PortfolioScenarioResult``         (dataclass)
- ``PRESET_SCENARIOS``                dict of named preset scenarios
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Scenario dataclasses ──────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Result of a What-If simulation for one customer."""

    customer_id:             Any
    base_churn_probability:  float
    new_churn_probability:   float
    delta_probability:       float          # new - base (negative = improvement)
    base_risk_category:      str
    new_risk_category:       str
    base_risk_score:         int
    new_risk_score:          int
    changes_applied:         dict[str, Any]  # feature → new_value
    monthly_charges:         float
    monthly_revenue_saved:   float          # if churn prevented

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id":            self.customer_id,
            "base_churn_probability": round(self.base_churn_probability, 4),
            "new_churn_probability":  round(self.new_churn_probability, 4),
            "delta_probability":      round(self.delta_probability, 4),
            "improvement_pct":        round(-self.delta_probability / max(self.base_churn_probability, 1e-6) * 100, 1),
            "base_risk_category":     self.base_risk_category,
            "new_risk_category":      self.new_risk_category,
            "base_risk_score":        self.base_risk_score,
            "new_risk_score":         self.new_risk_score,
            "changes_applied":        self.changes_applied,
            "monthly_charges":        round(self.monthly_charges, 2),
            "monthly_revenue_saved":  round(self.monthly_revenue_saved, 2),
        }


@dataclass
class PortfolioScenarioResult:
    """Aggregated result of applying a scenario to many customers."""

    scenario_name:              str
    total_customers:            int
    base_avg_churn_probability: float
    new_avg_churn_probability:  float
    customers_improved:         int
    customers_worsened:         int
    avg_probability_delta:      float
    base_expected_annual_loss:  float
    new_expected_annual_loss:   float
    annual_savings:             float
    individual_results:         list[ScenarioResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name":              self.scenario_name,
            "total_customers":            self.total_customers,
            "base_avg_churn_probability": round(self.base_avg_churn_probability, 4),
            "new_avg_churn_probability":  round(self.new_avg_churn_probability, 4),
            "customers_improved":         self.customers_improved,
            "customers_worsened":         self.customers_worsened,
            "avg_probability_delta":      round(self.avg_probability_delta, 4),
            "base_expected_annual_loss":  round(self.base_expected_annual_loss, 2),
            "new_expected_annual_loss":   round(self.new_expected_annual_loss, 2),
            "annual_savings":             round(self.annual_savings, 2),
        }


# ── Preset scenarios ──────────────────────────────────────────────────────────

PRESET_SCENARIOS: dict[str, dict[str, Any]] = {
    "upgrade_to_annual_contract": {
        "description": "Upgrade customer from Month-to-month to One year contract.",
        "changes":     {"Contract": "One year"},
    },
    "upgrade_to_two_year_contract": {
        "description": "Upgrade customer to Two year contract.",
        "changes":     {"Contract": "Two year"},
    },
    "switch_to_autopay": {
        "description": "Switch payment from electronic check to bank transfer.",
        "changes":     {"PaymentMethod": "Bank transfer (automatic)"},
    },
    "reduce_charges_10pct": {
        "description": "Apply 10 % discount on monthly charges.",
        "changes":     {"__modifier__": {"MonthlyCharges": 0.90}},  # multiplicative
    },
    "reduce_charges_15pct": {
        "description": "Apply 15 % loyalty discount on monthly charges.",
        "changes":     {"__modifier__": {"MonthlyCharges": 0.85}},
    },
    "add_online_security": {
        "description": "Add Online Security subscription.",
        "changes":     {"OnlineSecurity": "Yes"},
    },
    "add_tech_support": {
        "description": "Add Tech Support subscription.",
        "changes":     {"TechSupport": "Yes"},
    },
    "reduce_inactivity": {
        "description": "Simulate successful re-engagement (InactiveDays → 5).",
        "changes":     {"InactiveDays": 5},
    },
    "full_retention_bundle": {
        "description": "Apply contract upgrade + autopay + security (combined).",
        "changes":     {
            "Contract":       "One year",
            "PaymentMethod":  "Bank transfer (automatic)",
            "OnlineSecurity": "Yes",
            "InactiveDays":   10,
        },
    },
}


# ── What-If Simulator ──────────────────────────────────────────────────────────

class WhatIfSimulator:
    """
    Applies feature changes to customer records and re-runs prediction
    to measure churn probability impact.

    Usage
    -----
    ::
        from ml.models.predictor import ChurnPredictor
        predictor = ChurnPredictor().load()

        sim = WhatIfSimulator()
        result = sim.simulate_customer(
            customer_dict,
            scenario_changes={"Contract": "Two year"},
            predictor=predictor,
        )
        print(result.delta_probability)  # e.g. -0.23 (23 % reduction)
    """

    def _apply_changes(
        self,
        record: dict[str, Any],
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Return a modified copy of *record* with *changes* applied.

        Changes can be:
        - Direct value overrides: {"Contract": "Two year"}
        - Multiplicative modifiers via the special key ``__modifier__``:
          {"__modifier__": {"MonthlyCharges": 0.85}}
        """
        new_record = copy.deepcopy(record)

        # Handle multiplicative modifiers
        modifiers = changes.pop("__modifier__", {})
        for feat, multiplier in modifiers.items():
            if feat in new_record:
                try:
                    new_record[feat] = round(float(new_record[feat]) * multiplier, 2)
                except (ValueError, TypeError):
                    logger.warning("Could not apply multiplier to %s=%s", feat, new_record[feat])

        # Direct overrides
        new_record.update(changes)
        return new_record

    def simulate_customer(
        self,
        base_record: dict[str, Any],
        scenario_changes: dict[str, Any],
        predictor: Any,
        scenario_name: str = "custom",
    ) -> ScenarioResult:
        """
        Run a What-If scenario for one customer.

        Parameters
        ----------
        base_record      : original customer feature dict
        scenario_changes : dict of feature overrides (or preset scenario dict)
        predictor        : fitted ChurnPredictor instance
        scenario_name    : label for logging/display

        Returns
        -------
        ScenarioResult
        """
        # Support passing a preset scenario dict directly
        if "changes" in scenario_changes and "description" in scenario_changes:
            changes = copy.deepcopy(scenario_changes["changes"])
        else:
            changes = copy.deepcopy(scenario_changes)

        # Base prediction
        base_result = predictor.predict(base_record)
        base_prob   = base_result["churn_probability"]
        base_cat    = base_result["risk_category"]
        base_score  = base_result["risk_score"]

        # Apply changes and re-predict
        modified_record = self._apply_changes(base_record, changes)
        new_result      = predictor.predict(modified_record)
        new_prob        = new_result["churn_probability"]
        new_cat         = new_result["risk_category"]
        new_score       = new_result["risk_score"]

        monthly = float(base_record.get("MonthlyCharges", 0))
        delta   = new_prob - base_prob
        # Revenue saved = probability drop × monthly charges
        rev_saved = max(0.0, -delta) * monthly

        cid = base_record.get("customerID", base_record.get("customer_id", "unknown"))

        result = ScenarioResult(
            customer_id=cid,
            base_churn_probability=base_prob,
            new_churn_probability=new_prob,
            delta_probability=delta,
            base_risk_category=base_cat,
            new_risk_category=new_cat,
            base_risk_score=base_score,
            new_risk_score=new_score,
            changes_applied=changes,
            monthly_charges=monthly,
            monthly_revenue_saved=rev_saved,
        )

        logger.debug(
            "[WhatIf] %s  scenario=%s  Δprob=%.3f  Δscore=%d",
            cid, scenario_name, delta, new_score - base_score,
        )
        return result

    def simulate_portfolio(
        self,
        df: pd.DataFrame,
        portfolio_scenario: dict[str, Any],
        predictor: Any,
        scenario_name: str = "portfolio_scenario",
    ) -> PortfolioScenarioResult:
        """
        Apply *portfolio_scenario* to every row in *df* and aggregate results.

        Parameters
        ----------
        df                  : DataFrame of customer records (raw features)
        portfolio_scenario  : changes dict or preset scenario dict
        predictor           : fitted ChurnPredictor
        scenario_name       : label for the result

        Returns
        -------
        PortfolioScenarioResult
        """
        results: list[ScenarioResult] = []

        for _, row in df.iterrows():
            record = row.to_dict()
            try:
                res = self.simulate_customer(record, portfolio_scenario, predictor, scenario_name)
                results.append(res)
            except Exception as exc:
                logger.warning("What-If failed for row: %s", exc)

        if not results:
            raise RuntimeError("No valid What-If results generated.")

        base_probs = [r.base_churn_probability for r in results]
        new_probs  = [r.new_churn_probability  for r in results]
        charges    = [r.monthly_charges         for r in results]

        base_expected_annual = sum(p * c for p, c in zip(base_probs, charges)) * 12
        new_expected_annual  = sum(p * c for p, c in zip(new_probs,  charges)) * 12

        return PortfolioScenarioResult(
            scenario_name=scenario_name,
            total_customers=len(results),
            base_avg_churn_probability=float(np.mean(base_probs)),
            new_avg_churn_probability=float(np.mean(new_probs)),
            customers_improved=sum(1 for r in results if r.delta_probability < -0.01),
            customers_worsened=sum(1 for r in results if r.delta_probability >  0.01),
            avg_probability_delta=float(np.mean([r.delta_probability for r in results])),
            base_expected_annual_loss=base_expected_annual,
            new_expected_annual_loss=new_expected_annual,
            annual_savings=max(0.0, base_expected_annual - new_expected_annual),
            individual_results=results,
        )

    def sensitivity_analysis(
        self,
        base_record: dict[str, Any],
        feature: str,
        values: list[Any],
        predictor: Any,
    ) -> list[ScenarioResult]:
        """
        Sweep *feature* across *values* and return a ScenarioResult for each.

        Useful for plotting churn probability vs. a single feature (e.g. tenure).

        Parameters
        ----------
        base_record : base customer record
        feature     : feature name to vary
        values      : list of values to test
        predictor   : fitted ChurnPredictor

        Returns
        -------
        list[ScenarioResult] — one per value
        """
        results: list[ScenarioResult] = []
        for val in values:
            try:
                res = self.simulate_customer(
                    base_record,
                    {feature: val},
                    predictor,
                    scenario_name=f"{feature}={val}",
                )
                results.append(res)
            except Exception as exc:
                logger.warning("Sensitivity analysis failed for %s=%s: %s", feature, val, exc)

        return results
