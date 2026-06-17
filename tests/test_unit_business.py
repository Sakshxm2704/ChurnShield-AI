"""
tests/test_unit_business.py
-----------------------------
Unit tests: recommendation engine, revenue estimator, what-if simulator.
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd, numpy as np
from services.retention.recommendation_engine import RetentionRecommendationEngine

# ── shared records ────────────────────────────────────────────────────────────
HIGH = {"customerID":"H1","tenure":2,"MonthlyCharges":90.0,"InactiveDays":70,
        "Contract":"Month-to-month","PaymentMethod":"Electronic check","SeniorCitizen":0,
        "Partner":"No","InternetService":"Fiber optic","OnlineSecurity":"No","TechSupport":"No",
        "SubscriptionType":"Premium"}
LOW  = {"customerID":"L1","tenure":60,"MonthlyCharges":25.0,"InactiveDays":2,
        "Contract":"Two year","PaymentMethod":"Bank transfer (automatic)","SeniorCitizen":0,
        "Partner":"Yes","InternetService":"DSL","OnlineSecurity":"Yes","TechSupport":"Yes",
        "SubscriptionType":"Basic"}
MED  = {"customerID":"M1","tenure":18,"MonthlyCharges":55.0,"InactiveDays":28,
        "Contract":"One year","PaymentMethod":"Credit card (automatic)","SeniorCitizen":0,
        "Partner":"No","InternetService":"DSL","OnlineSecurity":"No","TechSupport":"No",
        "SubscriptionType":"Standard"}

# ═══════════════════════════════════════════════════════════════════════════════
class TestRecommendationEngine(unittest.TestCase):
    def setUp(self):
        from services.retention.recommendation_engine import RetentionRecommendationEngine
        self.engine=RetentionRecommendationEngine()
    def test_high_risk_gets_recommendations(self):
        recs=self.engine.recommend(HIGH,0.80); self.assertGreater(len(recs),0)
    def test_low_risk_below_threshold_no_recs(self):
        recs=self.engine.recommend(LOW,0.10); self.assertEqual(len(recs),0)
    def test_medium_risk_may_get_recs(self):
        recs=self.engine.recommend(MED,0.50); self.assertIsInstance(recs,list)
    def test_recs_have_required_fields(self):
        recs=self.engine.recommend(HIGH,0.80)
        for r in recs:
            self.assertIsInstance(r.action,str); self.assertIsInstance(r.priority,int)
            self.assertIsInstance(r.estimated_savings,float)
    def test_inactive_customer_gets_engagement_rec(self):
        cust={**HIGH,"InactiveDays":60}
        recs=self.engine.recommend(cust,0.70)
        actions=" ".join(r.rule_name for r in recs)
        self.assertTrue(any("inactiv" in a or "inactive" in a.lower() for a in [r.rule_name for r in recs]))
    def test_high_charges_gets_discount_rec(self):
        cust={**HIGH,"MonthlyCharges":85,"Contract":"Month-to-month"}
        recs=self.engine.recommend(cust,0.75)
        names=[r.rule_name for r in recs]
        self.assertTrue(any("discount" in n or "charges" in n or "upgrade" in n for n in names))
    def test_month_to_month_gets_contract_rec(self):
        cust={**HIGH,"Contract":"Month-to-month"}
        recs=self.engine.recommend(cust,0.70)
        self.assertTrue(any("contract" in r.rule_name or "upgrade" in r.rule_name for r in recs))
    def test_low_tenure_gets_onboarding_rec(self):
        cust={**HIGH,"tenure":3}
        recs=self.engine.recommend(cust,0.70)
        self.assertTrue(any("tenure" in r.rule_name or "onboard" in r.rule_name for r in recs))
    def test_max_recommendations_respected(self):
        eng=RetentionRecommendationEngine(max_recommendations=3)
        recs=eng.recommend(HIGH,0.80); self.assertLessEqual(len(recs),3)
    def test_priority_ordering(self):
        recs=self.engine.recommend(HIGH,0.80)
        if len(recs)>1:
            for i in range(len(recs)-1):
                self.assertLessEqual(recs[i].priority,recs[i+1].priority)
    def test_to_dict_serializable(self):
        recs=self.engine.recommend(HIGH,0.80)
        import json
        for r in recs:
            d=r.to_dict(); self.assertIsInstance(json.dumps(d),str)
    def test_savings_positive(self):
        recs=self.engine.recommend(HIGH,0.80)
        for r in recs: self.assertGreaterEqual(r.estimated_savings,0)
    def test_batch_recommend(self):
        df=pd.DataFrame([{**HIGH,"churn_probability":0.80,"segment":"Risky"},
                          {**LOW, "churn_probability":0.05,"segment":"Loyal"},
                          {**MED, "churn_probability":0.50,"segment":"Inactive"}])
        out=self.engine.recommend_batch(df)
        self.assertIn("recommendations",out.columns); self.assertIn("top_action",out.columns)

class TestRevenueEstimator(unittest.TestCase):
    def setUp(self):
        from analytics.revenue_estimator import RevenueEstimator, estimate_ltv
        self.est=RevenueEstimator(); self.ltv=estimate_ltv
    def test_ltv_positive(self): self.assertGreater(self.ltv(79.99,24),0)
    def test_ltv_longer_tenure_higher(self):
        self.assertGreater(self.ltv(79.99,36,expected_remaining_months=36),
                           self.ltv(79.99,36,expected_remaining_months=6))
    def test_customer_risk_structure(self):
        r=self.est.estimate_customer(HIGH,0.80)
        self.assertIsNotNone(r.expected_annual_loss); self.assertGreater(r.expected_annual_loss,0)
    def test_expected_loss_less_than_revenue(self):
        r=self.est.estimate_customer(HIGH,0.80)
        annual=HIGH["MonthlyCharges"]*12
        self.assertLessEqual(r.expected_annual_loss,annual*1.05)
    def test_low_prob_low_loss(self):
        r=self.est.estimate_customer(LOW,0.05)
        self.assertLess(r.expected_annual_loss,LOW["MonthlyCharges"]*2)
    def test_portfolio_risk_structure(self):
        df=pd.DataFrame([{**HIGH,"churn_probability":0.80},{**LOW,"churn_probability":0.05},
                          {**MED,"churn_probability":0.45}])
        df.rename(columns={"MonthlyCharges":"MonthlyCharges"},inplace=True)
        p=self.est.estimate_portfolio(df)
        self.assertGreater(p.total_customers,0)
        self.assertGreater(p.total_monthly_revenue,0)
        self.assertGreater(p.expected_annual_loss,0)
    def test_portfolio_counts(self):
        df=pd.DataFrame([{"MonthlyCharges":90,"churn_probability":0.80},
                          {"MonthlyCharges":25,"churn_probability":0.10},
                          {"MonthlyCharges":55,"churn_probability":0.50}])
        p=self.est.estimate_portfolio(df); self.assertEqual(p.total_customers,3)
    def test_to_dict_serializable(self):
        import json
        df=pd.DataFrame([{"MonthlyCharges":50,"churn_probability":0.50}])
        p=self.est.estimate_portfolio(df); s=json.dumps(p.to_dict())
        self.assertIsInstance(s,str)
    def test_missing_monthly_charges_raises(self):
        df=pd.DataFrame([{"churn_probability":0.5}])
        with self.assertRaises(ValueError): self.est.estimate_portfolio(df)

class TestWhatIfSimulator(unittest.TestCase):
    def setUp(self):
        from analytics.whatif_simulator import WhatIfSimulator, PRESET_SCENARIOS
        from backend.services.ml_service import get_model_metadata
        meta=get_model_metadata()
        if not meta.get("best_model"): self.skipTest("No trained model")
        from ml.models.predictor import ChurnPredictor
        self.predictor=ChurnPredictor("best_model").load()
        self.sim=WhatIfSimulator(); self.presets=PRESET_SCENARIOS
    def test_simulate_customer_returns_result(self):
        from analytics.whatif_simulator import ScenarioResult
        r=self.sim.simulate_customer(HIGH,{"Contract":"Two year"},self.predictor)
        self.assertIsInstance(r,ScenarioResult)
    def test_contract_upgrade_reduces_prob(self):
        r=self.sim.simulate_customer(HIGH,{"Contract":"Two year"},self.predictor)
        self.assertLess(r.new_churn_probability,r.base_churn_probability+0.1)
    def test_delta_is_new_minus_base(self):
        r=self.sim.simulate_customer(HIGH,{"Contract":"One year"},self.predictor)
        self.assertAlmostEqual(r.delta_probability,r.new_churn_probability-r.base_churn_probability,places=4)
    def test_preset_upgrade_annual(self):
        r=self.sim.simulate_customer(HIGH,self.presets["upgrade_to_annual_contract"],self.predictor)
        self.assertIsNotNone(r); self.assertLessEqual(r.new_churn_probability,1.0)
    def test_preset_reduce_charges(self):
        r=self.sim.simulate_customer(HIGH,self.presets["reduce_charges_15pct"],self.predictor)
        self.assertIsNotNone(r)
    def test_sensitivity_returns_list(self):
        results=self.sim.sensitivity_analysis(HIGH,"tenure",[1,12,24,36,48,60],self.predictor)
        self.assertIsInstance(results,list); self.assertGreater(len(results),0)
    def test_to_dict_serializable(self):
        import json
        r=self.sim.simulate_customer(HIGH,{"Contract":"Two year"},self.predictor)
        s=json.dumps(r.to_dict()); self.assertIsInstance(s,str)
    def test_monthly_revenue_saved_non_negative(self):
        r=self.sim.simulate_customer(HIGH,{"Contract":"Two year"},self.predictor)
        self.assertGreaterEqual(r.monthly_revenue_saved,0)
    def test_risk_categories_valid(self):
        r=self.sim.simulate_customer(HIGH,{"Contract":"Two year"},self.predictor)
        self.assertIn(r.base_risk_category,("High","Medium","Low"))
        self.assertIn(r.new_risk_category,("High","Medium","Low"))

if __name__=="__main__": unittest.main(verbosity=2)
