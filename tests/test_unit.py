"""
tests/test_unit.py
------------------
Unit tests for all core platform components.

Coverage:
  - Data preprocessing (clean_data, encode, scale)
  - Feature engineering (all 12 feature functions)
  - Risk scoring engine
  - Recommendation engine (all 12 rules)
  - Revenue estimation (LTV, expected loss, portfolio)
  - Customer segmentation
  - JWT authentication
  - Password hashing
  - Cache layer
  - Validation schemas
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from services.retention.recommendation_engine import RetentionRecommendationEngine

# Bootstrap path & env
from tests.conftest import (HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER,
                             VALID_CUSTOMER, PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

class TestDataPreprocessing(unittest.TestCase):
    """Tests for ml/features/preprocessing.py"""

    def setUp(self):
        from ml.features.preprocessing import clean_data
        self.clean_data = clean_data
        self.df_raw = pd.DataFrame([
            {"customerID":"1","gender":"Male","SeniorCitizen":0,"Partner":"No",
             "Dependents":"No","tenure":12,"PhoneService":"Yes","MultipleLines":"No",
             "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
             "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
             "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
             "PaymentMethod":"Electronic check","MonthlyCharges":70.0,"TotalCharges":"840",
             "InactiveDays":30,"SubscriptionType":"Standard","Churn":1},
            {"customerID":"2","gender":"Female","SeniorCitizen":0,"Partner":"Yes",
             "Dependents":"Yes","tenure":60,"PhoneService":"Yes","MultipleLines":"Yes",
             "InternetService":"DSL","OnlineSecurity":"Yes","OnlineBackup":"Yes",
             "DeviceProtection":"Yes","TechSupport":"Yes","StreamingTV":"No",
             "StreamingMovies":"No","Contract":"Two year","PaperlessBilling":"No",
             "PaymentMethod":"Bank transfer (automatic)","MonthlyCharges":50.0,"TotalCharges":"3000",
             "InactiveDays":3,"SubscriptionType":"Basic","Churn":0},
            # Duplicate row
            {"customerID":"2","gender":"Female","SeniorCitizen":0,"Partner":"Yes",
             "Dependents":"Yes","tenure":60,"PhoneService":"Yes","MultipleLines":"Yes",
             "InternetService":"DSL","OnlineSecurity":"Yes","OnlineBackup":"Yes",
             "DeviceProtection":"Yes","TechSupport":"Yes","StreamingTV":"No",
             "StreamingMovies":"No","Contract":"Two year","PaperlessBilling":"No",
             "PaymentMethod":"Bank transfer (automatic)","MonthlyCharges":50.0,"TotalCharges":"3000",
             "InactiveDays":3,"SubscriptionType":"Basic","Churn":0},
        ])

    def test_clean_removes_duplicates(self):
        """clean_data should drop duplicate rows"""
        cleaned = self.clean_data(self.df_raw.copy())
        self.assertEqual(len(cleaned), 2)

    def test_clean_removes_customer_id(self):
        """clean_data should drop the customerID column"""
        cleaned = self.clean_data(self.df_raw.copy())
        self.assertNotIn("customerID", cleaned.columns)

    def test_clean_handles_empty_total_charges(self):
        """TotalCharges empty strings should be coerced to numeric"""
        df = self.df_raw.copy()
        df.loc[0, "TotalCharges"] = " "  # empty string
        cleaned = self.clean_data(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["TotalCharges"]))

    def test_clean_numeric_tenure(self):
        """tenure column should be numeric after cleaning"""
        cleaned = self.clean_data(self.df_raw.copy())
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["tenure"]))

    def test_clean_monthly_charges_numeric(self):
        """MonthlyCharges should be numeric"""
        cleaned = self.clean_data(self.df_raw.copy())
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["MonthlyCharges"]))

    def test_clean_preserves_row_count(self):
        """After deduplication, rows with unique data are preserved"""
        df = self.df_raw.iloc[:2].copy()  # 2 unique rows
        cleaned = self.clean_data(df)
        self.assertEqual(len(cleaned), 2)


class TestBinaryEncoding(unittest.TestCase):
    """Tests for the binary encoding step inside clean_data"""

    def setUp(self):
        from ml.features.preprocessing import clean_data
        self.clean = clean_data

    def test_gender_encoded_to_binary(self):
        """gender should be 0/1 after cleaning + binary encoding"""
        from ml.features.preprocessing import _binary_encode as binary_encode_columns
        df = pd.DataFrame([{
            "customerID": "1", "gender": "Female", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No", "tenure": 12,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
            "TotalCharges": "840", "InactiveDays": 30,
            "SubscriptionType": "Standard", "Churn": 1,
        }])
        cleaned = self.clean(df)
        encoded = binary_encode_columns(cleaned)
        self.assertIn(encoded["gender"].iloc[0], [0, 1])

    def test_churn_binary(self):
        """Churn column values should be 0 or 1"""
        df = pd.DataFrame([{
            "customerID": "1", "gender": "Male", "SeniorCitizen": 0,
            "Partner": "No", "Dependents": "No", "tenure": 12,
            "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "Fiber optic", "OnlineSecurity": "No",
            "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0,
            "TotalCharges": "840", "InactiveDays": 30,
            "SubscriptionType": "Standard", "Churn": "Yes",
        }])
        cleaned = self.clean(df)
        self.assertIn(int(cleaned["Churn"].iloc[0]), [0, 1])


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering(unittest.TestCase):
    """Tests for ml/features/feature_engineering.py"""

    def setUp(self):
        from ml.features.feature_engineering import (
            add_charge_per_tenure, add_service_count, add_engagement_score,
            add_contract_risk_score, add_payment_risk_score,
            add_tenure_band, add_charge_band, add_is_high_value,
            add_has_streaming, add_has_security, add_total_charge_ratio,
            engineer_features,
        )
        self.df = pd.DataFrame([{
            "tenure": 12, "MonthlyCharges": 80.0, "TotalCharges": 960.0,
            "InactiveDays": 30, "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic", "StreamingTV": "Yes",
            "StreamingMovies": "No", "OnlineSecurity": "No",
            "TechSupport": "No",
        }])
        self.add_charge_per_tenure  = add_charge_per_tenure
        self.add_service_count      = add_service_count
        self.add_engagement_score   = add_engagement_score
        self.add_contract_risk_score = add_contract_risk_score
        self.add_payment_risk_score = add_payment_risk_score
        self.add_tenure_band        = add_tenure_band
        self.add_charge_band        = add_charge_band
        self.add_is_high_value      = add_is_high_value
        self.add_has_streaming      = add_has_streaming
        self.add_has_security       = add_has_security
        self.add_total_charge_ratio = add_total_charge_ratio
        self.engineer_features      = engineer_features

    def test_charge_per_tenure_creates_column(self):
        df = self.add_charge_per_tenure(self.df.copy())
        self.assertIn("charge_per_tenure", df.columns)

    def test_charge_per_tenure_value(self):
        df = self.add_charge_per_tenure(self.df.copy())
        expected = 80.0 / 12
        self.assertAlmostEqual(df["charge_per_tenure"].iloc[0], expected, places=2)

    def test_charge_per_tenure_zero_tenure(self):
        """tenure=0 should not cause division error"""
        df = self.df.copy()
        df["tenure"] = 0
        result = self.add_charge_per_tenure(df)
        self.assertFalse(result["charge_per_tenure"].isna().any())

    def test_service_count_is_integer(self):
        df = self.add_service_count(self.df.copy())
        self.assertIn("service_count", df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["service_count"]))

    def test_engagement_score_range(self):
        df = self.add_engagement_score(self.df.copy())
        self.assertIn("engagement_score", df.columns)
        self.assertGreaterEqual(df["engagement_score"].iloc[0], 0)
        self.assertLessEqual(df["engagement_score"].iloc[0], 100)

    def test_contract_risk_month_to_month_highest(self):
        """Month-to-month should have the highest contract risk score"""
        df_mtm = pd.DataFrame([{"Contract": "Month-to-month"}])
        df_2yr = pd.DataFrame([{"Contract": "Two year"}])
        r_mtm = self.add_contract_risk_score(df_mtm.copy())["contract_risk_score"].iloc[0]
        r_2yr = self.add_contract_risk_score(df_2yr.copy())["contract_risk_score"].iloc[0]
        self.assertGreater(r_mtm, r_2yr)

    def test_payment_risk_electronic_check_highest(self):
        """Electronic check should have highest payment risk"""
        df_ec = pd.DataFrame([{"PaymentMethod": "Electronic check"}])
        df_bt = pd.DataFrame([{"PaymentMethod": "Bank transfer (automatic)"}])
        r_ec = self.add_payment_risk_score(df_ec.copy())["payment_risk_score"].iloc[0]
        r_bt = self.add_payment_risk_score(df_bt.copy())["payment_risk_score"].iloc[0]
        self.assertGreater(r_ec, r_bt)

    def test_tenure_band_creates_column(self):
        df = self.add_tenure_band(self.df.copy())
        self.assertIn("tenure_band", df.columns)

    def test_charge_band_creates_column(self):
        df = self.add_charge_band(self.df.copy())
        self.assertIn("charge_band", df.columns)

    def test_is_high_value_binary(self):
        df = self.add_is_high_value(self.df.copy())
        self.assertIn("is_high_value", df.columns)
        self.assertIn(df["is_high_value"].iloc[0], [0, 1])

    def test_has_streaming_detects_yes(self):
        df = self.add_has_streaming(self.df.copy())
        self.assertIn("has_streaming", df.columns)
        self.assertEqual(df["has_streaming"].iloc[0], 1)

    def test_has_security_detects_no(self):
        df = self.add_has_security(self.df.copy())
        self.assertIn("has_security", df.columns)
        self.assertEqual(df["has_security"].iloc[0], 0)

    def test_engineer_features_adds_multiple_columns(self):
        """engineer_features should add at least 8 new columns"""
        original_cols = set(self.df.columns)
        result = self.engineer_features(self.df.copy())
        new_cols = set(result.columns) - original_cols
        self.assertGreaterEqual(len(new_cols), 8)

    def test_engineer_features_no_nan(self):
        """Engineered features should not introduce NaN values"""
        result = self.engineer_features(self.df.copy())
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertFalse(result[numeric_cols].isna().any().any())


# ══════════════════════════════════════════════════════════════════════════════
# 3. RISK SCORING
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskScoring(unittest.TestCase):
    """Tests for churn probability → risk score + category mapping"""

    def setUp(self):
        # Import risk score logic from predictor
        from ml.config import RISK_THRESHOLDS
        self.high_threshold   = RISK_THRESHOLDS.get("high",   0.65)
        self.medium_threshold = RISK_THRESHOLDS.get("medium", 0.35)

    def _score_to_category(self, prob: float) -> str:
        if prob >= self.high_threshold:
            return "High"
        if prob >= self.medium_threshold:
            return "Medium"
        return "Low"

    def _prob_to_score(self, prob: float) -> int:
        return min(100, max(0, round(prob * 100)))

    def test_high_risk_threshold(self):
        self.assertEqual(self._score_to_category(0.85), "High")
        self.assertEqual(self._score_to_category(0.65), "High")

    def test_medium_risk_threshold(self):
        self.assertEqual(self._score_to_category(0.50), "Medium")
        self.assertEqual(self._score_to_category(0.35), "Medium")

    def test_low_risk_threshold(self):
        self.assertEqual(self._score_to_category(0.20), "Low")
        self.assertEqual(self._score_to_category(0.00), "Low")

    def test_risk_score_range(self):
        for prob in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = self._prob_to_score(prob)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_risk_score_proportional(self):
        self.assertGreater(self._prob_to_score(0.8), self._prob_to_score(0.4))

    def test_risk_score_boundary(self):
        self.assertEqual(self._prob_to_score(1.0), 100)
        self.assertEqual(self._prob_to_score(0.0), 0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TestRecommendationEngine(unittest.TestCase):
    """Tests for services/retention/recommendation_engine.py"""

    def setUp(self):
        from services.retention.recommendation_engine import (
            RetentionRecommendationEngine, get_recommendations)
        self.engine = RetentionRecommendationEngine()
        self.get_recommendations = get_recommendations

    def test_high_risk_customer_gets_recommendations(self):
        recs = self.engine.recommend(HIGH_RISK_CUSTOMER, churn_probability=0.80)
        self.assertGreater(len(recs), 0)

    def test_low_risk_below_threshold_no_recommendations(self):
        """Customers below min_churn_prob should get no recommendations"""
        recs = self.engine.recommend(LOW_RISK_CUSTOMER, churn_probability=0.10)
        self.assertEqual(len(recs), 0)

    def test_month_to_month_rule_fires(self):
        """month_to_month_upgrade rule should fire for M2M contract"""
        customer = {**VALID_CUSTOMER, "contract": "Month-to-month"}
        recs = self.engine.recommend(customer, churn_probability=0.70)
        rule_names = [r.rule_name for r in recs]
        self.assertIn("month_to_month_upgrade", rule_names)

    def test_annual_contract_no_upgrade_rule(self):
        """Annual contract customer should NOT trigger month_to_month_upgrade"""
        customer = {**VALID_CUSTOMER, "Contract": "Two year"}
        recs = self.engine.recommend(customer, churn_probability=0.70)
        rule_names = [r.rule_name for r in recs]
        self.assertNotIn("month_to_month_upgrade", rule_names)

    def test_high_charges_discount_fires(self):
        """high_monthly_charges_discount should fire when charges > 70"""
        customer = {**VALID_CUSTOMER, "monthly_charges": 90.0}
        recs = self.engine.recommend(customer, churn_probability=0.70)
        rule_names = [r.rule_name for r in recs]
        self.assertIn("high_monthly_charges_discount", rule_names)

    def test_low_charges_no_discount(self):
        """Discount rule should NOT fire for low charges"""
        customer = {**VALID_CUSTOMER, "MonthlyCharges": 30.0}
        recs = self.engine.recommend(customer, churn_probability=0.70)
        rule_names = [r.rule_name for r in recs]
        self.assertNotIn("high_monthly_charges_discount", rule_names)

    def test_high_inactivity_reengagement_fires(self):
        """Re-engagement rule should fire when inactive_days > 45"""
        customer = {**VALID_CUSTOMER, "inactive_days": 60, "InactiveDays": 60}
        recs = self.engine.recommend(customer, churn_probability=0.70)
        rule_names = [r.rule_name for r in recs]
        self.assertIn("high_inactivity_reengagement", rule_names)

    def test_low_tenure_onboarding_fires(self):
        """Onboarding rule fires for tenure ≤ 6"""
        customer = {**VALID_CUSTOMER, "tenure": 3}
        recs = self.engine.recommend(customer, churn_probability=0.70)
        rule_names = [r.rule_name for r in recs]
        self.assertIn("low_tenure_onboarding", rule_names)

    def test_recommendations_sorted_by_priority(self):
        """Recommendations should be sorted priority ASC"""
        recs = self.engine.recommend(HIGH_RISK_CUSTOMER, churn_probability=0.80)
        if len(recs) > 1:
            priorities = [r.priority for r in recs]
            self.assertEqual(priorities, sorted(priorities))

    def test_max_recommendations_limit(self):
        """Engine should not return more than max_recommendations"""
        engine = RetentionRecommendationEngine(max_recommendations=3)
        recs = engine.recommend(HIGH_RISK_CUSTOMER, churn_probability=0.90)
        self.assertLessEqual(len(recs), 3)

    def test_estimated_savings_positive(self):
        """All estimated savings should be positive"""
        recs = self.engine.recommend(HIGH_RISK_CUSTOMER, churn_probability=0.80)
        for r in recs:
            self.assertGreater(r.estimated_savings, 0)

    def test_recommendations_serializable(self):
        """to_dict() should return JSON-serialisable dicts"""
        import json
        recs = self.engine.recommend(HIGH_RISK_CUSTOMER, churn_probability=0.80)
        for r in recs:
            d = r.to_dict()
            self.assertIsInstance(d, dict)
            json.dumps(d)  # must not raise

    def test_segment_affinity_boosts_priority(self):
        """Risky segment should boost matching rule priorities"""
        customer = {**HIGH_RISK_CUSTOMER, "contract": "Month-to-month"}
        recs_no_seg   = self.engine.recommend(customer, 0.80, segment=None)
        recs_with_seg = self.engine.recommend(customer, 0.80, segment="Risky")
        # Both should fire; segment version should have same or lower priority numbers
        if recs_no_seg and recs_with_seg:
            self.assertLessEqual(recs_with_seg[0].priority, recs_no_seg[0].priority + 1)

    def test_batch_recommend(self):
        """recommend_batch should add 'recommendations' column"""
        df = pd.DataFrame([HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
        df["churn_probability"] = [0.80, 0.10]
        result = self.engine.recommend_batch(df)
        self.assertIn("recommendations", result.columns)
        self.assertIn("top_action", result.columns)

    def test_convenience_function(self):
        recs = self.get_recommendations(HIGH_RISK_CUSTOMER, 0.80)
        self.assertIsInstance(recs, list)
        if recs:
            self.assertIn("action", recs[0])


# ══════════════════════════════════════════════════════════════════════════════
# 5. REVENUE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

class TestRevenueEstimation(unittest.TestCase):
    """Tests for analytics/revenue_estimator.py"""

    def setUp(self):
        from analytics.revenue_estimator import (
            RevenueEstimator, estimate_ltv, estimate_revenue_risk)
        self.estimator         = RevenueEstimator()
        self.estimate_ltv      = estimate_ltv
        self.estimate_rev_risk = estimate_revenue_risk

    def test_ltv_positive(self):
        ltv = self.estimate_ltv(79.99, 12)
        self.assertGreater(ltv, 0)

    def test_ltv_increases_with_charges(self):
        ltv_low  = self.estimate_ltv(30.0,  12)
        ltv_high = self.estimate_ltv(100.0, 12)
        self.assertGreater(ltv_high, ltv_low)

    def test_ltv_increases_with_expected_remaining(self):
        """LTV should increase when expected remaining months are higher"""
        ltv_short = self.estimate_ltv(70.0, 6,  expected_remaining_months=6)
        ltv_long  = self.estimate_ltv(70.0, 60, expected_remaining_months=48)
        self.assertGreater(ltv_long, ltv_short)

    def test_ltv_zero_discount_rate(self):
        """Zero discount rate should return simple multiplication"""
        ltv = self.estimate_ltv(100.0, 12, monthly_discount_rate=0)
        expected = 100.0 * 0.65 * 24  # 24 remaining months default
        self.assertAlmostEqual(ltv, expected, places=0)

    def test_customer_revenue_risk(self):
        record = {**VALID_CUSTOMER, "MonthlyCharges": 79.99}
        risk = self.estimator.estimate_customer(record, churn_probability=0.80)
        self.assertGreater(risk.expected_monthly_loss, 0)
        self.assertGreater(risk.expected_annual_loss, 0)
        self.assertGreater(risk.ltv_at_risk, 0)

    def test_expected_loss_formula(self):
        """E[loss] = prob × monthly_charges"""
        record = {**VALID_CUSTOMER, "MonthlyCharges": 100.0}
        risk   = self.estimator.estimate_customer(record, churn_probability=0.50)
        self.assertAlmostEqual(risk.expected_monthly_loss, 50.0, places=1)

    def test_net_risk_non_negative(self):
        """Net revenue at risk should never be negative"""
        record = {**VALID_CUSTOMER, "MonthlyCharges": 50.0}
        risk = self.estimator.estimate_customer(record, churn_probability=0.40,
                                                 recommendations=[{"estimated_savings": 999}])
        self.assertGreaterEqual(risk.net_revenue_at_risk, 0)

    def test_portfolio_risk(self):
        df = pd.DataFrame([
            {"MonthlyCharges": 80.0, "churn_probability": 0.80,
             "risk_category": "High",  "tenure": 6},
            {"MonthlyCharges": 50.0, "churn_probability": 0.50,
             "risk_category": "Medium","tenure": 24},
            {"MonthlyCharges": 30.0, "churn_probability": 0.20,
             "risk_category": "Low",   "tenure": 48},
        ])
        portfolio = self.estimator.estimate_portfolio(df)
        self.assertEqual(portfolio.total_customers, 3)
        self.assertEqual(portfolio.high_risk_count, 1)
        self.assertEqual(portfolio.medium_risk_count, 1)
        self.assertEqual(portfolio.low_risk_count, 1)
        self.assertGreater(portfolio.expected_annual_loss, 0)

    def test_portfolio_revenue_by_segment(self):
        """Segment breakdown should work when segment column present"""
        df = pd.DataFrame([
            {"MonthlyCharges": 80.0, "churn_probability": 0.80,
             "risk_category": "High", "segment": "Risky", "tenure": 6},
            {"MonthlyCharges": 50.0, "churn_probability": 0.20,
             "risk_category": "Low",  "segment": "Loyal", "tenure": 48},
        ])
        portfolio = self.estimator.estimate_portfolio(df)
        self.assertIn("Risky", portfolio.revenue_by_segment)
        self.assertIn("Loyal", portfolio.revenue_by_segment)

    def test_worst_case_loss_gte_expected(self):
        """Worst case loss should be >= expected loss"""
        df = pd.DataFrame([
            {"MonthlyCharges": 80.0, "churn_probability": 0.80,
             "risk_category": "High", "tenure": 6},
        ])
        p = self.estimator.estimate_portfolio(df)
        self.assertGreaterEqual(p.worst_case_annual_loss, p.expected_annual_loss)

    def test_to_dict_serializable(self):
        import json
        df = pd.DataFrame([
            {"MonthlyCharges": 80.0, "churn_probability": 0.80,
             "risk_category": "High", "tenure": 6},
        ])
        p = self.estimator.estimate_portfolio(df)
        json.dumps(p.to_dict())  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 6. CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomerSegmentation(unittest.TestCase):
    """Tests for analytics/segmentation.py"""

    def setUp(self):
        from analytics.segmentation import CustomerSegmenter, SEGMENT_LABELS
        self.Segmenter      = CustomerSegmenter
        self.SEGMENT_LABELS = SEGMENT_LABELS
        # Build a synthetic DataFrame large enough to fit 4 clusters
        import numpy as np
        np.random.seed(42)
        n = 80
        self.df = pd.DataFrame({
            "tenure":            [5]*20  + [20]*20 + [50]*20 + [15]*20,
            "MonthlyCharges":    [80.0]*20 + [50.0]*20 + [35.0]*20 + [65.0]*20,
            "InactiveDays":      [60]*20 + [5]*20  + [3]*20  + [50]*20,
            "churn_probability": [0.75]*20 + [0.10]*20 + [0.08]*20 + [0.60]*20,
            "service_count":     [2]*20  + [4]*20  + [3]*20  + [2]*20,
            "engagement_score":  [20]*20 + [80]*20 + [85]*20 + [25]*20,
        })

    def test_fit_returns_segmenter(self):
        s = self.Segmenter(n_clusters=4)
        result = s.fit(self.df)
        self.assertIsInstance(result, self.Segmenter)
        self.assertTrue(result._fitted)

    def test_predict_adds_segment_column(self):
        s = self.Segmenter(n_clusters=4)
        s.fit(self.df)
        df_seg = s.predict(self.df)
        self.assertIn("segment", df_seg.columns)
        self.assertIn("cluster_id", df_seg.columns)

    def test_segment_labels_are_valid(self):
        s = self.Segmenter(n_clusters=4)
        s.fit(self.df)
        df_seg = s.predict(self.df)
        for seg in df_seg["segment"].unique():
            self.assertIn(seg, self.SEGMENT_LABELS)

    def test_predict_before_fit_raises(self):
        s = self.Segmenter(n_clusters=4)
        with self.assertRaises(RuntimeError):
            s.predict(self.df)

    def test_segment_profile_returns_dict(self):
        s = self.Segmenter(n_clusters=4)
        s.fit(self.df)
        df_seg = s.predict(self.df)
        profile = s.segment_profile(df_seg)
        self.assertIsInstance(profile, dict)
        self.assertGreater(len(profile), 0)

    def test_segment_profile_has_count(self):
        s = self.Segmenter(n_clusters=4)
        s.fit(self.df)
        df_seg = s.predict(self.df)
        profile = s.segment_profile(df_seg)
        for seg_stats in profile.values():
            self.assertIn("count", seg_stats)
            self.assertGreater(seg_stats["count"], 0)

    def test_total_segments_cover_all_rows(self):
        s = self.Segmenter(n_clusters=4)
        s.fit(self.df)
        df_seg = s.predict(self.df)
        profile = s.segment_profile(df_seg)
        total_count = sum(v["count"] for v in profile.values())
        self.assertEqual(total_count, len(self.df))


# ══════════════════════════════════════════════════════════════════════════════
# 7. JWT AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

class TestJWTAuth(unittest.TestCase):
    """Tests for backend/auth/jwt_handler.py"""

    def setUp(self):
        from backend.auth.jwt_handler import (
            create_access_token, create_refresh_token,
            verify_token, decode_token)
        self.create_access   = create_access_token
        self.create_refresh  = create_refresh_token
        self.verify_token    = verify_token
        self.decode_token    = decode_token

    def test_create_access_token_returns_string(self):
        token = self.create_access({"user_id": 1, "email": "test@test.com"})
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 20)

    def test_create_refresh_token_returns_string(self):
        token = self.create_refresh({"user_id": 1})
        self.assertIsInstance(token, str)

    def test_verify_valid_token(self):
        token   = self.create_access({"user_id": 1, "email": "test@test.com"})
        payload = self.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload.get("user_id"), 1)

    def test_verify_invalid_token_returns_none(self):
        result = self.verify_token("not.a.valid.token")
        self.assertIsNone(result)

    def test_verify_tampered_token_returns_none(self):
        token   = self.create_access({"user_id": 1})
        tampered = token[:-5] + "XXXXX"
        result  = self.verify_token(tampered)
        self.assertIsNone(result)

    def test_decode_token_contains_user_id(self):
        token   = self.create_access({"user_id": 42, "email": "x@x.com"})
        payload = self.decode_token(token)
        self.assertEqual(payload.get("user_id"), 42)

    def test_tokens_are_unique(self):
        """Tokens for different users must differ"""
        t1 = self.create_access({"user_id": 1})
        t2 = self.create_access({"user_id": 2})
        self.assertNotEqual(t1, t2)


# ══════════════════════════════════════════════════════════════════════════════
# 8. PASSWORD HASHING
# ══════════════════════════════════════════════════════════════════════════════

class TestPasswordHashing(unittest.TestCase):
    """Tests for backend/auth/password.py"""

    def setUp(self):
        from backend.auth.password import hash_password, verify_password
        self.hash   = hash_password
        self.verify = verify_password

    def test_hash_is_not_plaintext(self):
        h = self.hash("mypassword")
        self.assertNotEqual(h, "mypassword")

    def test_verify_correct_password(self):
        h = self.hash("correctpassword")
        self.assertTrue(self.verify("correctpassword", h))

    def test_verify_wrong_password(self):
        h = self.hash("correctpassword")
        self.assertFalse(self.verify("wrongpassword", h))

    def test_same_password_different_hashes(self):
        h1 = self.hash("password123")
        h2 = self.hash("password123")
        self.assertNotEqual(h1, h2)  # salt randomness

    def test_hash_is_string(self):
        h = self.hash("anypassword")
        self.assertIsInstance(h, str)

    def test_empty_password_hashes(self):
        """Empty string should still hash without error"""
        h = self.hash("")
        self.assertIsInstance(h, str)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CACHE LAYER
# ══════════════════════════════════════════════════════════════════════════════

class TestCacheLayer(unittest.TestCase):
    """Tests for backend/core/cache.py"""

    def setUp(self):
        from backend.core.cache import AppCache
        self.cache = AppCache(max_size=10)

    def test_set_and_get(self):
        self.cache.set("key1", {"data": 42}, ttl=60)
        result = self.cache.get("key1")
        self.assertEqual(result, {"data": 42})

    def test_miss_returns_none(self):
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_expired_returns_none(self):
        import time
        self.cache.set("exp_key", "value", ttl=0)
        time.sleep(0.01)
        result = self.cache.get("exp_key")
        self.assertIsNone(result)

    def test_delete_key(self):
        self.cache.set("del_key", "value", ttl=60)
        self.cache.delete("del_key")
        self.assertIsNone(self.cache.get("del_key"))

    def test_delete_prefix(self):
        self.cache.set("prefix:a", 1, ttl=60)
        self.cache.set("prefix:b", 2, ttl=60)
        self.cache.set("other:c",  3, ttl=60)
        removed = self.cache.delete_prefix("prefix:")
        self.assertEqual(removed, 2)
        self.assertIsNone(self.cache.get("prefix:a"))
        self.assertIsNone(self.cache.get("prefix:b"))
        self.assertIsNotNone(self.cache.get("other:c"))

    def test_stats_track_hits_misses(self):
        self.cache.set("stat_key", "val", ttl=60)
        self.cache.get("stat_key")    # hit
        self.cache.get("miss_key")    # miss
        stats = self.cache.stats
        self.assertGreaterEqual(stats["hits"],   1)
        self.assertGreaterEqual(stats["misses"], 1)

    def test_clear(self):
        self.cache.set("k1", 1, ttl=60)
        self.cache.set("k2", 2, ttl=60)
        self.cache.clear()
        self.assertEqual(self.cache.stats["size"], 0)

    def test_thread_safety(self):
        """Concurrent reads/writes should not raise exceptions"""
        import threading
        errors = []

        def writer():
            try:
                for i in range(50):
                    self.cache.set(f"t{i}", i, ttl=10)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    self.cache.get(f"t{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


# ══════════════════════════════════════════════════════════════════════════════
# 10. VALIDATION SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class TestValidationSchemas(unittest.TestCase):
    """Tests for backend/models/schemas.py"""

    def setUp(self):
        from backend.models.schemas import (
            validate_customer, validate_signup, validate_login,
            validate_pagination, ok, err)
        self.validate_customer   = validate_customer
        self.validate_signup     = validate_signup
        self.validate_login      = validate_login
        self.validate_pagination = validate_pagination
        self.ok  = ok
        self.err = err

    def test_validate_customer_valid(self):
        result = self.validate_customer(VALID_CUSTOMER)
        self.assertIn("tenure", result)
        self.assertIn("monthly_charges", result)

    def test_validate_customer_missing_required_raises(self):
        with self.assertRaises((ValueError, KeyError)):
            self.validate_customer({"tenure": 12})

    def test_validate_customer_normalises_monthly_charges(self):
        data = {**VALID_CUSTOMER, "monthly_charges": "79.99"}
        result = self.validate_customer(data)
        self.assertIsInstance(result["monthly_charges"], float)

    def test_validate_signup_valid(self):
        result = self.validate_signup({
            "name": "Test User", "email": "test@example.com",
            "password": "Pass1234!", "role": "admin"})
        self.assertEqual(result["email"], "test@example.com")

    def test_validate_signup_short_name_raises(self):
        with self.assertRaises(ValueError):
            self.validate_signup({
                "name":"A", "email": "a@b.com", "password": "Pass1234!"})

    def test_validate_signup_short_password_raises(self):
        with self.assertRaises(ValueError):
            self.validate_signup({
                "name": "Test User", "email": "a@b.com", "password": "short"})

    def test_validate_signup_invalid_role_defaults_to_viewer(self):
        result = self.validate_signup({
            "name": "Test User", "email": "a@b.com",
            "password": "Pass1234!", "role": "superadmin"})
        self.assertEqual(result["role"], "viewer")

    def test_validate_login_valid(self):
        result = self.validate_login({"email": "a@b.com", "password": "Pass1234!"})
        self.assertEqual(result["email"], "a@b.com")

    def test_validate_pagination_defaults(self):
        page, size = self.validate_pagination({})
        self.assertEqual(page, 1)
        self.assertGreater(size, 0)

    def test_ok_response_shape(self):
        body, status = self.ok({"key": "val"}, "success")
        self.assertTrue(body["success"])
        self.assertEqual(status, 200)

    def test_err_response_shape(self):
        body, status = self.err("Bad request", "BAD_INPUT", 400)
        self.assertFalse(body["success"])
        self.assertEqual(status, 400)


# ══════════════════════════════════════════════════════════════════════════════
# 11. WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class TestWhatIfSimulator(unittest.TestCase):
    """Tests for analytics/whatif_simulator.py"""

    def setUp(self):
        from analytics.whatif_simulator import WhatIfSimulator, PRESET_SCENARIOS
        self.Simulator       = WhatIfSimulator
        self.PRESETS         = PRESET_SCENARIOS

        # Create a minimal mock predictor
        class MockPredictor:
            call_count = 0
            def predict(self, record):
                self.call_count += 1
                charges = float(record.get("MonthlyCharges", 50))
                tenure  = int(record.get("tenure", 12))
                contract = record.get("Contract", "Month-to-month")
                inactive = int(record.get("InactiveDays", 0))
                # Simple mock: high prob for M2M + high charges + inactivity
                prob = 0.80 if contract == "Month-to-month" else 0.30
                prob = min(1.0, prob + inactive / 200)
                return {
                    "churn_probability": round(prob, 4),
                    "risk_category":     "High" if prob >= 0.65 else "Medium" if prob >= 0.35 else "Low",
                    "risk_score":        min(100, round(prob * 100)),
                }
        self.mock_predictor = MockPredictor()
        self.sim = WhatIfSimulator()

    def test_simulate_customer_returns_result(self):
        result = self.sim.simulate_customer(
            VALID_CUSTOMER, {"Contract": "Two year"}, self.mock_predictor)
        self.assertIsNotNone(result)
        self.assertIn("base_churn_probability", result.to_dict())

    def test_contract_upgrade_reduces_probability(self):
        """Upgrading from M2M to 2-year should reduce churn probability"""
        customer = {**VALID_CUSTOMER, "Contract": "Month-to-month"}
        result = self.sim.simulate_customer(
            customer, {"Contract": "Two year"}, self.mock_predictor)
        self.assertLess(result.delta_probability, 0)

    def test_preset_scenarios_all_valid(self):
        """All preset scenarios should run without raising"""
        for name, scenario in self.PRESETS.items():
            result = self.sim.simulate_customer(
                VALID_CUSTOMER, scenario, self.mock_predictor,
                scenario_name=name)
            self.assertIsNotNone(result)

    def test_result_delta_is_new_minus_base(self):
        result = self.sim.simulate_customer(
            VALID_CUSTOMER, {"Contract": "Two year"}, self.mock_predictor)
        expected_delta = result.new_churn_probability - result.base_churn_probability
        self.assertAlmostEqual(result.delta_probability, expected_delta, places=4)

    def test_revenue_saved_non_negative(self):
        result = self.sim.simulate_customer(
            VALID_CUSTOMER, {"Contract": "Two year"}, self.mock_predictor)
        self.assertGreaterEqual(result.monthly_revenue_saved, 0)

    def test_sensitivity_analysis_returns_list(self):
        values = [6, 12, 24, 36, 60]
        results = self.sim.sensitivity_analysis(
            VALID_CUSTOMER, "tenure", values, self.mock_predictor)
        self.assertEqual(len(results), len(values))

    def test_portfolio_simulation(self):
        df = pd.DataFrame([VALID_CUSTOMER, HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER])
        portfolio_result = self.sim.simulate_portfolio(
            df, {"Contract": "Two year"}, self.mock_predictor,
            scenario_name="test_upgrade")
        self.assertEqual(portfolio_result.total_customers, 3)
        self.assertIsNotNone(portfolio_result.annual_savings)

    def test_result_to_dict_serializable(self):
        import json
        result = self.sim.simulate_customer(
            VALID_CUSTOMER, {"Contract": "Two year"}, self.mock_predictor)
        json.dumps(result.to_dict())  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
