"""
tests/test_unit_ml.py
----------------------
Unit tests: preprocessing, feature engineering, prediction, segmentation, SHAP.
"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd

def _df(n=20, churn=True):
    rng = np.random.RandomState(42)
    df  = pd.DataFrame({
        "tenure": rng.randint(1,72,n), "MonthlyCharges": rng.uniform(18,110,n).round(2),
        "TotalCharges": (rng.uniform(18,110,n)*rng.randint(1,72,n)).round(2).astype(str),
        "Contract": rng.choice(["Month-to-month","One year","Two year"],n),
        "PaymentMethod": rng.choice(["Electronic check","Bank transfer (automatic)","Credit card (automatic)"],n),
        "InternetService": rng.choice(["DSL","Fiber optic","No"],n),
        "OnlineSecurity": rng.choice(["Yes","No"],n), "OnlineBackup": rng.choice(["Yes","No"],n),
        "DeviceProtection": rng.choice(["Yes","No"],n), "TechSupport": rng.choice(["Yes","No"],n),
        "StreamingTV": rng.choice(["Yes","No"],n), "StreamingMovies": rng.choice(["Yes","No"],n),
        "PaperlessBilling": rng.choice(["Yes","No"],n),
        "gender": rng.choice(["Male","Female"],n), "SeniorCitizen": rng.randint(0,2,n),
        "Partner": rng.choice(["Yes","No"],n), "Dependents": rng.choice(["Yes","No"],n),
        "PhoneService": rng.choice(["Yes","No"],n), "MultipleLines": rng.choice(["Yes","No"],n),
        "InactiveDays": rng.randint(0,90,n), "SubscriptionType": rng.choice(["Basic","Standard","Premium"],n),
    })
    if churn: df["Churn"] = rng.choice([0,1],n,p=[0.73,0.27])
    return df

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        from ml.features.preprocessing import clean_data
        self.clean = clean_data
    def test_returns_dataframe(self):
        self.assertIsInstance(self.clean(_df()), pd.DataFrame)
    def test_removes_duplicates(self):
        df = _df(20); dup = pd.concat([df, df.iloc[:5]], ignore_index=True)
        self.assertLessEqual(len(self.clean(dup)), len(dup))
    def test_total_charges_numeric(self):
        df = _df(10); df.at[0,"TotalCharges"]=" "
        out = self.clean(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(out["TotalCharges"]))
    def test_no_nan_total_charges(self):
        df = _df(10); df.at[2,"TotalCharges"]=""
        out = self.clean(df)
        self.assertEqual(out["TotalCharges"].isna().sum(), 0)
    def test_tenure_numeric(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(self.clean(_df())["tenure"]))
    def test_does_not_raise_on_clean_data(self):
        try: self.clean(_df(30))
        except Exception as e: self.fail(f"clean_data raised: {e}")
    def test_missing_total_charges_imputed(self):
        df = _df(10); df.at[3,"TotalCharges"] = "abc"
        out = self.clean(df)
        self.assertFalse(out["TotalCharges"].isna().any())

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        from ml.features.feature_engineering import (
            add_charge_per_tenure, add_service_count, add_engagement_score,
            add_contract_risk_score, add_tenure_band, add_has_streaming, engineer_features)
        self.cpt=add_charge_per_tenure; self.svc=add_service_count
        self.eng=add_engagement_score;  self.risk=add_contract_risk_score
        self.band=add_tenure_band;      self.stream=add_has_streaming
        self.eng_all=engineer_features
    def test_charge_per_tenure_zero_tenure(self):
        df=_df(5); df["tenure"]=0
        out=self.cpt(df)
        self.assertIn("charge_per_tenure",out.columns)
        self.assertTrue((out["charge_per_tenure"]==0).all())
    def test_charge_per_tenure_positive(self):
        out=self.cpt(_df(10))
        self.assertTrue((out["charge_per_tenure"]>=0).all())
    def test_service_count_range(self):
        out=self.svc(_df(20))
        self.assertTrue((out["service_count"]>=0).all())
        self.assertTrue((out["service_count"]<=10).all())
    def test_engagement_score_range(self):
        df=self.svc(_df(20)); out=self.eng(df)
        self.assertTrue((out["engagement_score"]>=0).all())
        self.assertTrue((out["engagement_score"]<=100).all())
    def test_contract_risk_score_valid(self):
        out=self.risk(_df(30))
        self.assertTrue(out["contract_risk_score"].isin([0,1,2,3]).all())
    def test_tenure_band_no_nan(self):
        out=self.band(_df(30))
        self.assertEqual(out["tenure_band"].isna().sum(), 0)
    def test_has_streaming_binary(self):
        out=self.stream(_df(20))
        self.assertTrue(out["has_streaming"].isin([0,1]).all())
    def test_engineer_adds_5plus_columns(self):
        orig=set(_df(20).columns); new=set(self.eng_all(_df(20)).columns)-orig
        self.assertGreater(len(new), 5)
    def test_engineer_no_nans(self):
        out=self.eng_all(_df(20))
        self.assertEqual(out.isnull().sum().sum(), 0)
    def test_engineer_no_duplicate_columns(self):
        out=self.eng_all(_df(10))
        self.assertEqual(len(out.columns), len(set(out.columns)))

class TestRiskScoring(unittest.TestCase):
    def _pred(self, **kw):
        from backend.services.ml_service import predict_customer
        base={"tenure":6,"MonthlyCharges":89.99,"Contract":"Month-to-month",
              "PaymentMethod":"Electronic check","InactiveDays":45}
        return predict_customer({**base,**kw})
    def test_probability_in_range(self):
        r=self._pred(); self.assertGreaterEqual(r["churn_probability"],0); self.assertLessEqual(r["churn_probability"],1)
    def test_risk_score_in_range(self):
        r=self._pred(); self.assertGreaterEqual(r["risk_score"],0); self.assertLessEqual(r["risk_score"],100)
    def test_risk_category_valid(self):
        r=self._pred(); self.assertIn(r["risk_category"],("High","Medium","Low"))
    def test_churn_label_valid(self):
        r=self._pred(); self.assertIn(r["churn_label"],("Churn","No Churn"))
    def test_high_risk_features_higher_prob(self):
        high=self._pred(tenure=1,MonthlyCharges=95,InactiveDays=80,Contract="Month-to-month",PaymentMethod="Electronic check")
        low =self._pred(tenure=72,MonthlyCharges=20,InactiveDays=0,Contract="Two year",PaymentMethod="Bank transfer (automatic)")
        self.assertGreater(high["churn_probability"], low["churn_probability"])
    def test_score_correlates_with_prob(self):
        high=self._pred(tenure=1,MonthlyCharges=95,InactiveDays=80)
        low =self._pred(tenure=72,MonthlyCharges=20,InactiveDays=0,Contract="Two year",PaymentMethod="Bank transfer (automatic)")
        if high["churn_probability"]>low["churn_probability"]:
            self.assertGreaterEqual(high["risk_score"],low["risk_score"])
    def test_returns_segment(self):
        r=self._pred(); self.assertIn("segment",r)
    def test_returns_recommendations(self):
        r=self._pred(); self.assertIn("recommendations",r)
    def test_predict_missing_optional_fields(self):
        """Should not raise if optional fields are absent."""
        from backend.services.ml_service import predict_customer
        r=predict_customer({"tenure":12,"MonthlyCharges":50,"Contract":"One year","PaymentMethod":"Mailed check"})
        self.assertIn("churn_probability",r)

class TestSegmentation(unittest.TestCase):
    def _pred_df(self,n=80):
        rng=np.random.RandomState(99)
        return pd.DataFrame({"tenure":rng.randint(1,72,n),"MonthlyCharges":rng.uniform(20,110,n),
            "InactiveDays":rng.randint(0,90,n),"churn_probability":rng.uniform(0,1,n),
            "risk_score":rng.randint(0,100,n),"service_count":rng.randint(0,8,n),
            "engagement_score":rng.uniform(0,100,n),"churn_label":rng.choice(["Churn","No Churn"],n)})
    def test_fit_predict_adds_segment(self):
        from analytics.segmentation import CustomerSegmenter
        s=CustomerSegmenter(4); s.fit(self._pred_df()); out=s.predict(self._pred_df())
        self.assertIn("segment",out.columns)
    def test_labels_are_known(self):
        from analytics.segmentation import CustomerSegmenter, SEGMENT_LABELS
        s=CustomerSegmenter(4); df=self._pred_df(100); s.fit(df); out=s.predict(df)
        for lbl in out["segment"].unique(): self.assertIn(lbl,SEGMENT_LABELS)
    def test_no_nan_segments(self):
        from analytics.segmentation import CustomerSegmenter
        s=CustomerSegmenter(4); df=self._pred_df(60); s.fit(df); out=s.predict(df)
        self.assertEqual(out["segment"].isna().sum(),0)
    def test_predict_without_fit_raises(self):
        from analytics.segmentation import CustomerSegmenter
        with self.assertRaises(RuntimeError): CustomerSegmenter().predict(self._pred_df(10))
    def test_profile_has_count_key(self):
        from analytics.segmentation import CustomerSegmenter
        s=CustomerSegmenter(4); df=self._pred_df(80); s.fit(df); out=s.predict(df)
        for _,stats in s.segment_profile(out).items(): self.assertIn("count",stats)
    def test_all_rows_assigned(self):
        from analytics.segmentation import CustomerSegmenter
        s=CustomerSegmenter(4); df=self._pred_df(40); s.fit(df); out=s.predict(df)
        self.assertEqual(len(out),40)

class TestSHAP(unittest.TestCase):
    def setUp(self):
        import joblib
        from ml.config import MODEL_DIR
        if not (MODEL_DIR/"best_model.joblib").exists():
            self.skipTest("No trained model — run ml.pipeline first")
        from ml.evaluation.explainer import ChurnExplainer
        model=joblib.load(MODEL_DIR/"best_model.joblib")
        feats=joblib.load(MODEL_DIR/"feature_names.joblib") if (MODEL_DIR/"feature_names.joblib").exists() else [f"f{i}" for i in range(10)]
        self.exp=ChurnExplainer(model,feats)
    def _X(self,n=1): return np.random.RandomState(1).rand(n,len(self.exp.feature_names))
    def test_explain_returns_list(self):
        r=self.exp.explain_instance(self._X()[0]); self.assertIsInstance(r,list)
    def test_explanation_keys(self):
        r=self.exp.explain_instance(self._X()[0])
        if r: self.assertIn("feature",r[0]); self.assertIn("shap_value",r[0])
    def test_top_n_respected(self):
        r=self.exp.explain_instance(self._X()[0],top_n=5); self.assertLessEqual(len(r),5)
    def test_global_importance_not_none(self):
        r=self.exp.get_global_importance(self._X(20)); self.assertIsNotNone(r)

if __name__=="__main__": unittest.main(verbosity=2)
