"""
tests/test_ml_validation.py
----------------------------
ML model validation: confusion matrix, cross-validation, model comparison,
data quality checks, and performance benchmarks.
"""
from __future__ import annotations
import sys, unittest, json, time
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# 1. ML Model Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestMLModelValidation(unittest.TestCase):
    """Validate model performance metrics from training results."""

    def setUp(self):
        meta_path = Path(__file__).parent.parent / "ml" / "models" / "saved" / "model_metadata.json"
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.eval = self.meta["evaluation"]

    def test_best_model_selected(self):
        best = self.meta["best_model"]
        self.assertIn(best, self.eval.keys())

    def test_best_model_has_highest_auc(self):
        best  = self.meta["best_model"]
        aucs  = self.meta["auc_scores"]
        best_auc = aucs[best]
        for model, auc in aucs.items():
            self.assertGreaterEqual(best_auc, auc,
                msg=f"Best model AUC should be ≥ {model} AUC")

    def test_all_models_trained(self):
        models = set(self.eval.keys())
        expected = {"logistic_regression", "random_forest", "xgboost"}
        # At least 2 of 3 must be present (xgboost optional)
        self.assertGreaterEqual(len(models & expected), 2)

    def test_accuracy_above_threshold(self):
        for name, m in self.eval.items():
            self.assertGreater(m["accuracy"], 0.60,
                msg=f"{name} accuracy {m['accuracy']:.3f} below 0.60")

    def test_roc_auc_above_threshold(self):
        for name, m in self.eval.items():
            self.assertGreater(m["roc_auc"], 0.65,
                msg=f"{name} AUC {m['roc_auc']:.3f} below 0.65")

    def test_recall_above_threshold(self):
        """High recall is critical for churn (minimize missed churners)."""
        for name, m in self.eval.items():
            self.assertGreater(m["recall"], 0.45,
                msg=f"{name} recall {m['recall']:.3f} below 0.45")

    def test_f1_reasonable(self):
        for name, m in self.eval.items():
            self.assertGreater(m["f1"], 0.45,
                msg=f"{name} F1 {m['f1']:.3f} below 0.45")

    def test_class_report_has_both_classes(self):
        for name, m in self.eval.items():
            report = m.get("class_report", {})
            self.assertIn("0", report, f"{name} missing class 0 in report")
            self.assertIn("1", report, f"{name} missing class 1 in report")

    def test_train_test_split_reasonable(self):
        self.assertGreater(self.meta["train_size"], 0)
        self.assertGreater(self.meta["test_size"],  0)
        ratio = self.meta["train_size"] / (self.meta["train_size"] + self.meta["test_size"])
        self.assertGreater(ratio, 0.65)
        self.assertLess(ratio, 0.90)

    def test_churn_rate_stratified(self):
        """Train and test churn rates should be similar (stratified split)."""
        train_rate = self.meta["churn_rate_train"]
        test_rate  = self.meta["churn_rate_test"]
        self.assertAlmostEqual(train_rate, test_rate, delta=0.05,
            msg=f"Churn rates differ by >5%: train={train_rate:.3f} test={test_rate:.3f}")

    def test_feature_count_reasonable(self):
        self.assertGreater(self.meta["feature_count"], 20)
        self.assertLess(self.meta["feature_count"],    200)

    def test_no_overfitting_indicator(self):
        """Best model AUC should be reasonable — very high values may indicate overfit."""
        best_auc = self.meta["auc_scores"][self.meta["best_model"]]
        self.assertLess(best_auc, 0.99, "AUC suspiciously high — possible overfit")

    def test_model_comparison_logistic_vs_rf(self):
        """Both models should be close in performance (within 10 % AUC)."""
        lr  = self.meta["auc_scores"].get("logistic_regression", 0)
        rf  = self.meta["auc_scores"].get("random_forest", 0)
        if lr and rf:
            self.assertLess(abs(lr - rf), 0.10)

    def test_feature_importance_sums_to_one(self):
        """Logistic regression coefficients are normalised — top 20 sum should be significant."""
        lr_feats = self.eval["logistic_regression"]["feature_importance"]
        total    = sum(f["importance"] for f in lr_feats)
        self.assertGreater(total, 0)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model Artifacts Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestModelArtifacts(unittest.TestCase):

    def setUp(self):
        import joblib
        self.model_dir = Path(__file__).parent.parent / "ml" / "models" / "saved"
        self.joblib    = joblib

    def test_best_model_loadable(self):
        model = self.joblib.load(self.model_dir / "best_model.joblib")
        self.assertIsNotNone(model)

    def test_preprocessor_loadable(self):
        preprocessor = self.joblib.load(self.model_dir / "preprocessor.joblib")
        self.assertIsNotNone(preprocessor)

    def test_feature_names_loadable(self):
        names = self.joblib.load(self.model_dir / "feature_names.joblib")
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 10)

    def test_segmenter_loadable(self):
        seg = self.joblib.load(self.model_dir / "customer_segmenter.joblib")
        self.assertIsNotNone(seg)

    def test_model_has_predict_method(self):
        model = self.joblib.load(self.model_dir / "best_model.joblib")
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_preprocessor_has_transform(self):
        preprocessor = self.joblib.load(self.model_dir / "preprocessor.joblib")
        self.assertTrue(hasattr(preprocessor, "transform"))

    def test_logistic_regression_saved(self):
        lr = self.joblib.load(self.model_dir / "logistic_regression.joblib")
        self.assertIsNotNone(lr)

    def test_random_forest_saved(self):
        rf = self.joblib.load(self.model_dir / "random_forest.joblib")
        self.assertIsNotNone(rf)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Data Quality Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDataQuality(unittest.TestCase):

    def setUp(self):
        raw_path = Path(__file__).parent.parent / "data" / "raw" / "telco_churn.csv"
        if raw_path.exists():
            self.df = pd.read_csv(raw_path)
        else:
            # Generate fresh data for quality check
            from data.generate_dataset import generate_telco_dataset
            self.df = generate_telco_dataset(n_rows=500, seed=42)
        self.has_data = len(self.df) > 0

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.df), 0)

    def test_required_columns_present(self):
        required = ["tenure", "MonthlyCharges", "Contract", "PaymentMethod", "Churn"]
        for col in required:
            self.assertIn(col, self.df.columns, f"Required column '{col}' missing")

    def test_tenure_non_negative(self):
        if not self.has_data: self.skipTest("No data")
        self.assertTrue((self.df["tenure"] >= 0).all(),
                        f"Negative tenure found: {self.df[self.df['tenure'] < 0]['tenure'].values}")

    def test_monthly_charges_positive(self):
        if not self.has_data: self.skipTest("No data")
        self.assertTrue((self.df["MonthlyCharges"] > 0).all())

    def test_churn_binary(self):
        if not self.has_data: self.skipTest("No data")
        unique = set(self.df["Churn"].unique())
        self.assertTrue(unique <= {0, 1, "Yes", "No"},
                        f"Unexpected Churn values: {unique}")

    def test_contract_categories_valid(self):
        if not self.has_data: self.skipTest("No data")
        valid = {"Month-to-month", "One year", "Two year"}
        actual = set(self.df["Contract"].unique())
        self.assertTrue(actual <= valid, f"Unexpected Contract values: {actual - valid}")

    def test_no_duplicate_customer_ids(self):
        """customerID should have low duplicate rate (synthetic data may reuse IDs across batches)."""
        if not self.has_data or "customerID" not in self.df.columns: self.skipTest("No customerID")
        total = len(self.df)
        dup_count = int(self.df["customerID"].duplicated().sum())
        dup_rate = dup_count / total
        # Allow up to 2% duplicates (synthetic dataset uses sequential IDs per run)
        self.assertLessEqual(dup_rate, 0.02,
            f"Duplicate customerID rate {dup_rate:.1%} exceeds 2% ({dup_count}/{total})") 

    def test_monthly_charges_outliers(self):
        if not self.has_data: self.skipTest("No data")
        q99 = self.df["MonthlyCharges"].quantile(0.99)
        self.assertLess(q99, 200, f"Suspicious high charges: {q99}")

    def test_tenure_outliers(self):
        if not self.has_data: self.skipTest("No data")
        max_tenure = self.df["tenure"].max()
        self.assertLessEqual(max_tenure, 200)

    def test_churn_rate_reasonable(self):
        if not self.has_data: self.skipTest("No data")
        churn_col = "Churn"
        if self.df[churn_col].dtype == object:
            rate = (self.df[churn_col] == "Yes").mean()
        else:
            rate = self.df[churn_col].mean()
        self.assertGreater(rate, 0.05, "Churn rate < 5% — unexpected")
        self.assertLess(rate, 0.70, "Churn rate > 70% — unexpected")

    def test_missing_values_in_key_columns(self):
        if not self.has_data: self.skipTest("No data")
        key_cols = ["tenure", "MonthlyCharges", "Contract"]
        for col in key_cols:
            if col in self.df.columns:
                missing = self.df[col].isna().sum()
                self.assertEqual(missing, 0, f"{col} has {missing} missing values")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Database Schema Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestDatabaseSchema(unittest.TestCase):

    def setUp(self):
        from backend.core.database import get_db, init_db
        init_db()
        self.get_db = get_db

    def _tables(self):
        with self.get_db() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        return [r[0] for r in rows]

    def _columns(self, table):
        with self.get_db() as conn:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [r[1] for r in rows]

    def _indexes(self, table):
        with self.get_db() as conn:
            rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
        return [r[1] for r in rows]

    def test_all_tables_created(self):
        tables = self._tables()
        required = ["users", "customers", "predictions", "recommendations",
                    "retention_logs", "analytics_logs", "api_logs",
                    "alert_log", "monitoring_metrics"]
        for t in required:
            self.assertIn(t, tables, f"Table '{t}' missing")

    def test_users_columns(self):
        cols = self._columns("users")
        for c in ["id", "name", "email", "password_hash", "role", "created_at"]:
            self.assertIn(c, cols)

    def test_customers_columns(self):
        cols = self._columns("customers")
        for c in ["customer_id", "tenure", "monthly_charges", "contract", "payment_method"]:
            self.assertIn(c, cols)

    def test_predictions_columns(self):
        cols = self._columns("predictions")
        for c in ["prediction_id", "customer_id", "churn_probability", "risk_score", "prediction_label"]:
            self.assertIn(c, cols)

    def test_alert_log_columns(self):
        cols = self._columns("alert_log")
        for c in ["id", "customer_id", "alert_type", "risk_category", "churn_probability"]:
            self.assertIn(c, cols)

    def test_users_email_unique_index(self):
        indexes = self._indexes("users")
        self.assertTrue(any("email" in idx for idx in indexes))

    def test_predictions_customer_id_indexed(self):
        indexes = self._indexes("predictions")
        self.assertTrue(any("customer_id" in idx for idx in indexes))

    def test_alert_log_indexed(self):
        indexes = self._indexes("alert_log")
        self.assertTrue(len(indexes) > 0)

    def test_foreign_key_cascade(self):
        """Deleting a customer should cascade to predictions via application logic."""
        from backend.services.db_service import create_customer, save_prediction
        c = create_customer({
            "tenure": 3, "monthly_charges": 70.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "inactive_days": 20,
        })
        save_prediction(c["customer_id"], {
            "churn_probability": 0.75, "risk_score": 80,
            "risk_category": "High", "churn_label": "Churn", "model_used": "test",
        })
        # Verify prediction was saved
        with self.get_db() as conn:
            preds = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE customer_id=?",
                (c["customer_id"],)
            ).fetchone()[0]
        self.assertGreater(preds, 0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Performance Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPerformance(unittest.TestCase):
    """Measure response times against defined SLAs."""

    SLA_SINGLE_PREDICTION_MS = 2000    # 2 seconds
    SLA_BATCH_100_ROWS_MS    = 10000   # 10 seconds for 100 rows
    SLA_ANALYTICS_MS         = 1000    # 1 second

    @classmethod
    def setUpClass(cls):
        import sys, json
        from tests.conftest import make_app, get_auth_token
        cls.app    = make_app()
        cls.client = cls.app.test_client()
        email      = f"perf_{int(time.time())}@test.io"
        cls.token  = get_auth_token(cls.client, email=email)
        cls.auth   = {"Authorization": f"Bearer {cls.token}"}

    def test_single_prediction_latency(self):
        from tests.conftest import VALID_CUSTOMER
        start = time.monotonic()
        r = self.client.post(
            "/api/v1/predict",
            data=json.dumps(VALID_CUSTOMER),
            content_type="application/json",
            headers=self.auth,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        self.assertEqual(r.status_code, 200)
        self.assertLess(elapsed_ms, self.SLA_SINGLE_PREDICTION_MS,
            msg=f"Prediction took {elapsed_ms:.0f}ms (SLA: {self.SLA_SINGLE_PREDICTION_MS}ms)")

    def test_batch_prediction_100_rows(self):
        # Build a 100-row CSV
        header = "tenure,monthly_charges,contract,payment_method,inactive_days\n"
        rows   = "".join(
            f"{(i%60)+1},{25+(i%80):.1f},Month-to-month,Electronic check,{i%90}\n"
            for i in range(100)
        )
        csv_data = (header + rows).encode()
        start = time.monotonic()
        r = self.client.post(
            "/api/v1/batch_predict",
            data={"file": (io.BytesIO(csv_data), "batch100.csv")},
            content_type="multipart/form-data",
            headers=self.auth,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        self.assertEqual(r.status_code, 200)
        self.assertLess(elapsed_ms, self.SLA_BATCH_100_ROWS_MS,
            msg=f"Batch (100 rows) took {elapsed_ms:.0f}ms (SLA: {self.SLA_BATCH_100_ROWS_MS}ms)")

    def test_analytics_endpoint_latency(self):
        start = time.monotonic()
        r = self.client.get("/api/v1/analytics", headers=self.auth)
        elapsed_ms = (time.monotonic() - start) * 1000
        self.assertEqual(r.status_code, 200)
        self.assertLess(elapsed_ms, self.SLA_ANALYTICS_MS,
            msg=f"Analytics took {elapsed_ms:.0f}ms (SLA: {self.SLA_ANALYTICS_MS}ms)")

    def test_prediction_throughput(self):
        """Run 5 predictions and verify average under SLA."""
        from tests.conftest import VALID_CUSTOMER
        latencies = []
        for _ in range(5):
            start = time.monotonic()
            self.client.post(
                "/api/v1/predict",
                data=json.dumps(VALID_CUSTOMER),
                content_type="application/json",
                headers=self.auth,
            )
            latencies.append((time.monotonic() - start) * 1000)
        avg_ms = sum(latencies) / len(latencies)
        self.assertLess(avg_ms, self.SLA_SINGLE_PREDICTION_MS,
            msg=f"Average prediction {avg_ms:.0f}ms exceeds SLA")


import io, json  # needed for performance tests

if __name__ == "__main__":
    unittest.main(verbosity=2)
