"""
ml/pipeline.py
--------------
Master pipeline orchestrator for the Churn Intelligence Platform.

Executes the full ML pipeline in a single command:

  1. Generate / load synthetic Telco dataset
  2. Feature engineering
  3. Preprocessing (clean, encode, scale, split)
  4. Train all models (Logistic Regression, Random Forest, XGBoost)
  5. Evaluate all models with full metrics suite
  6. Select & save the best model
  7. Build SHAP explainer
  8. Batch-predict all customers
  9. Run KMeans customer segmentation
 10. Apply retention recommendation engine
 11. Compute portfolio revenue risk
 12. Save all artefacts

Usage
-----
    python -m ml.pipeline                        # full run
    python -m ml.pipeline --no-generate          # skip data generation (use existing CSV)
    python -m ml.pipeline --rows 15000           # generate with custom row count
    python -m ml.pipeline --no-shap              # skip SHAP (faster)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Project root on sys.path (for script execution) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import (
    DATA_RAW,
    MODEL_DIR,
    RAW_DATASET_PATH,
    REPORTS_DIR,
)
from ml.features.feature_engineering import engineer_features
from ml.features.preprocessing import run_preprocessing_pipeline
from ml.models.trainer import run_training_pipeline
from ml.evaluation.evaluator import evaluate_all_models, print_evaluation_report
from ml.evaluation.explainer import build_explainer, ChurnExplainer
from ml.models.predictor import ChurnPredictor

from analytics.segmentation import segment_customers
from analytics.revenue_estimator import RevenueEstimator
from services.retention.recommendation_engine import RetentionRecommendationEngine

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ml.pipeline")

# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_generate_data(n_rows: int, seed: int) -> None:
    """Step 1 — Generate synthetic Telco dataset."""
    logger.info("── Step 1: Data Generation ──────────────────────────────────")
    from data.generate_dataset import generate_telco_dataset

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df = generate_telco_dataset(n_rows=n_rows, seed=seed)
    df.to_csv(RAW_DATASET_PATH, index=False)

    churn_rate = df["Churn"].mean() * 100
    logger.info(
        "Dataset generated: %d rows, churn rate %.1f %% → %s",
        len(df), churn_rate, RAW_DATASET_PATH,
    )


def step_preprocess() -> tuple:
    """
    Step 2–3 — Feature engineering + preprocessing.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, feature_names, clean_df
    """
    logger.info("── Step 2: Feature Engineering & Preprocessing ──────────────")

    # run_preprocessing_pipeline loads, cleans, engineers features, and splits
    result = run_preprocessing_pipeline(save_artefacts=True)
    X_train, X_test, y_train, y_test, preprocessor, feature_names, clean_df = result

    logger.info(
        "Features: %d  |  Train: %d  |  Test: %d",
        len(feature_names), len(y_train), len(y_test),
    )
    return X_train, X_test, y_train, y_test, preprocessor, feature_names, clean_df


def step_train_and_evaluate(
    X_train, X_test, y_train, y_test, feature_names
) -> tuple:
    """
    Step 4–6 — Train, evaluate, select best model.

    Returns
    -------
    models, best_name, best_model, training_metadata
    """
    logger.info("── Step 3–5: Training & Evaluation ──────────────────────────")

    models, best_name, best_model, metadata = run_training_pipeline(
        X_train, X_test, y_train, y_test, feature_names, save=True
    )

    print_evaluation_report(metadata["evaluation"])
    logger.info("Best model: %s  (AUC %.4f)", best_name, metadata["auc_scores"][best_name])
    return models, best_name, best_model, metadata


def step_build_explainer(
    best_model,
    X_train: np.ndarray,
    feature_names: list[str],
    use_shap: bool = True,
) -> ChurnExplainer:
    """Step 7 — Build SHAP explainer."""
    logger.info("── Step 6: SHAP Explainability ──────────────────────────────")

    if use_shap:
        explainer = build_explainer(best_model, X_train, feature_names, n_background=200)
    else:
        from ml.evaluation.explainer import ChurnExplainer
        explainer = ChurnExplainer(best_model, feature_names)

    # Save explainer
    explainer_path = MODEL_DIR / "explainer.joblib"
    joblib.dump(explainer, explainer_path)
    logger.info("Explainer saved → %s", explainer_path)
    return explainer


def step_batch_predict(clean_df: pd.DataFrame, predictor: ChurnPredictor) -> pd.DataFrame:
    """Step 8 — Batch predict all customers, keeping engineered features for segmentation."""
    logger.info("── Step 7: Batch Prediction ─────────────────────────────────")

    # Add engineered features to the clean_df BEFORE batch predict so they are
    # available in the returned DataFrame for downstream segmentation.
    from ml.features.feature_engineering import engineer_features as _eng
    clean_enriched = _eng(clean_df.copy())

    df_pred = predictor.predict_batch(clean_enriched)

    high   = (df_pred["risk_category"] == "High").sum()
    medium = (df_pred["risk_category"] == "Medium").sum()
    low    = (df_pred["risk_category"] == "Low").sum()

    logger.info(
        "Predictions complete — High: %d | Medium: %d | Low: %d",
        high, medium, low,
    )
    return df_pred


def step_segment(df_pred: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Step 9 — KMeans customer segmentation."""
    logger.info("── Step 8: Customer Segmentation ────────────────────────────")

    df_seg, segmenter = segment_customers(df_pred, save=True)
    profile = segmenter.segment_profile(df_seg)

    for seg, stats in profile.items():
        logger.info(
            "  %-12s  n=%4d (%4.1f %%)  avg_churn=%.2f  avg_charges=$%.0f",
            seg,
            stats["count"],
            stats["pct_of_total"],
            stats.get("avg_churn_probability", 0),
            stats.get("avg_MonthlyCharges", 0),
        )
    return df_seg, profile


def step_recommendations(df_seg: pd.DataFrame) -> pd.DataFrame:
    """Step 10 — Apply retention recommendation engine."""
    logger.info("── Step 9: Retention Recommendations ────────────────────────")

    engine = RetentionRecommendationEngine()
    df_rec = engine.recommend_batch(df_seg)

    with_recs = (df_rec["recommendations"].apply(len) > 0).sum()
    logger.info("Customers with retention recommendations: %d", with_recs)
    return df_rec


def step_revenue_risk(df_rec: pd.DataFrame) -> dict:
    """Step 11 — Portfolio revenue risk estimation."""
    logger.info("── Step 10: Revenue Risk Estimation ─────────────────────────")

    # Add estimated_savings column from recommendations
    df_rec = df_rec.copy()
    df_rec["estimated_savings"] = df_rec["recommendations"].apply(
        lambda recs: sum(r.estimated_savings for r in recs) if recs else 0.0
    )

    estimator = RevenueEstimator()
    portfolio  = estimator.estimate_portfolio(df_rec)
    rev_dict   = portfolio.to_dict()

    logger.info("Total monthly revenue:      $%.0f", portfolio.total_monthly_revenue)
    logger.info("Expected annual loss:        $%.0f", portfolio.expected_annual_loss)
    logger.info("Recoverable (interventions): $%.0f", portfolio.recoverable_revenue_annual)
    logger.info("Net revenue at risk:         $%.0f", portfolio.net_revenue_at_risk_annual)
    return rev_dict


def step_save_outputs(
    df_rec: pd.DataFrame,
    metadata: dict,
    segment_profile: dict,
    revenue_risk: dict,
    best_name: str,
) -> None:
    """Step 12 — Save all output artefacts."""
    logger.info("── Step 11: Saving Outputs ──────────────────────────────────")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Scored dataset (CSV)
    scored_path = REPORTS_DIR / "scored_customers.csv"
    # Drop non-serialisable recommendation objects before saving
    cols_to_drop = [c for c in ["recommendations"] if c in df_rec.columns]
    df_rec.drop(columns=cols_to_drop).to_csv(scored_path, index=False)
    logger.info("Scored dataset saved → %s", scored_path)

    # Full pipeline report (JSON)
    report = {
        "best_model":      best_name,
        "training":        metadata,
        "segment_profile": segment_profile,
        "revenue_risk":    revenue_risk,
    }
    report_path = REPORTS_DIR / "pipeline_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Pipeline report saved → %s", report_path)


# ── Master orchestrator ───────────────────────────────────────────────────────

def run_full_pipeline(
    generate: bool = True,
    n_rows: int = 7_000,
    seed: int = 42,
    use_shap: bool = True,
) -> dict:
    """
    Execute all pipeline steps in sequence.

    Parameters
    ----------
    generate : bool  — generate fresh data (True) or use existing CSV (False)
    n_rows   : int   — number of rows to generate
    seed     : int   — random seed for reproducibility
    use_shap : bool  — build SHAP explainer (slower but richer output)

    Returns
    -------
    dict containing metadata, segment_profile, and revenue_risk
    """
    t_start = time.perf_counter()
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   Churn Intelligence Platform — Full ML Pipeline             ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    # 1. Data
    if generate:
        step_generate_data(n_rows, seed)
    else:
        logger.info("Skipping data generation — using existing %s", RAW_DATASET_PATH)

    # 2–3. Preprocess
    X_train, X_test, y_train, y_test, preprocessor, feature_names, clean_df = step_preprocess()

    # 4–6. Train & evaluate
    models, best_name, best_model, metadata = step_train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )

    # 7. Explainer
    explainer = step_build_explainer(best_model, X_train, feature_names, use_shap)

    # 8. Batch predict (reload from disk to validate saved artefacts)
    predictor = ChurnPredictor("best_model").load()
    df_pred   = step_batch_predict(clean_df, predictor)

    # 9. Segmentation
    df_seg, segment_profile = step_segment(df_pred)

    # 10. Recommendations
    df_rec = step_recommendations(df_seg)

    # 11. Revenue risk
    revenue_risk = step_revenue_risk(df_rec)

    # 12. Save outputs
    step_save_outputs(df_rec, metadata, segment_profile, revenue_risk, best_name)

    elapsed = time.perf_counter() - t_start
    logger.info(
        "╔══════════════════════════════════════════════════════════════╗"
    )
    logger.info(
        "║   Pipeline complete in %.1f s                               ║", elapsed
    )
    logger.info(
        "╚══════════════════════════════════════════════════════════════╝"
    )

    return {
        "best_model":      best_name,
        "metadata":        metadata,
        "segment_profile": segment_profile,
        "revenue_risk":    revenue_risk,
        "elapsed_seconds": round(elapsed, 1),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Churn Intelligence ML pipeline."
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip data generation and use existing CSV."
    )
    parser.add_argument(
        "--rows", type=int, default=7_000,
        help="Number of synthetic rows to generate (default: 7000)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data generation (default: 42)."
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP explainer fitting (faster run)."
    )
    args = parser.parse_args()

    result = run_full_pipeline(
        generate=not args.no_generate,
        n_rows=args.rows,
        seed=args.seed,
        use_shap=not args.no_shap,
    )
    sys.exit(0)
