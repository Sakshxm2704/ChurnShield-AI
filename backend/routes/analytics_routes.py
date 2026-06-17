"""
backend/routes/analytics_routes.py
------------------------------------
Analytics, recommendations, segments, and What-If endpoints:
  GET  /api/v1/analytics              — portfolio KPI dashboard
  GET  /api/v1/analytics/revenue      — revenue risk breakdown
  GET  /api/v1/analytics/feature_importance — SHAP global importance
  GET  /api/v1/recommendations        — list all retention recommendations
  GET  /api/v1/segments               — customer segment summary
  POST /api/v1/whatif                 — What-If scenario simulation
"""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from backend.auth.dependencies   import require_auth, require_role
from backend.models.schemas      import validate_customer, validate_pagination, ok, err
from backend.services.db_service import (
    get_analytics_summary, list_all_recommendations, get_segments_summary,
)
from backend.services.ml_service import get_feature_importance, get_portfolio_stats
from backend.core.cache          import cache, TTL_ANALYTICS, TTL_SEGMENTS, TTL_FEATURES

logger = logging.getLogger(__name__)
bp     = Blueprint("analytics", __name__, url_prefix="/api/v1")


# ── GET /analytics ────────────────────────────────────────────────────────────
@bp.route("/analytics", methods=["GET"])
@require_auth
def analytics(current_user):
    """
    Return live platform KPIs: risk distribution, revenue at risk, trend data.
    ---
    tags: [Analytics]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Dashboard analytics summary
    """
    cache_key = f"analytics:portfolio"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(ok(cached)[0]), 200

    db_stats   = get_analytics_summary()
    ml_stats   = get_portfolio_stats()

    # Merge both sources; db_stats takes precedence for live counts
    merged = {**ml_stats, **db_stats}
    cache.set(cache_key, merged, ttl=TTL_ANALYTICS)
    return jsonify(ok(merged)[0]), 200


# ── GET /analytics/revenue ────────────────────────────────────────────────────
@bp.route("/analytics/revenue", methods=["GET"])
@require_auth
def revenue_analytics(current_user):
    """
    Revenue risk estimation from the ML-scored portfolio.
    ---
    tags: [Analytics]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Revenue risk breakdown (expected loss, recoverable, LTV at risk)
    """
    from backend.core.config import settings
    import pandas as pd, json
    scored_path = settings.ML_REPORTS_DIR / "scored_customers.csv"
    if not scored_path.exists():
        return jsonify(ok({
            "message": "No scored data found. Run ml/pipeline.py first.",
            "expected_annual_loss": 0,
        })[0]), 200

    try:
        df = pd.read_csv(scored_path)
        from analytics.revenue_estimator import RevenueEstimator
        estimator = RevenueEstimator()
        portfolio = estimator.estimate_portfolio(df)
        return jsonify(ok(portfolio.to_dict())[0]), 200
    except Exception as e:
        logger.exception("Revenue analytics failed")
        return jsonify(err(f"Revenue estimation error: {e}", "MODEL_ERROR")[0]), 503


# ── GET /analytics/feature_importance ────────────────────────────────────────
@bp.route("/analytics/feature_importance", methods=["GET"])
@require_auth
def feature_importance(current_user):
    """
    Return global SHAP / model feature importance.
    ---
    tags: [Analytics]
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: top_n
        schema: {type: integer, default: 15}
    responses:
      200:
        description: Ranked feature importance list
    """
    top_n = min(50, max(1, int(request.args.get("top_n", 15))))
    try:
        importance = get_feature_importance(top_n=top_n)
    except Exception as e:
        return jsonify(err(str(e), "MODEL_ERROR")[0]), 503
    return jsonify(ok({"feature_importance": importance, "count": len(importance)})[0]), 200


# ── GET /analytics/model_metrics ─────────────────────────────────────────────
@bp.route("/analytics/model_metrics", methods=["GET"])
@require_auth
def model_metrics(current_user):
    """
    Return training metrics for all trained models.
    ---
    tags: [Analytics]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Model evaluation results (accuracy, F1, AUC, etc.)
    """
    from backend.services.ml_service import get_model_metadata
    meta = get_model_metadata()
    if not meta:
        return jsonify(ok({"message": "No model metadata found."})[0]), 200
    return jsonify(ok({
        "best_model":  meta.get("best_model"),
        "models":      meta.get("evaluation", {}),
        "auc_scores":  meta.get("auc_scores", {}),
        "train_size":  meta.get("train_size"),
        "test_size":   meta.get("test_size"),
        "churn_rate":  meta.get("churn_rate_train"),
    })[0]), 200


# ── GET /recommendations ──────────────────────────────────────────────────────
@bp.route("/recommendations", methods=["GET"])
@require_auth
def recommendations(current_user):
    """
    List all retention recommendations (paginated).
    ---
    tags: [Recommendations]
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: page
        schema: {type: integer, default: 1}
      - in: query
        name: page_size
        schema: {type: integer, default: 20}
    responses:
      200:
        description: Paginated recommendations list
    """
    page, page_size = validate_pagination(request.args)
    result = list_all_recommendations(page, page_size)
    return jsonify(ok(result)[0]), 200


# ── GET /segments ─────────────────────────────────────────────────────────────
@bp.route("/segments", methods=["GET"])
@require_auth
def segments(current_user):
    """
    Return customer segment distribution and per-segment statistics.
    ---
    tags: [Segments]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Segment profiles (Loyal / Premium / Risky / Inactive)
    """
    # Try to load from ML segmenter output first
    from backend.core.config import settings
    import pandas as pd, json as _json
    scored_path = settings.ML_REPORTS_DIR / "scored_customers.csv"

    if scored_path.exists() and "segment" in pd.read_csv(scored_path, nrows=1).columns:
        try:
            df = pd.read_csv(scored_path)
            from analytics.segmentation import CustomerSegmenter
            seg_path = settings.ML_MODEL_DIR / "customer_segmenter.joblib"
            segmenter = CustomerSegmenter.load(seg_path)
            profile   = segmenter.segment_profile(df)
            return jsonify(ok({
                "segments": profile,
                "source":   "ml_segmenter",
                "total":    len(df),
            })[0]), 200
        except Exception as e:
            logger.warning("ML segmenter failed, falling back to DB: %s", e)

    # Fallback: DB-derived segments
    result = get_segments_summary()
    result["source"] = "database_derived"
    return jsonify(ok(result)[0]), 200


# ── POST /whatif ──────────────────────────────────────────────────────────────
@bp.route("/whatif", methods=["POST"])
@require_auth
def whatif(current_user):
    """
    What-If scenario simulation: apply feature changes and see new churn probability.
    ---
    tags: [Analytics]
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [customer_data, scenario]
            properties:
              customer_data:
                type: object
                description: Base customer record (same schema as /predict)
              scenario:
                type: object
                description: Feature overrides to simulate
                example: {"Contract": "One year", "PaymentMethod": "Bank transfer (automatic)"}
              scenario_name:
                type: string
                example: "upgrade_to_annual_contract"
    responses:
      200:
        description: Scenario simulation result with delta probability
      400:
        description: Validation error
    """
    data = request.get_json(silent=True) or {}

    if "customer_data" not in data:
        return jsonify(err("'customer_data' is required.", "VALIDATION_ERROR")[0]), 400

    scenario_changes = data.get("scenario", {})
    scenario_name    = data.get("scenario_name", "custom")

    # Support named preset scenarios
    if scenario_name != "custom" and not scenario_changes:
        from analytics.whatif_simulator import PRESET_SCENARIOS
        if scenario_name not in PRESET_SCENARIOS:
            return jsonify(err(
                f"Unknown preset scenario '{scenario_name}'. "
                f"Available: {list(PRESET_SCENARIOS.keys())}",
                "INVALID_SCENARIO")[0]), 400
        scenario_changes = PRESET_SCENARIOS[scenario_name]

    try:
        clean = validate_customer(data["customer_data"])
    except ValueError as e:
        return jsonify(err(str(e), "VALIDATION_ERROR")[0]), 400

    try:
        from analytics.whatif_simulator import WhatIfSimulator
        from backend.services.ml_service import get_predictor
        predictor = get_predictor()
        sim    = WhatIfSimulator()
        result = sim.simulate_customer(clean, scenario_changes, predictor, scenario_name)
        return jsonify(ok(result.to_dict(), "Scenario simulation complete.")[0]), 200
    except Exception as e:
        logger.exception("What-If simulation failed")
        return jsonify(err(f"Simulation error: {e}", "SIMULATION_ERROR")[0]), 503


# ── GET /analytics/presets ────────────────────────────────────────────────────
@bp.route("/analytics/whatif_presets", methods=["GET"])
@require_auth
def whatif_presets(current_user):
    """
    List available What-If preset scenario names and descriptions.
    ---
    tags: [Analytics]
    security:
      - BearerAuth: []
    responses:
      200:
        description: Available preset scenarios
    """
    try:
        from analytics.whatif_simulator import PRESET_SCENARIOS
        presets = [{"name": k, "description": v.get("description", "")}
                   for k, v in PRESET_SCENARIOS.items()]
    except Exception:
        presets = []
    return jsonify(ok({"presets": presets, "count": len(presets)})[0]), 200
