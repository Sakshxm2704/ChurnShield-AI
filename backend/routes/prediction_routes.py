"""
backend/routes/prediction_routes.py
-------------------------------------
Prediction endpoints:
  POST /api/v1/predict             — single customer churn prediction
  POST /api/v1/batch_predict       — CSV upload for bulk predictions
  GET  /api/v1/predictions         — list prediction history (paginated)
  GET  /api/v1/predictions/<id>    — single prediction detail
  GET  /api/v1/predictions/explain/<prediction_id>  — SHAP explanation
"""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone

from flask import Blueprint, jsonify, request, send_file

from backend.auth.dependencies   import require_auth, require_role
from backend.models.schemas      import validate_customer, validate_pagination, ok, err
from backend.services.db_service import (
    create_customer, save_prediction, save_recommendations,
    list_predictions, get_latest_prediction, get_prediction_history,
    get_customer, save_history,
)
from backend.services.ml_service import predict_customer, predict_from_csv
from services.alerts.alert_service import alert_service
from backend.services.monitoring    import monitor
from backend.core.config             import settings

logger   = logging.getLogger(__name__)
pred_log = logging.getLogger("api.prediction")

bp = Blueprint("predictions", __name__, url_prefix="/api/v1")


# ── POST /predict ─────────────────────────────────────────────────────────────
@bp.route("/predict", methods=["POST"])
@require_auth
def predict(current_user):
    """
    Generate a full churn prediction for one customer.
    ---
    tags: [Predictions]
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [tenure, monthly_charges, contract, payment_method]
            properties:
              tenure:            {type: integer, minimum: 0, example: 12}
              monthly_charges:   {type: number,  example: 79.99}
              contract:          {type: string,  example: "Month-to-month"}
              payment_method:    {type: string,  example: "Electronic check"}
              inactive_days:     {type: integer, minimum: 0, example: 30}
              subscription_type: {type: string,  example: "Standard"}
              internet_service:  {type: string,  example: "Fiber optic"}
              online_security:   {type: string,  example: "No"}
              tech_support:      {type: string,  example: "No"}
              gender:            {type: string,  example: "Male"}
              senior_citizen:    {type: integer, example: 0}
              save_customer:     {type: boolean, default: true,
                                  description: "Persist customer + prediction to DB"}
              include_shap:      {type: boolean, default: true}
    responses:
      200:
        description: Prediction result with risk score, recommendations, and SHAP
      400:
        description: Validation error
      503:
        description: ML model not loaded
    """
    data = request.get_json(silent=True) or {}
    save_flag  = data.pop("save_customer", True)
    shap_flag  = data.pop("include_shap", True)

    try:
        clean = validate_customer(data)
    except ValueError as e:
        return jsonify(err(str(e), "VALIDATION_ERROR")[0]), 400

    try:
        result = predict_customer(clean, include_shap=shap_flag)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify(err(f"ML inference error: {e}", "MODEL_ERROR")[0]), 503

    customer_id = None
    if save_flag:
        try:
            customer = create_customer(clean)
            customer_id = customer["customer_id"]
            prediction_id = save_prediction(customer_id, result)
            result["prediction_id"] = prediction_id
            result["customer_id"]   = customer_id
            if result.get("recommendations"):
                save_recommendations(customer_id, result["recommendations"])
            # Fire automated alert for high/medium risk customers
            try:
                alert = alert_service.evaluate_and_alert(
                    result, clean, customer_id, prediction_id
                )
                if alert:
                    result["alert"] = {
                        "triggered":     True,
                        "risk_category": alert["risk_category"],
                        "alert_id":      alert.get("alert_id"),
                    }
            except Exception as ae:
                logger.warning("Alert dispatch failed (non-fatal): %s", ae)
            # Invalidate analytics cache so next dashboard load is fresh
            cache.delete_prefix("analytics:")
        except Exception as e:
            logger.warning("Failed to persist prediction: %s", e)

    pred_log.info(
        "Prediction: customer=%s  prob=%.4f  risk=%s  score=%d  user=%s",
        customer_id or "anon",
        result["churn_probability"],
        result["risk_category"],
        result["risk_score"],
        current_user["email"],
    )

    # Record to this user's "My History"
    try:
        save_history(
            user_id=current_user["id"],
            entry_type="single_prediction",
            title=f"Single Prediction — {result.get('risk_category','—')} risk",
            summary=(f"{round(result.get('churn_probability',0)*100,1)}% churn probability, "
                     f"score {result.get('risk_score','—')}/100"),
            risk_category=result.get("risk_category"),
            reference_id=result.get("prediction_id"),
            payload=clean,
            result=result,
        )
    except Exception as he:
        logger.warning("Failed to save history entry (non-fatal): %s", he)

    # Record monitoring metrics
    try:
        monitor.record_prediction(
            risk_category=result.get("risk_category", "Low"),
            user_id=current_user.get("id"),
        )
    except Exception:
        pass

    # Record to monitoring counters
    try:
        monitor.record_prediction(
            risk_category=result.get("risk_category", "Low"),
            user_id=current_user.get("id"),
        )
    except Exception:
        pass

    return jsonify(ok(result, "Prediction generated successfully.")[0]), 200


# ── POST /batch_predict ───────────────────────────────────────────────────────
@bp.route("/batch_predict", methods=["POST"])
@require_auth
def batch_predict(current_user):
    """
    Bulk churn prediction via CSV file upload.
    ---
    tags: [Predictions]
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            required: [file]
            properties:
              file:
                type: string
                format: binary
                description: CSV file (max 10,000 rows)
              export:
                type: boolean
                default: false
                description: If true, return scored CSV file instead of JSON
    responses:
      200:
        description: Batch prediction summary + first 100 rows preview
      400:
        description: No file uploaded or invalid CSV
    """
    if "file" not in request.files:
        return jsonify(err("No file provided. Upload a CSV under the 'file' field.", "NO_FILE")[0]), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify(err("File must be a .csv file.", "INVALID_FILE")[0]), 400

    export = request.form.get("export", "false").lower() == "true"

    try:
        csv_bytes = f.read()
        batch_result = predict_from_csv(csv_bytes)
    except ValueError as e:
        return jsonify(err(str(e), "CSV_ERROR")[0]), 400
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify(err(f"Batch prediction error: {e}", "MODEL_ERROR")[0]), 503

    pred_log.info(
        "Batch prediction: %d rows  high=%d  medium=%d  low=%d  user=%s",
        batch_result["total"],
        batch_result["high_risk"],
        batch_result["medium_risk"],
        batch_result["low_risk"],
        current_user["email"],
    )

    # Record to this user's "My History" (store summary only — not the full CSV)
    try:
        save_history(
            user_id=current_user["id"],
            entry_type="batch_prediction",
            title=f"Batch Prediction — {batch_result.get('total', 0)} rows",
            summary=(f"High: {batch_result.get('high_risk',0)}  "
                     f"Medium: {batch_result.get('medium_risk',0)}  "
                     f"Low: {batch_result.get('low_risk',0)}"),
            payload={"filename": f.filename},
            result={k: v for k, v in batch_result.items()
                    if k not in ("csv_output", "rows")},
        )
    except Exception as he:
        logger.warning("Failed to save history entry (non-fatal): %s", he)

    # Return scored CSV file for download
    if export:
        csv_io = io.BytesIO(batch_result["csv_output"].encode("utf-8"))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return send_file(
            csv_io,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"churn_predictions_{timestamp}.csv",
        )

    # Remove full CSV from JSON response (too large)
    batch_result.pop("csv_output", None)
    return jsonify(ok(batch_result, f"Batch prediction complete. {batch_result['total']} rows processed.")[0]), 200


# ── GET /batch_predict/export ─────────────────────────────────────────────────
@bp.route("/batch_predict/export", methods=["POST"])
@require_auth
def batch_predict_export(current_user):
    """
    Upload CSV and receive a scored CSV file as download.
    Alias for POST /batch_predict with export=true.
    ---
    tags: [Predictions]
    security:
      - BearerAuth: []
    """
    if "file" not in request.files:
        return jsonify(err("No file provided.", "NO_FILE")[0]), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify(err("File must be a .csv file.", "INVALID_FILE")[0]), 400

    try:
        batch_result = predict_from_csv(f.read())
    except ValueError as e:
        return jsonify(err(str(e), "CSV_ERROR")[0]), 400
    except Exception as e:
        return jsonify(err(f"Batch error: {e}", "MODEL_ERROR")[0]), 503

    csv_io = io.BytesIO(batch_result["csv_output"].encode("utf-8"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return send_file(csv_io, mimetype="text/csv", as_attachment=True,
                     download_name=f"churn_predictions_{timestamp}.csv")


# ── GET /predictions ──────────────────────────────────────────────────────────
@bp.route("/predictions", methods=["GET"])
@require_auth
def list_preds(current_user):
    """
    List prediction history (paginated).
    ---
    tags: [Predictions]
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: page
        schema: {type: integer, default: 1}
      - in: query
        name: page_size
        schema: {type: integer, default: 20}
      - in: query
        name: risk_label
        schema: {type: string, enum: [High, Medium, Low]}
    responses:
      200:
        description: Paginated list of predictions
    """
    page, page_size = validate_pagination(request.args)
    risk_label = request.args.get("risk_label")
    result = list_predictions(page, page_size, risk_label)
    return jsonify(ok(result)[0]), 200


# ── GET /predictions/<prediction_id> ─────────────────────────────────────────
@bp.route("/predictions/<int:prediction_id>", methods=["GET"])
@require_auth
def get_pred(prediction_id: int, current_user):
    """
    Retrieve a single prediction by ID, including SHAP values.
    ---
    tags: [Predictions]
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: prediction_id
        required: true
        schema: {type: integer}
    responses:
      200:
        description: Prediction detail
      404:
        description: Prediction not found
    """
    from backend.core.database import get_db, row_to_dict
    with get_db() as conn:
        row = conn.execute("SELECT * FROM predictions WHERE prediction_id=?",
                           (prediction_id,)).fetchone()
    pred = row_to_dict(row)
    if not pred:
        return jsonify(err("Prediction not found.", "NOT_FOUND")[0]), 404

    # Parse SHAP values from JSON string
    if pred.get("shap_values"):
        try:
            pred["shap_values"] = json.loads(pred["shap_values"])
        except Exception:
            pass
    return jsonify(ok(pred)[0]), 200


# ══════════════════════════════════════════════════════════════════════════════
# GROQ AI ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@bp.route("/groq/predict", methods=["POST"])
@require_auth
def groq_predict_route(current_user):
    """
    Groq AI se prediction — kisi bhi industry ke liye.
    Body: customer data + industry field
    """
    from backend.services.groq_service import groq_predict, is_groq_available
    from backend.models.schemas import ok, err

    if not is_groq_available():
        return jsonify(err(
            "Groq API key not configured. Add GROQ_API_KEY to your .env file.",
            "GROQ_NOT_CONFIGURED", 503
        )[0]), 503

    data = request.get_json(silent=True) or {}
    industry = data.pop("industry", "general")

    if not data:
        return jsonify(err("Customer data required.", "MISSING_DATA", 400)[0]), 400

    try:
        result = groq_predict(data, industry)
        try:
            save_history(
                user_id=current_user["id"],
                entry_type="groq_prediction",
                title=f"Groq AI Prediction — {industry.title()}",
                summary=(f"{round(result.get('churn_probability',0)*100,1)}% churn probability"
                         if result.get("churn_probability") is not None else "Groq AI analysis"),
                risk_category=result.get("risk_category"),
                payload={"industry": industry, **data},
                result=result,
            )
        except Exception as he:
            logger.warning("Failed to save history entry (non-fatal): %s", he)
        return jsonify(ok(result, "Groq AI prediction successful.")[0]), 200
    except RuntimeError as e:
        return jsonify(err(str(e), "GROQ_ERROR", 503)[0]), 503
    except Exception as e:
        logger.error("Groq predict error: %s", e)
        return jsonify(err("Groq prediction failed.", "GROQ_ERROR", 500)[0]), 500


@bp.route("/groq/batch", methods=["POST"])
@require_auth
def groq_batch_route(current_user):
    """
    Groq AI se CSV batch prediction — kisi bhi industry ke liye.
    Form data: file (CSV) + industry
    """
    from backend.services.groq_service import groq_batch_predict, is_groq_available
    from backend.models.schemas import ok, err
    import pandas as pd
    import io

    if not is_groq_available():
        return jsonify(err(
            "Groq API key not configured.",
            "GROQ_NOT_CONFIGURED", 503
        )[0]), 503

    if "file" not in request.files:
        return jsonify(err("CSV file required.", "MISSING_FILE", 400)[0]), 400

    industry = request.form.get("industry", "general")
    f = request.files["file"]

    try:
        df = pd.read_csv(io.BytesIO(f.read()))
        if len(df) > 100:
            return jsonify(err(
                "Groq batch limit is 100 rows (API cost). Use ML batch for larger files.",
                "TOO_MANY_ROWS", 400
            )[0]), 400

        customers = df.to_dict(orient="records")
        results   = groq_batch_predict(customers, industry)

        high   = sum(1 for r in results if r.get("risk_category") == "High")
        medium = sum(1 for r in results if r.get("risk_category") == "Medium")
        low    = sum(1 for r in results if r.get("risk_category") == "Low")

        try:
            save_history(
                user_id=current_user["id"],
                entry_type="groq_batch",
                title=f"Groq AI Batch — {len(results)} rows ({industry.title()})",
                summary=f"High: {high}  Medium: {medium}  Low: {low}",
                payload={"industry": industry, "filename": f.filename},
                result={"total": len(results), "high_risk": high,
                        "medium_risk": medium, "low_risk": low, "industry": industry},
            )
        except Exception as he:
            logger.warning("Failed to save history entry (non-fatal): %s", he)

        return jsonify(ok({
            "total":    len(results),
            "high_risk":   high,
            "medium_risk": medium,
            "low_risk":    low,
            "industry":    industry,
            "source":      "groq_ai",
            "results":     results[:50],  # First 50 in response
        }, "Groq batch prediction complete.")[0]), 200

    except Exception as e:
        logger.error("Groq batch error: %s", e)
        return jsonify(err(str(e), "GROQ_BATCH_ERROR", 500)[0]), 500


@bp.route("/groq/status", methods=["GET"])
@require_auth
def groq_status(current_user):
    """Groq API ki status check karo."""
    from backend.services.groq_service import is_groq_available, get_supported_industries
    from backend.models.schemas import ok

    return jsonify(ok({
        "available":   is_groq_available(),
        "model":       settings.GROQ_MODEL,
        "industries":  get_supported_industries(),
    }, "Groq status.")[0]), 200


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM TRAINING ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@bp.route("/train/upload", methods=["POST"])
@require_auth
def upload_training_data(current_user):
    """
    User apna CSV upload kare → custom model train ho.
    Form: file (CSV) + industry + target_col (optional, default: 'churn')
    """
    from backend.services.custom_training_service import train_custom_model
    from backend.models.schemas import ok, err

    if "file" not in request.files:
        return jsonify(err("CSV file required.", "MISSING_FILE", 400)[0]), 400

    f          = request.files["file"]
    industry   = request.form.get("industry", "general")
    target_col = request.form.get("target_col", "churn")

    try:
        csv_content = f.read()
        result = train_custom_model(csv_content, industry, target_col)
        return jsonify(ok(result, "Custom model training started!")[0]), 200
    except ValueError as e:
        return jsonify(err(str(e), "VALIDATION_ERROR", 400)[0]), 400
    except Exception as e:
        logger.error("Custom training error: %s", e)
        return jsonify(err(str(e), "TRAINING_ERROR", 500)[0]), 500


@bp.route("/train/status", methods=["GET"])
@require_auth
def training_status(current_user):
    """Custom model training ki current status."""
    from backend.services.custom_training_service import get_training_status
    from backend.models.schemas import ok
    return jsonify(ok(get_training_status(), "Training status.")[0]), 200


@bp.route("/train/models", methods=["GET"])
@require_auth
def list_custom_models(current_user):
    """Saare trained custom models ki list."""
    from backend.services.custom_training_service import list_custom_models
    from backend.models.schemas import ok
    models = list_custom_models()
    return jsonify(ok({"models": models, "count": len(models)}, "Custom models.")[0]), 200
