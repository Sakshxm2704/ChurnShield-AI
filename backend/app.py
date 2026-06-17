"""
backend/app.py
--------------
Flask application factory for the ChurnShield AI API.

Architecture
------------
  backend/
  ├── app.py                  ← This file (application factory + entry point)
  ├── core/
  │   ├── config.py           ← Settings (env-driven)
  │   ├── database.py         ← SQLite layer (swap URL for PostgreSQL)
  │   └── logging_config.py   ← Structured JSON/text logging
  ├── auth/
  │   ├── jwt_handler.py      ← JWT create / verify
  │   ├── password.py         ← scrypt password hashing
  │   └── dependencies.py     ← require_auth / require_role decorators
  ├── middleware/
  │   └── request_logger.py   ← Timing, CORS, global error handler
  ├── models/
  │   └── schemas.py          ← Input validation + response helpers
  ├── routes/
  │   ├── auth_routes.py      ← /api/v1/auth/*
  │   ├── prediction_routes.py← /api/v1/predict, /batch_predict, /predictions
  │   ├── customer_routes.py  ← /api/v1/customers
  │   ├── analytics_routes.py ← /api/v1/analytics, /recommendations, /segments
  │   └── history_routes.py   ← /api/v1/history (per-user "My History")
  └── services/
      ├── ml_service.py       ← ML artefact loader + inference helpers
      └── db_service.py       ← All SQL operations

Run
---
    # Development
    python -m backend.app

    # Production
    gunicorn "backend.app:create_app()" -w 4 -b 0.0.0.0:8000

    # Docker
    docker compose -f docker/docker-compose.yml up
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify

# ── Ensure project root is on sys.path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config       import settings
from backend.core.database     import init_db
from backend.core.logging_config import setup_logging
from backend.middleware.request_logger import register_middleware

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Application factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> Flask:
    """
    Build and configure the Flask application.

    Returns a fully wired Flask instance ready for ``flask run`` or gunicorn.
    """
    # ── Logging (first) ───────────────────────────────────────────────────────
    setup_logging()
    logger.info("Creating application [%s env=%s]", settings.APP_NAME, settings.APP_ENV)

    # ── Flask instance ─────────────────────────────────────────────────────────
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY              = settings.SECRET_KEY,
        JSON_SORT_KEYS          = False,
        MAX_CONTENT_LENGTH      = 50 * 1024 * 1024,  # 50 MB CSV upload limit
        PROPAGATE_EXCEPTIONS    = True,
    )

    # ── Database (create tables if not exist) ──────────────────────────────────
    init_db()
    logger.info("Database initialised.")

    # ── Middleware ─────────────────────────────────────────────────────────────
    register_middleware(app)

    # ── Register blueprints ───────────────────────────────────────────────────
    _register_blueprints(app)

    # ── Built-in utility routes ────────────────────────────────────────────────
    _register_utility_routes(app)

    # ── OpenAPI / Swagger UI ──────────────────────────────────────────────────
    _register_openapi(app)

    logger.info(
        "Application ready. API docs: http://%s:%d/docs",
        settings.API_HOST, settings.API_PORT,
    )
    return app


# ── Blueprint registration ────────────────────────────────────────────────────

def _register_blueprints(app: Flask) -> None:
    from backend.routes.auth_routes       import bp as auth_bp
    from backend.routes.prediction_routes import bp as pred_bp
    from backend.routes.customer_routes   import bp as cust_bp
    from backend.routes.analytics_routes  import bp as anlyt_bp
    from backend.routes.monitoring_routes import bp as mon_bp
    from backend.routes.history_routes    import bp as hist_bp

    for bp in (auth_bp, pred_bp, cust_bp, anlyt_bp, mon_bp, hist_bp):
        app.register_blueprint(bp)
        logger.debug("Blueprint registered: %s", bp.name)


# ── Utility routes ─────────────────────────────────────────────────────────────

def _register_utility_routes(app: Flask) -> None:
    """Liveness probe, readiness probe, and version endpoint."""

    @app.get("/health")
    def health():
        """
        Liveness probe.
        ---
        tags: [System]
        responses:
          200:
            description: Service is alive
        """
        return jsonify({"status": "ok", "version": settings.APP_VERSION}), 200

    @app.get("/ready")
    def ready():
        """
        Readiness probe — checks DB connectivity.
        ---
        tags: [System]
        responses:
          200:
            description: Service ready
          503:
            description: Database unreachable
        """
        try:
            from backend.core.database import get_db
            with get_db() as conn:
                conn.execute("SELECT 1")
            return jsonify({"status": "ready", "db": "ok"}), 200
        except Exception as exc:
            logger.error("Readiness check failed: %s", exc)
            return jsonify({"status": "not_ready", "db": "error", "detail": str(exc)}), 503

    @app.get("/api/v1/info")
    def api_info():
        """
        Return API version and ML model metadata.
        ---
        tags: [System]
        responses:
          200:
            description: API info
        """
        from backend.services.ml_service import get_model_metadata
        meta = get_model_metadata()
        return jsonify({
            "success":     True,
            "app_name":    settings.APP_NAME,
            "version":     settings.APP_VERSION,
            "environment": settings.APP_ENV,
            "best_model":  meta.get("best_model", "not_trained"),
            "docs":        "/docs",
            "openapi":     "/openapi.json",
        }), 200


# ── OpenAPI specification + Swagger UI ───────────────────────────────────────

def _register_openapi(app: Flask) -> None:
    """Serve the OpenAPI 3.0 spec and an inline Swagger UI."""

    # Build the spec at registration time (avoids circular imports on /docs)
    spec = _build_openapi_spec()

    @app.get("/openapi.json")
    def openapi_spec():
        """Serve the raw OpenAPI 3.0 JSON specification."""
        return jsonify(spec), 200

    @app.get("/docs")
    def swagger_ui():
        """Serve the interactive Swagger UI (HTML)."""
        html = _swagger_ui_html(
            title=settings.APP_NAME,
            spec_url="/openapi.json",
        )
        from flask import Response
        return Response(html, content_type="text/html")

    @app.get("/redoc")
    def redoc_ui():
        """Serve the ReDoc documentation UI."""
        html = _redoc_html(title=settings.APP_NAME, spec_url="/openapi.json")
        from flask import Response
        return Response(html, content_type="text/html")


def _build_openapi_spec() -> dict:
    """Return the full OpenAPI 3.0.3 specification as a Python dict."""
    return {
        "openapi": "3.0.3",
        "info": {
            "title":       settings.APP_NAME,
            "description": (
                "## Enterprise AI-Powered Customer Churn Intelligence & Retention Platform\n\n"
                "Provides ML-driven churn predictions, SHAP explainability, KMeans customer "
                "segmentation, retention recommendation engine, revenue risk estimation, "
                "and What-If scenario simulation.\n\n"
                "### Authentication\n"
                "All protected endpoints require a `Bearer <token>` header.  "
                "Obtain a token from `POST /api/v1/auth/login`.\n\n"
                "### Quick Start\n"
                "1. `POST /api/v1/auth/signup` — create an account\n"
                "2. `POST /api/v1/auth/login` — get your token\n"
                "3. `POST /api/v1/predict` — score a customer"
            ),
            "version":        settings.APP_VERSION,
            "contact":        {"name": "ChurnShield AI Team", "email": "api@churnplatform.io"},
            "license":        {"name": "MIT"},
        },
        "servers": [
            {"url": "http://localhost:8000",  "description": "Local development"},
            {"url": "https://api.churnplatform.io", "description": "Production"},
        ],
        "tags": [
            {"name": "Auth",            "description": "User registration, login, and token management"},
            {"name": "Predictions",     "description": "Single and batch churn predictions with SHAP"},
            {"name": "Customers",       "description": "Customer CRUD and prediction history"},
            {"name": "Analytics",       "description": "Portfolio KPIs, revenue risk, feature importance"},
            {"name": "Recommendations", "description": "AI-generated retention actions"},
            {"name": "Segments",        "description": "KMeans customer segments"},
            {"name": "System",          "description": "Health and readiness probes"},
        ],
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type":         "http",
                    "scheme":       "bearer",
                    "bearerFormat": "JWT",
                    "description":  "JWT access token from POST /api/v1/auth/login",
                }
            },
            "schemas": _openapi_schemas(),
            "responses": {
                "Unauthorized": {
                    "description": "Authentication required or token expired.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                },
                "Forbidden": {
                    "description": "Insufficient role permissions.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                },
                "NotFound": {
                    "description": "Resource not found.",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                },
            },
        },
        "paths": _openapi_paths(),
    }


def _openapi_schemas() -> dict:
    return {
        "ErrorResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": False},
                "error":   {"type": "string",  "example": "Authentication required."},
                "code":    {"type": "string",  "example": "MISSING_TOKEN"},
            },
        },
        "SuccessResponse": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "message": {"type": "string"},
                "data":    {"type": "object"},
            },
        },
        "CustomerInput": {
            "type":     "object",
            "required": ["tenure", "monthly_charges", "contract", "payment_method"],
            "properties": {
                "tenure":            {"type": "integer",  "minimum": 0,    "example": 12,     "description": "Months with company"},
                "monthly_charges":   {"type": "number",   "minimum": 0,    "example": 79.99},
                "contract":          {"type": "string",   "enum": ["Month-to-month", "One year", "Two year"], "example": "Month-to-month"},
                "payment_method":    {"type": "string",   "enum": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]},
                "inactive_days":     {"type": "integer",  "minimum": 0,    "example": 45},
                "subscription_type": {"type": "string",   "enum": ["Basic", "Standard", "Premium"], "example": "Standard"},
                "internet_service":  {"type": "string",   "enum": ["DSL", "Fiber optic", "No"]},
                "online_security":   {"type": "string",   "example": "No"},
                "tech_support":      {"type": "string",   "example": "No"},
                "gender":            {"type": "string",   "example": "Male"},
                "senior_citizen":    {"type": "integer",  "enum": [0, 1],  "example": 0},
                "partner":           {"type": "string",   "example": "No"},
                "total_charges":     {"type": "number",   "example": 959.88},
                "save_customer":     {"type": "boolean",  "default": True, "description": "Persist to database"},
                "include_shap":      {"type": "boolean",  "default": True, "description": "Include SHAP explanations"},
            },
        },
        "PredictionResponse": {
            "type": "object",
            "properties": {
                "churn_probability":  {"type": "number",  "example": 0.7342},
                "risk_score":         {"type": "integer", "example": 84},
                "risk_category":      {"type": "string",  "example": "High"},
                "churn_label":        {"type": "string",  "example": "Churn"},
                "segment":            {"type": "string",  "example": "Risky"},
                "recommendations":    {"type": "array",   "items": {"$ref": "#/components/schemas/Recommendation"}},
                "revenue_risk":       {"type": "object"},
                "shap_explanation":   {"type": "array",   "items": {"type": "object"}},
                "model_used":         {"type": "string",  "example": "best_model"},
            },
        },
        "Recommendation": {
            "type": "object",
            "properties": {
                "rule_name":          {"type": "string"},
                "action":             {"type": "string"},
                "priority":           {"type": "integer"},
                "estimated_savings":  {"type": "number"},
                "triggered_by":       {"type": "string"},
            },
        },
        "SignupRequest": {
            "type":     "object",
            "required": ["name", "email", "password"],
            "properties": {
                "name":     {"type": "string", "example": "Jane Smith"},
                "email":    {"type": "string", "format": "email", "example": "jane@example.com"},
                "password": {"type": "string", "minLength": 8, "example": "secureP@ss1"},
                "role":     {"type": "string", "enum": ["admin", "analyst", "viewer", "retention"], "default": "viewer"},
            },
        },
        "LoginRequest": {
            "type":     "object",
            "required": ["email", "password"],
            "properties": {
                "email":    {"type": "string", "format": "email"},
                "password": {"type": "string"},
            },
        },
        "TokenResponse": {
            "type": "object",
            "properties": {
                "access_token":  {"type": "string"},
                "refresh_token": {"type": "string"},
                "token_type":    {"type": "string", "example": "bearer"},
                "expires_in":    {"type": "integer", "example": 3600},
            },
        },
    }


def _openapi_paths() -> dict:  # noqa: C901 (complex but flat)
    """Return the paths object for the OpenAPI spec."""
    auth_security = [{"BearerAuth": []}]
    return {
        # ── System ─────────────────────────────────────────────────────────
        "/health": {
            "get": {
                "tags": ["System"], "summary": "Liveness probe",
                "operationId": "health",
                "responses": {"200": {"description": "Service alive"}},
            }
        },
        "/ready": {
            "get": {
                "tags": ["System"], "summary": "Readiness probe (DB check)",
                "operationId": "ready",
                "responses": {
                    "200": {"description": "Ready"},
                    "503": {"description": "Database unreachable"},
                },
            }
        },
        "/api/v1/info": {
            "get": {
                "tags": ["System"], "summary": "API info and model metadata",
                "operationId": "apiInfo",
                "responses": {"200": {"description": "API metadata"}},
            }
        },
        # ── Auth ────────────────────────────────────────────────────────────
        "/api/v1/auth/signup": {
            "post": {
                "tags": ["Auth"], "summary": "Register a new user",
                "operationId": "signup",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/SignupRequest"}}},
                },
                "responses": {
                    "201": {"description": "User created"},
                    "409": {"description": "Email already registered"},
                },
            }
        },
        "/api/v1/auth/login": {
            "post": {
                "tags": ["Auth"], "summary": "Login and receive JWT tokens",
                "operationId": "login",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/LoginRequest"}}},
                },
                "responses": {
                    "200": {
                        "description": "Login successful",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TokenResponse"}}},
                    },
                    "401": {"description": "Invalid credentials"},
                },
            }
        },
        "/api/v1/auth/refresh": {
            "post": {
                "tags": ["Auth"], "summary": "Refresh access token",
                "operationId": "refreshToken",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {
                        "type": "object",
                        "properties": {"refresh_token": {"type": "string"}},
                    }}},
                },
                "responses": {"200": {"description": "New access token"}},
            }
        },
        "/api/v1/auth/me": {
            "get": {
                "tags": ["Auth"], "summary": "Get current user profile",
                "operationId": "me",
                "security": auth_security,
                "responses": {
                    "200": {"description": "User profile"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                },
            }
        },
        "/api/v1/auth/logout": {
            "post": {
                "tags": ["Auth"], "summary": "Logout (client-side token discard)",
                "operationId": "logout",
                "security": auth_security,
                "responses": {"200": {"description": "Logged out"}},
            }
        },
        # ── History ──────────────────────────────────────────────────────────
        "/api/v1/history": {
            "post": {
                "tags": ["History"], "summary": "Record a history entry for the logged-in user",
                "operationId": "createHistoryEntry",
                "security": auth_security,
                "responses": {
                    "201": {"description": "Created"},
                    "400": {"description": "Validation error"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                },
            },
            "get": {
                "tags": ["History"], "summary": "List the logged-in user's own history",
                "operationId": "listHistory",
                "security": auth_security,
                "parameters": [
                    {"in": "query", "name": "page",       "schema": {"type": "integer", "default": 1}},
                    {"in": "query", "name": "page_size",  "schema": {"type": "integer", "default": 20}},
                    {"in": "query", "name": "entry_type",  "schema": {"type": "string"}},
                ],
                "responses": {
                    "200": {"description": "Paginated history entries"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                },
            }
        },
        "/api/v1/history/{history_id}": {
            "get": {
                "tags": ["History"], "summary": "Reopen one history entry",
                "operationId": "getHistoryEntry",
                "security": auth_security,
                "parameters": [
                    {"in": "path", "name": "history_id", "required": True, "schema": {"type": "integer"}},
                ],
                "responses": {
                    "200": {"description": "Full history entry"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"description": "Not found"},
                },
            },
            "delete": {
                "tags": ["History"], "summary": "Delete one of the user's own history entries",
                "operationId": "deleteHistoryEntry",
                "security": auth_security,
                "parameters": [
                    {"in": "path", "name": "history_id", "required": True, "schema": {"type": "integer"}},
                ],
                "responses": {
                    "200": {"description": "Deleted"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"description": "Not found"},
                },
            }
        },
        # ── Predictions ─────────────────────────────────────────────────────
        "/api/v1/predict": {
            "post": {
                "tags": ["Predictions"],
                "summary": "Predict churn for a single customer",
                "description": (
                    "Runs the full prediction pipeline:\n"
                    "1. Feature engineering\n"
                    "2. Churn probability + risk score\n"
                    "3. Customer segmentation\n"
                    "4. Retention recommendations\n"
                    "5. Revenue risk estimation\n"
                    "6. SHAP feature explanations"
                ),
                "operationId": "predict",
                "security": auth_security,
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CustomerInput"}}},
                },
                "responses": {
                    "200": {
                        "description": "Full prediction result",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/PredictionResponse"}}},
                    },
                    "400": {"description": "Validation error"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "503": {"description": "ML model not loaded"},
                },
            }
        },
        "/api/v1/batch_predict": {
            "post": {
                "tags": ["Predictions"],
                "summary": "Bulk churn prediction via CSV upload",
                "description": "Upload a CSV (max 10,000 rows). Returns JSON summary + first 100 rows. Set `export=true` to download a scored CSV.",
                "operationId": "batchPredict",
                "security": auth_security,
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "file":   {"type": "string", "format": "binary"},
                                    "export": {"type": "boolean", "default": False},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Batch summary or scored CSV file"},
                    "400": {"description": "Invalid CSV"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                },
            }
        },
        "/api/v1/batch_predict/export": {
            "post": {
                "tags": ["Predictions"],
                "summary": "Upload CSV → download scored CSV",
                "operationId": "batchPredictExport",
                "security": auth_security,
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {"file": {"type": "string", "format": "binary"}},
                            }
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Scored CSV file download", "content": {"text/csv": {}}},
                    "400": {"description": "Invalid CSV"},
                },
            }
        },
        "/api/v1/predictions": {
            "get": {
                "tags": ["Predictions"], "summary": "List prediction history (paginated)",
                "operationId": "listPredictions",
                "security": auth_security,
                "parameters": [
                    {"name": "page",       "in": "query", "schema": {"type": "integer", "default": 1}},
                    {"name": "page_size",  "in": "query", "schema": {"type": "integer", "default": 20}},
                    {"name": "risk_label", "in": "query", "schema": {"type": "string",  "enum": ["High", "Medium", "Low"]}},
                ],
                "responses": {"200": {"description": "Paginated predictions"}},
            }
        },
        "/api/v1/predictions/{prediction_id}": {
            "get": {
                "tags": ["Predictions"], "summary": "Get a prediction by ID",
                "operationId": "getPrediction",
                "security": auth_security,
                "parameters": [{"name": "prediction_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {
                    "200": {"description": "Prediction detail with SHAP values"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                },
            }
        },
        # ── Customers ────────────────────────────────────────────────────────
        "/api/v1/customers": {
            "get": {
                "tags": ["Customers"], "summary": "List customers (paginated)",
                "operationId": "listCustomers",
                "security": auth_security,
                "parameters": [
                    {"name": "page",      "in": "query", "schema": {"type": "integer", "default": 1}},
                    {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 20}},
                    {"name": "risk",      "in": "query", "schema": {"type": "string",  "enum": ["High", "Medium", "Low"]}},
                ],
                "responses": {"200": {"description": "Paginated customer list"}},
            },
            "post": {
                "tags": ["Customers"], "summary": "Create customer record",
                "operationId": "createCustomer",
                "security": auth_security,
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CustomerInput"}}},
                },
                "responses": {"201": {"description": "Customer created"}},
            },
        },
        "/api/v1/customers/{customer_id}": {
            "get": {
                "tags": ["Customers"], "summary": "Get customer by ID with latest prediction",
                "operationId": "getCustomer",
                "security": auth_security,
                "parameters": [{"name": "customer_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {
                    "200": {"description": "Customer detail"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                },
            },
            "put": {
                "tags": ["Customers"], "summary": "Update customer",
                "operationId": "updateCustomer",
                "security": auth_security,
                "parameters": [{"name": "customer_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"type": "object"}}},
                },
                "responses": {
                    "200": {"description": "Customer updated"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                },
            },
        },
        "/api/v1/customers/{customer_id}/history": {
            "get": {
                "tags": ["Customers"], "summary": "Get prediction history for a customer",
                "operationId": "customerHistory",
                "security": auth_security,
                "parameters": [{"name": "customer_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {
                    "200": {"description": "Prediction history list"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                },
            }
        },
        # ── Analytics ────────────────────────────────────────────────────────
        "/api/v1/analytics": {
            "get": {
                "tags": ["Analytics"], "summary": "Portfolio KPI dashboard",
                "description": "Returns merged ML portfolio stats + live DB counts.",
                "operationId": "analytics",
                "security": auth_security,
                "responses": {"200": {"description": "Dashboard KPIs"}},
            }
        },
        "/api/v1/analytics/revenue": {
            "get": {
                "tags": ["Analytics"], "summary": "Revenue risk breakdown",
                "operationId": "revenueAnalytics",
                "security": auth_security,
                "responses": {"200": {"description": "Expected loss, recoverable, LTV at risk"}},
            }
        },
        "/api/v1/analytics/feature_importance": {
            "get": {
                "tags": ["Analytics"], "summary": "Global SHAP feature importance",
                "operationId": "featureImportance",
                "security": auth_security,
                "parameters": [{"name": "top_n", "in": "query", "schema": {"type": "integer", "default": 15}}],
                "responses": {"200": {"description": "Ranked feature list"}},
            }
        },
        "/api/v1/analytics/model_metrics": {
            "get": {
                "tags": ["Analytics"], "summary": "Training metrics for all models",
                "operationId": "modelMetrics",
                "security": auth_security,
                "responses": {"200": {"description": "Accuracy, F1, AUC per model"}},
            }
        },
        "/api/v1/analytics/whatif_presets": {
            "get": {
                "tags": ["Analytics"], "summary": "List What-If preset scenarios",
                "operationId": "whatifPresets",
                "security": auth_security,
                "responses": {"200": {"description": "Available preset scenarios"}},
            }
        },
        # ── Recommendations ──────────────────────────────────────────────────
        "/api/v1/recommendations": {
            "get": {
                "tags": ["Recommendations"], "summary": "List all retention recommendations",
                "operationId": "listRecommendations",
                "security": auth_security,
                "parameters": [
                    {"name": "page",      "in": "query", "schema": {"type": "integer", "default": 1}},
                    {"name": "page_size", "in": "query", "schema": {"type": "integer", "default": 20}},
                ],
                "responses": {"200": {"description": "Paginated recommendations"}},
            }
        },
        # ── Segments ─────────────────────────────────────────────────────────
        "/api/v1/segments": {
            "get": {
                "tags": ["Segments"], "summary": "Customer segment profiles",
                "description": "Returns KMeans segment statistics: Loyal, Premium, Risky, Inactive.",
                "operationId": "segments",
                "security": auth_security,
                "responses": {"200": {"description": "Segment profiles"}},
            }
        },
        # ── What-If ──────────────────────────────────────────────────────────
        "/api/v1/whatif": {
            "post": {
                "tags": ["Analytics"], "summary": "What-If scenario simulation",
                "description": (
                    "Apply feature changes to a customer record and re-run prediction "
                    "to measure the churn probability delta.\n\n"
                    "Supports custom changes or named preset scenarios."
                ),
                "operationId": "whatif",
                "security": auth_security,
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "customer_data":  {"$ref": "#/components/schemas/CustomerInput"},
                                    "scenario":       {"type": "object", "example": {"Contract": "Two year"}},
                                    "scenario_name":  {"type": "string", "example": "upgrade_to_annual_contract"},
                                },
                            },
                            "example": {
                                "customer_data": {
                                    "tenure": 6, "monthly_charges": 79.99,
                                    "contract": "Month-to-month",
                                    "payment_method": "Electronic check",
                                    "inactive_days": 50,
                                },
                                "scenario_name": "upgrade_to_annual_contract",
                            },
                        }
                    },
                },
                "responses": {
                    "200": {"description": "Scenario simulation result with delta probability"},
                    "400": {"description": "Validation error"},
                    "503": {"description": "Simulation error"},
                },
            }
        },
    }


def _swagger_ui_html(title: str, spec_url: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — API Docs</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui.css">
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0f1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    .topbar {{ background: linear-gradient(135deg, #1a1d27 0%, #0f1117 100%); padding: 16px 24px;
               display: flex; align-items: center; gap: 14px; border-bottom: 1px solid #2a2d3a; }}
    .topbar-logo {{ width: 36px; height: 36px; background: linear-gradient(135deg, #6366f1, #8b5cf6);
                    border-radius: 8px; display: flex; align-items: center; justify-content: center;
                    font-size: 18px; }}
    .topbar-title {{ color: #e2e8f0; font-size: 17px; font-weight: 600; letter-spacing: -0.3px; }}
    .topbar-subtitle {{ color: #64748b; font-size: 12px; margin-top: 1px; }}
    .topbar-badge {{ margin-left: auto; background: #1e3a5f; color: #60a5fa;
                     padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
    #swagger-ui .swagger-ui {{ background: transparent; }}
    #swagger-ui .swagger-ui .info {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px; padding: 24px; margin-bottom: 20px; }}
    #swagger-ui .swagger-ui .info .title {{ color: #e2e8f0 !important; }}
    #swagger-ui .swagger-ui .scheme-container {{ background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; }}
    .swagger-ui .opblock-tag {{ color: #a78bfa !important; font-weight: 600; }}
    .swagger-ui .opblock.opblock-post .opblock-summary {{ border-color: #6366f1; }}
    .swagger-ui .opblock.opblock-get  .opblock-summary {{ border-color: #22d3ee; }}
    .swagger-ui .btn.authorize {{ background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; color: white; }}
    .swagger-ui .topbar {{ display: none; }}
    .swagger-wrapper {{ max-width: 1200px; margin: 0 auto; padding: 0 16px 40px; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="topbar-logo">🧠</div>
    <div>
      <div class="topbar-title">{title}</div>
      <div class="topbar-subtitle">API Documentation</div>
    </div>
    <span class="topbar-badge">v{settings.APP_VERSION}</span>
  </div>
  <div class="swagger-wrapper">
    <div id="swagger-ui"></div>
  </div>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/5.17.14/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({{
      url: "{spec_url}",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
      layout: "BaseLayout",
      deepLinking: true,
      displayRequestDuration: true,
      filter: true,
      tryItOutEnabled: true,
      persistAuthorization: true,
      requestSnippetsEnabled: true,
    }});
  </script>
</body>
</html>"""


def _redoc_html(title: str, spec_url: str) -> str:
    return f"""<!DOCTYPE html>
<html>
  <head>
    <title>{title} — ReDoc</title>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
    <style>body {{ margin: 0; padding: 0; }}</style>
  </head>
  <body>
    <redoc spec-url="{spec_url}" theme='{{"colors":{{"primary":{{"main":"#6366f1"}}}}}}'></redoc>
    <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
  </body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_app()
    logger.info(
        "Starting development server on http://%s:%d",
        settings.API_HOST, settings.API_PORT,
    )
    app.run(
        host=settings.API_HOST,
        port=settings.API_PORT,
        debug=settings.DEBUG,
        use_reloader=settings.API_RELOAD,
    )
