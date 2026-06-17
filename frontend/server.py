"""
frontend/server.py
------------------
Flask development server that serves the Churn Intelligence Platform dashboard.

Features
--------
- Serves dashboard.html at /
- Transparent proxy for /api/* → backend API (avoids CORS in dev)
- /health endpoint for container readiness
- Demo-mode fallback with realistic mock data when backend is offline
- PDF report generation at /report/generate

Run
---
    # Development (auto-reload)
    python -m frontend.server

    # With custom API target
    API_URL=http://api.prod:8000 python -m frontend.server
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, Response, jsonify, render_template_string, request, send_file

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("frontend")

# ── Config ────────────────────────────────────────────────────────────────────
API_URL      = os.getenv("API_URL", "http://localhost:8000")
FRONTEND_DIR = Path(__file__).resolve().parent
DASHBOARD    = FRONTEND_DIR / "dashboard.html"
PORT         = int(os.getenv("FRONTEND_PORT", "8501"))
DEBUG        = os.getenv("APP_ENV", "development") == "development"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def index():
    """Serve the main dashboard HTML."""
    if not DASHBOARD.exists():
        return "<h2>Dashboard not found — run the build step first.</h2>", 404
    return DASHBOARD.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "churn-frontend", "version": "1.0.0"}), 200


# ══════════════════════════════════════════════════════════════════════════════
# API Proxy  (dev only — avoids CORS; in prod use nginx upstream)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def proxy_api(path: str):
    """
    Transparent reverse proxy: /api/* → backend API_URL/api/*.
    Falls back to demo data when the backend is unreachable.
    """
    target = f"{API_URL}/api/{path}"
    qs     = request.query_string.decode()
    if qs:
        target += "?" + qs

    # Forward headers (exclude hop-by-hop)
    skip_headers = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {
        k: v for k, v in request.headers
        if k.lower() not in skip_headers
    }

    try:
        body = request.get_data() or None
        req  = urllib.request.Request(target, data=body, headers=headers, method=request.method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body    = resp.read()
            status_code  = resp.status
            content_type = resp.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:
        resp_body    = e.read()
        status_code  = e.code
        content_type = e.headers.get("Content-Type", "application/json")
    except (urllib.error.URLError, OSError) as exc:
        # Backend unreachable — try demo fallback
        logger.warning("Backend unreachable (%s), serving demo data for /%s", exc, path)
        demo = _demo_fallback(path, request.method, request.query_string.decode())
        if demo is not None:
            return jsonify(demo), 200
        return jsonify({"success": False, "error": f"Backend unreachable: {exc}", "code": "BACKEND_DOWN"}), 503

    return Response(resp_body, status=status_code, content_type=content_type)


# ══════════════════════════════════════════════════════════════════════════════
# Demo fallback data  (realistic mock when backend is offline)
# ══════════════════════════════════════════════════════════════════════════════

def _demo_fallback(path: str, method: str, qs: str) -> dict | None:
    """Return demo JSON for common endpoints when the backend is unreachable."""
    p = path.lstrip("/")

    if p == "v1/analytics" and method == "GET":
        return {
            "success": True, "message": "Demo mode",
            "data": {
                "total_customers": 7042, "high_risk": 1869, "medium_risk": 1824,
                "low_risk": 3349, "avg_churn_probability": 0.2654,
                "churn_rate": 0.2654, "total_monthly_revenue": 456244.0,
                "expected_monthly_loss": 120987.0, "expected_annual_loss": 1451844.0,
                "api_calls_today": 0, "predictions_today": 0,
                "segment_distribution": {"Loyal": 1890, "Premium": 1901, "Inactive": 2043, "Risky": 1208},
                "_demo_mode": True,
            }
        }

    if p == "v1/analytics/revenue" and method == "GET":
        return {
            "success": True, "data": {
                "total_customers": 7042, "high_risk_count": 1869,
                "medium_risk_count": 1824, "low_risk_count": 3349,
                "total_monthly_revenue": 456244.0,
                "expected_monthly_loss": 120987.0,
                "expected_annual_loss": 1451844.0,
                "worst_case_annual_loss": 2678648.0,
                "recoverable_revenue_annual": 145184.0,
                "net_revenue_at_risk_annual": 1306660.0,
                "avg_churn_probability": 0.2654,
                "total_ltv_at_risk": 3264649.0,
                "revenue_by_segment": {"Loyal": 245890, "Premium": 612340, "Risky": 389440, "Inactive": 204174},
                "churn_rate_by_segment": {"Loyal": 0.08, "Premium": 0.22, "Risky": 0.71, "Inactive": 0.65},
            }
        }

    if p == "v1/analytics/model_metrics" and method == "GET":
        return {
            "success": True, "data": {
                "best_model": "logistic_regression",
                "auc_scores": {"logistic_regression": 0.7701, "random_forest": 0.7484, "gradient_boosting": 0.7362},
                "models": {
                    "logistic_regression": {"accuracy": 0.684, "precision": 0.561, "recall": 0.757, "f1": 0.644, "roc_auc": 0.7701},
                    "random_forest":       {"accuracy": 0.688, "precision": 0.566, "recall": 0.753, "f1": 0.646, "roc_auc": 0.7484},
                    "gradient_boosting":   {"accuracy": 0.681, "precision": 0.584, "recall": 0.547, "f1": 0.565, "roc_auc": 0.7362},
                },
                "train_size": 5600, "test_size": 1400, "churn_rate": 0.2654,
            }
        }

    if p == "v1/analytics/feature_importance" and method == "GET":
        feats = [
            {"feature": "Contract_Two year",                   "importance": 0.891},
            {"feature": "Contract_One year",                   "importance": 0.764},
            {"feature": "PaymentMethod_Electronic check",       "importance": 0.623},
            {"feature": "tenure",                              "importance": 0.587},
            {"feature": "InternetService_Fiber optic",         "importance": 0.521},
            {"feature": "MonthlyCharges",                      "importance": 0.498},
            {"feature": "InactiveDays",                        "importance": 0.465},
            {"feature": "OnlineSecurity_No",                   "importance": 0.412},
            {"feature": "TechSupport_No",                      "importance": 0.378},
            {"feature": "charge_per_tenure",                   "importance": 0.334},
            {"feature": "TotalCharges",                        "importance": 0.298},
            {"feature": "engagement_score",                    "importance": 0.276},
            {"feature": "PaperlessBilling_Yes",                "importance": 0.243},
            {"feature": "SeniorCitizen",                       "importance": 0.198},
            {"feature": "StreamingTV_Yes",                     "importance": 0.167},
        ]
        return {"success": True, "data": {"feature_importance": feats, "count": len(feats)}}

    if p == "v1/segments" and method == "GET":
        return {
            "success": True, "data": {
                "segments": {
                    "Loyal":    {"count": 1890, "pct_of_total": 26.8, "avg_churn_probability": 0.08, "avg_MonthlyCharges": 42.3, "avg_tenure": 48.2, "churn_rate": 8.2},
                    "Premium":  {"count": 1901, "pct_of_total": 27.0, "avg_churn_probability": 0.22, "avg_MonthlyCharges": 89.5, "avg_tenure": 28.6, "churn_rate": 22.4},
                    "Risky":    {"count": 1208, "pct_of_total": 17.2, "avg_churn_probability": 0.71, "avg_MonthlyCharges": 78.2, "avg_tenure": 9.4,  "churn_rate": 71.0},
                    "Inactive": {"count": 2043, "pct_of_total": 29.0, "avg_churn_probability": 0.65, "avg_MonthlyCharges": 61.7, "avg_tenure": 18.3, "churn_rate": 65.1},
                },
                "source": "demo", "total": 7042,
            }
        }

    if p == "v1/customers" and method == "GET":
        customers = [
            {"customer_id": i+1, "gender": ["Male","Female"][i%2],
             "tenure": 3+i*4, "monthly_charges": round(25+i*8.5, 2),
             "contract": ["Month-to-month","One year","Two year"][i%3],
             "payment_method": ["Electronic check","Bank transfer (automatic)","Credit card (automatic)"][i%3],
             "inactive_days": i*7, "subscription_type": ["Basic","Standard","Premium"][i%3]}
            for i in range(15)
        ]
        return {"success": True, "data": {"items": customers, "total": 7042, "page": 1, "page_size": 15, "pages": 470}}

    if p == "v1/recommendations" and method == "GET":
        recs = [
            {"recommendation_id": i+1, "customer_id": i+1,
             "rule_name": ["month_to_month_upgrade","high_monthly_charges_discount","high_inactivity_reengagement"][i%3],
             "recommended_action": ["Offer annual contract upgrade with 1 free month",
                                    "Apply 15% loyalty discount for 3 billing cycles",
                                    "Send personalised re-engagement campaign"][i%3],
             "estimated_savings": round(45.0 + i*12.3, 2), "priority": (i%3)+1, "is_completed": i%4==0}
            for i in range(12)
        ]
        return {"success": True, "data": {"items": recs, "total": 1843, "page": 1, "page_size": 20, "pages": 93}}

    if p == "v1/analytics/whatif_presets" and method == "GET":
        return {
            "success": True, "data": {
                "presets": [
                    {"name": "upgrade_to_annual_contract",  "description": "Upgrade from Month-to-month to One year contract"},
                    {"name": "upgrade_to_two_year_contract","description": "Upgrade to Two year contract"},
                    {"name": "switch_to_autopay",           "description": "Switch payment to bank transfer auto-pay"},
                    {"name": "reduce_charges_10pct",        "description": "Apply 10% discount on monthly charges"},
                    {"name": "reduce_charges_15pct",        "description": "Apply 15% loyalty discount"},
                    {"name": "add_online_security",         "description": "Add Online Security subscription"},
                    {"name": "reduce_inactivity",           "description": "Re-engagement (InactiveDays → 5)"},
                    {"name": "full_retention_bundle",       "description": "Contract upgrade + autopay + security (combined)"},
                ],
                "count": 8,
            }
        }

    if p == "v1/predictions" and method == "GET":
        preds = [
            {"prediction_id": i+1, "customer_id": i+1,
             "churn_probability": round(0.3+i*0.06, 4),
             "risk_score": 30+i*7,
             "prediction_label": ["Low","Medium","High"][min(i//3,2)],
             "churn_label": "Churn" if i > 6 else "No Churn",
             "model_used": "logistic_regression",
             "created_at": datetime.now(timezone.utc).isoformat()}
            for i in range(10)
        ]
        return {"success": True, "data": {"items": preds, "total": 0, "page": 1, "page_size": 20, "pages": 0}}

    if p in ("v1/auth/login", "v1/auth/signup"):
        # Auth endpoints should not be proxied in demo mode — let them fail naturally
        return None

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PDF Report Generation
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/report/generate")
def generate_report():
    """
    Generate a PDF churn analysis report from provided data.
    Receives JSON payload, returns PDF bytes.
    
    This is a server-side fallback; the dashboard also generates PDF client-side
    via jsPDF. This endpoint handles richer multi-page reports.
    """
    try:
        payload = request.get_json(silent=True) or {}
        report_type = payload.get("type", "portfolio")

        # Build HTML report and return as downloadable file
        html = _build_report_html(payload, report_type)
        
        # Return HTML for browser printing as PDF (no wkhtmltopdf dependency needed)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return Response(
            html,
            status=200,
            headers={
                "Content-Type": "text/html",
                "Content-Disposition": f'inline; filename="churn_report_{timestamp}.html"',
            },
        )
    except Exception as exc:
        logger.exception("Report generation failed")
        return jsonify({"success": False, "error": str(exc)}), 500


def _build_report_html(data: dict, report_type: str) -> str:
    """Build a print-ready HTML report."""
    now   = datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M UTC")
    title = "Customer Churn Intelligence Report"

    if report_type == "prediction":
        d   = data.get("prediction", {})
        cust = data.get("customer", {})
        sections = f"""
        <div class="section">
          <h2>Prediction Summary</h2>
          <table>
            <tr><td>Churn Probability</td><td class="val {'red' if d.get('churn_probability',0)>0.65 else 'amber' if d.get('churn_probability',0)>0.35 else 'green'}">{round((d.get('churn_probability',0))*100,1)}%</td></tr>
            <tr><td>Risk Score</td><td class="val">{d.get('risk_score','—')}/100</td></tr>
            <tr><td>Risk Category</td><td class="val">{d.get('risk_category','—')}</td></tr>
            <tr><td>Churn Label</td><td class="val">{d.get('churn_label','—')}</td></tr>
            <tr><td>Segment</td><td class="val">{d.get('segment','—')}</td></tr>
            <tr><td>Model Used</td><td class="val">{d.get('model_used','—')}</td></tr>
          </table>
        </div>
        <div class="section">
          <h2>Customer Profile</h2>
          <table>
            {''.join(f"<tr><td>{k.replace('_',' ').title()}</td><td class='val'>{v}</td></tr>" for k,v in cust.items() if k != 'save_customer')}
          </table>
        </div>
        """
        recs = d.get("recommendations", [])
        if recs:
            sections += f"""
        <div class="section">
          <h2>Retention Recommendations ({len(recs)})</h2>
          {''.join(f'<div class="rec"><strong>Priority {r.get("priority",1)}:</strong> {r.get("action",r.get("recommended_action",""))} <em>(Est. savings: ${r.get("estimated_savings",0):.0f}/yr)</em></div>' for r in recs)}
        </div>"""
    else:
        analytics = data.get("analytics", {})
        sections = f"""
        <div class="section">
          <h2>Portfolio Overview</h2>
          <div class="kpi-grid">
            <div class="kpi"><div class="kpi-val">{analytics.get('total_customers','—')}</div><div class="kpi-lbl">Total Customers</div></div>
            <div class="kpi red"><div class="kpi-val">{analytics.get('high_risk','—')}</div><div class="kpi-lbl">High Risk</div></div>
            <div class="kpi amber"><div class="kpi-val">{round((analytics.get('churn_rate',0))*100,1)}%</div><div class="kpi-lbl">Churn Rate</div></div>
            <div class="kpi red"><div class="kpi-val">${analytics.get('expected_annual_loss',0):,.0f}</div><div class="kpi-lbl">Annual Revenue at Risk</div></div>
          </div>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    @page {{ size: A4; margin: 20mm; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #1a202c; background: white; }}
    .header {{ background: linear-gradient(135deg, #0d1117 0%, #1a1d27 100%); color: white; padding: 32px 40px; margin-bottom: 32px; }}
    .header h1 {{ font-size: 22px; font-weight: 700; margin-bottom: 4px; }}
    .header .sub {{ font-size: 12px; opacity: 0.7; }}
    .section {{ margin: 0 40px 28px; }}
    .section h2 {{ font-size: 15px; font-weight: 600; color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #f0f4f8; }}
    td:first-child {{ color: #64748b; width: 40%; }}
    .val {{ font-weight: 600; color: #1a202c; }}
    .red {{ color: #e53e3e !important; }}
    .amber {{ color: #dd6b20 !important; }}
    .green {{ color: #2f855a !important; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .kpi {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }}
    .kpi-val {{ font-size: 20px; font-weight: 700; margin-bottom: 4px; color: #1a202c; }}
    .kpi.red .kpi-val {{ color: #e53e3e; }}
    .kpi.amber .kpi-val {{ color: #dd6b20; }}
    .kpi-lbl {{ font-size: 11px; color: #64748b; }}
    .rec {{ background: #f8fafc; border-left: 3px solid #6366f1; padding: 10px 14px; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 12px; }}
    .footer {{ margin: 40px 40px 0; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 10px; color: #94a3b8; display: flex; justify-content: space-between; }}
    @media print {{ .header {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }} }}
  </style>
  <script>window.onload = () => window.print();</script>
</head>
<body>
  <div class="header">
    <h1>🧠 {title}</h1>
    <div class="sub">Generated: {now} &nbsp;|&nbsp; Churn Intelligence Platform v1.0.0</div>
  </div>
  {sections}
  <div class="footer">
    <span>Churn Intelligence Platform</span>
    <span>Confidential — Internal Use Only</span>
    <span>{now}</span>
  </div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   Churn Intelligence Platform — Dashboard Server             ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("  Dashboard : http://localhost:%d", PORT)
    logger.info("  Backend   : %s", API_URL)
    logger.info("  Mode      : %s", "development" if DEBUG else "production")
    logger.info("─" * 64)

    app.run(host="0.0.0.0", port=PORT, debug=DEBUG, use_reloader=DEBUG)
