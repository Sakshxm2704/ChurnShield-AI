"""
backend/services/groq_service.py
---------------------------------
Groq AI integration for universal churn prediction.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.core.config import settings

logger = logging.getLogger(__name__)

INDUSTRY_CONTEXT = {
    "telecom":    "telecom/internet service provider. Key factors: contract type, tenure, monthly charges, payment method, service usage.",
    "banking":    "bank or financial institution. Key factors: account age, transaction frequency, balance, loan status, digital engagement.",
    "ecommerce":  "e-commerce/retail platform. Key factors: purchase frequency, last purchase date, average order value, returns, engagement.",
    "saas":       "SaaS/software subscription company. Key factors: feature usage, login frequency, support tickets, contract tier, renewal history.",
    "healthcare": "healthcare/insurance provider. Key factors: appointment frequency, claim history, plan type, engagement with wellness programs.",
    "gaming":     "gaming/entertainment platform. Key factors: session frequency, last active date, spending, subscription tier, social engagement.",
    "general":    "subscription-based business. Key factors: tenure, charges, engagement, contract type, payment behavior.",
}


def groq_predict(customer_data: dict[str, Any], industry: str = "general") -> dict[str, Any]:
    if not is_groq_available():
        raise RuntimeError("Groq API key not configured. Add GROQ_API_KEY to your .env file.")

    industry_ctx = INDUSTRY_CONTEXT.get(industry.lower(), INDUSTRY_CONTEXT["general"])
    prompt       = _build_prediction_prompt(customer_data, industry_ctx)
    raw_response = _call_groq(prompt)
    result       = _parse_groq_response(raw_response, customer_data)

    result["source"]     = "groq_ai"
    result["industry"]   = industry
    result["model_used"] = f"groq/{settings.GROQ_MODEL}"

    logger.info("Groq prediction: industry=%s prob=%.3f risk=%s", industry, result["churn_probability"], result["risk_category"])
    return result


def groq_batch_predict(customers: list[dict[str, Any]], industry: str = "general") -> list[dict[str, Any]]:
    results = []
    for i, customer in enumerate(customers):
        try:
            result = groq_predict(customer, industry)
            result["row_number"] = i + 1
            results.append(result)
        except Exception as e:
            logger.warning("Groq batch failed for row %d: %s", i + 1, e)
            results.append({"row_number": i+1, "error": str(e), "churn_probability": 0.5, "risk_category": "Medium", "risk_score": 50, "churn_label": "Unknown", "source": "groq_ai_error"})
    return results


def is_groq_available() -> bool:
    return bool(settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key_here")


def _build_prediction_prompt(customer_data: dict, industry_ctx: str) -> str:
    data_str = "\n".join([f"  - {k}: {v}" for k, v in customer_data.items() if k not in ("save_customer", "include_shap", "use_groq")])

    return f"""You are an expert customer churn analyst for a {industry_ctx}

Analyze this customer data CAREFULLY and predict churn risk using the scoring rules below.

CUSTOMER DATA:
{data_str}

SCORING RULES — follow these strictly to calculate churn_probability:

Base score: 0.50

TENURE scoring (months with company):
- 0-6 months   → very new customer → add +0.25
- 7-12 months  → new customer      → add +0.15
- 13-24 months → moderate tenure   → add +0.05
- 25-48 months → established       → subtract -0.10
- 49+ months   → very loyal        → subtract -0.20

MONTHLY CHARGES scoring:
- $80 or more → high financial pressure → add +0.15
- $50 to $79  → moderate charges        → add +0.05
- $20 to $49  → affordable, satisfied   → subtract -0.15

INACTIVE DAYS scoring:
- 60+ days   → very disengaged  → add +0.30
- 30-59 days → disengaged       → add +0.15
- 15-29 days → slightly inactive → add +0.05
- 0-14 days  → active, engaged  → subtract -0.25

CONTRACT TYPE scoring:
- Month-to-month → easy to cancel    → add +0.20
- One year       → some commitment   → subtract -0.05
- Two year       → strong commitment → subtract -0.35

PAYMENT METHOD scoring:
- Electronic check             → highest churn risk → add +0.10
- Mailed check                 → moderate risk      → add +0.05
- Bank transfer / Credit card  → loyal customer     → subtract -0.10

CALCULATION:
1. Start at 0.50
2. Add or subtract each applicable score
3. Clamp final value between 0.05 and 0.95
4. Round to 2 decimal places

RISK CATEGORIES:
- churn_probability >= 0.65           → "High",   churn_label = "Churn"
- churn_probability 0.35 to 0.64     → "Medium", churn_label = "Churn"
- churn_probability < 0.35            → "Low",    churn_label = "No Churn"

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "churn_probability": <calculated float>,
  "risk_category": "<High|Medium|Low>",
  "churn_label": "<Churn|No Churn>",
  "risk_score": <round(churn_probability * 100)>,
  "segment": "<Risky|Inactive|Premium|Loyal>",
  "explanation": "<2-3 sentences explaining key factors>",
  "top_risk_factors": [
    {{"factor": "<factor>", "impact": "<High|Medium|Low>", "detail": "<specific value>"}},
    {{"factor": "<factor>", "impact": "<High|Medium|Low>", "detail": "<specific value>"}},
    {{"factor": "<factor>", "impact": "<High|Medium|Low>", "detail": "<specific value>"}}
  ],
  "recommendations": [
    {{"action": "<specific retention action>", "priority": 1, "estimated_savings": <monthly_charges * churn_probability>, "triggered_by": "<reason>"}},
    {{"action": "<second retention action>",   "priority": 2, "estimated_savings": <monthly_charges * churn_probability * 0.5>, "triggered_by": "<reason>"}}
  ],
  "revenue_risk": {{
    "expected_monthly_loss": <monthly_charges * churn_probability>,
    "expected_annual_loss":  <monthly_charges * churn_probability * 12>,
    "ltv_at_risk":           <monthly_charges * churn_probability * 24 * 0.65>
  }}
}}"""


def _call_groq(prompt: str) -> str:
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests library not installed. Run: pip3 install requests")

    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model":       settings.GROQ_MODEL or "llama-3.3-70b-versatile",
        "messages":    [
            {"role": "system", "content": "You are a precise churn prediction AI. Follow scoring rules exactly. Respond with valid JSON only — no markdown, no text outside JSON."},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens":  1200,
    }

    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 401: raise RuntimeError("Invalid Groq API key. Check GROQ_API_KEY in .env")
        if resp.status_code == 429: raise RuntimeError("Groq rate limit exceeded. Please wait and try again.")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        logger.error("Groq HTTP error %s: %s", resp.status_code, resp.text)
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        logger.error("Groq request failed: %s", e)
        raise RuntimeError(f"Groq API call failed: {e}")


def _parse_groq_response(raw: str, original_data: dict) -> dict[str, Any]:
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"): clean = clean[4:]
        clean = clean.strip()

        parsed     = json.loads(clean)
        prob       = max(0.05, min(0.95, float(parsed.get("churn_probability", 0.5))))
        risk_score = max(0, min(100, int(round(prob * 100))))

        if prob >= 0.65:   risk_cat = "High"
        elif prob >= 0.35: risk_cat = "Medium"
        else:              risk_cat = "Low"

        monthly  = float(original_data.get("monthly_charges") or original_data.get("MonthlyCharges") or 50)
        rev_risk = parsed.get("revenue_risk") or {}
        if not rev_risk.get("expected_annual_loss"):
            rev_risk = {
                "expected_monthly_loss": round(prob * monthly, 2),
                "expected_annual_loss":  round(prob * monthly * 12, 2),
                "ltv_at_risk":           round(prob * monthly * 24 * 0.65, 2),
            }

        return {
            "churn_probability": round(prob, 4),
            "risk_score":        risk_score,
            "risk_category":     risk_cat,
            "churn_label":       "Churn" if prob >= 0.5 else "No Churn",
            "segment":           parsed.get("segment", "Risky" if prob >= 0.65 else "Loyal"),
            "explanation":       parsed.get("explanation", ""),
            "shap_explanation":  parsed.get("top_risk_factors", []),
            "recommendations":   parsed.get("recommendations", []),
            "revenue_risk":      rev_risk,
        }

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to parse Groq response: %s", e)
        monthly = float(original_data.get("monthly_charges") or original_data.get("MonthlyCharges") or 50)
        return {
            "churn_probability": 0.5, "risk_score": 50, "risk_category": "Medium",
            "churn_label": "Unknown", "segment": "Unknown",
            "explanation": "Could not parse AI response. Please try again.",
            "shap_explanation": [], "recommendations": [],
            "revenue_risk": {"expected_monthly_loss": round(0.5*monthly,2), "expected_annual_loss": round(0.5*monthly*12,2), "ltv_at_risk": round(0.5*monthly*24*0.65,2)},
        }


def get_supported_industries() -> list[dict]:
    return [
        {"id": "telecom",    "name": "Telecom / Internet",    "icon": "📡"},
        {"id": "banking",    "name": "Banking / Finance",      "icon": "🏦"},
        {"id": "ecommerce",  "name": "E-Commerce / Retail",    "icon": "🛒"},
        {"id": "saas",       "name": "SaaS / Software",        "icon": "💻"},
        {"id": "healthcare", "name": "Healthcare / Insurance", "icon": "🏥"},
        {"id": "gaming",     "name": "Gaming / Entertainment", "icon": "🎮"},
        {"id": "general",    "name": "General / Other",        "icon": "🏢"},
    ]