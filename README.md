<div align="center">

# 🧠 Churn Intelligence Platform

### Enterprise AI-Powered Customer Churn Intelligence & Retention System

[![Python](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ed?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Predict customer churn · Segment at-risk customers · Generate AI retention actions · Monitor revenue risk**

[Demo](#demo) · [Quick Start](#quick-start) · [API Docs](#api-documentation) · [Architecture](#architecture) · [Deploy](#deployment)

</div>

---

## 📸 Screenshots

> _Screenshots show the live running platform_

| Dashboard KPIs | Single Prediction | Batch Prediction |
|:-:|:-:|:-:|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Prediction](docs/screenshots/prediction.png) | ![Batch](docs/screenshots/batch.png) |

| Analytics Charts | SHAP Explainability | What-If Simulator |
|:-:|:-:|:-:|
| ![Analytics](docs/screenshots/analytics.png) | ![SHAP](docs/screenshots/shap.png) | ![WhatIf](docs/screenshots/whatif.png) |

> **To capture screenshots:** Run the platform, open `http://localhost:8501`, and save screenshots to `docs/screenshots/`

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- **3 trained models**: Logistic Regression, Random Forest, XGBoost (GradientBoosting)
- **Auto best-model selection** based on ROC-AUC score
- **32 features** after engineering (tenure, charges, contract, inactivity, engagement score, etc.)
- **SHAP explainability** — global feature importance + per-prediction waterfall charts
- **KMeans segmentation** — Loyal, Premium, Risky, Inactive clusters

### 🎯 Prediction & Risk Scoring
- Churn probability `[0.0 – 1.0]` with calibrated confidence
- Risk score `[0 – 100]` normalized for display
- Risk categories: **High** (≥65%) · **Medium** (35-65%) · **Low** (<35%)
- Batch CSV prediction (up to 10,000 rows) with downloadable scored output

### 🔔 Automated Alert System
- Auto-triggers on every high-risk prediction (prob ≥ 0.65)
- Medium-risk alerts when inactivity ≥ 30 days
- Alert log persisted to database with email status tracking
- HTML email templates with risk details + top recommendation

### 💡 Retention Intelligence
- **12 business rules** engine: contract upgrades, discounts, re-engagement, bundles
- Segment-affinity priority boosting
- **Revenue loss estimation**: DCF-based LTV, expected annual loss, recoverable revenue
- **What-If simulator**: 9 preset scenarios + custom feature changes

### 📊 Monitoring Dashboard
- Real-time API call counters (rolling 15-min window)
- Active user tracking
- 7-day churn trend chart
- 24-hour API activity heatmap
- Cache hit/miss statistics

### 🏗️ Production Architecture
- JWT authentication with role-based access control (admin/analyst/retention/viewer)
- Two-tier LRU cache (in-memory + SQLite persistence)
- Structured JSON logging to rotating files
- OpenAPI 3.0 spec + Swagger UI at `/docs`
- Docker Compose with PostgreSQL, Redis, pgAdmin

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         Dashboard (Flask + HTML/CSS/JS + Plotly)            │
│                      Port 8501                              │
└─────────────────────┬───────────────────────────────────────┘
                      │  HTTP REST + JWT
┌─────────────────────▼───────────────────────────────────────┐
│              Flask REST API  (Port 8000)                     │
│  /auth · /predict · /customers · /analytics · /monitoring   │
│  JWT Middleware · CORS · Request Logging · Cache Layer       │
└──────┬──────────────┬──────────────────────────┬────────────┘
       │              │                          │
┌──────▼──────┐ ┌─────▼────────┐ ┌──────────────▼──────────┐
│  ML Service │ │  DB Service  │ │   Alert + Monitor Svc   │
│  predict()  │ │  SQLite/PG   │ │   alert_log · SMTP      │
│  SHAP       │ │  20+ indexes │ │   rolling counters      │
└──────┬──────┘ └──────────────┘ └─────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────┐
│                    ML Pipeline                               │
│  Preprocessing → 3 Models → Segmentation → Revenue → SHAP  │
└─────────────────────────────────────────────────────────────┘
```

See [deploy/architecture.md](deploy/architecture.md) for the full layered diagram with data flows.

---

## 📁 Project Structure

```
churn-platform/
├── backend/                    # Flask REST API
│   ├── app.py                  # App factory + OpenAPI spec
│   ├── auth/                   # JWT, password hashing, auth decorators
│   ├── core/                   # Config, DB engine, cache, logging
│   ├── middleware/             # CORS, request timing, error handlers
│   ├── models/                 # Pydantic-style validation schemas
│   ├── routes/                 # auth · prediction · customer · analytics · monitoring
│   └── services/               # ml_service · db_service · monitoring
│
├── ml/                         # Machine learning pipeline
│   ├── pipeline.py             # Master orchestrator (11 steps)
│   ├── features/               # Feature engineering + preprocessing
│   ├── models/                 # Trainer + predictor
│   └── evaluation/             # Evaluator + SHAP explainer
│
├── analytics/                  # Analytics engines
│   ├── segmentation.py         # KMeans customer segmentation
│   ├── revenue_estimator.py    # DCF-based revenue risk
│   └── whatif_simulator.py     # Scenario simulation engine
│
├── services/
│   ├── alerts/                 # Alert generation + email dispatch
│   └── retention/              # Retention recommendation engine
│
├── frontend/
│   ├── server.py               # Flask dashboard server + API proxy
│   ├── dashboard.html          # Full single-page enterprise UI
│   └── utils/                  # API client helpers
│
├── data/
│   ├── generate_dataset.py     # Synthetic Telco dataset generator
│   └── seed_demo.py            # Demo data seeder (50 customers)
│
├── database/
│   ├── models.py               # SQLAlchemy ORM models
│   └── init_db.py              # Schema + seed CLI
│
├── deploy/
│   ├── architecture.md         # Full architecture diagrams
│   └── render.yaml             # Render.com blueprint
│
├── docker/
│   ├── docker-compose.yml      # Full production stack
│   ├── Dockerfile.backend      # Backend container
│   └── Dockerfile.frontend     # Frontend container
│
├── streamlit_app.py            # Streamlit Cloud entry point
├── .env.example                # Environment variable template
├── requirements.txt            # Full Python dependencies
└── README.md                   # This file
```

---

## ⚡ Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.11 | Runtime |
| pip | latest | Package manager |
| Docker (optional) | ≥ 24.0 | Containerised deployment |

### 1 — Clone & Setup

```bash
git clone https://github.com/your-username/churn-platform.git
cd churn-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install flask werkzeug PyJWT scikit-learn joblib numpy pandas python-dotenv
```

### 2 — Configure Environment

```bash
cp .env.example .env
# Edit .env — minimum required:
#   SECRET_KEY=<generate with: python -c "import secrets; print(secrets.token_hex(32))">
```

### 3 — Train the ML Pipeline

```bash
# Generates data, trains 3 models, segments customers, saves all artefacts
python -m ml.pipeline --no-shap --rows 5000
# Full pipeline with SHAP (slower but complete):
# python -m ml.pipeline --rows 7000
```

### 4 — Seed Demo Data

```bash
# Creates 3 users + 50 customers + predictions + alerts
python -m data.seed_demo
```

### 5 — Start the API

```bash
python -m backend.app
# → API:       http://localhost:8000
# → Swagger:   http://localhost:8000/docs
# → Health:    http://localhost:8000/health
```

### 6 — Start the Dashboard

```bash
python -m frontend.server
# → Dashboard: http://localhost:8501
# → Login:     admin@demo.io / Demo1234!
```

---

## 🐳 Docker Compose

```bash
# Copy and configure environment
cp .env.example .env
# Set SECRET_KEY in .env

# Start all services (PostgreSQL + Redis + Backend + Frontend)
docker compose -f docker/docker-compose.yml up -d

# Train models inside container
docker compose -f docker/docker-compose.yml exec backend \
  python -m ml.pipeline --no-shap --rows 5000

# Seed demo data
docker compose -f docker/docker-compose.yml exec backend \
  python -m data.seed_demo

# View logs
docker compose -f docker/docker-compose.yml logs -f backend

# Open pgAdmin (dev profile)
docker compose -f docker/docker-compose.yml --profile dev up -d pgadmin
# → http://localhost:5050  (admin@churnplatform.io / admin1234)

# Stop everything
docker compose -f docker/docker-compose.yml down
```

**Service URLs:**

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Health | http://localhost:8000/health |
| pgAdmin | http://localhost:5050 |

---

## 📖 API Documentation

Interactive docs available at **`http://localhost:8000/docs`** after starting the server.

### Authentication

```bash
# 1. Create account
curl -X POST http://localhost:8000/api/v1/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"name":"Jane Smith","email":"jane@example.com","password":"SecureP@ss1","role":"analyst"}'

# 2. Login → get token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"jane@example.com","password":"SecureP@ss1"}'
# Response: {"data": {"access_token": "eyJ...", ...}}

# 3. Use token in all subsequent requests
TOKEN="eyJ..."
```

### Core Endpoints

#### `POST /api/v1/predict` — Single Customer Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 6,
    "monthly_charges": 89.99,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "inactive_days": 45,
    "subscription_type": "Premium",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "save_customer": true
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "churn_probability": 0.7842,
    "risk_score": 91,
    "risk_category": "High",
    "churn_label": "Churn",
    "segment": "Risky",
    "model_used": "logistic_regression",
    "recommendations": [
      {
        "rule_name": "low_tenure_onboarding",
        "action": "Assign a dedicated Customer Success Manager...",
        "priority": 1,
        "estimated_savings": 162.42
      }
    ],
    "revenue_risk": {
      "expected_annual_loss": 841.67,
      "ltv_at_risk": 1243.88,
      "recoverable_revenue": 180.0
    },
    "alert": {"triggered": true, "risk_category": "High"},
    "customer_id": 42,
    "prediction_id": 87
  }
}
```

#### `POST /api/v1/batch_predict` — Bulk CSV Prediction

```bash
# Upload CSV, get JSON summary + optional scored CSV download
curl -X POST http://localhost:8000/api/v1/batch_predict \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@customers.csv"

# Download scored CSV
curl -X POST http://localhost:8000/api/v1/batch_predict/export \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@customers.csv" -o scored_customers.csv
```

#### `GET /api/v1/analytics` — Portfolio KPIs

```bash
curl http://localhost:8000/api/v1/analytics \
  -H "Authorization: Bearer $TOKEN"
```

#### `POST /api/v1/whatif` — Scenario Simulation

```bash
curl -X POST http://localhost:8000/api/v1/whatif \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_data": {"tenure": 6, "monthly_charges": 89.99, "contract": "Month-to-month", ...},
    "scenario_name": "upgrade_to_annual_contract"
  }'
# Returns: base_prob=0.78 → new_prob=0.44 → improvement=43.6%
```

#### `GET /api/v1/monitoring` — Platform Metrics

```bash
curl http://localhost:8000/api/v1/monitoring \
  -H "Authorization: Bearer $TOKEN"
```

### All Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/auth/signup` | — | Register user |
| POST | `/api/v1/auth/login` | — | Login, get JWT |
| GET | `/api/v1/auth/me` | ✓ | Current user profile |
| POST | `/api/v1/predict` | ✓ | Single prediction |
| POST | `/api/v1/batch_predict` | ✓ | CSV bulk prediction |
| GET | `/api/v1/predictions` | ✓ | Prediction history |
| GET | `/api/v1/customers` | ✓ | List customers |
| GET | `/api/v1/customers/{id}` | ✓ | Customer + latest prediction |
| GET | `/api/v1/analytics` | ✓ | Portfolio KPIs |
| GET | `/api/v1/analytics/revenue` | ✓ | Revenue risk breakdown |
| GET | `/api/v1/analytics/feature_importance` | ✓ | SHAP feature ranking |
| GET | `/api/v1/analytics/model_metrics` | ✓ | Model training metrics |
| POST | `/api/v1/whatif` | ✓ | What-If simulation |
| GET | `/api/v1/recommendations` | ✓ | Retention recommendations |
| GET | `/api/v1/segments` | ✓ | Customer segments |
| GET | `/api/v1/monitoring` | ✓ | Real-time platform metrics |
| GET | `/api/v1/monitoring/alerts` | ✓ | Alert history |
| GET | `/api/v1/monitoring/cache` | ✓ | Cache statistics |
| GET | `/health` | — | Liveness probe |
| GET | `/ready` | — | Readiness probe |
| GET | `/docs` | — | Swagger UI |

---

## 🚀 Deployment

### Option A: Render.com (Recommended — Free Tier)

Render hosts both the API and dashboard as separate web services.

1. **Fork this repository** to your GitHub account

2. **Create a Render account** at [render.com](https://render.com)

3. **New → Blueprint** → connect your fork

4. Render reads `deploy/render.yaml` automatically

5. Set secret environment variables in the Render dashboard:
   - `SECRET_KEY` — generate with `python -c "import secrets; print(secrets.token_hex(32))"`
   - `SMTP_USER` + `SMTP_PASSWORD` (optional, for email alerts)
   - `ALERT_EMAIL_RECIPIENT` (optional)

6. After deploy, set `FRONTEND_URL` in the API service to your dashboard URL

**Build command** (auto-runs on Render):
```bash
pip install flask werkzeug PyJWT scikit-learn joblib numpy pandas python-dotenv
python -m ml.pipeline --no-shap --rows 5000
python -m data.seed_demo
```

### Option B: Streamlit Cloud

1. Fork this repo to GitHub

2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → New app

3. Set:
   - **Repository**: your fork
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.12

4. Add secrets in Streamlit Cloud Settings:
   ```toml
   API_URL = "https://your-render-api.onrender.com"
   ```

5. The app embeds the dashboard via iframe + proxies API calls

### Option C: Docker Compose (Self-hosted)

```bash
# Production-ready single-command deploy
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml exec backend python -m ml.pipeline --no-shap
docker compose -f docker/docker-compose.yml exec backend python -m data.seed_demo
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | **required** | JWT signing secret (≥32 chars) |
| `APP_ENV` | `development` | `development` / `production` |
| `DATABASE_URL` | `sqlite:///./data/churn_platform.db` | DB connection string |
| `API_PORT` | `8000` | Backend server port |
| `FRONTEND_PORT` | `8501` | Dashboard server port |
| `FRONTEND_URL` | `http://localhost:8501` | Used in alert email links |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | JWT expiry |
| `ML_HIGH_RISK_THRESHOLD` | `0.65` | High-risk probability cutoff |
| `ML_MEDIUM_RISK_THRESHOLD` | `0.35` | Medium-risk probability cutoff |
| `ALERT_HIGH_RISK_THRESHOLD` | `0.65` | Alert trigger threshold |
| `SMTP_HOST` | `smtp.gmail.com` | SMTP server |
| `SMTP_USER` | _(empty)_ | Email sender address |
| `SMTP_PASSWORD` | _(empty)_ | Email app password |
| `ALERT_EMAIL_RECIPIENT` | _(empty)_ | Alert notification recipient |
| `CACHE_TTL_ANALYTICS` | `300` | Analytics cache TTL (seconds) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

Full list with descriptions: [`.env.example`](.env.example)

---

## 🧪 Running Tests

```bash
# Quick smoke test (no dependencies)
python -c "
from backend.app import create_app
import json, io
app = create_app()
c = app.test_client()
c.post('/api/v1/auth/signup', data=json.dumps({'name':'Test User','email':'test@t.io','password':'Pass1234!','role':'admin'}), content_type='application/json')
r = c.post('/api/v1/auth/login', data=json.dumps({'email':'test@t.io','password':'Pass1234!'}), content_type='application/json')
token = r.get_json()['data']['access_token']
auth = {'Authorization': f'Bearer {token}'}
r = c.post('/api/v1/predict', data=json.dumps({'tenure':6,'monthly_charges':89.99,'contract':'Month-to-month','payment_method':'Electronic check','inactive_days':45}), content_type='application/json', headers=auth)
d = r.get_json()['data']
print(f'predict: prob={d[\"churn_probability\"]} risk={d[\"risk_category\"]} score={d[\"risk_score\"]}')
"

# Module import check
python -c "
mods = ['backend.app','backend.services.ml_service','backend.services.monitoring',
        'services.alerts.alert_service','analytics.segmentation','ml.pipeline']
for m in mods:
    __import__(m)
    print(f'  OK {m}')
print('All imports OK')
"
```

---

## 🏆 ML Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression ⭐ | 0.684 | 0.561 | 0.757 | 0.644 | **0.770** |
| Random Forest | 0.688 | 0.566 | 0.753 | 0.646 | 0.748 |
| XGBoost | 0.681 | 0.584 | 0.547 | 0.565 | 0.736 |

- **Training set**: 5,600 customers (stratified)
- **Test set**: 1,400 customers
- **Dataset churn rate**: 26.5%
- **Features**: 32 (after engineering)

---

## 🔧 ML Pipeline CLI

```bash
# Full pipeline (generate + train + segment + recommend + revenue)
python -m ml.pipeline

# Skip data generation (use existing CSV)
python -m ml.pipeline --no-generate

# Custom dataset size
python -m ml.pipeline --rows 10000

# Skip SHAP (faster, no explainability)
python -m ml.pipeline --no-shap

# Seed demo data
python -m data.seed_demo
python -m data.seed_demo --reset        # drop + reseed
python -m data.seed_demo --customers 200
```

---

## 🔐 Default Demo Credentials

After running `python -m data.seed_demo`:

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@demo.io | Demo1234! |
| Analyst | analyst@demo.io | Demo1234! |
| Viewer | viewer@demo.io | Demo1234! |

---

## 📦 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| API | Flask | 3.1 |
| Auth | PyJWT + Werkzeug scrypt | 2.7 / 3.1 |
| ML | scikit-learn + XGBoost | 1.8 / 2.0 |
| Explainability | SHAP | 0.45 |
| Data | NumPy + Pandas | 2.4 / 3.0 |
| Database | SQLite → PostgreSQL | 3 / 15 |
| Frontend | HTML/CSS/JS + Plotly | CDN |
| Charts | Plotly.js | 5.x |
| Containers | Docker + Compose | 24.x |
| Cloud (API) | Render.com | - |
| Cloud (UI) | Streamlit Cloud | 1.35 |
| PDF | jsPDF | CDN |
| Logging | Python logging (rotating) | stdlib |

---

## 🗂️ Resume Talking Points

This project demonstrates:

- **End-to-end ML system design**: raw data → feature engineering → training → evaluation → production inference
- **REST API architecture**: Flask blueprints, JWT RBAC, OpenAPI spec, middleware chain
- **Production patterns**: LRU caching, structured logging, background thread email dispatch, health probes
- **Business intelligence**: Revenue DCF estimation, SHAP explainability, KMeans segmentation, What-If simulation
- **DevOps**: Docker Compose multi-service stack, Render + Streamlit Cloud deployment configs
- **Data engineering**: SQLite with 9 tables, 20+ composite indexes, cascade deletes, UTC timestamps

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ as an enterprise AI portfolio project

**[⭐ Star this repo](https://github.com/your-username/churn-platform)** if you find it useful!

</div>
