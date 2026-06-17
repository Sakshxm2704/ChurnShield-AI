# Churn Intelligence Platform — Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│              CHURN INTELLIGENCE PLATFORM v1.0                            │
│                  Enterprise AI Retention System                          │
└──────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════╗
║  PRESENTATION LAYER  (Port 8501)                                         ║
║                                                                          ║
║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  ║
║  │Dashboard │ │Prediction│ │Analytics │ │Segments  │ │ Monitoring   │  ║
║  │  KPIs    │ │ + WhatIf │ │+ Revenue │ │  + SHAP  │ │  + Alerts    │  ║
║  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────────┘  ║
║  Plotly Charts · Swagger UI · JWT Auth · PDF Reports · Dark Theme        ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↕  HTTP / REST + JSON
╔══════════════════════════════════════════════════════════════════════════╗
║  API LAYER  Flask 3.1  (Port 8000)  /api/v1/*                            ║
║                                                                          ║
║  auth · customers · predictions · analytics · recommendations · monitor  ║
║  JWT Bearer Auth · CORS · Request Timing · Structured Logging            ║
╚══════════════════════════════════════════════════════════════════════════╝
                              ↕
╔═══════════════════════════╦══════════════════╦════════════════════════════╗
║  ML PIPELINE               ║  SERVICE LAYER   ║  DATA LAYER               ║
║                            ║                  ║                           ║
║  Feature Engineering  →    ║  ML Service      ║  SQLite / PostgreSQL      ║
║  Preprocessing        →    ║  DB Service      ║  users                    ║
║  LogReg / RF / XGB    →    ║  Alert Service   ║  customers                ║
║  Best Model Selection →    ║  Monitoring Svc  ║  predictions              ║
║  SHAP Explainability  →    ║  Cache (LRU)     ║  recommendations          ║
║  KMeans Segmentation  →    ║  Recommendation  ║  alert_log                ║
║  Revenue Estimator    →    ║  Engine          ║  api_logs                 ║
║  WhatIf Simulator          ║                  ║  monitoring_metrics       ║
╚═══════════════════════════╩══════════════════╩════════════════════════════╝
```

## Prediction Data Flow

```
POST /api/v1/predict
  → JWT Auth → Validate Input → Map Fields
  → Feature Engineering → Preprocess → ML Predict
  → Risk Score → KMeans Segment → Recommendations
  → Revenue Estimation → SHAP Explanation
  → Save to DB → Trigger Alert (if high-risk) → Return JSON
```

## Deployment

```
Local:   backend:8000  +  frontend:8501  +  SQLite
Docker:  postgres:5432  +  redis:6379  +  backend  +  frontend
Cloud:   Render API  +  Streamlit Cloud dashboard
```

## Database Schema

```
users → (predictions) → customers → recommendations
                    ↓              → retention_logs
                alert_log          → alert_log
api_logs  /  monitoring_metrics  /  cache_store
```
