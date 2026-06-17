# ================================================================
# Churn Intelligence Platform — Developer Makefile
# Usage: make <target>
# ================================================================

.PHONY: help install train seed run-api run-ui run-all test clean docker-up docker-down

help:
	@echo "Churn Intelligence Platform"
	@echo ""
	@echo "  make install     Install Python dependencies"
	@echo "  make train       Run ML training pipeline"
	@echo "  make seed        Seed demo data"
	@echo "  make run-api     Start backend API (port 8000)"
	@echo "  make run-ui      Start dashboard (port 8501)"
	@echo "  make run-all     Start API + dashboard in parallel"
	@echo "  make test        Run end-to-end smoke test"
	@echo "  make docker-up   Start full stack via Docker Compose"
	@echo "  make docker-down Stop Docker stack"
	@echo "  make clean       Remove generated files"

install:
	pip install flask werkzeug PyJWT scikit-learn joblib numpy pandas python-dotenv

train:
	python -m ml.pipeline --no-shap --rows 5000

seed:
	python -m data.seed_demo

run-api:
	python -m backend.app

run-ui:
	python -m frontend.server

run-all:
	@echo "Starting API and Dashboard..."
	python -m backend.app &
	sleep 2
	python -m frontend.server

test:
	python -c " \
from backend.app import create_app; import json, io; app = create_app(); c = app.test_client(); \
c.post('/api/v1/auth/signup', data=json.dumps({'name':'Test User','email':'test@e.io','password':'Pass1234!','role':'admin'}), content_type='application/json'); \
r = c.post('/api/v1/auth/login', data=json.dumps({'email':'test@e.io','password':'Pass1234!'}), content_type='application/json'); \
token = r.get_json()['data']['access_token']; auth = {'Authorization': f'Bearer {token}'}; \
r = c.post('/api/v1/predict', data=json.dumps({'tenure':6,'monthly_charges':89.99,'contract':'Month-to-month','payment_method':'Electronic check','inactive_days':45}), content_type='application/json', headers=auth); \
d = r.get_json()['data']; print(f'predict: prob={d[\"churn_probability\"]} risk={d[\"risk_category\"]}'); \
print('ALL TESTS PASSED'); \
"

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-reset:
	docker compose -f docker/docker-compose.yml down -v
	docker compose -f docker/docker-compose.yml up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f data/churn_platform.db
	rm -f ml/models/saved/best_model.joblib ml/models/saved/preprocessor.joblib
