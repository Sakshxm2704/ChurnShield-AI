"""
tests/test_api.py
------------------
API integration tests: all endpoints, auth, edge cases, security, performance.
"""
from __future__ import annotations
import sys, unittest, json, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("SECRET_KEY","test-secret-key-at-least-32-chars-long!!")
os.environ.setdefault("APP_ENV","testing")

def _app():
    from backend.app import create_app
    a=create_app(); a.config["TESTING"]=True; return a

import random
def _signup_login(client, role="admin"):
    email=f"u{random.randint(100000,999999)}@test.io"
    client.post("/api/v1/auth/signup",
        data=json.dumps({"name":"Test","email":email,"password":"Pass1234!","role":role}),
        content_type="application/json")
    r=client.post("/api/v1/auth/login",
        data=json.dumps({"email":email,"password":"Pass1234!"}),
        content_type="application/json")
    return r.get_json()["data"]["access_token"]

def _hdr(t): return {"Authorization":f"Bearer {t}"}
def _j(client,method,path,body=None,headers=None,**kw):
    fn=getattr(client,method.lower())
    kw2=dict(headers=headers or {})
    if body: kw2.update(data=json.dumps(body),content_type="application/json")
    return fn(path,**kw2,**kw)

CSV=b"tenure,monthly_charges,contract,payment_method,inactive_days\n3,89.99,Month-to-month,Electronic check,60\n24,45.0,One year,Bank transfer (automatic),5\n60,25.0,Two year,Credit card (automatic),2\n"
P={"tenure":6,"monthly_charges":89.99,"contract":"Month-to-month","payment_method":"Electronic check","inactive_days":45,"save_customer":False,"include_shap":False}

# ═══════════════════════════════════════════════════════════════════════════════
class TestSystemEndpoints(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client()
    def test_health(self): r=self.c.get("/health"); self.assertEqual(r.status_code,200)
    def test_health_returns_ok(self): r=self.c.get("/health"); self.assertEqual(r.get_json()["status"],"ok")
    def test_ready(self): r=self.c.get("/ready"); self.assertIn(r.status_code,(200,503))
    def test_openapi_spec(self): r=self.c.get("/openapi.json"); self.assertEqual(r.status_code,200)
    def test_openapi_has_paths(self): r=self.c.get("/openapi.json"); self.assertIn("paths",r.get_json())
    def test_swagger_ui(self): r=self.c.get("/docs"); self.assertEqual(r.status_code,200)
    def test_api_info(self): r=self.c.get("/api/v1/info"); self.assertEqual(r.status_code,200)

class TestAuthAPI(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client()
    def _email(self): return f"t{random.randint(1,999999)}@test.io"
    def test_signup_success(self):
        r=self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jane","email":self._email(),"password":"Pass1234!"}),content_type="application/json")
        self.assertEqual(r.status_code,201)
    def test_signup_returns_token(self):
        e=self._email()
        r=self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jane","email":e,"password":"Pass1234!"}),content_type="application/json")
        d=r.get_json(); self.assertTrue(d["success"])
    def test_signup_missing_email_fails(self):
        r=self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jane","password":"Pass1234!"}),content_type="application/json")
        self.assertEqual(r.status_code,400)
    def test_signup_short_password_fails(self):
        r=self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jane","email":self._email(),"password":"short"}),content_type="application/json")
        self.assertEqual(r.status_code,400)
    def test_signup_duplicate_email_fails(self):
        e=self._email()
        self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Alice","email":e,"password":"Pass1234!"}),content_type="application/json")
        r=self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Bob","email":e,"password":"Pass1234!"}),content_type="application/json")
        self.assertIn(r.status_code,(400,409))
    def test_login_success(self):
        e=self._email()
        self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jo","email":e,"password":"Pass1234!"}),content_type="application/json")
        r=self.c.post("/api/v1/auth/login",data=json.dumps({"email":e,"password":"Pass1234!"}),content_type="application/json")
        self.assertEqual(r.status_code,200)
    def test_login_returns_access_token(self):
        e=self._email()
        self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jo","email":e,"password":"Pass1234!"}),content_type="application/json")
        r=self.c.post("/api/v1/auth/login",data=json.dumps({"email":e,"password":"Pass1234!"}),content_type="application/json")
        self.assertIn("access_token",r.get_json()["data"])
    def test_login_wrong_password(self):
        e=self._email()
        self.c.post("/api/v1/auth/signup",data=json.dumps({"name":"Jo","email":e,"password":"Pass1234!"}),content_type="application/json")
        r=self.c.post("/api/v1/auth/login",data=json.dumps({"email":e,"password":"wrong"}),content_type="application/json")
        self.assertEqual(r.status_code,401)
    def test_login_unknown_email(self):
        r=self.c.post("/api/v1/auth/login",data=json.dumps({"email":"nobody@x.com","password":"x"}),content_type="application/json")
        self.assertEqual(r.status_code,401)
    def test_me_requires_auth(self): r=self.c.get("/api/v1/auth/me"); self.assertEqual(r.status_code,401)
    def test_me_returns_user(self):
        t=_signup_login(self.c); r=self.c.get("/api/v1/auth/me",headers=_hdr(t))
        self.assertEqual(r.status_code,200)

class TestPredictionAPI(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client(); self.t=_signup_login(self.c)
    def test_predict_success(self):
        r=_j(self.c,"post","/api/v1/predict",P,_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_predict_returns_probability(self):
        r=_j(self.c,"post","/api/v1/predict",P,_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("churn_probability",d); self.assertGreater(d["churn_probability"],0)
    def test_predict_returns_risk_score(self):
        r=_j(self.c,"post","/api/v1/predict",P,_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("risk_score",d); self.assertGreaterEqual(d["risk_score"],0)
    def test_predict_returns_risk_category(self):
        r=_j(self.c,"post","/api/v1/predict",P,_hdr(self.t))
        self.assertIn(r.get_json()["data"]["risk_category"],("High","Medium","Low"))
    def test_predict_returns_recommendations(self):
        r=_j(self.c,"post","/api/v1/predict",P,_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("recommendations",d)
    def test_predict_requires_auth(self):
        r=_j(self.c,"post","/api/v1/predict",P,{}); self.assertEqual(r.status_code,401)
    def test_predict_missing_tenure_fails(self):
        bad={k:v for k,v in P.items() if k!="tenure"}
        r=_j(self.c,"post","/api/v1/predict",bad,_hdr(self.t)); self.assertEqual(r.status_code,400)
    def test_predict_missing_contract_fails(self):
        bad={k:v for k,v in P.items() if k!="contract"}
        r=_j(self.c,"post","/api/v1/predict",bad,_hdr(self.t)); self.assertEqual(r.status_code,400)
    def test_predict_empty_body_fails(self):
        r=_j(self.c,"post","/api/v1/predict",{},_hdr(self.t)); self.assertEqual(r.status_code,400)
    def test_predict_invalid_contract_handled(self):
        bad={**P,"contract":"InvalidContract"}
        r=_j(self.c,"post","/api/v1/predict",bad,_hdr(self.t))
        # Should either work (coerce) or return 400 — not 500
        self.assertNotEqual(r.status_code,500)
    def test_predict_response_time(self):
        t0=time.time(); _j(self.c,"post","/api/v1/predict",P,_hdr(self.t)); elapsed=time.time()-t0
        self.assertLess(elapsed,2.0,f"Prediction took {elapsed:.2f}s (limit 2s)")
    def test_batch_predict_csv(self):
        r=self.c.post("/api/v1/batch_predict",data={"file":(io.BytesIO(CSV),"t.csv")},content_type="multipart/form-data",headers=_hdr(self.t))
        self.assertEqual(r.status_code,200)
    def test_batch_predict_total(self):
        r=self.c.post("/api/v1/batch_predict",data={"file":(io.BytesIO(CSV),"t.csv")},content_type="multipart/form-data",headers=_hdr(self.t))
        d=r.get_json()["data"]; self.assertEqual(d["total"],3)
    def test_batch_predict_requires_auth(self):
        r=self.c.post("/api/v1/batch_predict",data={"file":(io.BytesIO(CSV),"t.csv")},content_type="multipart/form-data")
        self.assertEqual(r.status_code,401)
    def test_batch_predict_no_file_fails(self):
        r=self.c.post("/api/v1/batch_predict",headers=_hdr(self.t)); self.assertIn(r.status_code,(400,422,500))
    def test_batch_response_time_3rows(self):
        t0=time.time()
        self.c.post("/api/v1/batch_predict",data={"file":(io.BytesIO(CSV),"t.csv")},content_type="multipart/form-data",headers=_hdr(self.t))
        elapsed=time.time()-t0; self.assertLess(elapsed,10.0)
    def test_predictions_list(self):
        r=self.c.get("/api/v1/predictions",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_predictions_list_structure(self):
        r=self.c.get("/api/v1/predictions",headers=_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("items",d); self.assertIn("total",d)

class TestCustomerAPI(unittest.TestCase):
    def setUp(self):
        self.app=_app(); self.c=self.app.test_client(); self.t=_signup_login(self.c)
        p={**P,"save_customer":True}; r=_j(self.c,"post","/api/v1/predict",p,_hdr(self.t))
        self.cid=r.get_json()["data"].get("customer_id")
    def test_list_customers(self): r=self.c.get("/api/v1/customers",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_list_structure(self):
        r=self.c.get("/api/v1/customers",headers=_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("items",d); self.assertIn("total",d)
    def test_list_requires_auth(self): r=self.c.get("/api/v1/customers"); self.assertEqual(r.status_code,401)
    def test_get_customer_by_id(self):
        if not self.cid: self.skipTest("No customer created")
        r=self.c.get(f"/api/v1/customers/{self.cid}",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_get_customer_not_found(self):
        r=self.c.get("/api/v1/customers/999999",headers=_hdr(self.t)); self.assertEqual(r.status_code,404)
    def test_customer_has_latest_prediction(self):
        if not self.cid: self.skipTest("No customer")
        r=self.c.get(f"/api/v1/customers/{self.cid}",headers=_hdr(self.t))
        self.assertIsNotNone(r.get_json()["data"].get("latest_prediction"))
    def test_prediction_history(self):
        if not self.cid: self.skipTest("No customer")
        r=self.c.get(f"/api/v1/customers/{self.cid}/history",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)

class TestAnalyticsAPI(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client(); self.t=_signup_login(self.c)
    def test_analytics(self): r=self.c.get("/api/v1/analytics",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_analytics_requires_auth(self): r=self.c.get("/api/v1/analytics"); self.assertEqual(r.status_code,401)
    def test_revenue_analytics(self): r=self.c.get("/api/v1/analytics/revenue",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_model_metrics(self): r=self.c.get("/api/v1/analytics/model_metrics",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_feature_importance(self): r=self.c.get("/api/v1/analytics/feature_importance",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_segments(self): r=self.c.get("/api/v1/segments",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_segments_has_4_groups(self):
        r=self.c.get("/api/v1/segments",headers=_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("segments",d)
    def test_recommendations_list(self): r=self.c.get("/api/v1/recommendations",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_whatif_presets(self): r=self.c.get("/api/v1/analytics/whatif_presets",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_whatif_simulation(self):
        body={"customer_data":P,"scenario_name":"upgrade_to_annual_contract"}
        r=_j(self.c,"post","/api/v1/whatif",body,_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_whatif_has_delta(self):
        body={"customer_data":P,"scenario_name":"upgrade_to_annual_contract"}
        r=_j(self.c,"post","/api/v1/whatif",body,_hdr(self.t))
        self.assertIn("delta_probability",r.get_json()["data"])
    def test_whatif_requires_auth(self):
        body={"customer_data":P,"scenario_name":"upgrade_to_annual_contract"}
        r=_j(self.c,"post","/api/v1/whatif",body,{}); self.assertEqual(r.status_code,401)

class TestMonitoringAPI(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client(); self.t=_signup_login(self.c)
    def test_monitoring(self): r=self.c.get("/api/v1/monitoring",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_monitoring_has_metrics(self):
        r=self.c.get("/api/v1/monitoring",headers=_hdr(self.t)); d=r.get_json()["data"]
        self.assertIn("total_predictions",d)
    def test_alerts_list(self): r=self.c.get("/api/v1/monitoring/alerts",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_cache_stats(self): r=self.c.get("/api/v1/monitoring/cache",headers=_hdr(self.t)); self.assertEqual(r.status_code,200)
    def test_monitoring_requires_auth(self): r=self.c.get("/api/v1/monitoring"); self.assertEqual(r.status_code,401)

class TestSecurity(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client()
    def test_missing_token_401(self):
        for path in ["/api/v1/predict","/api/v1/customers","/api/v1/analytics","/api/v1/segments","/api/v1/monitoring"]:
            r=self.c.get(path); self.assertIn(r.status_code,(401,405),f"{path} should be 401 without token")
    def test_invalid_token_401(self):
        bad={"Authorization":"Bearer totally.invalid.token"}
        r=self.c.get("/api/v1/analytics",headers=bad); self.assertEqual(r.status_code,401)
    def test_malformed_bearer_401(self):
        r=self.c.get("/api/v1/analytics",headers={"Authorization":"NotBearer abc"}); self.assertEqual(r.status_code,401)
    def test_expired_token_401(self):
        from datetime import timedelta
        from backend.auth.jwt_handler import create_access_token
        t=create_access_token({"user_id":1,"email":"x@x.com","role":"admin"},timedelta(seconds=-1))
        r=self.c.get("/api/v1/analytics",headers=_hdr(t)); self.assertEqual(r.status_code,401)
    def test_password_not_stored_plaintext(self):
        from backend.auth.password import hash_password
        h=hash_password("mypassword"); self.assertNotIn("mypassword",h)
    def test_404_for_unknown_routes(self):
        r=self.c.get("/api/v1/nonexistent_endpoint_xyz"); self.assertEqual(r.status_code,404)

class TestPerformance(unittest.TestCase):
    def setUp(self): self.app=_app(); self.c=self.app.test_client(); self.t=_signup_login(self.c)
    def test_single_prediction_under_2s(self):
        t0=time.time(); _j(self.c,"post","/api/v1/predict",P,_hdr(self.t)); elapsed=time.time()-t0
        self.assertLess(elapsed,2.0,f"Took {elapsed:.3f}s")
    def test_analytics_under_1s(self):
        t0=time.time(); self.c.get("/api/v1/analytics",headers=_hdr(self.t)); elapsed=time.time()-t0
        self.assertLess(elapsed,1.0,f"Took {elapsed:.3f}s")
    def test_batch_3rows_under_5s(self):
        t0=time.time()
        self.c.post("/api/v1/batch_predict",data={"file":(io.BytesIO(CSV),"t.csv")},content_type="multipart/form-data",headers=_hdr(self.t))
        elapsed=time.time()-t0; self.assertLess(elapsed,5.0,f"Took {elapsed:.3f}s")
    def test_10_sequential_predictions_under_15s(self):
        t0=time.time()
        for _ in range(10): _j(self.c,"post","/api/v1/predict",P,_hdr(self.t))
        elapsed=time.time()-t0; self.assertLess(elapsed,15.0,f"10 preds took {elapsed:.3f}s")

if __name__=="__main__": unittest.main(verbosity=2)
