"""
tests/test_unit_auth_db.py
---------------------------
Unit tests: password hashing, JWT, cache, DB CRUD, input validation.
"""
from __future__ import annotations
import sys, unittest, json, time, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("SECRET_KEY","test-secret-key-at-least-32-chars-long!!")
os.environ.setdefault("APP_ENV","testing")

# ── Isolation mixin ───────────────────────────────────────────────────────────
class _DBMixin:
    def setUp(self):
        self._tmp=tempfile.NamedTemporaryFile(suffix=".db",delete=False); self._tmp.close()
        from backend.core import database as dm, config as cm
        self._orig=cm.settings.DATABASE_URL
        cm.settings.DATABASE_URL=f"sqlite:///{self._tmp.name}"
        if hasattr(dm._local,"conn") and dm._local.conn:
            try: dm._local.conn.close()
            except: pass
            dm._local.conn=None
        dm.init_db()
        import importlib, backend.services.db_service as dbs
        importlib.reload(dbs)
        self.db=dbs
    def tearDown(self):
        from backend.core import database as dm, config as cm
        if hasattr(dm._local,"conn") and dm._local.conn:
            try: dm._local.conn.close()
            except: pass
            dm._local.conn=None
        cm.settings.DATABASE_URL=self._orig
        try: os.unlink(self._tmp.name)
        except: pass

# ═══════════════════════════════════════════════════════════════════════════════
class TestPasswordHashing(unittest.TestCase):
    def setUp(self):
        from backend.auth.password import hash_password, verify_password
        self.hp=hash_password; self.vp=verify_password
    def test_returns_string(self): self.assertIsInstance(self.hp("pw"),str)
    def test_not_plaintext(self): self.assertNotEqual(self.hp("s"),"s")
    def test_verify_correct(self): h=self.hp("x"); self.assertTrue(self.vp("x",h))
    def test_verify_wrong(self): h=self.hp("a"); self.assertFalse(self.vp("b",h))
    def test_salted_unique(self): h1=self.hp("s"); h2=self.hp("s"); self.assertNotEqual(h1,h2)
    def test_verify_after_salt(self): h1=self.hp("s"); h2=self.hp("s"); self.assertTrue(self.vp("s",h1)); self.assertTrue(self.vp("s",h2))
    def test_empty_password(self): h=self.hp(""); self.assertIsInstance(h,str)
    def test_unicode(self): h=self.hp("pässwørd"); self.assertTrue(self.vp("pässwørd",h))
    def test_long_password(self): pw="x"*200; h=self.hp(pw); self.assertTrue(self.vp(pw,h))

class TestJWT(unittest.TestCase):
    def setUp(self):
        from backend.auth.jwt_handler import create_access_token, verify_token, create_refresh_token
        self.create=create_access_token; self.verify=verify_token; self.refresh=create_refresh_token
    def test_create_returns_string(self): self.assertIsInstance(self.create({"uid":1}),str)
    def test_token_has_3_parts(self): t=self.create({"uid":1}); self.assertEqual(len(t.split(".")),3)
    def test_verify_valid_token(self): t=self.create({"uid":1,"role":"admin"}); p=self.verify(t); self.assertIsNotNone(p)
    def test_verify_contains_payload(self): t=self.create({"uid":1,"role":"admin"}); p=self.verify(t); self.assertEqual(p["uid"],1)
    def test_verify_expired_returns_none(self):
        from datetime import timedelta
        t=self.create({"uid":1},expires_delta=timedelta(seconds=-1))
        self.assertIsNone(self.verify(t))
    def test_verify_invalid_signature(self): self.assertIsNone(self.verify("bad.token.here"))
    def test_refresh_token_valid(self): t=self.refresh({"uid":2}); self.assertIsNotNone(self.verify(t))
    def test_different_payloads_differ(self):
        t1=self.create({"uid":1}); t2=self.create({"uid":2}); self.assertNotEqual(t1,t2)
    def test_empty_payload(self): t=self.create({}); p=self.verify(t); self.assertIsNotNone(p)

class TestCache(unittest.TestCase):
    def setUp(self):
        from backend.core.cache import AppCache
        self.cache=AppCache(max_size=50)
    def test_set_and_get(self): self.cache.set("k","v"); self.assertEqual(self.cache.get("k"),"v")
    def test_miss_returns_none(self): self.assertIsNone(self.cache.get("nonexistent"))
    def test_ttl_expiry(self):
        self.cache.set("k2","v2",ttl=0); time.sleep(0.01)
        result=self.cache.get("k2"); self.assertIsNone(result)
    def test_delete(self): self.cache.set("k","v"); self.cache.delete("k"); self.assertIsNone(self.cache.get("k"))
    def test_delete_prefix(self):
        self.cache.set("a:1","v1"); self.cache.set("a:2","v2"); self.cache.set("b:1","v3")
        self.cache.delete_prefix("a:"); self.assertIsNone(self.cache.get("a:1")); self.assertEqual(self.cache.get("b:1"),"v3")
    def test_clear(self): self.cache.set("x","y"); self.cache.clear(); self.assertIsNone(self.cache.get("x"))
    def test_stats_hit_rate(self): self.cache.set("k","v"); self.cache.get("k"); self.cache.get("miss"); s=self.cache.stats; self.assertGreater(s["hits"],0)
    def test_overwrite(self): self.cache.set("k","v1"); self.cache.set("k","v2"); self.assertEqual(self.cache.get("k"),"v2")
    def test_store_dict(self): self.cache.set("k",{"a":1}); self.assertEqual(self.cache.get("k"),{"a":1})

class TestDatabaseCRUD(_DBMixin, unittest.TestCase):
    def test_create_user(self):
        u=self.db.create_user("Alice","alice_crud@test.com","hash"); self.assertEqual(u["name"],"Alice")
    def test_get_user_by_email(self):
        self.db.create_user("Bob","bob_crud@test.com","hash"); u=self.db.get_user_by_email("bob_crud@test.com")
        self.assertIsNotNone(u); self.assertEqual(u["email"],"bob_crud@test.com")
    def test_get_user_missing(self): self.assertIsNone(self.db.get_user_by_email("nobody@x.com"))
    def test_duplicate_email_raises(self):
        self.db.create_user("C","dup_crud@test.com","h")
        with self.assertRaises(Exception): self.db.create_user("D","dup_crud@test.com","h")
    def test_create_customer(self):
        c=self.db.create_customer({"tenure":10,"monthly_charges":50,"contract":"One year","payment_method":"Mailed check","inactive_days":5,"subscription_type":"Basic"})
        self.assertIn("customer_id",c); self.assertGreater(c["customer_id"],0)
    def test_get_customer(self):
        c=self.db.create_customer({"tenure":5,"monthly_charges":40,"contract":"Month-to-month","payment_method":"Electronic check","inactive_days":20,"subscription_type":"Standard"})
        fetched=self.db.get_customer(c["customer_id"]); self.assertEqual(fetched["customer_id"],c["customer_id"])
    def test_get_missing_customer(self): self.assertIsNone(self.db.get_customer(999999))
    def test_save_prediction(self):
        c=self.db.create_customer({"tenure":5,"monthly_charges":40,"contract":"Month-to-month","payment_method":"Electronic check","inactive_days":0,"subscription_type":"Basic"})
        pid=self.db.save_prediction(c["customer_id"],{"churn_probability":0.7,"risk_score":85,"risk_category":"High","model_used":"lr","churn_label":"Churn","segment":"Risky"})
        self.assertIsInstance(pid,int); self.assertGreater(pid,0)
    def test_list_customers_pagination(self):
        for i in range(5): self.db.create_customer({"tenure":i+1,"monthly_charges":30+i,"contract":"Month-to-month","payment_method":"Electronic check","inactive_days":i,"subscription_type":"Basic"})
        r=self.db.list_customers(page=1,page_size=3)
        self.assertIsInstance(r,dict); self.assertIn("items",r); self.assertLessEqual(len(r["items"]),3)
    def test_save_recommendations(self):
        c=self.db.create_customer({"tenure":3,"monthly_charges":80,"contract":"Month-to-month","payment_method":"Electronic check","inactive_days":50,"subscription_type":"Premium"})
        n=self.db.save_recommendations(c["customer_id"],[{"rule_name":"r1","action":"Offer discount","estimated_savings":100,"priority":1}])
        self.assertGreater(n,0)
    def test_list_predictions_empty(self):
        r=self.db.list_predictions(); self.assertIsInstance(r,dict)

class TestInputValidation(unittest.TestCase):
    def setUp(self):
        from backend.models.schemas import validate_customer, validate_signup, validate_pagination, validate_login
        self.vc=validate_customer; self.vs=validate_signup; self.vp=validate_pagination; self.vl=validate_login
    def test_valid_customer_passes(self):
        r=self.vc({"tenure":6,"monthly_charges":89.99,"contract":"Month-to-month","payment_method":"Electronic check"})
        self.assertIn("tenure",r)
    def test_negative_tenure_clamped(self):
        r=self.vc({"tenure":-5,"monthly_charges":50,"contract":"Month-to-month","payment_method":"Electronic check"})
        self.assertGreaterEqual(r["tenure"],0)
    def test_missing_required_raises(self):
        with self.assertRaises((ValueError,KeyError)): self.vc({"tenure":6})
    def test_valid_signup(self):
        r=self.vs({"name":"Jane","email":"jane@x.com","password":"Pass1234!"}); self.assertEqual(r["email"],"jane@x.com")
    def test_short_password_raises(self):
        with self.assertRaises(ValueError): self.vs({"name":"Jane","email":"j@x.com","password":"short"})
    def test_invalid_email_raises(self):
        with self.assertRaises(ValueError): self.vs({"name":"Jane","email":"notanemail","password":"Pass1234!"})
    def test_pagination_defaults(self):
        page, page_size = self.vp({})
        self.assertIsInstance(page, int); self.assertGreaterEqual(page, 1)
        self.assertIsInstance(page_size, int)
    def test_pagination_page_clamped(self):
        page, _ = self.vp({"page": -5}); self.assertGreaterEqual(page, 1)

if __name__=="__main__": unittest.main(verbosity=2)
