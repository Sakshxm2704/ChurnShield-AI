-- ============================================================
-- docker/postgres/init/01_extensions.sql
-- Runs automatically on first container start.
-- ============================================================

-- UUID helper (used by application-layer UUID generation)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- pg_trgm for fuzzy text search on customer names / emails
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- unaccent for accent-insensitive search
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- ── Performance settings (override in postgresql.conf for prod) ──
-- These are session-level hints; adjust for your hardware.
ALTER SYSTEM SET work_mem = '64MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET random_page_cost = 1.1;   -- assume SSD
ALTER SYSTEM SET effective_io_concurrency = 200;

SELECT pg_reload_conf();
