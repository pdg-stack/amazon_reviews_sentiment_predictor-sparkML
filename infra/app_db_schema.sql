-- Run against the `app` database (after CREATE DATABASE app, done
-- separately since that statement can't run inside the DB it creates).
-- See docs/DEPLOYMENT.md for the full one-time setup sequence.

CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    review_title TEXT,
    review_text TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    client_ip TEXT,
    user_agent TEXT,
    api_key_id TEXT,
    received_at TIMESTAMPTZ NOT NULL,
    responded_at TIMESTAMPTZ NOT NULL,
    model_version INT,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_received_at ON predictions (received_at);

CREATE TABLE IF NOT EXISTS drift_checks (
    id BIGSERIAL PRIMARY KEY,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL,
    sample_size INT,
    window_seconds INT,
    prediction_confidence_psi DOUBLE PRECISION,
    review_length_psi DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_drift_checks_checked_at ON drift_checks (checked_at);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO app_user;
