-- Reference schema (SQLAlchemy create_all is the source of truth at runtime).
-- Tables: visits, conversation_turns, visit_artifacts

CREATE TABLE IF NOT EXISTS visits (
    visit_id VARCHAR(128) PRIMARY KEY,
    status VARCHAR(64),
    follow_up_iteration INTEGER NOT NULL DEFAULT 0,
    confidence DOUBLE PRECISION,
    previous_confidence DOUBLE PRECISION,
    source_file TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id SERIAL PRIMARY KEY,
    visit_id VARCHAR(128) NOT NULL REFERENCES visits(visit_id) ON DELETE CASCADE,
    round INTEGER NOT NULL DEFAULT 0,
    turn_index INTEGER NOT NULL DEFAULT 0,
    role VARCHAR(64),
    speaker VARCHAR(64),
    text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS visit_artifacts (
    id SERIAL PRIMARY KEY,
    visit_id VARCHAR(128) NOT NULL REFERENCES visits(visit_id) ON DELETE CASCADE,
    artifact_type VARCHAR(64) NOT NULL,
    local_path TEXT,
    s3_key TEXT,
    s3_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
