-- Phase P10.4 Autonomous End-to-End Model Rescue Gate (additive, user_version = 10)
CREATE TABLE IF NOT EXISTS autonomous_rescue_sessions (
    project TEXT NOT NULL REFERENCES projects(id),
    session_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('initialized', 'profiling', 'synthesizing', 'failure_intercepted', 'hypothesizing', 'iterating', 'optimizing', 'finalized', 'failed')),
    dataset_id TEXT NOT NULL,
    initial_candidate_id TEXT,
    apex_candidate_id TEXT,
    bundle_path TEXT,
    turns_executed INTEGER NOT NULL DEFAULT 0,
    profile_json TEXT,
    session_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL,
    PRIMARY KEY(project, session_id)
);
CREATE INDEX IF NOT EXISTS idx_rescue_sessions_proj_status ON autonomous_rescue_sessions(project, status);
CREATE INDEX IF NOT EXISTS idx_rescue_sessions_apex ON autonomous_rescue_sessions(project, apex_candidate_id);
