CREATE TABLE IF NOT EXISTS candidate_evaluations (
    project TEXT NOT NULL REFERENCES projects(id),
    evaluation_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    environment_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('evaluating', 'evaluated', 'failed', 'rejected')),
    folds INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    primary_metric REAL,
    metrics_json TEXT NOT NULL,
    slices_json TEXT,
    scorecard_hash TEXT,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, evaluation_id)
);
CREATE INDEX IF NOT EXISTS idx_candidate_evaluations_project ON candidate_evaluations(project, candidate_id);

CREATE TABLE IF NOT EXISTS validation_query_counters (
    project TEXT NOT NULL,
    session_id TEXT NOT NULL,
    query_count INTEGER NOT NULL DEFAULT 0,
    max_queries INTEGER NOT NULL DEFAULT 50,
    PRIMARY KEY(project, session_id)
);
