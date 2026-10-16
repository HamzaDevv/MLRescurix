-- Phase P10 candidate synthesis and refinement records (additive, user_version = 10)
CREATE TABLE IF NOT EXISTS candidate_synthesis_records (
    project TEXT NOT NULL REFERENCES projects(id),
    synthesis_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK(task_type IN ('binary', 'multiclass', 'regression')),
    model_family TEXT NOT NULL,
    framework TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, synthesis_id)
);
CREATE INDEX IF NOT EXISTS candidate_synthesis_project_cand ON candidate_synthesis_records(project, candidate_id);
