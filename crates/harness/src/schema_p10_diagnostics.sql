-- Phase P10.2 Structured Diagnostic Feedback Bridge (additive, user_version = 10)
CREATE TABLE IF NOT EXISTS diagnostic_receipts (
    project TEXT NOT NULL REFERENCES projects(id),
    diagnostic_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    failure_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    root_cause_file TEXT,
    root_cause_line INTEGER,
    root_cause_symbol TEXT,
    exception_class TEXT,
    traceback_summary TEXT NOT NULL,
    resource_telemetry_json TEXT NOT NULL,
    remediations_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, diagnostic_id)
);
CREATE INDEX IF NOT EXISTS idx_diagnostic_receipts_project_cand ON diagnostic_receipts(project, candidate_id);
CREATE INDEX IF NOT EXISTS idx_diagnostic_receipts_failure_type ON diagnostic_receipts(project, failure_type);
