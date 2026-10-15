-- Phase P8 feasibility reports, execution plans, and micro-pilots (additive, user_version = 10)

CREATE TABLE IF NOT EXISTS feasibility_reports (
  project TEXT NOT NULL REFERENCES projects(id),
  id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  model_family TEXT NOT NULL,
  verdict TEXT NOT NULL CHECK(verdict IN ('locally_feasible', 'feasible_with_adaptation', 'requires_external_compute', 'unknown')),
  confidence TEXT NOT NULL CHECK(confidence IN ('high', 'medium', 'low')),
  predicted_runtime_ms INTEGER NOT NULL,
  predicted_peak_memory_bytes INTEGER NOT NULL,
  cv_fold_count INTEGER NOT NULL,
  include_final_refit INTEGER NOT NULL CHECK(include_final_refit IN (0, 1)),
  concurrent_worker_leases INTEGER NOT NULL,
  report_json TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, id)
);

CREATE TABLE IF NOT EXISTS execution_plans (
  project TEXT NOT NULL REFERENCES projects(id),
  id TEXT NOT NULL,
  report_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  model_family TEXT NOT NULL,
  selected_preset TEXT NOT NULL,
  allocated_threads INTEGER NOT NULL,
  allocated_memory_bytes INTEGER NOT NULL,
  allocated_scratch_bytes INTEGER NOT NULL,
  locked_split_protocol_preserved INTEGER NOT NULL CHECK(locked_split_protocol_preserved = 1),
  plan_json TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, id),
  FOREIGN KEY(project, report_id) REFERENCES feasibility_reports(project, id)
);

CREATE TABLE IF NOT EXISTS micro_pilot_runs (
  project TEXT NOT NULL REFERENCES projects(id),
  trial_id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  model_family TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  feature_count INTEGER NOT NULL,
  elapsed_ms INTEGER NOT NULL,
  peak_rss_bytes INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('succeeded', 'timed_out', 'memory_exceeded', 'failed')),
  details_json TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, trial_id)
);
