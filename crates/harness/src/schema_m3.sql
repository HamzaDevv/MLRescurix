CREATE TABLE scientific_jobs_old AS SELECT * FROM scientific_jobs;
DROP TABLE scientific_jobs;
CREATE TABLE scientific_jobs(
  project TEXT NOT NULL,
  job TEXT NOT NULL,
  dataset TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN('audit','baselines','finalize','trial')),
  spec TEXT NOT NULL,
  result_sha256 TEXT,
  PRIMARY KEY(project,job),
  FOREIGN KEY(project,job) REFERENCES jobs(project,id),
  FOREIGN KEY(project,dataset) REFERENCES scientific_datasets(project,id)
);
INSERT INTO scientific_jobs(project,job,dataset,operation,spec,result_sha256)
  SELECT project,job,dataset,operation,spec,result_sha256 FROM scientific_jobs_old;
DROP TABLE scientific_jobs_old;

CREATE TABLE search_sessions(
  project TEXT NOT NULL REFERENCES projects(id),
  id TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  spec TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('created','screening','evaluating','confirming','completed','provisional','failed','cancelled')),
  created_ms INTEGER NOT NULL,
  updated_ms INTEGER NOT NULL,
  apex_candidate_id TEXT,
  selection_record TEXT,
  PRIMARY KEY(project,id),
  FOREIGN KEY(project,dataset_id) REFERENCES scientific_datasets(project,id)
);

CREATE TABLE experiment_dag(
  project TEXT NOT NULL REFERENCES projects(id),
  session_id TEXT NOT NULL,
  id TEXT NOT NULL,
  recipe TEXT NOT NULL CHECK(recipe IN ('dummy','linear','tree')),
  parameters TEXT NOT NULL,
  seed INTEGER NOT NULL,
  fidelity REAL NOT NULL CHECK(fidelity > 0.0 AND fidelity <= 1.0),
  node_type TEXT NOT NULL CHECK(node_type IN ('baseline','screening','full_fidelity','repeat')),
  parent_id TEXT,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project,id),
  FOREIGN KEY(project,session_id) REFERENCES search_sessions(project,id)
);

CREATE TABLE optimizer_trials(
  project TEXT NOT NULL REFERENCES projects(id),
  session_id TEXT NOT NULL,
  study_name TEXT NOT NULL,
  optuna_trial_id INTEGER NOT NULL,
  experiment_id TEXT NOT NULL,
  fidelity REAL NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('asked','dispatched','committed','told','failed')),
  score REAL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project,session_id,study_name,optuna_trial_id),
  FOREIGN KEY(project,experiment_id) REFERENCES experiment_dag(project,id)
);

CREATE TABLE experiment_attempts(
  project TEXT NOT NULL REFERENCES projects(id),
  experiment_id TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  job_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('dispatched','running','succeeded','failed','interrupted','timed_out')),
  started_ms INTEGER NOT NULL,
  ended_ms INTEGER,
  PRIMARY KEY(project,experiment_id,attempt),
  FOREIGN KEY(project,experiment_id) REFERENCES experiment_dag(project,id),
  FOREIGN KEY(project,job_id) REFERENCES jobs(project,id)
);

CREATE TABLE trial_results(
  project TEXT NOT NULL REFERENCES projects(id),
  experiment_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('evaluated','infeasible','failed','pruned')),
  primary_metric REAL,
  metrics TEXT NOT NULL,
  constraints TEXT NOT NULL,
  is_feasible INTEGER NOT NULL CHECK(is_feasible IN (0,1)),
  train_ms REAL NOT NULL,
  model_sha256 TEXT NOT NULL,
  predictions_sha256 TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  committed_ms INTEGER NOT NULL,
  PRIMARY KEY(project,experiment_id),
  FOREIGN KEY(project,experiment_id) REFERENCES experiment_dag(project,id)
);

PRAGMA user_version=4;
