CREATE TABLE scientific_ensembles(
  project TEXT NOT NULL REFERENCES projects(id),
  ensemble_id TEXT NOT NULL,
  recipe_type TEXT NOT NULL CHECK(recipe_type IN ('voting', 'stacking', 'averaging')),
  parent_ids TEXT NOT NULL,
  weights TEXT,
  meta_params TEXT,
  is_feasible INTEGER NOT NULL CHECK(is_feasible IN (0, 1)),
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, ensemble_id)
);

CREATE TABLE feature_transforms(
  project TEXT NOT NULL REFERENCES projects(id),
  transform_id TEXT NOT NULL,
  transform_spec TEXT NOT NULL,
  parent_dataset TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, transform_id)
);

CREATE TABLE calibration_records(
  project TEXT NOT NULL REFERENCES projects(id),
  calibration_id TEXT NOT NULL,
  base_candidate_id TEXT NOT NULL,
  method TEXT NOT NULL CHECK(method IN ('isotonic', 'sigmoid')),
  brier_score REAL,
  log_loss REAL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, calibration_id)
);

PRAGMA user_version=6;
