CREATE TABLE scientific_datasets(
  project TEXT NOT NULL REFERENCES projects(id),
  id TEXT NOT NULL,
  path TEXT NOT NULL,
  bytes INTEGER NOT NULL CHECK(bytes BETWEEN 1 AND 33554432),
  sha256 TEXT NOT NULL,
  registered_ms INTEGER NOT NULL,
  PRIMARY KEY(project,id)
);
CREATE TABLE scientific_jobs(
  project TEXT NOT NULL,
  job TEXT NOT NULL,
  dataset TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN('audit','baselines','finalize')),
  spec TEXT NOT NULL,
  result_sha256 TEXT,
  PRIMARY KEY(project,job),
  FOREIGN KEY(project,job) REFERENCES jobs(project,id),
  FOREIGN KEY(project,dataset) REFERENCES scientific_datasets(project,id)
);
CREATE TABLE final_test_gates(
  project TEXT NOT NULL REFERENCES projects(id),
  access_id TEXT NOT NULL,
  dataset_sha256 TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  frozen_model_sha256 TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL CHECK(status IN('reserved','consumed')),
  result_sha256 TEXT,
  PRIMARY KEY(project,access_id),
  UNIQUE(project,dataset_sha256)
);
PRAGMA user_version=3;
