CREATE TABLE IF NOT EXISTS custom_protocols (
  project TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  dataset_sha256 TEXT NOT NULL,
  protocol_sha256 TEXT NOT NULL,
  protocol_json TEXT NOT NULL,
  PRIMARY KEY(project, dataset_id),
  FOREIGN KEY(project, dataset_id) REFERENCES scientific_datasets(project, id)
);
CREATE TRIGGER IF NOT EXISTS custom_protocol_no_update BEFORE UPDATE ON custom_protocols
BEGIN SELECT RAISE(ABORT, 'custom protocol is immutable'); END;
CREATE TRIGGER IF NOT EXISTS custom_protocol_no_delete BEFORE DELETE ON custom_protocols
BEGIN SELECT RAISE(ABORT, 'custom protocol is immutable'); END;

CREATE TABLE IF NOT EXISTS assessment_lockbox_consumptions (
  project TEXT NOT NULL,
  dataset_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  artifact_sha256 TEXT NOT NULL,
  predictions_sha256 TEXT NOT NULL,
  metric TEXT NOT NULL,
  score REAL NOT NULL,
  receipt_identity TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  consumed_at_ms INTEGER NOT NULL,
  PRIMARY KEY(project, dataset_id),
  FOREIGN KEY(project, dataset_id) REFERENCES scientific_datasets(project, id)
);
CREATE TRIGGER IF NOT EXISTS assessment_consumption_no_update BEFORE UPDATE ON assessment_lockbox_consumptions
BEGIN SELECT RAISE(ABORT, 'assessment consumption is immutable'); END;
CREATE TRIGGER IF NOT EXISTS assessment_consumption_no_delete BEFORE DELETE ON assessment_lockbox_consumptions
BEGIN SELECT RAISE(ABORT, 'assessment consumption is immutable'); END;
