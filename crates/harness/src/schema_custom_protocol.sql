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
