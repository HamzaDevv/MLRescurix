-- Rebuild without renaming the old table, so child foreign-key references
-- continue to point to scientific_datasets. The caller turns foreign keys off
-- before the transaction and checks them before committing.
CREATE TABLE scientific_datasets_v12 (
  project TEXT NOT NULL REFERENCES projects(id),
  id TEXT NOT NULL,
  path TEXT NOT NULL,
  bytes INTEGER NOT NULL CHECK(bytes BETWEEN 1 AND 100663296),
  sha256 TEXT NOT NULL,
  registered_ms INTEGER NOT NULL,
  PRIMARY KEY(project,id)
);
INSERT INTO scientific_datasets_v12 SELECT project,id,path,bytes,sha256,registered_ms FROM scientific_datasets;
DROP TABLE scientific_datasets;
ALTER TABLE scientific_datasets_v12 RENAME TO scientific_datasets;
PRAGMA user_version=12;
