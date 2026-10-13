CREATE TABLE experiment_dag_v7(
  project TEXT NOT NULL REFERENCES projects(id),
  session_id TEXT NOT NULL,
  id TEXT NOT NULL,
  recipe TEXT NOT NULL CHECK(recipe IN ('dummy','linear','tree','hist_boosting')),
  parameters TEXT NOT NULL,
  seed INTEGER NOT NULL,
  fidelity REAL NOT NULL CHECK(fidelity > 0.0 AND fidelity <= 1.0),
  node_type TEXT NOT NULL CHECK(node_type IN ('baseline','screening','full_fidelity','repeat')),
  parent_id TEXT,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project,id),
  FOREIGN KEY(project,session_id) REFERENCES search_sessions(project,id)
);
INSERT INTO experiment_dag_v7 SELECT * FROM experiment_dag;
DROP TABLE experiment_dag;
ALTER TABLE experiment_dag_v7 RENAME TO experiment_dag;
PRAGMA user_version=7;
