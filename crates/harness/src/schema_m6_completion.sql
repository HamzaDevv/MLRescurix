CREATE TABLE scientific_jobs_v8(
  project TEXT NOT NULL, job TEXT NOT NULL, dataset TEXT NOT NULL,
  operation TEXT NOT NULL CHECK(operation IN('audit','baselines','finalize','trial','pipeline')),
  spec TEXT NOT NULL, result_sha256 TEXT,
  PRIMARY KEY(project,job),
  FOREIGN KEY(project,job) REFERENCES jobs(project,id),
  FOREIGN KEY(project,dataset) REFERENCES scientific_datasets(project,id)
);
INSERT INTO scientific_jobs_v8 SELECT * FROM scientific_jobs;
DROP TABLE scientific_jobs;
ALTER TABLE scientific_jobs_v8 RENAME TO scientific_jobs;
CREATE TABLE pipeline_receipts(
 project TEXT NOT NULL, job TEXT NOT NULL, candidate TEXT NOT NULL,
 dataset_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 environment_identity TEXT NOT NULL, protocol_sha256 TEXT NOT NULL,
 model_sha256 TEXT NOT NULL, predictions_sha256 TEXT NOT NULL,
 oof_sha256 TEXT NOT NULL, receipt_json TEXT NOT NULL,
 PRIMARY KEY(project,candidate),
 FOREIGN KEY(project,job) REFERENCES jobs(project,id)
);
CREATE TABLE content_decisions(
 project TEXT NOT NULL REFERENCES projects(id), candidate TEXT NOT NULL,
 binding_sha256 TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN('user_approval','critic_audit')),
 actor TEXT NOT NULL, approved INTEGER NOT NULL CHECK(approved IN(0,1)),
 rationale TEXT NOT NULL, binding_json TEXT NOT NULL, created_ms INTEGER NOT NULL,
 PRIMARY KEY(project,candidate,kind,created_ms)
);
CREATE TABLE provider_turn_receipts(
 project TEXT NOT NULL REFERENCES projects(id), task TEXT NOT NULL, turn INTEGER NOT NULL,
 action TEXT NOT NULL, response_json TEXT,
 status TEXT NOT NULL CHECK(status IN('awaiting_response','response_received','executing_tools','completed','outcome_unknown')),
 updated_ms INTEGER NOT NULL,
 PRIMARY KEY(project,task,turn)
);
CREATE TABLE outbound_views(
 project TEXT NOT NULL REFERENCES projects(id), request_sha256 TEXT NOT NULL,
 view_json TEXT NOT NULL, actor TEXT NOT NULL, approved INTEGER NOT NULL CHECK(approved IN(0,1)),
 expires_ms INTEGER NOT NULL,
 PRIMARY KEY(project,request_sha256)
);
PRAGMA user_version=8;
