CREATE TABLE jobs(
  id TEXT NOT NULL,
  project TEXT NOT NULL REFERENCES projects(id),
  task TEXT NOT NULL,
  generation INTEGER NOT NULL,
  spec TEXT NOT NULL,
  action TEXT,
  status TEXT NOT NULL CHECK(status IN('queued','running','terminating','recovering','succeeded','failed','timed_out','cancelled','rejected','interrupted')),
  reason TEXT,
  pid INTEGER,
  submitted_ms INTEGER NOT NULL,
  started_ms INTEGER,
  ended_ms INTEGER,
  stage TEXT,
  log_path TEXT,
  artifacts TEXT NOT NULL DEFAULT '[]',
  cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK(cancel_requested IN(0,1)),
  recovered_deadline_ms INTEGER,
  PRIMARY KEY(project,id),
  FOREIGN KEY(project,task) REFERENCES tasks(project,id)
);
CREATE INDEX jobs_queue ON jobs(status,project,submitted_ms,id);
CREATE TABLE scheduler(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE project_weights(project TEXT PRIMARY KEY REFERENCES projects(id),weight INTEGER NOT NULL CHECK(weight BETWEEN 1 AND 10));
CREATE TABLE managed_artifacts(project TEXT NOT NULL REFERENCES projects(id),job TEXT NOT NULL,name TEXT NOT NULL,bytes INTEGER NOT NULL,sha256 TEXT NOT NULL,pinned INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(project,job,name),FOREIGN KEY(project,job) REFERENCES jobs(project,id));
CREATE TABLE environments(id TEXT PRIMARY KEY,python TEXT NOT NULL,identity TEXT NOT NULL,path TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN('building','ready','deleting','failed')));
CREATE TABLE environment_leases(environment TEXT NOT NULL REFERENCES environments(id),project TEXT NOT NULL REFERENCES projects(id),job TEXT NOT NULL,PRIMARY KEY(environment,project,job));
CREATE TABLE job_telemetry(project TEXT NOT NULL,job TEXT NOT NULL,sampled_ms INTEGER NOT NULL,cpu_percent REAL,rss_bytes INTEGER,peak_rss_bytes INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(project,job),FOREIGN KEY(project,job) REFERENCES jobs(project,id));
PRAGMA user_version=2;
