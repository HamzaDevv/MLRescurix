CREATE TABLE agent_roles(
  project TEXT NOT NULL REFERENCES projects(id),
  role_id TEXT NOT NULL,
  version TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  system_prompt TEXT NOT NULL,
  capabilities TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, role_id)
);

CREATE TABLE agent_tasks(
  project TEXT NOT NULL REFERENCES projects(id),
  task_id TEXT NOT NULL,
  role_id TEXT NOT NULL,
  parent_task_id TEXT,
  status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'completed', 'failed', 'cancelled')),
  hypothesis TEXT,
  intent TEXT NOT NULL,
  assigned_branch TEXT,
  created_ms INTEGER NOT NULL,
  updated_ms INTEGER NOT NULL,
  PRIMARY KEY(project, task_id)
);

CREATE TABLE agent_checkpoints(
  project TEXT NOT NULL REFERENCES projects(id),
  checkpoint_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  role_id TEXT NOT NULL,
  handoff_version INTEGER NOT NULL,
  context_tokens INTEGER NOT NULL,
  state_snapshot TEXT NOT NULL,
  active_jobs TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, checkpoint_id)
);

CREATE TABLE achievements(
  project TEXT NOT NULL REFERENCES projects(id),
  achievement_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  evidence_sha256 TEXT,
  metrics TEXT,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, achievement_id)
);

CREATE TABLE handoffs(
  project TEXT NOT NULL REFERENCES projects(id),
  handoff_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  from_role TEXT NOT NULL,
  to_role TEXT NOT NULL,
  version INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending', 'accepted', 'rejected', 'superseded')),
  summary TEXT NOT NULL,
  context_payload TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, handoff_id)
);

CREATE TABLE scientific_findings(
  project TEXT NOT NULL REFERENCES projects(id),
  finding_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  hypothesis_id TEXT NOT NULL,
  verdict TEXT NOT NULL CHECK(verdict IN ('supported', 'rejected', 'inconclusive')),
  rationale TEXT NOT NULL,
  metrics TEXT NOT NULL,
  artifacts TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, finding_id)
);

CREATE TABLE branch_workspaces(
  project TEXT NOT NULL REFERENCES projects(id),
  branch_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('active', 'merged', 'abandoned', 'clean')),
  base_commit TEXT NOT NULL,
  current_commit TEXT NOT NULL,
  head_sha256 TEXT NOT NULL,
  dirty_files TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  updated_ms INTEGER NOT NULL,
  PRIMARY KEY(project, branch_id)
);

CREATE TABLE agent_tools(
  project TEXT NOT NULL REFERENCES projects(id),
  tool_name TEXT NOT NULL,
  version TEXT NOT NULL,
  descriptor_json TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  environment_id TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, tool_name, version)
);

CREATE TABLE provider_accounting(
  project TEXT NOT NULL REFERENCES projects(id),
  request_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  role_id TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  completion_tokens INTEGER NOT NULL,
  cost_millicents INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  status TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, request_id)
);

PRAGMA user_version=5;
