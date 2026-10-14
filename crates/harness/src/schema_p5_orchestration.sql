-- Phase P5 orchestration and collaboration tables (additive, user_version = 10)

CREATE TABLE IF NOT EXISTS task_assignments (
  project TEXT NOT NULL REFERENCES projects(id),
  task_id TEXT NOT NULL,
  role_id TEXT NOT NULL,
  goal TEXT NOT NULL,
  hypothesis TEXT,
  immutable_inputs TEXT NOT NULL,
  allowed_tools TEXT NOT NULL,
  branch_ownership TEXT NOT NULL,
  limits TEXT NOT NULL,
  expected_artifacts TEXT NOT NULL,
  acceptance_criteria TEXT NOT NULL,
  dependency_ids TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, task_id)
);

CREATE TABLE IF NOT EXISTS task_completions (
  project TEXT NOT NULL REFERENCES projects(id),
  task_id TEXT NOT NULL,
  verdict TEXT NOT NULL CHECK(verdict IN ('succeeded', 'failed', 'inconclusive')),
  experiment_id TEXT,
  candidate_id TEXT,
  source_hash TEXT,
  metric_delta TEXT,
  protocol TEXT,
  failures TEXT NOT NULL,
  bounded_learnings TEXT NOT NULL,
  next_actions TEXT NOT NULL,
  artifacts_produced TEXT NOT NULL,
  completed_ms INTEGER NOT NULL,
  PRIMARY KEY(project, task_id)
);

CREATE TABLE IF NOT EXISTS steering_directives (
  project TEXT NOT NULL REFERENCES projects(id),
  directive_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  actor TEXT NOT NULL,
  epoch INTEGER NOT NULL,
  directive TEXT NOT NULL,
  created_ms INTEGER NOT NULL,
  applied INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(project, directive_id)
);

CREATE TABLE IF NOT EXISTS user_approvals (
  project TEXT NOT NULL REFERENCES projects(id),
  approval_id TEXT NOT NULL,
  action_type TEXT NOT NULL CHECK(action_type IN ('holdout_access', 'deployment_handoff', 'final_test')),
  target_id TEXT NOT NULL,
  actor TEXT NOT NULL,
  approved INTEGER NOT NULL CHECK(approved IN (0, 1)),
  created_ms INTEGER NOT NULL,
  expires_ms INTEGER NOT NULL,
  PRIMARY KEY(project, approval_id)
);

CREATE INDEX IF NOT EXISTS idx_task_assignments_project_role ON task_assignments(project, role_id);
CREATE INDEX IF NOT EXISTS idx_task_completions_project ON task_completions(project);
CREATE INDEX IF NOT EXISTS idx_steering_directives_task ON steering_directives(project, task_id, epoch);
CREATE INDEX IF NOT EXISTS idx_user_approvals_lookup ON user_approvals(project, action_type, target_id, approved, expires_ms);
