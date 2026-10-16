-- Phase P10.3 Multi-Fidelity Iteration & Hypothesis Loop (additive, user_version = 10)
CREATE TABLE IF NOT EXISTS iteration_hypotheses (
    project TEXT NOT NULL REFERENCES projects(id),
    hypothesis_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    parent_candidate_id TEXT,
    mutation_type TEXT NOT NULL CHECK(mutation_type IN ('feature_engineering', 'hyperparameter_tuning', 'architecture_change', 'ensembling')),
    hypothesis_statement TEXT NOT NULL,
    expected_metric TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('maximize', 'minimize')),
    target_threshold REAL,
    minimum_improvement REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'admitted', 'rejected', 'confirmed', 'refuted', 'inconclusive')),
    baseline_metric REAL,
    observed_metric REAL,
    metric_delta REAL,
    settlement_reason TEXT,
    hypothesis_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    settled_ms INTEGER,
    PRIMARY KEY(project, hypothesis_id)
);
CREATE INDEX IF NOT EXISTS idx_iteration_hypotheses_proj_cand ON iteration_hypotheses(project, candidate_id);
CREATE INDEX IF NOT EXISTS idx_iteration_hypotheses_status ON iteration_hypotheses(project, status);

CREATE TABLE IF NOT EXISTS pareto_frontier_records (
    project TEXT NOT NULL REFERENCES projects(id),
    frontier_id TEXT NOT NULL,
    primary_metric_name TEXT NOT NULL,
    optimization_direction TEXT NOT NULL CHECK(optimization_direction IN ('maximize', 'minimize')),
    records_json TEXT NOT NULL,
    frontier_candidate_ids_json TEXT NOT NULL,
    non_dominated_count INTEGER NOT NULL,
    dominated_count INTEGER NOT NULL,
    evaluated_at_ms INTEGER NOT NULL,
    PRIMARY KEY(project, frontier_id)
);
CREATE INDEX IF NOT EXISTS idx_pareto_frontier_project ON pareto_frontier_records(project);
