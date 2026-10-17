-- MLRescurix v2 Reflexive Agent-First Architecture Schema Migration: user_version 10 -> 11

-- 1. Immutable Dataset Topology Contracts (Locked by M2-EDA)
CREATE TABLE IF NOT EXISTS dataset_topology_contracts (
    project TEXT NOT NULL REFERENCES projects(id),
    dataset_id TEXT NOT NULL,
    target_column TEXT NOT NULL,
    task_type TEXT NOT NULL CHECK(task_type IN ('binary_classification', 'multiclass_classification', 'regression', 'survival', 'time_series')),
    id_columns_json TEXT NOT NULL,
    entity_group_col TEXT,
    temporal_col TEXT,
    cv_scheme TEXT NOT NULL CHECK(cv_scheme IN ('stratified_kfold', 'stratified_group_kfold', 'group_kfold', 'purged_time_series_split', 'kfold')),
    cv_splits INTEGER NOT NULL DEFAULT 5 CHECK(cv_splits >= 2),
    column_classifications_json TEXT NOT NULL,
    coercions_json TEXT NOT NULL DEFAULT '{}',
    clean_feature_columns_json TEXT NOT NULL,
    row_count INTEGER NOT NULL CHECK(row_count >= 0),
    column_count INTEGER NOT NULL CHECK(column_count >= 0),
    class_balance_json TEXT,
    summary_digest TEXT,
    contract_json TEXT NOT NULL,
    locked_at_ms INTEGER NOT NULL,
    PRIMARY KEY(project, dataset_id)
);
CREATE INDEX IF NOT EXISTS idx_topology_project_dataset ON dataset_topology_contracts(project, dataset_id);

-- 2. Persistent Research Ledger Epochs (M1 Orchestrator Closed-Loop Tracking)
CREATE TABLE IF NOT EXISTS research_ledger_epochs (
    project TEXT NOT NULL REFERENCES projects(id),
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL CHECK(epoch >= 0),
    hypotheses_json TEXT NOT NULL,
    accepted_features_json TEXT NOT NULL,
    rejected_features_json TEXT NOT NULL,
    ablation_results_json TEXT,
    model_results_json TEXT,
    ensemble_result_json TEXT,
    critique_receipt_id TEXT,
    m1_reflection TEXT,
    best_cv_score REAL,
    apex_candidate_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('active', 'converged', 'vetoed', 'aborted', 'finalized')),
    epoch_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, session_id, epoch)
);
CREATE INDEX IF NOT EXISTS idx_ledger_session_epoch ON research_ledger_epochs(project, session_id, epoch);
CREATE INDEX IF NOT EXISTS idx_ledger_status ON research_ledger_epochs(project, status);

-- 3. Feature Ablation Experiment Log (M2-FE Paired CV Results)
CREATE TABLE IF NOT EXISTS feature_ablations (
    project TEXT NOT NULL REFERENCES projects(id),
    ablation_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL CHECK(epoch >= 0),
    hypothesis_id TEXT NOT NULL,
    feature_names_json TEXT NOT NULL,
    baseline_cv_score REAL NOT NULL,
    ablated_cv_score REAL NOT NULL,
    cv_score_delta REAL NOT NULL,
    train_val_gap_delta REAL NOT NULL,
    fold_win_count INTEGER NOT NULL,
    total_folds INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ACCEPTED', 'REJECTED', 'PENDING')),
    importance_scores_json TEXT,
    rejection_reason TEXT,
    evaluated_at_ms INTEGER NOT NULL,
    PRIMARY KEY(project, ablation_id)
);
CREATE INDEX IF NOT EXISTS idx_feature_ablations_session ON feature_ablations(project, session_id, epoch);
CREATE INDEX IF NOT EXISTS idx_feature_ablations_status ON feature_ablations(project, status);

-- 4. Structured Adversarial Critique Receipts (M2-Critic Audit Output)
CREATE TABLE IF NOT EXISTS critique_receipts (
    project TEXT NOT NULL REFERENCES projects(id),
    critique_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    epoch INTEGER NOT NULL CHECK(epoch >= 0),
    status TEXT NOT NULL CHECK(status IN ('APPROVED', 'VETO')),
    blocking_flaws_json TEXT NOT NULL,
    soft_warnings_json TEXT NOT NULL DEFAULT '[]',
    audit_checks_json TEXT NOT NULL,
    error_slice_hypotheses_json TEXT NOT NULL DEFAULT '[]',
    reviewer_notes TEXT,
    receipt_json TEXT NOT NULL,
    evaluated_at_ms INTEGER NOT NULL,
    PRIMARY KEY(project, critique_id)
);
CREATE INDEX IF NOT EXISTS idx_critique_receipts_candidate ON critique_receipts(project, candidate_id);
CREATE INDEX IF NOT EXISTS idx_critique_receipts_session_epoch ON critique_receipts(project, session_id, epoch);
CREATE INDEX IF NOT EXISTS idx_critique_receipts_status ON critique_receipts(project, status);
