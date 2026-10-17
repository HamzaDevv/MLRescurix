use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::params;
use serde_json::{json, Value};

use mlrescurix_harness::{
    agents::{
        gateway::{ProviderGateway, ScriptedProviderFixture},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
    },
    core::{Operation, Policy, Request},
    execution,
    rescue::{
        get_rescue_session, list_rescue_sessions,
        record_rescue_session, update_rescue_session_status, AutonomousRescueSession,
        DatasetProfile, RescueTurnRecord,
    },
    Runtime,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy { max_tasks: 1024, budget: b, capacity: c }
}

fn test_context(base: &str, code: &str) -> mlrescurix_harness::contracts::Context {
    mlrescurix_harness::contracts::Context {
        role_id: "lead".into(),
        role_version: "1".into(),
        prompt_hash: "prompt-hash-p104".into(),
        model: None,
        phase: "autonomous_rescue".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "synthesize_candidate".into(),
        stop_conditions: vec!["budget".into()],
    }
}

fn setup_p10_project(root: &Path, project: &str, task_id: &str, role: &str) -> AgentOrchestrator {
    if !root.join("OWNED").exists() {
        if root.exists() {
            let _ = fs::remove_dir_all(root);
        }
        Runtime::init(root, test_policy(100)).unwrap();
    }
    execution::migrate(root).unwrap();
    let r = Runtime::open(root).unwrap();
    let branch = format!("b-{task_id}");

    let _ = r.command(Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P10.4 autonomous rescue gate test".into(),
            finalization_id: format!("final-{project}"),
            policy: test_policy(100),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("branch-{task_id}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: branch.clone(),
            base_hash: "base-p104".into(),
            code_hash: "code-p104".into(),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("task-{task_id}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.into(),
            branch_id: branch.clone(),
            context: test_context("base-p104", "code-p104"),
        },
    }).unwrap();

    let now = now_ms();
    let _ = r.command(Request {
        command_id: format!("claim-{task_id}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::Claim {
            task_id: task_id.into(),
            owner: format!("agent-{task_id}"),
            lease_id: format!("lease-{task_id}"),
            now_ms: now,
            expires_ms: now + 3_600_000,
            expected_generation: 1,
        },
    }).unwrap();

    let config = OrchestratorConfig {
        project_id: project.into(),
        task_id: task_id.into(),
        role_id: role.into(),
        branch_id: branch,
        model: "dummy-model".into(),
        max_turns: 15,
    };

    let gw: Arc<dyn ProviderGateway> = Arc::new(ScriptedProviderFixture::new(vec![]));
    AgentOrchestrator::new(root, config, gw).unwrap()
}

fn create_synthetic_csv(path: &Path, num_rows: usize) -> (String, i64) {
    let mut lines = Vec::new();
    lines.push("row_id,feature_num1,feature_num2,feature_cat,target".to_string());
    for i in 0..num_rows {
        let cat = match i % 3 {
            0 => "alpha",
            1 => "beta",
            _ => "gamma",
        };
        let target = if (i % 2 == 0 && i % 3 != 0) || (i % 5 == 0) { 1 } else { 0 };
        lines.push(format!("r_{i},{:.3},{:.3},{},{}", (i as f64) * 0.1, (i as f64) * -0.05, cat, target));
    }
    let content = lines.join("\n");
    fs::write(path, &content).unwrap();
    let bytes = content.len() as i64;
    use sha2::{Digest, Sha256};
    let sha256 = format!("{:x}", Sha256::digest(content.as_bytes()));
    (sha256, bytes)
}

#[test]
fn p10_4_schema_v10_persistence_and_rescue_session_model() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path();
    let _ = setup_p10_project(root, "p-rescue-db", "t-rescue-db", "lead");

    // 1. Verify PRAGMA user_version remains strictly 10
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let version: i32 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert!(version >= 10, "PRAGMA user_version must be at least 10, got {version}");

    // 2. Verify autonomous_rescue_sessions table and indexes exist
    let table_exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='autonomous_rescue_sessions')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(table_exists, "autonomous_rescue_sessions table must exist");

    let idx_exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_rescue_sessions_proj_status')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(idx_exists, "idx_rescue_sessions_proj_status index must exist");

    // 3. Record an AutonomousRescueSession domain model
    let now = now_ms();
    let session = AutonomousRescueSession {
        schema_version: "v2".into(),
        session_id: "rescue-test-001".into(),
        project_id: "p-rescue-db".into(),
        dataset_id: "ds_test".into(),
        status: "initialized".into(),
        initial_candidate_id: Some("cand_base".into()),
        apex_candidate_id: None,
        bundle_path: None,
        turns_executed: 1,
        dataset_profile: Some(DatasetProfile {
            columns: vec!["row_id".into(), "f1".into(), "f2".into(), "target".into()],
            row_count: 500,
            column_count: 4,
            null_counts: BTreeMap::new(),
            null_percentages: BTreeMap::new(),
            cardinalities: BTreeMap::new(),
            dtypes: BTreeMap::new(),
            feature_columns: vec!["f1".into(), "f2".into()],
            target_column: Some("target".into()),
            id_column: Some("row_id".into()),
            class_balance: Some(json!({"0": 250, "1": 250})),
            inferred_task_type: "binary_classification".into(),
        }),
        turns: vec![RescueTurnRecord {
            turn_index: 1,
            phase: "profiling".into(),
            action: "profile_dataset".into(),
            candidate_id: None,
            status: "completed".into(),
            details: json!({"rows": 500}),
        }],
        failure_interception: None,
        hypotheses: vec![],
        pareto_frontier_id: None,
        finalization: None,
        resource_accounting: None,
        created_ms: now,
        updated_ms: now,
    };

    record_rescue_session(root, "p-rescue-db", &session).unwrap();

    // 4. Query via get_rescue_session
    let loaded = get_rescue_session(root, "p-rescue-db", "rescue-test-001").unwrap();
    assert_eq!(loaded.session_id, "rescue-test-001");
    assert_eq!(loaded.status, "initialized");
    assert_eq!(loaded.dataset_profile.as_ref().unwrap().row_count, 500);

    // 5. Query via list_rescue_sessions
    let sessions = list_rescue_sessions(root, "p-rescue-db", 10).unwrap();
    assert_eq!(sessions.len(), 1);

    // 6. Update status
    update_rescue_session_status(
        root,
        "p-rescue-db",
        "rescue-test-001",
        "finalized",
        Some("cand_opt"),
        Some("/workflows/p-rescue-db/bundle"),
        6,
    ).unwrap();

    let updated = get_rescue_session(root, "p-rescue-db", "rescue-test-001").unwrap();
    assert_eq!(updated.status, "finalized");
    assert_eq!(updated.apex_candidate_id, Some("cand_opt".into()));
    assert_eq!(updated.turns_executed, 6);
    assert_eq!(updated.bundle_path, Some("/workflows/p-rescue-db/bundle".into()));
}

#[test]
fn p10_4_dataset_profiling_tool_dispatch() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path();
    let mut orchestrator = setup_p10_project(root, "p-prof-test", "t-prof-test", "lead");

    // Create synthetic dataset CSV
    let ds_csv = root.join("dataset.csv");
    let (ds_sha, ds_bytes) = create_synthetic_csv(&ds_csv, 600);

    // Register into scientific_datasets
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params!["p-prof-test", "ds_synth", ds_csv.to_string_lossy(), ds_bytes, ds_sha, now_ms()],
    ).unwrap();

    // 1. Dispatch profile_dataset tool directly
    let res = orchestrator.dispatch_tool_call(
        "profile_dataset",
        &json!({
            "dataset_id": "ds_synth",
            "target_column": "target",
            "id_column": "row_id",
        }).to_string(),
    ).unwrap();

    let prof: Value = serde_json::from_str(&res).unwrap();
    assert_eq!(prof["row_count"].as_u64(), Some(600));
    assert_eq!(prof["column_count"].as_u64(), Some(5));
    assert_eq!(prof["target_column"].as_str(), Some("target"));
    assert_eq!(prof["inferred_task_type"].as_str(), Some("binary_classification"));

    // 2. Dispatch dataset_query with profile: true
    let query_res = orchestrator.dispatch_tool_call(
        "dataset_query",
        &json!({
            "dataset_id": "ds_synth",
            "profile": true,
            "target_column": "target",
        }).to_string(),
    ).unwrap();

    let qval: Value = serde_json::from_str(&query_res).unwrap();
    assert_eq!(qval["row_count"].as_u64(), Some(600));
    assert_eq!(qval["dataset_id"].as_str(), Some("ds_synth"));
    assert!(qval["profile"].is_object(), "dataset_query must include profile object when profile=true");
    assert_eq!(qval["profile"]["inferred_task_type"].as_str(), Some("binary_classification"));
}

#[test]
fn p10_4_full_autonomous_rescue_closed_loop_and_bundle_delivery() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path();
    let mut orchestrator = setup_p10_project(root, "p-auto-rescue", "t-auto-rescue", "lead");

    // 1. Register synthetic dataset
    let ds_csv = root.join("dataset.csv");
    let (ds_sha, ds_bytes) = create_synthetic_csv(&ds_csv, 600);

    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params!["p-auto-rescue", "ds_main", ds_csv.to_string_lossy(), ds_bytes, ds_sha, now_ms()],
    ).unwrap();

    // 2. Synthesize baseline candidate
    let synth_res = orchestrator.dispatch_tool_call(
        "synthesize_candidate",
        &json!({
            "candidate_id": "cand_baseline",
            "task_type": "binary",
            "target_column": "target",
            "model_family": "boosting",
            "framework": "lightgbm",
            "candidate_path": "candidates/cand_baseline",
        }).to_string(),
    ).unwrap();
    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"].as_str(), Some("synthesized"));

    // 3. Propose iteration hypothesis
    let hyp_res = orchestrator.dispatch_tool_call(
        "propose_hypothesis",
        &json!({
            "candidate_id": "cand_baseline",
            "mutation_type": "hyperparameter_tuning",
            "hypothesis_statement": "Tune regularization and learning rate for LightGBM baseline",
            "expected_metric": "balanced_accuracy",
            "direction": "maximize",
            "minimum_improvement": 0.005,
        }).to_string(),
    ).unwrap();
    let hyp_val: Value = serde_json::from_str(&hyp_res).unwrap();
    let hyp_id = hyp_val["hypothesis_id"].as_str().unwrap();

    // 4. Multi-fidelity candidate screening (Tier 0 fast screen + Tier 1 CV)
    let screen_res = orchestrator.dispatch_tool_call(
        "screen_candidate_fidelity",
        &json!({
            "candidate_id": "cand_baseline",
            "dataset_id": "ds_main",
            "hypothesis_id": hyp_id,
            "candidate_path": "candidates/cand_baseline",
            "folds": 2,
            "seed": 2026,
        }).to_string(),
    ).unwrap();
    let screen_val: Value = serde_json::from_str(&screen_res).unwrap();
    assert_eq!(screen_val["status"].as_str(), Some("completed"));

    // 5. Query Pareto frontier
    let pareto_res = orchestrator.dispatch_tool_call(
        "query_pareto_frontier",
        &json!({}).to_string(),
    ).unwrap();
    let pareto_val: Value = serde_json::from_str(&pareto_res).unwrap();
    let records = pareto_val["records"].as_array().unwrap();
    assert!(!records.is_empty(), "Pareto frontier must contain evaluated baseline record");
    assert_eq!(records[0]["candidate_id"].as_str(), Some("cand_baseline"));

    // 6. Independent critic audit submission
    let mut critic_orchestrator = setup_p10_project(root, "p-auto-rescue", "t-critic-auto", "critic");
    let audit_res = critic_orchestrator.dispatch_tool_call(
        "submit_critic_audit",
        &json!({
            "candidate_id": "cand_baseline",
            "verdict": "approved",
            "protocol_compliance": true,
            "evidence_summary": "Passed strict multi-fidelity screening, balanced accuracy > 0.60, zero data leakage verified.",
        }).to_string(),
    ).unwrap();
    let audit_val: Value = serde_json::from_str(&audit_res).unwrap();
    assert_eq!(audit_val["audit_status"].as_str(), Some("approved"));

    // 7. Record authenticated user approval for final test gate
    mlrescurix_harness::agents::task_contract::register_user_approval(
        root,
        "p-auto-rescue",
        "appr-cand-01",
        "final_test",
        "cand_baseline",
        "human_reviewer",
        true,
        3_600_000,
    ).unwrap();

    // Also authorize test_access_id
    let test_access_id = "test-gate-p104-001";
    mlrescurix_harness::agents::task_contract::register_user_approval(
        root,
        "p-auto-rescue",
        "appr-gate-01",
        "final_test",
        test_access_id,
        "human_reviewer",
        true,
        3_600_000,
    ).unwrap();

    // 8. Request finalization producing Apex-certified mlrescurix.bundle.v2
    let finalize_res = orchestrator.dispatch_tool_call(
        "request_finalization",
        &json!({
            "candidate_id": "cand_baseline",
            "candidate_path": "candidates/cand_baseline",
            "dataset_id": "ds_main",
            "test_access_id": test_access_id,
            "rationale": "Autonomous rescue complete: optimal non-dominated Pareto candidate verified.",
            "task_type": "binary_classification",
            "target_column": "target",
            "primary_metric": "balanced_accuracy",
        }).to_string(),
    ).unwrap();

    let fin_val: Value = serde_json::from_str(&finalize_res).unwrap();
    assert_eq!(fin_val["status"].as_str(), Some("finalized_and_delivered"));
    assert_eq!(fin_val["candidate_id"].as_str(), Some("cand_baseline"));
    assert_eq!(fin_val["test_access_id"].as_str(), Some(test_access_id));

    // 9. Verify bundle on disk using Rust bundle verifier
    let bundle_dir = root.join("workflows").join("p-auto-rescue").join("bundle");
    assert!(bundle_dir.is_dir());

    let bundle_report = mlrescurix_harness::bundle::verify_bundle(&bundle_dir, false).unwrap();
    assert_eq!(bundle_report.status, "integrity_verified");
    assert!(bundle_report.manifest_verified);
    assert!(bundle_report.contracts_verified);
    assert_eq!(bundle_report.format, "mlrescurix.bundle.v2");

    // 10. Verify project state transitioned to 'completed'
    let proj_state: String = db.query_row(
        "SELECT state FROM projects WHERE id=?1",
        params!["p-auto-rescue"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(proj_state, "completed");
}

#[test]
fn p10_4_negative_holdout_access_refusal_without_approval() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path();
    let mut orchestrator = setup_p10_project(root, "p-neg-test", "t-neg-test", "lead");

    // Register synthetic dataset
    let ds_csv = root.join("dataset.csv");
    let (ds_sha, ds_bytes) = create_synthetic_csv(&ds_csv, 300);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params!["p-neg-test", "ds_neg", ds_csv.to_string_lossy(), ds_bytes, ds_sha, now_ms()],
    ).unwrap();

    // Synthesize candidate
    orchestrator.dispatch_tool_call(
        "synthesize_candidate",
        &json!({
            "candidate_id": "cand_unapproved",
            "task_type": "binary",
            "target_column": "target",
            "candidate_path": "candidates/cand_unapproved",
        }).to_string(),
    ).unwrap();

    // Attempt finalization without critic audit or user approval
    let err = orchestrator.dispatch_tool_call(
        "request_finalization",
        &json!({
            "candidate_id": "cand_unapproved",
            "rationale": "Attempting premature unapproved finalization",
        }).to_string(),
    ).unwrap_err();

    assert!(err.contains("Finalization refused"), "Must refuse finalization without critic audit: {err}");

    // Submit critic audit as approved, but omit user approval
    let mut critic_orchestrator = setup_p10_project(root, "p-neg-test", "t-neg-critic", "critic");
    critic_orchestrator.dispatch_tool_call(
        "submit_critic_audit",
        &json!({
            "candidate_id": "cand_unapproved",
            "verdict": "approved",
            "protocol_compliance": true,
            "evidence_summary": "Passed critic audit",
        }).to_string(),
    ).unwrap();

    let err2 = orchestrator.dispatch_tool_call(
        "request_finalization",
        &json!({
            "candidate_id": "cand_unapproved",
            "rationale": "Attempting finalization without user approval",
        }).to_string(),
    ).unwrap_err();

    assert!(err2.contains("user approval"), "Must refuse finalization without user approval: {err2}");
}

#[test]
fn p10_4_negative_role_rbac_authorization() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path();
    // Setup with 'auditor' role (which lacks package_candidate_bundle and request_finalization)
    let mut orchestrator = setup_p10_project(root, "p-rbac-test", "t-rbac-test", "auditor");

    // 1. Auditor attempts to call package_candidate_bundle
    let err = orchestrator.dispatch_tool_call(
        "package_candidate_bundle",
        &json!({
            "candidate_id": "cand_x",
            "dataset_id": "ds_x",
        }).to_string(),
    ).unwrap_err();

    assert!(err.contains("Role 'auditor' is not authorized to invoke tool 'package_candidate_bundle'"), "RBAC violation must be rejected: {err}");

    // 2. Auditor attempts to call request_finalization
    let err_fin = orchestrator.dispatch_tool_call(
        "request_finalization",
        &json!({
            "candidate_id": "cand_x",
            "rationale": "Unauthorized finalization",
        }).to_string(),
    ).unwrap_err();

    assert!(err_fin.contains("Role 'auditor' is not authorized to invoke tool 'request_finalization'"), "RBAC violation must be rejected: {err_fin}");
}
