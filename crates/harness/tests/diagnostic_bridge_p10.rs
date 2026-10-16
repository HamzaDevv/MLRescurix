use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use serde_json::{json, Value};

use mlrescurix_harness::{
    agents::{
        gateway::{ProviderGateway, ScriptedProviderFixture},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
        workspace::BranchWorkspaceManager,
    },
    core::{Operation, Policy, Request},
    diagnostics::{
        get_diagnostic_receipt, list_diagnostic_receipts, record_diagnostic_receipt,
        DiagnosticBridge,
    },
    execution,
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
        prompt_hash: "prompt-hash-p102".into(),
        model: None,
        phase: "diagnostic_bridge".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "inspect_diagnostics".into(),
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
            intent: "P10.2 diagnostic bridge test".into(),
            finalization_id: format!("final-{project}"),
            policy: test_policy(100),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("branch-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: branch.clone(),
            base_hash: "base-p10".into(),
            code_hash: "code-p10".into(),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("task-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.into(),
            branch_id: branch.clone(),
            context: test_context("base-p10", "code-p10"),
        },
    }).unwrap();

    let now = now_ms();
    let _ = r.command(Request {
        command_id: format!("claim-{project}"),
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
        max_turns: 10,
    };

    let gw: Arc<dyn ProviderGateway> = Arc::new(ScriptedProviderFixture::new(vec![]));
    AgentOrchestrator::new(root, config, gw).unwrap()
}

#[test]
fn p10_2_schema_v10_persistence_and_receipt_retrieval() {
    let td = tempfile::tempdir().unwrap();
    let _ = setup_p10_project(td.path(), "p-diag-db", "t-diag-db", "lead");

    // Verify PRAGMA user_version remains 10
    let db = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let version: i32 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version, 10, "PRAGMA user_version must remain exactly 10");

    // Record receipts for various failure modes
    let r1 = DiagnosticBridge::build_receipt(
        "cand_oom_01",
        "fit",
        Some(137),
        "Killed\nProcess killed by cgroup oom killer",
        "",
        "Killed by cgroup",
        None,
        Some(512 * 1024 * 1024),
        Some(4200),
        Some(45000),
        Some(true),
        Some(json!({"fold": 0})),
    );
    record_diagnostic_receipt(td.path(), "p-diag-db", &r1).unwrap();

    let r2 = DiagnosticBridge::build_receipt(
        "cand_nan_01",
        "fit",
        Some(1),
        "FloatingPointError: Loss is nan in epoch 4\n  File \"candidate/model.py\", line 42, in train",
        "",
        "Loss is nan",
        None,
        None,
        Some(1500),
        Some(45000),
        None,
        Some(json!({"fold": 1})),
    );
    record_diagnostic_receipt(td.path(), "p-diag-db", &r2).unwrap();

    // Query single receipt
    let fetched = get_diagnostic_receipt(td.path(), "p-diag-db", &r1.diagnostic_id).unwrap();
    assert_eq!(fetched.diagnostic_id, r1.diagnostic_id);
    assert_eq!(fetched.failure_type, "cgroup_oom");
    assert_eq!(fetched.resource_telemetry.oom_killed, Some(true));
    assert!(fetched.remediation_suggestions.len() >= 2);

    // List receipts filtered by candidate
    let list_c1 = list_diagnostic_receipts(td.path(), "p-diag-db", Some("cand_oom_01"), None, 10).unwrap();
    assert_eq!(list_c1.len(), 1);
    assert_eq!(list_c1[0].diagnostic_id, r1.diagnostic_id);

    // List receipts filtered by failure_type
    let list_nan = list_diagnostic_receipts(td.path(), "p-diag-db", None, Some("non_finite_loss"), 10).unwrap();
    assert_eq!(list_nan.len(), 1);
    assert_eq!(list_nan[0].diagnostic_id, r2.diagnostic_id);

    // List all
    let all = list_diagnostic_receipts(td.path(), "p-diag-db", None, None, 10).unwrap();
    assert_eq!(all.len(), 2);
}

#[test]
fn p10_2_inspect_diagnostics_tool_and_discovery() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-diag-tool", "t-diag-tool", "lead");

    // 1. Tool discovery
    let out = orch.dispatch_tool_call("discover_capabilities", "{}").unwrap();
    let val: Value = serde_json::from_str(&out).unwrap();
    let tools = val["tools"].as_array().expect("tools array");
    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(tool_names.contains(&"inspect_diagnostics"), "ToolRegistry missing inspect_diagnostics");

    // 2. Insert dummy receipt
    let receipt = DiagnosticBridge::build_receipt(
        "cand_tool_01",
        "evaluate",
        Some(1),
        "ValueError: shapes (10, 4) and (10, 5) not aligned\n  File \"candidate/model.py\", line 12, in predict",
        "",
        "Dimension mismatch",
        None,
        None,
        Some(800),
        None,
        None,
        None,
    );
    record_diagnostic_receipt(td.path(), "p-diag-tool", &receipt).unwrap();

    // 3. Query via inspect_diagnostics tool
    let tool_out = orch.dispatch_tool_call("inspect_diagnostics", &json!({
        "candidate_id": "cand_tool_01",
    }).to_string()).unwrap();

    let res: Value = serde_json::from_str(&tool_out).unwrap();
    assert_eq!(res["count"], 1);
    let diags = res["diagnostics"].as_array().unwrap();
    assert_eq!(diags[0]["failure_type"], "shape_mismatch");
    assert_eq!(diags[0]["root_cause"]["line_number"], 12);
    assert_eq!(diags[0]["root_cause"]["symbol"], "predict");

    // 4. Query by exact diagnostic_id
    let single_out = orch.dispatch_tool_call("inspect_diagnostics", &json!({
        "diagnostic_id": receipt.diagnostic_id,
    }).to_string()).unwrap();
    let single_res: Value = serde_json::from_str(&single_out).unwrap();
    assert_eq!(single_res["diagnostic_id"], receipt.diagnostic_id);

    // 5. RBAC enforcement: critic role cannot invoke inspect_diagnostics
    let mut critic_orch = setup_p10_project(td.path(), "p-diag-critic", "t-diag-critic", "critic");
    let unauth = critic_orch.dispatch_tool_call("inspect_diagnostics", "{}");
    assert!(unauth.is_err(), "Critic role must be refused inspect_diagnostics");
    assert!(unauth.unwrap_err().contains("not authorized"));
}

#[test]
fn p10_2_fault_injection_cgroup_oom_in_sandbox() {
    let td = tempfile::tempdir().unwrap();
    let _orch = setup_p10_project(td.path(), "p-diag-oom", "t-diag-oom", "lead");

    // Direct diagnosis test simulating Docker container killed with code 137 (SIGKILL OOM)
    let diag = DiagnosticBridge::build_receipt(
        "cand_oom_test",
        "fit",
        Some(137),
        "Killed\nProcess exceeded cgroup v2 memory limit",
        "",
        "",
        None,
        Some(256 * 1024 * 1024),
        Some(3200),
        Some(30000),
        Some(true),
        None,
    );
    assert_eq!(diag.failure_type, "cgroup_oom");
    assert_eq!(diag.resource_telemetry.oom_killed, Some(true));
    assert!(diag.remediation_suggestions.iter().any(|s| s.contains("mini-batch")));

    // Save and verify query
    record_diagnostic_receipt(td.path(), "p-diag-oom", &diag).unwrap();
    let list = list_diagnostic_receipts(td.path(), "p-diag-oom", Some("cand_oom_test"), None, 5).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].failure_type, "cgroup_oom");
}

#[test]
fn p10_2_fault_injection_non_finite_loss_and_shape_mismatch() {
    let td = tempfile::tempdir().unwrap();
    let _orch = setup_p10_project(td.path(), "p-diag-faults", "t-diag-faults", "lead");

    // 1. Non-finite loss fault
    let diag_nan = DiagnosticBridge::build_receipt(
        "cand_fault_nan",
        "fit",
        Some(1),
        "FloatingPointError: Loss is nan in epoch 2\n  File \"candidate/model.py\", line 75, in fit",
        "",
        "Loss is nan in epoch 2",
        None,
        None,
        Some(1200),
        None,
        None,
        None,
    );
    assert_eq!(diag_nan.failure_type, "non_finite_loss");
    assert_eq!(diag_nan.root_cause.line_number, Some(75));
    assert_eq!(diag_nan.root_cause.symbol.as_deref(), Some("fit"));
    assert!(diag_nan.remediation_suggestions.iter().any(|s| s.contains("epsilon") || s.contains("gradient clipping")));

    record_diagnostic_receipt(td.path(), "p-diag-faults", &diag_nan).unwrap();

    // 2. Shape mismatch fault
    let diag_shape = DiagnosticBridge::build_receipt(
        "cand_fault_shape",
        "predict",
        Some(1),
        "ValueError: Number of features of the model must match the input. Model n_features is 15 and input n_features is 10\n  File \"candidate/model.py\", line 99, in predict",
        "",
        "Number of features mismatch",
        None,
        None,
        Some(400),
        None,
        None,
        None,
    );
    assert_eq!(diag_shape.failure_type, "shape_mismatch");
    assert_eq!(diag_shape.root_cause.line_number, Some(99));
    assert_eq!(diag_shape.root_cause.symbol.as_deref(), Some("predict"));
    assert!(diag_shape.remediation_suggestions.iter().any(|s| s.contains("identical columns") || s.contains("one-hot")));

    record_diagnostic_receipt(td.path(), "p-diag-faults", &diag_shape).unwrap();

    let all_faults = list_diagnostic_receipts(td.path(), "p-diag-faults", None, None, 10).unwrap();
    assert_eq!(all_faults.len(), 2);
}

#[test]
fn p10_2_fault_injection_execution_timeout_clamp() {
    let td = tempfile::tempdir().unwrap();
    let _orch = setup_p10_project(td.path(), "p-diag-to", "t-diag-to", "lead");

    // Simulated execution timeout in sandbox_exec or candidate runner
    let diag = DiagnosticBridge::build_receipt(
        "cand_timeout_test",
        "sandbox_exec",
        Some(124),
        "Sandbox execution timed out after 30000ms",
        "",
        "Execution deadline exceeded",
        None,
        Some(512 * 1024 * 1024),
        Some(30000),
        Some(30000),
        None,
        None,
    );
    assert_eq!(diag.failure_type, "execution_timeout");
    assert_eq!(diag.resource_telemetry.exit_code, Some(124));
    assert_eq!(diag.resource_telemetry.wall_elapsed_ms, Some(30000));
    assert!(diag.remediation_suggestions.iter().any(|s| s.contains("Vectorize") || s.contains("early stopping")));

    record_diagnostic_receipt(td.path(), "p-diag-to", &diag).unwrap();

    let list = list_diagnostic_receipts(td.path(), "p-diag-to", Some("cand_timeout_test"), Some("execution_timeout"), 5).unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].failure_type, "execution_timeout");
}

#[test]
fn p10_2_fault_injection_candidate_evaluation_runtime_exception() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-diag-eval", "t-diag-eval", "lead");

    // 1. Synthesize candidate with custom module containing an intentional runtime error
    let bad_code = r#"
def faulty_transform(X):
    # Injected runtime exception
    return X["non_existent_column_123"]
"#;
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_bad_code",
        "task_type": "binary",
        "target_column": "target",
        "model_family": "linear",
        "framework": "scikit-learn",
        "candidate_path": "candidate",
        "class_order": [0, 1],
        "custom_modules": {
            "transforms/bad_transform.py": bad_code
        }
    }).to_string()).unwrap();

    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");

    // Patch entrypoint.py to call faulty_transform on fit
    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-diag-eval", "b-t-diag-eval");
    let entrypoint_p = branch_dir.join("candidate").join("entrypoint.py");
    let original_entry = fs::read_to_string(&entrypoint_p).unwrap();
    let patched_entry = original_entry.replace(
        "    model.fit(X_transformed, y)",
        "    from transforms.bad_transform import faulty_transform\n    faulty_transform(X_df)\n    model.fit(X_transformed, y)",
    );
    assert_ne!(patched_entry, original_entry, "Patching entrypoint must modify content");
    fs::write(&entrypoint_p, &patched_entry).unwrap();

    // Recompute source hash and update manifest so source_hash verification passes
    let manifest_p = branch_dir.join("candidate").join("manifest.json");
    let mut manifest: Value = serde_json::from_str(&fs::read_to_string(&manifest_p).unwrap()).unwrap();
    let mut files = vec![
        ("entrypoint.py", patched_entry.as_bytes().to_vec()),
        ("model.py", fs::read(branch_dir.join("candidate").join("model.py")).unwrap()),
        ("features.py", fs::read(branch_dir.join("candidate").join("features.py")).unwrap()),
        ("transforms/bad_transform.py", bad_code.as_bytes().to_vec()),
    ];
    files.sort_by(|a, b| a.0.cmp(&b.0));
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for (rel_path, content) in files {
        let file_sha = format!("{:x}", Sha256::digest(&content));
        let line = format!("{}:{}:{}\n", rel_path, content.len(), file_sha);
        hasher.update(line.as_bytes());
    }
    let new_src_hash = format!("{:x}", hasher.finalize());
    manifest["source_hash"] = json!(new_src_hash);
    fs::write(&manifest_p, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    // Register dummy dataset
    let ds_csv = td.path().join("dummy.csv");
    fs::write(&ds_csv, "row_id,f1,f2,target\n0,1.0,2.0,0\n1,2.0,3.0,1\n2,3.0,4.0,0\n3,4.0,5.0,1\n").unwrap();
    execution::register_dataset(td.path(), "p-diag-eval", "ds_dummy", &ds_csv).unwrap();

    // 2. Evaluate submission: runner should fail and record rich DiagnosticReceipt
    let eval_err = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_bad_code",
        "candidate_path": "candidate",
        "dataset_id": "ds_dummy",
        "folds": 2
    }).to_string()).unwrap_err();
    assert!(eval_err.contains("candidate evaluation failed"));

    // Verify diagnostic receipt was persisted to SQLite
    let persisted = list_diagnostic_receipts(td.path(), "p-diag-eval", Some("cand_bad_code"), None, 5).unwrap();
    assert_eq!(persisted.len(), 1);
    assert_eq!(persisted[0].candidate_id, "cand_bad_code");
    assert_eq!(persisted[0].failure_type, "runtime_exception");

    // 3. Inspect via inspect_diagnostics tool
    let tool_diag = orch.dispatch_tool_call("inspect_diagnostics", &json!({
        "candidate_id": "cand_bad_code"
    }).to_string()).unwrap();
    let tool_val: Value = serde_json::from_str(&tool_diag).unwrap();
    assert_eq!(tool_val["count"], 1);
}
