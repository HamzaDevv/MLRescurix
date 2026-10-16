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
        workspace::BranchWorkspaceManager,
    },
    core::{Operation, Policy, Request},
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
        prompt_hash: "prompt-hash-p10".into(),
        model: None,
        phase: "candidate_synthesis".into(),
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
            intent: "P10 candidate synthesis test".into(),
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
fn p10_1_schema_validation_and_tool_discovery() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-synth-disc", "t-synth-disc", "lead");

    let out = orch.dispatch_tool_call("discover_capabilities", "{}").unwrap();
    let val: Value = serde_json::from_str(&out).unwrap();
    let tools = val["tools"].as_array().expect("tools array");
    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();

    assert!(tool_names.contains(&"synthesize_candidate"), "ToolRegistry missing synthesize_candidate");
    assert!(tool_names.contains(&"sandbox_exec"), "ToolRegistry missing sandbox_exec");
    assert!(tool_names.contains(&"evaluate_submission"), "ToolRegistry missing evaluate_submission");
}

#[test]
fn p10_2_synthesize_lightgbm_candidate_and_evaluate_submission() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-synth-lgb", "t-synth-lgb", "lead");

    // 1. Synthesize LightGBM candidate
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_lgb_01",
        "task_type": "binary",
        "target_column": "target",
        "model_family": "boosting",
        "framework": "lightgbm",
        "candidate_path": "candidates/cand_lgb_01",
        "class_order": ["no", "yes"],
        "hyperparameters": {
            "n_estimators": 30,
            "learning_rate": 0.1
        },
        "feature_pipeline": {
            "scaling": "standard",
            "categorical_encoding": "onehot"
        }
    }).to_string()).unwrap();

    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");
    assert_eq!(synth_val["candidate_id"], "cand_lgb_01");
    let src_hash = synth_val["source_hash"].as_str().expect("source_hash string");
    assert!(!src_hash.is_empty());

    // Verify candidate files in branch workspace
    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-synth-lgb", "b-t-synth-lgb");
    let cand_dir = branch_dir.join("candidates").join("cand_lgb_01");
    assert!(cand_dir.join("entrypoint.py").is_file());
    assert!(cand_dir.join("model.py").is_file());
    assert!(cand_dir.join("features.py").is_file());
    assert!(cand_dir.join("manifest.json").is_file());

    // 2. Register synthetic classification dataset
    let mut csv_rows = vec!["row_id,feat_1,feat_2,cat_slice,target".to_string()];
    for i in 0..120 {
        let f1 = (i as f64) * 0.15;
        let f2 = (i as f64) * -0.8;
        let slice = if i < 40 { "low" } else if i < 80 { "mid" } else { "high" };
        let target = if i >= 60 { "yes" } else { "no" };
        csv_rows.push(format!("{i},{f1:.2},{f2:.2},{slice},{target}"));
    }
    let ds_csv = td.path().join("churn_dataset.csv");
    fs::write(&ds_csv, csv_rows.join("\n")).unwrap();

    let reg_ds = execution::register_dataset(td.path(), "p-synth-lgb", "ds_churn", &ds_csv).unwrap();
    assert!(reg_ds["sha256"].as_str().is_some());

    // 3. Evaluate submission through official evaluation gate
    let eval_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_lgb_01",
        "candidate_path": "candidates/cand_lgb_01",
        "dataset_id": "ds_churn",
        "folds": 3,
        "seed": 2026,
        "diagnostic_slices": ["cat_slice"]
    }).to_string()).unwrap();

    let eval_val: Value = serde_json::from_str(&eval_res).unwrap();
    assert_eq!(eval_val["status"], "evaluated");
    assert_eq!(eval_val["candidate_id"], "cand_lgb_01");
    let primary = eval_val["primary_metric"].as_f64().expect("primary metric float");
    assert!(primary >= 0.70, "Primary metric balanced_accuracy must exceed 0.70, got: {primary}");

    // Verify durable record in candidate_evaluations
    let db = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let count: i64 = db.query_row(
        "SELECT COUNT(*) FROM candidate_evaluations WHERE project=?1 AND candidate_id=?2",
        params!["p-synth-lgb", "cand_lgb_01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 1);

    // Verify durable record in candidate_synthesis_records
    let synth_count: i64 = db.query_row(
        "SELECT COUNT(*) FROM candidate_synthesis_records WHERE project=?1 AND candidate_id=?2",
        params!["p-synth-lgb", "cand_lgb_01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(synth_count, 1);
}

#[test]
fn p10_3_synthesize_pytorch_candidate_and_evaluate_submission() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-synth-pt", "t-synth-pt", "model_specialist");

    // 1. Synthesize PyTorch neural network candidate
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_pytorch_mlp",
        "task_type": "binary",
        "target_column": "label",
        "model_family": "neural",
        "framework": "pytorch",
        "candidate_path": "candidate",
        "class_order": [0, 1],
        "hyperparameters": {
            "hidden_dim": 32,
            "epochs": 5,
            "learning_rate": 0.01,
            "batch_size": 16
        },
        "feature_pipeline": {
            "scaling": "standard"
        }
    }).to_string()).unwrap();

    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");

    // 2. Register synthetic dataset
    let mut csv_rows = vec!["row_id,f1,f2,f3,label".to_string()];
    for i in 0..100 {
        let f1 = (i as f64) * 0.2;
        let f2 = (i as f64) * -0.5;
        let f3 = (i as f64) * 1.2;
        let label = if i >= 50 { 1 } else { 0 };
        csv_rows.push(format!("{i},{f1:.2},{f2:.2},{f3:.2},{label}"));
    }
    let ds_csv = td.path().join("nn_data.csv");
    fs::write(&ds_csv, csv_rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-synth-pt", "ds_nn", &ds_csv).unwrap();

    // 3. Evaluate submission
    let eval_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_pytorch_mlp",
        "candidate_path": "candidate",
        "dataset_id": "ds_nn",
        "folds": 3,
        "seed": 2026
    }).to_string()).unwrap();

    let eval_val: Value = serde_json::from_str(&eval_res).unwrap();
    assert_eq!(eval_val["status"], "evaluated");
    let primary = eval_val["primary_metric"].as_f64().expect("primary metric float");
    assert!(primary >= 0.60, "PyTorch candidate balanced_accuracy must be valid, got: {primary}");
}

#[test]
fn p10_4_synthesize_regression_candidate_and_evaluate_submission() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-synth-reg", "t-synth-reg", "feature_engineer");

    // 1. Synthesize Random Forest regression candidate
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_rf_reg_01",
        "task_type": "regression",
        "target_column": "house_value",
        "model_family": "tree",
        "framework": "scikit-learn",
        "candidate_path": "candidates/cand_rf_reg_01",
        "hyperparameters": {
            "n_estimators": 20,
            "max_depth": 5
        },
        "feature_pipeline": {
            "scaling": "minmax"
        }
    }).to_string()).unwrap();

    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");

    // 2. Register synthetic regression dataset
    let mut csv_rows = vec!["row_id,sqft,bedrooms,house_value".to_string()];
    for i in 0..100 {
        let sqft = 1000.0 + (i as f64) * 25.0;
        let beds = 2 + (i % 3);
        let house_val = sqft * 150.0 + (beds as f64) * 20000.0;
        csv_rows.push(format!("{i},{sqft:.1},{beds},{house_val:.1}"));
    }
    let ds_csv = td.path().join("housing_data.csv");
    fs::write(&ds_csv, csv_rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-synth-reg", "ds_housing", &ds_csv).unwrap();

    // 3. Evaluate regression submission
    let eval_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_rf_reg_01",
        "candidate_path": "candidates/cand_rf_reg_01",
        "dataset_id": "ds_housing",
        "folds": 3,
        "seed": 2026
    }).to_string()).unwrap();

    let eval_val: Value = serde_json::from_str(&eval_res).unwrap();
    assert_eq!(eval_val["status"], "evaluated");
    let metrics = &eval_val["metrics"];
    assert!(metrics["mae"].as_f64().is_some());
    assert!(metrics["rmse"].as_f64().is_some());
}

#[test]
fn p10_5_custom_modules_and_integrity_guards() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-synth-cust", "t-synth-cust", "lead");

    // 1. Synthesize candidate with custom module injection
    let custom_py = "def agent_transformer(x):\n    return x * 2.0\n";
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_cust_mod",
        "task_type": "binary",
        "target_column": "target",
        "model_family": "linear",
        "framework": "scikit-learn",
        "candidate_path": "candidate",
        "class_order": [0, 1],
        "custom_modules": {
            "transforms/agent_math.py": custom_py
        }
    }).to_string()).unwrap();

    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");

    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-synth-cust", "b-t-synth-cust");
    let cust_file = branch_dir.join("candidate").join("transforms").join("agent_math.py");
    assert!(cust_file.is_file());
    assert_eq!(fs::read_to_string(&cust_file).unwrap(), custom_py);

    // 2. Tampering detection: modify candidate file without updating manifest source_hash
    let entrypoint_p = branch_dir.join("candidate").join("entrypoint.py");
    let original_entry = fs::read_to_string(&entrypoint_p).unwrap();
    fs::write(&entrypoint_p, format!("{original_entry}\n# Tampered line\n")).unwrap();

    // Register dummy dataset
    let ds_csv = td.path().join("dummy.csv");
    fs::write(&ds_csv, "row_id,f1,target\n0,1.0,0\n1,2.0,1\n2,3.0,0\n3,4.0,1\n").unwrap();
    execution::register_dataset(td.path(), "p-synth-cust", "ds_dummy", &ds_csv).unwrap();

    // evaluate_submission MUST detect source_hash mismatch and reject
    let tamper_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_cust_mod",
        "candidate_path": "candidate",
        "dataset_id": "ds_dummy",
        "folds": 2
    }).to_string());
    assert!(tamper_res.is_err(), "evaluate_submission must refuse tampered candidate");
    let err = tamper_res.unwrap_err();
    assert!(err.contains("source_hash mismatch"), "Expected source_hash mismatch error, got: {err}");
}

#[test]
fn p10_6_rbac_refusal_for_unauthorized_role() {
    let td = tempfile::tempdir().unwrap();
    // critic role does not have synthesize_candidate capability
    let mut orch = setup_p10_project(td.path(), "p-synth-rbac", "t-synth-rbac", "critic");

    let refusal = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_unauth",
        "task_type": "binary",
        "target_column": "target"
    }).to_string());

    assert!(refusal.is_err());
    let err = refusal.unwrap_err();
    assert!(err.contains("Role 'critic' is not authorized to invoke tool 'synthesize_candidate'"), "got: {err}");
}
