use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::params;
use serde_json::{json, Value};

use mlrescurix_harness::{
    agents::{
        gateway::{ChatCompletionResponse, ScriptedProviderFixture, TokenUsage},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
        roles::{get_role, get_standard_roles},
        task_contract::{get_pending_steering_directives, record_steering_directive},
        tools::ToolRegistry,
        workspace::BranchWorkspaceManager,
    },
    contracts::Context,
    core::{Operation, Policy, Request},
    execution,
    Runtime,
};


fn test_policy(limit: i64, workers: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 10_000);
    b.insert("wall_ms".into(), limit * 60_000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), workers);
    c.insert("cpu_threads".into(), 4);
    Policy { max_tasks: 1024, budget: b, capacity: c }
}

fn test_context(role_id: &str) -> Context {
    Context {
        role_id: role_id.into(),
        role_version: "2".into(),
        prompt_hash: "prompt-hash-reflexive".into(),
        model: None,
        phase: "reflexive_research".into(),
        hypothesis: Some("hyp-reflexive".into()),
        base_hash: "base-main".into(),
        code_hash: "code-main".into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "step".into(),
        stop_conditions: vec!["budget".into()],
    }
}

fn setup_test_runtime_and_task(
    root: &Path,
    project_id: &str,
    task_id: &str,
    role_id: &str,
) -> Runtime {
    let policy = test_policy(200, 4);
    let rt = Runtime::init(root, policy.clone()).unwrap();
    execution::migrate(root).unwrap();

    // Create project
    let res = rt.command(Request {
        command_id: format!("cmd-{project_id}-init"),
        project_id: project_id.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "Reflexive loop integration test".into(),
            finalization_id: format!("fin-{project_id}"),
            policy,
        },
    }).unwrap();
    assert_eq!(res.status, "applied");

    // Create branch
    let res = rt.command(Request {
        command_id: format!("cmd-{project_id}-b-init"),
        project_id: project_id.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: "main".into(),
            base_hash: "base-main".into(),
            code_hash: "code-main".into(),
        },
    }).unwrap();
    assert_eq!(res.status, "applied");

    // Create task
    let res = rt.command(Request {
        command_id: format!("cmd-{project_id}-t-init"),
        project_id: project_id.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.into(),
            branch_id: "main".into(),
            context: test_context(role_id),
        },
    }).unwrap();
    assert_eq!(res.status, "applied");

    // Claim task for orchestrator
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
    let res = rt.command(Request {
        command_id: format!("cmd-{project_id}-claim"),
        project_id: project_id.into(),
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
    assert_eq!(res.status, "applied");

    // Initialize branch workspace
    BranchWorkspaceManager::init_branch(root, project_id, "main", "main").unwrap();

    rt
}

fn get_titanic_dataset_path() -> PathBuf {
    let repo_dataset = PathBuf::from(".mlrescurix-state/kaggle-titanic/titanic.csv");
    if repo_dataset.is_file() {
        return fs::canonicalize(&repo_dataset).unwrap();
    }
    let alt_dataset = PathBuf::from("../../.mlrescurix-state/kaggle-titanic/titanic.csv");
    if alt_dataset.is_file() {
        return fs::canonicalize(&alt_dataset).unwrap();
    }
    panic!("titanic.csv dataset not found in repo");
}

// =============================================================================
// Test 1: Scientific Tools Registry Exposure
// =============================================================================

#[test]
fn test_scientific_tools_registry_exposure() {
    let tools = ToolRegistry::get_trusted_tools();
    let tool_names: Vec<String> = tools.iter().map(|t| t.function.name.clone()).collect();

    // Assert all 5 Category A Agent-First scientific tools are exposed
    for expected in [
        "eda_forensics",
        "ablation_sprint",
        "model_portfolio_eval",
        "critic_audit",
        "reflexive_orchestrate",
    ] {
        assert!(
            tool_names.contains(&expected.to_string()),
            "ToolRegistry must contain '{expected}', available tools: {tool_names:?}"
        );
    }

    // Inspect schemas and parameter invariants
    let eda_tool = tools.iter().find(|t| t.function.name == "eda_forensics").unwrap();
    assert_eq!(eda_tool.tool_type, "function");
    assert!(eda_tool.function.description.contains("eda_forensics_suite"));
    let eda_params = &eda_tool.function.parameters;
    assert_eq!(eda_params["type"], "object");
    assert!(eda_params["properties"]["dataset_path"].is_object());
    assert!(eda_params["required"].as_array().unwrap().contains(&json!("dataset_path")));

    let ablation_tool = tools.iter().find(|t| t.function.name == "ablation_sprint").unwrap();
    assert!(ablation_tool.function.description.contains("ablation"));
    let abl_params = &ablation_tool.function.parameters;
    assert_eq!(abl_params["type"], "object");
    assert!(abl_params["properties"]["dataset_path"].is_object());
    assert!(abl_params["properties"]["candidate_features_py"].is_object());

    let portfolio_tool = tools.iter().find(|t| t.function.name == "model_portfolio_eval").unwrap();
    assert!(portfolio_tool.function.description.contains("portfolio"));
    let port_params = &portfolio_tool.function.parameters;
    assert_eq!(port_params["type"], "object");
    assert!(port_params["properties"]["dataset_path"].is_object());

    let critic_tool = tools.iter().find(|t| t.function.name == "critic_audit").unwrap();
    assert!(critic_tool.function.description.contains("critic_audit_candidate"));
    let crit_params = &critic_tool.function.parameters;
    assert_eq!(crit_params["type"], "object");
    assert!(crit_params["properties"]["candidate"].is_object());
    assert!(crit_params["properties"]["topology"].is_object());

    let reflexive_tool = tools.iter().find(|t| t.function.name == "reflexive_orchestrate").unwrap();
    assert!(reflexive_tool.function.description.contains("reflexive loops"));
    let reflx_params = &reflexive_tool.function.parameters;
    assert_eq!(reflx_params["type"], "object");
    assert!(reflx_params["properties"]["dataset_path"].is_object());
}

// =============================================================================
// Test 2: Role RBAC Authorization & Rejection
// =============================================================================

#[test]
fn test_scientific_tools_role_rbac_and_unauthorized_rejection() {
    let roles = get_standard_roles();
    assert_eq!(roles.len(), 8);

    // Verify role capability mappings
    let lead = get_role("lead").unwrap();
    for tool in ["eda_forensics", "ablation_sprint", "model_portfolio_eval", "critic_audit", "reflexive_orchestrate"] {
        assert!(lead.capabilities.contains(&tool.to_string()), "lead must have capability '{tool}'");
    }

    let auditor = get_role("auditor").unwrap();
    assert!(auditor.capabilities.contains(&"eda_forensics".to_string()));
    assert!(!auditor.capabilities.contains(&"critic_audit".to_string()));

    let fe = get_role("feature_engineer").unwrap();
    assert!(fe.capabilities.contains(&"ablation_sprint".to_string()));
    assert!(fe.capabilities.contains(&"eda_forensics".to_string()));
    assert!(!fe.capabilities.contains(&"reflexive_orchestrate".to_string()));

    let ms = get_role("model_specialist").unwrap();
    assert!(ms.capabilities.contains(&"model_portfolio_eval".to_string()));
    assert!(ms.capabilities.contains(&"ablation_sprint".to_string()));
    assert!(!ms.capabilities.contains(&"eda_forensics".to_string()));

    let critic = get_role("critic").unwrap();
    assert!(critic.capabilities.contains(&"critic_audit".to_string()));
    assert!(!critic.capabilities.contains(&"ablation_sprint".to_string()));

    // Test runtime RBAC rejection via orchestrator
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-rbac", "t-verifier", "verifier");

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-rbac".into(),
            task_id: "t-verifier".into(),
            role_id: "verifier".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 3,
        },
        gateway,
    ).unwrap();

    // Verifier is not authorized to invoke ablation_sprint
    let err = orch.dispatch_tool_call("ablation_sprint", "{}").unwrap_err();
    assert!(
        err.contains("not authorized to invoke tool 'ablation_sprint'"),
        "Expected RBAC authorization rejection error, got: {err}"
    );

    // Verifier is not authorized to invoke eda_forensics
    let err = orch.dispatch_tool_call("eda_forensics", "{}").unwrap_err();
    assert!(
        err.contains("not authorized to invoke tool 'eda_forensics'"),
        "Expected RBAC authorization rejection error, got: {err}"
    );
}

// =============================================================================
// Test 3: M1 Boss Turn-Boundary Steering & Attempt Epoch Fencing
// =============================================================================

#[test]
fn test_critic_audit_veto_steering_and_attempt_epoch_fencing() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-steer", "t-lead", "lead");

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1 Assistant response
        ChatCompletionResponse {
            id: "resp-1".into(),
            model: "fixture".into(),
            content: Some("Proceeding with guidance".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 10, completion_tokens: 10, total_tokens: 20 },
            latency_ms: 10,
        },
    ]));

    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-steer".into(),
            task_id: "t-lead".into(),
            role_id: "lead".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 5,
        },
        gateway,
    ).unwrap();

    // 1. Synthesize CritiqueReceipt with VETO status
    let critique_receipt = json!({
        "critique_id": "crit-veto-01",
        "candidate_id": "cand-naive-overfit",
        "session_id": "sess-test-01",
        "epoch": 1,
        "status": "VETO",
        "blocking_flaws": [
            "Train-val gap exceeded: 0.165 > 0.120 (Severe Overfitting on training fold)",
            "Model monoculture: only 1 model family evaluated"
        ],
        "error_slice_hypotheses": [
            "Decompose Ticket quasi-identifiers into group size and fare per member",
            "Evaluate Gradient Boosting and Regularized ElasticNet alongside tree baseline"
        ],
        "audit_checks": {
            "target_memorization": {"passed": false, "reason": "train_val_gap 0.165 > 0.12"},
            "model_monoculture": {"passed": false, "reason": "only 1 family"}
        }
    });

    // 2. Invoke handle_critique_receipt
    let guidance_opt = orch.handle_critique_receipt(&critique_receipt, None).unwrap();
    assert!(guidance_opt.is_some(), "Expected GuidanceContract on VETO");
    let guidance = guidance_opt.unwrap();

    assert_eq!(guidance.epoch, 2);
    assert_eq!(guidance.target_task_id, "t-lead");
    assert_eq!(guidance.actor, "m1_boss");
    assert_eq!(guidance.veto_reasons.len(), 2);
    assert_eq!(guidance.recommended_actions.len(), 2);
    assert!(guidance.directive.contains("[M1 BOSS GUIDANCE CONTRACT"));
    assert!(guidance.directive.contains("Adversarial M2-Critic VETO issued"));

    // Verify directive is recorded in SQLite steering_directives table with applied=0
    let pending = get_pending_steering_directives(&root, "p-steer", "t-lead").unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].directive_id, format!("guidance-p-steer-ep2"));
    assert_eq!(pending[0].epoch, 2);

    // 3. Test Attempt Epoch Fencing:
    // Insert a STALE directive from prior epoch 0
    record_steering_directive(
        &root,
        "p-steer",
        "steer-stale-ep0",
        "t-lead",
        "m1_boss",
        0, // stale epoch 0 < worker epoch 1 (worker will be at epoch 1 when guidance is applied)
        "Stale directive from previous epoch that should be fenced off",
    ).unwrap();

    // Verify both are currently in pending list
    let pending_before_step = get_pending_steering_directives(&root, "p-steer", "t-lead").unwrap();
    assert_eq!(pending_before_step.len(), 2);

    // 4. Advance worker turn via step():
    // The worker at turn boundary will drain steering directives:
    // - "guidance-p-steer-ep1" (epoch 1 >= worker epoch 0): applied! Bumps handoff_version to 1.
    // - "steer-stale-ep0" (epoch 0 < worker epoch 1): rejected under Attempt Epoch Fencing!
    let turn_res = orch.step().unwrap();
    assert_eq!(turn_res.turn_index, 1);

    // Verify no directives remain pending
    let pending_after = get_pending_steering_directives(&root, "p-steer", "t-lead").unwrap();
    assert_eq!(pending_after.len(), 0, "All pending directives must be consumed or fenced");

    // Verify achievements contain the epoch fencing rejection
    let stale_achievement = orch.context.achievements.iter().find(|a| a.kind == "epoch_fencing_stale_directive_rejected");
    assert!(
        stale_achievement.is_some(),
        "Expected epoch_fencing_stale_directive_rejected achievement to be recorded, found: {:?}",
        orch.context.achievements
    );
    let ach = stale_achievement.unwrap();
    assert!(ach.summary.contains("Stale steering directive 'steer-stale-ep0' from epoch 0 rejected under epoch fencing"));
}

// =============================================================================
// Test 4: EDA Forensics Tool Dispatch & SQLite Topology Synchronization
// =============================================================================

#[test]
fn test_eda_forensics_tool_dispatch_and_sqlite_sync() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-eda", "t-lead", "lead");
    let titanic_path = get_titanic_dataset_path();

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-eda".into(),
            task_id: "t-lead".into(),
            role_id: "lead".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 3,
        },
        gateway,
    ).unwrap();

    let args = json!({
        "dataset_path": titanic_path.to_string_lossy(),
        "target_column": "Survived",
        "problem_statement": "Predict Titanic passenger survival with CV Balanced Accuracy >= 81.5%"
    }).to_string();

    let result_str = orch.dispatch_tool_call("eda_forensics", &args).unwrap();
    let result: Value = serde_json::from_str(&result_str).unwrap();

    assert_eq!(result["status"], "completed");
    let topo = result.get("topology").or_else(|| result.get("dataset_topology")).unwrap();
    assert_eq!(topo["target_column"], "Survived");
    assert_eq!(topo["task_type"], "binary_classification");
    assert!(topo["clean_feature_columns"].as_array().unwrap().len() >= 5);

    // Verify SQLite dataset_topology_contracts synchronization
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let target_col: String = conn.query_row(
        "SELECT target_column FROM dataset_topology_contracts WHERE project=?1",
        params!["p-eda"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(target_col, "Survived");

    let clean_features_json: String = conn.query_row(
        "SELECT clean_feature_columns_json FROM dataset_topology_contracts WHERE project=?1",
        params!["p-eda"],
        |r| r.get(0),
    ).unwrap();
    let clean_features: Vec<String> = serde_json::from_str(&clean_features_json).unwrap();
    assert!(clean_features.contains(&"Pclass".to_string()));
    assert!(clean_features.contains(&"Sex".to_string()));
}

// =============================================================================
// Test 5: Feature Ablation Sprint Tool Dispatch & SQLite Synchronization
// =============================================================================

#[test]
fn test_ablation_sprint_tool_dispatch_and_sqlite_sync() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-abl", "t-fe", "feature_engineer");
    let titanic_path = get_titanic_dataset_path();

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-abl".into(),
            task_id: "t-fe".into(),
            role_id: "feature_engineer".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 3,
        },
        gateway,
    ).unwrap();

    // Feature function extracting title honorifics
    let fe_py = r#"
import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if 'Name' in df.columns:
        out['is_mr'] = df['Name'].str.contains('Mr.', na=False).astype(float)
        out['is_mrs'] = df['Name'].str.contains('Mrs.', na=False).astype(float)
        out['is_miss'] = df['Name'].str.contains('Miss.', na=False).astype(float)
    if 'Age' in df.columns:
        out['is_child'] = (df['Age'] < 12).astype(float)
    return out
"#;

    let args = json!({
        "dataset_path": titanic_path.to_string_lossy(),
        "candidate_features_py": fe_py,
        "hypothesis_id": "hyp_title_honorifics",
        "target_column": "Survived",
        "session_id": "sess_fe_01",
        "epoch": 1
    }).to_string();

    let result_str = orch.dispatch_tool_call("ablation_sprint", &args).unwrap();
    let result: Value = serde_json::from_str(&result_str).unwrap();

    assert!(result.get("ablation_id").is_some());
    assert!(result.get("status").is_some());
    assert!(result.get("baseline_cv_score").is_some());
    assert!(result.get("ablated_cv_score").is_some());
    assert!(result.get("cv_score_delta").is_some());

    // Verify SQLite feature_ablations synchronization
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM feature_ablations WHERE project=?1",
        params!["p-abl"],
        |r| r.get(0),
    ).unwrap();
    assert!(count >= 1, "Expected at least 1 feature_ablations row in SQLite, got {count}");

    let status: String = conn.query_row(
        "SELECT status FROM feature_ablations WHERE project=?1",
        params!["p-abl"],
        |r| r.get(0),
    ).unwrap();
    assert!(matches!(status.as_str(), "ACCEPTED" | "REJECTED"));
}

// =============================================================================
// Test 6: Model Portfolio Evaluation Tool Dispatch
// =============================================================================

#[test]
fn test_model_portfolio_eval_tool_dispatch() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-port", "t-ms", "model_specialist");
    let titanic_path = get_titanic_dataset_path();

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-port".into(),
            task_id: "t-ms".into(),
            role_id: "model_specialist".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 3,
        },
        gateway,
    ).unwrap();

    let args = json!({
        "dataset_path": titanic_path.to_string_lossy(),
        "target_column": "Survived",
        "families": ["hist_gbdt", "penalized_linear"]
    }).to_string();

    let result_str = orch.dispatch_tool_call("model_portfolio_eval", &args).unwrap();
    let result: Value = serde_json::from_str(&result_str).unwrap();

    assert_eq!(result["status"], "evaluated");
    let families = result["families_evaluated"].as_array().unwrap();
    assert!(families.len() >= 2);

    let portfolio = &result["portfolio"];
    for fam in families {
        let fam_str = fam.as_str().unwrap();
        let fam_res = &portfolio[fam_str];
        assert!(fam_res.get("mean_cv_score").is_some());
        assert!(fam_res.get("train_val_gap").is_some());
        assert!(fam_res.get("cv_scores").is_some());
        assert!(fam_res.get("worst_slices").is_some());
    }
}

// =============================================================================
// Test 7: Adversarial Critic Audit Tool Dispatch & SQLite Synchronization
// =============================================================================

#[test]
fn test_critic_audit_tool_dispatch_and_sqlite_sync() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-crit", "t-critic", "critic");

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-crit".into(),
            task_id: "t-critic".into(),
            role_id: "critic".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 3,
        },
        gateway,
    ).unwrap();

    let candidate_val = json!({
        "candidate_id": "cand-balanced-ensemble",
        "model_family": "hist_gbdt",
        "cv_score": 0.825,
        "train_score": 0.840,
        "train_val_gap": 0.015,
        "distinct_families_evaluated": ["hist_gbdt", "catboost", "penalized_linear"],
        "evaluated_model_families": ["hist_gbdt", "catboost", "penalized_linear"],
        "features_used": ["Pclass", "Sex", "Age", "Fare", "Embarked", "Name__honorific", "Ticket__group_size"],
        "cv_scheme": "stratified_kfold",
        "cv_splits": 5
    });

    let topology_val = json!({
        "target_column": "Survived",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_kfold",
        "cv_splits": 5,
        "id_columns": ["PassengerId"],
        "column_classifications": {
            "PassengerId": "DROP_SURROGATE_ID",
            "Name": "SHORT_TEXT_EMBED",
            "Ticket": "QUASI_ID_DECOMPOSE"
        }
    });

    let args = json!({
        "candidate": candidate_val,
        "topology": topology_val,
        "session_id": "sess_crit_01",
        "epoch": 2
    }).to_string();

    let result_str = orch.dispatch_tool_call("critic_audit", &args).unwrap();
    let result: Value = serde_json::from_str(&result_str).unwrap();

    assert!(result.get("critique_id").is_some());
    assert_eq!(result["candidate_id"], "cand-balanced-ensemble");
    assert_eq!(result["status"], "APPROVED");
    assert_eq!(result["blocking_flaws"].as_array().unwrap().len(), 0);

    // Verify SQLite critique_receipts synchronization
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM critique_receipts WHERE project=?1",
        params!["p-crit"],
        |r| r.get(0),
    ).unwrap();
    assert!(count >= 1, "Expected at least 1 critique_receipts row in SQLite, got {count}");

    let status: String = conn.query_row(
        "SELECT status FROM critique_receipts WHERE project=?1",
        params!["p-crit"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(status, "APPROVED");
}

// =============================================================================
// Test 8: End-to-End Reflexive Orchestration & Full SQLite v11 Synchronization
// =============================================================================

#[test]
fn test_reflexive_orchestrate_end_to_end_and_sqlite_v11_sync() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let _rt = setup_test_runtime_and_task(&root, "p-reflx", "t-lead", "lead");
    let titanic_path = get_titanic_dataset_path();

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(
        &root,
        OrchestratorConfig {
            project_id: "p-reflx".into(),
            task_id: "t-lead".into(),
            role_id: "lead".into(),
            branch_id: "main".into(),
            model: "fixture".into(),
            max_turns: 5,
        },
        gateway,
    ).unwrap();

    let args = json!({
        "dataset_path": titanic_path.to_string_lossy(),
        "target_column": "Survived",
        "max_epochs": 2,
        "inject_naive_first_epoch": true,
        "session_id": "sess_acceptance_harness"
    }).to_string();

    let result_str = orch.dispatch_tool_call("reflexive_orchestrate", &args).unwrap();
    let result: Value = serde_json::from_str(&result_str).unwrap();

    assert!(matches!(result["status"].as_str(), Some("finalized" | "completed")));
    assert!(result.get("apex_candidate_id").is_some());
    let best_cv = result["best_cv_score"].as_f64().unwrap();
    assert!(
        best_cv >= 0.815,
        "Expected final reflexive loop best CV score >= 81.5%, got {best_cv:.4}"
    );

    let ledger = &result["research_ledger"];
    let epochs = ledger["epochs"].as_array().unwrap();
    assert!(epochs.len() >= 2);

    // Assert Epoch 1 was VETOed
    assert_eq!(epochs[1]["critique"]["status"], "VETO");
    // Assert Epoch 2 was APPROVED
    assert_eq!(epochs[2]["critique"]["status"], "APPROVED");

    // Comprehensive SQLite v11 Table Synchronization Verification:
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    // 1. dataset_topology_contracts
    let topo_count: i64 = conn.query_row(
        "SELECT count(*) FROM dataset_topology_contracts WHERE project=?1",
        params!["p-reflx"],
        |r| r.get(0),
    ).unwrap();
    assert!(topo_count >= 1, "dataset_topology_contracts must have rows");

    // 2. research_ledger_epochs
    let epoch_count: i64 = conn.query_row(
        "SELECT count(*) FROM research_ledger_epochs WHERE project=?1",
        params!["p-reflx"],
        |r| r.get(0),
    ).unwrap();
    assert!(epoch_count >= 2, "research_ledger_epochs must have at least 2 epochs logged");

    // 3. feature_ablations
    let abl_count: i64 = conn.query_row(
        "SELECT count(*) FROM feature_ablations WHERE project=?1",
        params!["p-reflx"],
        |r| r.get(0),
    ).unwrap();
    assert!(abl_count >= 1, "feature_ablations must have rows logged");

    // 4. critique_receipts
    let crit_count: i64 = conn.query_row(
        "SELECT count(*) FROM critique_receipts WHERE project=?1",
        params!["p-reflx"],
        |r| r.get(0),
    ).unwrap();
    assert!(crit_count >= 2, "critique_receipts must have at least 2 critiques logged (Epoch 1 VETO and Epoch 2 APPROVED)");
}
