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
    iteration::{
        compute_pareto_frontier, get_hypothesis, get_latest_pareto_frontier,
        get_pareto_frontier, list_hypotheses, record_hypothesis, record_pareto_frontier,
        settle_hypothesis, ExpectedMetricDelta, FullTierBudget, HypothesisOutcome,
        IterationHypothesis, MultiFidelityBudget, ParetoRecord, ProxyTierBudget,
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
        prompt_hash: "prompt-hash-p103".into(),
        model: None,
        phase: "iteration_loop".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "propose_hypothesis".into(),
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
            intent: "P10.3 iteration loop test".into(),
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
            base_hash: "base-p103".into(),
            code_hash: "code-p103".into(),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("task-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.into(),
            branch_id: branch.clone(),
            context: test_context("base-p103", "code-p103"),
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
fn p10_3_schema_v10_persistence_and_pareto_tracking() {
    let td = tempfile::tempdir().unwrap();
    let _ = setup_p10_project(td.path(), "p-iter-db", "t-iter-db", "lead");

    // 1. Verify PRAGMA user_version remains exactly 10
    let db = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let version: i32 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version, 10, "PRAGMA user_version must remain exactly 10");

    // 2. Insert hypothesis via Rust API
    let hyp = IterationHypothesis {
        schema_version: "v2".into(),
        hypothesis_id: "hyp-cand-01".into(),
        candidate_id: "cand_01".into(),
        parent_candidate_id: Some("cand_baseline".into()),
        mutation_type: "feature_engineering".into(),
        hypothesis_statement: "Adding log transform to skewed features boosts ROC AUC".into(),
        rationale: Some("Distribution inspection showed skewness > 3.0".into()),
        expected_metric_delta: ExpectedMetricDelta {
            metric: "roc_auc".into(),
            direction: "maximize".into(),
            target_threshold: Some(0.85),
            minimum_improvement: 0.02,
        },
        multi_fidelity_budget: MultiFidelityBudget {
            proxy_tier: ProxyTierBudget {
                max_samples: 300,
                fidelity_level: Some(0.25),
                screening_threshold: Some(0.75),
                timeout_ms: Some(15000),
            },
            full_tier: FullTierBudget {
                folds: 5,
                seed: Some(2026),
                timeout_ms: Some(45000),
            },
        },
        outcome: Some(HypothesisOutcome {
            status: "pending".into(),
            baseline_metric: Some(0.80),
            proxy_metric_observed: None,
            full_metric_observed: None,
            metric_delta: None,
            reason: None,
            settled_at_ms: None,
        }),
        created_ms: now_ms(),
    };

    record_hypothesis(td.path(), "p-iter-db", &hyp).unwrap();

    // 3. Query hypothesis
    let fetched = get_hypothesis(td.path(), "p-iter-db", "hyp-cand-01").unwrap();
    assert_eq!(fetched.candidate_id, "cand_01");
    assert_eq!(fetched.mutation_type, "feature_engineering");
    assert_eq!(fetched.outcome.as_ref().unwrap().status, "pending");

    // 4. Settle hypothesis
    let mut to_settle = fetched.clone();
    settle_hypothesis(
        &mut to_settle,
        Some(0.86),
        Some(0.80),
        false,
        None,
    );
    record_hypothesis(td.path(), "p-iter-db", &to_settle).unwrap();

    let settled = get_hypothesis(td.path(), "p-iter-db", "hyp-cand-01").unwrap();
    let outcome = settled.outcome.unwrap();
    assert_eq!(outcome.status, "confirmed");
    assert!((outcome.metric_delta.unwrap() - 0.06).abs() < 1e-6);
    assert!(outcome.settled_at_ms.is_some());

    // 5. Test native Pareto frontier tracking:
    // Candidate A: 0.85 score, 100ms latency, 50MB peak memory (non-dominated: highest score)
    // Candidate B: 0.82 score, 30ms latency, 20MB peak memory (non-dominated: fastest & lowest memory)
    // Candidate C: 0.80 score, 120ms latency, 60MB peak memory (dominated by A and B)
    let records = vec![
        ParetoRecord {
            candidate_id: "cand_A".into(),
            parent_candidate_id: None,
            validation_score: 0.85,
            latency_ms: 100.0,
            peak_memory_bytes: 50 * 1024 * 1024,
            domination_status: "non_dominated".into(),
            dominated_by: vec![],
            metadata: None,
        },
        ParetoRecord {
            candidate_id: "cand_B".into(),
            parent_candidate_id: None,
            validation_score: 0.82,
            latency_ms: 30.0,
            peak_memory_bytes: 20 * 1024 * 1024,
            domination_status: "non_dominated".into(),
            dominated_by: vec![],
            metadata: None,
        },
        ParetoRecord {
            candidate_id: "cand_C".into(),
            parent_candidate_id: None,
            validation_score: 0.80,
            latency_ms: 120.0,
            peak_memory_bytes: 60 * 1024 * 1024,
            domination_status: "non_dominated".into(),
            dominated_by: vec![],
            metadata: None,
        },
    ];

    let frontier = compute_pareto_frontier(&records, "roc_auc", "maximize", "p-iter-db", None);
    assert_eq!(frontier.non_dominated_count, 2);
    assert_eq!(frontier.records.len(), 3);
    assert_eq!(frontier.frontier_candidate_ids.len(), 2);

    let cand_a_res = frontier.records.iter().find(|r| r.candidate_id == "cand_A").unwrap();
    assert_eq!(cand_a_res.domination_status, "non_dominated");

    let cand_b_res = frontier.records.iter().find(|r| r.candidate_id == "cand_B").unwrap();
    assert_eq!(cand_b_res.domination_status, "non_dominated");

    let cand_c_res = frontier.records.iter().find(|r| r.candidate_id == "cand_C").unwrap();
    assert_eq!(cand_c_res.domination_status, "dominated");
    assert!(cand_c_res.dominated_by.contains(&"cand_A".to_string()));
    assert!(cand_c_res.dominated_by.contains(&"cand_B".to_string()));

    // Record Pareto frontier in SQLite and read back
    record_pareto_frontier(td.path(), "p-iter-db", &frontier).unwrap();
    let retrieved_latest = get_latest_pareto_frontier(td.path(), "p-iter-db").unwrap().expect("latest frontier");
    assert_eq!(retrieved_latest.frontier_id, frontier.frontier_id);
    assert_eq!(retrieved_latest.non_dominated_count, 2);

    // Also verify get_pareto_frontier by exact ID and list_hypotheses
    let by_id = get_pareto_frontier(td.path(), "p-iter-db", &frontier.frontier_id).unwrap();
    assert_eq!(by_id.frontier_id, frontier.frontier_id);

    let hyps = list_hypotheses(td.path(), "p-iter-db", Some("cand_01"), None, 10).unwrap();
    assert_eq!(hyps.len(), 1);
}

#[test]
fn p10_3_tool_discovery_and_propose_hypothesis() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-hyp-disc", "t-hyp-disc", "lead");

    // 1. Tool discovery
    let out = orch.dispatch_tool_call("discover_capabilities", "{}").unwrap();
    let val: Value = serde_json::from_str(&out).unwrap();
    let tools = val["tools"].as_array().expect("tools array");
    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();

    assert!(tool_names.contains(&"propose_hypothesis"), "ToolRegistry missing propose_hypothesis");
    assert!(tool_names.contains(&"screen_candidate_fidelity"), "ToolRegistry missing screen_candidate_fidelity");
    assert!(tool_names.contains(&"query_pareto_frontier"), "ToolRegistry missing query_pareto_frontier");

    // 2. Dispatch propose_hypothesis
    let hyp_res = orch.dispatch_tool_call("propose_hypothesis", &json!({
        "candidate_id": "cand_rf_01",
        "parent_candidate_id": "cand_baseline",
        "mutation_type": "hyperparameter_tuning",
        "hypothesis_statement": "Increasing n_estimators from 20 to 100 reduces generalization error",
        "rationale": "High variance observed in baseline CV folds",
        "metric": "balanced_accuracy",
        "direction": "maximize",
        "target_threshold": 0.82,
        "minimum_improvement": 0.03,
        "proxy_max_samples": 250,
        "screening_threshold": 0.70,
        "full_folds": 3
    }).to_string()).unwrap();

    let hyp_val: Value = serde_json::from_str(&hyp_res).unwrap();
    assert_eq!(hyp_val["candidate_id"], "cand_rf_01");
    assert_eq!(hyp_val["parent_candidate_id"], "cand_baseline");
    assert_eq!(hyp_val["mutation_type"], "hyperparameter_tuning");
    assert_eq!(hyp_val["outcome"]["status"], "pending");

    let hyp_id = hyp_val["hypothesis_id"].as_str().expect("hypothesis_id string");
    assert!(!hyp_id.is_empty());

    // 3. Verify SQLite durability
    let db = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let count: i64 = db.query_row(
        "SELECT COUNT(*) FROM iteration_hypotheses WHERE project=?1 AND candidate_id=?2",
        params!["p-hyp-disc", "cand_rf_01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn p10_3_proxy_screening_tier_early_discard() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-tier0-discard", "t-tier0-discard", "feature_engineer");

    // 1. Synthesize baseline candidate
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_fast_discard",
        "task_type": "binary",
        "target_column": "target",
        "model_family": "boosting",
        "framework": "lightgbm",
        "candidate_path": "candidates/cand_fast_discard",
        "class_order": ["neg", "pos"],
        "hyperparameters": {
            "n_estimators": 10,
            "learning_rate": 0.05
        },
        "feature_pipeline": {
            "scaling": "standard"
        }
    }).to_string()).unwrap();
    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");

    // 2. Register synthetic dataset
    let mut csv_rows = vec!["row_id,f1,f2,target".to_string()];
    for i in 0..120 {
        let f1 = (i as f64) * 0.1;
        let f2 = (i as f64) * -0.5;
        let target = if i >= 60 { "pos" } else { "neg" };
        csv_rows.push(format!("{i},{f1:.2},{f2:.2},{target}"));
    }
    let ds_csv = td.path().join("data_discard.csv");
    fs::write(&ds_csv, csv_rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-tier0-discard", "ds_discard", &ds_csv).unwrap();

    // 3. Propose hypothesis with unreachable screening threshold (e.g. 1.01)
    let prop_res = orch.dispatch_tool_call("propose_hypothesis", &json!({
        "candidate_id": "cand_fast_discard",
        "mutation_type": "architecture_change",
        "hypothesis_statement": "Radical architecture shift will achieve near-perfect score",
        "metric": "balanced_accuracy",
        "direction": "maximize",
        "target_threshold": 0.99,
        "screening_threshold": 1.01
    }).to_string()).unwrap();
    let prop_val: Value = serde_json::from_str(&prop_res).unwrap();
    let hyp_id = prop_val["hypothesis_id"].as_str().unwrap();

    // 4. Run multi-fidelity screening: Tier 0 fast proxy screen must discard candidate early!
    let screen_res = orch.dispatch_tool_call("screen_candidate_fidelity", &json!({
        "candidate_id": "cand_fast_discard",
        "dataset_id": "ds_discard",
        "candidate_path": "candidates/cand_fast_discard",
        "hypothesis_id": hyp_id,
        "screening_threshold": 1.01,
        "tier": "progressive"
    }).to_string()).unwrap();

    let screen_val: Value = serde_json::from_str(&screen_res).unwrap();
    assert_eq!(screen_val["status"], "rejected_early", "Candidate must be rejected at Tier 0 proxy");
    assert_eq!(screen_val["tier_reached"], 0);
    assert_eq!(screen_val["proxy_screening"]["status"], "rejected");
    assert!(screen_val["full_evaluation"].is_null(), "Tier 1 full CV must NOT be executed on rejected candidate");

    // Verify hypothesis settlement in SQLite is marked refuted
    let fetched = get_hypothesis(td.path(), "p-tier0-discard", hyp_id).unwrap();
    let outcome = fetched.outcome.expect("settled outcome");
    assert_eq!(outcome.status, "refuted");
    assert!(outcome.reason.unwrap().contains("screening threshold"));

    // Verify candidate_evaluations has NO full evaluation record
    let db = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let eval_count: i64 = db.query_row(
        "SELECT COUNT(*) FROM candidate_evaluations WHERE project=?1 AND candidate_id=?2",
        params!["p-tier0-discard", "cand_fast_discard"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(eval_count, 0, "No official candidate_evaluations record should exist for discarded candidate");
}

#[test]
fn p10_3_proxy_admission_full_cv_and_pareto_frontier() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p10_project(td.path(), "p-tier1-pareto", "t-tier1-pareto", "model_specialist");

    // 1. Synthesize candidate
    let synth_res = orch.dispatch_tool_call("synthesize_candidate", &json!({
        "candidate_id": "cand_admitted_01",
        "task_type": "binary",
        "target_column": "target",
        "model_family": "boosting",
        "framework": "lightgbm",
        "candidate_path": "candidates/cand_admitted_01",
        "class_order": ["class_0", "class_1"],
        "hyperparameters": {
            "n_estimators": 25,
            "learning_rate": 0.1
        },
        "feature_pipeline": {
            "scaling": "standard"
        }
    }).to_string()).unwrap();
    let synth_val: Value = serde_json::from_str(&synth_res).unwrap();
    assert_eq!(synth_val["status"], "synthesized");

    // 2. Register synthetic dataset
    let mut csv_rows = vec!["row_id,f1,f2,target".to_string()];
    for i in 0..120 {
        let f1 = (i as f64) * 0.15;
        let f2 = (i as f64) * -0.7;
        let target = if i >= 60 { "class_1" } else { "class_0" };
        csv_rows.push(format!("{i},{f1:.2},{f2:.2},{target}"));
    }
    let ds_csv = td.path().join("data_admit.csv");
    fs::write(&ds_csv, csv_rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-tier1-pareto", "ds_admit", &ds_csv).unwrap();

    // 3. Propose hypothesis with reachable screening threshold (0.50) and baseline 0.65
    let prop_res = orch.dispatch_tool_call("propose_hypothesis", &json!({
        "candidate_id": "cand_admitted_01",
        "parent_candidate_id": "cand_base",
        "mutation_type": "hyperparameter_tuning",
        "hypothesis_statement": "Boosting estimators tuned to 25 achieves >= 0.70 balanced accuracy",
        "metric": "balanced_accuracy",
        "direction": "maximize",
        "target_threshold": 0.70,
        "minimum_improvement": 0.02,
        "proxy_max_samples": 200,
        "screening_threshold": 0.50,
        "full_folds": 3
    }).to_string()).unwrap();
    let prop_val: Value = serde_json::from_str(&prop_res).unwrap();
    let hyp_id = prop_val["hypothesis_id"].as_str().unwrap();

    // 4. Run multi-fidelity screening: Tier 0 proxy passes -> Tier 1 full CV executes
    let screen_res = orch.dispatch_tool_call("screen_candidate_fidelity", &json!({
        "candidate_id": "cand_admitted_01",
        "dataset_id": "ds_admit",
        "candidate_path": "candidates/cand_admitted_01",
        "hypothesis_id": hyp_id,
        "baseline_score": 0.65,
        "screening_threshold": 0.50,
        "folds": 3,
        "seed": 2026,
        "tier": "progressive"
    }).to_string()).unwrap();

    let screen_val: Value = serde_json::from_str(&screen_res).unwrap();
    assert_eq!(screen_val["status"], "completed");
    assert_eq!(screen_val["tier_reached"], 1);
    assert_eq!(screen_val["proxy_screening"]["status"], "admitted");

    let full_eval = &screen_val["full_evaluation"];
    assert_eq!(full_eval["status"], "evaluated");
    let primary = full_eval["primary_metric"].as_f64().expect("primary metric float");
    assert!(primary >= 0.70, "Primary metric must exceed 0.70, got: {primary}");

    // Verify hypothesis settlement in SQLite
    let fetched = get_hypothesis(td.path(), "p-tier1-pareto", hyp_id).unwrap();
    let outcome = fetched.outcome.expect("settled outcome");
    assert_eq!(outcome.status, "confirmed");
    assert!(outcome.metric_delta.unwrap() > 0.0);

    // Verify durable record in candidate_evaluations
    let db = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let eval_count: i64 = db.query_row(
        "SELECT COUNT(*) FROM candidate_evaluations WHERE project=?1 AND candidate_id=?2",
        params!["p-tier1-pareto", "cand_admitted_01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(eval_count, 1);

    // 5. Query Pareto frontier via tool
    let pareto_tool_res = orch.dispatch_tool_call("query_pareto_frontier", &json!({
        "primary_metric": "balanced_accuracy",
        "direction": "maximize"
    }).to_string()).unwrap();

    let pareto_val: Value = serde_json::from_str(&pareto_tool_res).unwrap();
    assert_eq!(pareto_val["non_dominated_count"], 1);
    assert_eq!(pareto_val["records"].as_array().unwrap().len(), 1);
    let frontier_ids = pareto_val["frontier_candidate_ids"].as_array().expect("frontier_candidate_ids array");
    assert_eq!(frontier_ids.len(), 1);
    assert_eq!(frontier_ids[0], "cand_admitted_01");
}

#[test]
fn p10_3_rbac_authorization_enforcement() {
    let td = tempfile::tempdir().unwrap();

    // Setup project with unauthorized 'critic' role
    let mut critic_orch = setup_p10_project(td.path(), "p-rbac-test", "t-rbac-critic", "critic");

    // 1. propose_hypothesis must be refused for critic
    let res1 = critic_orch.dispatch_tool_call("propose_hypothesis", &json!({
        "candidate_id": "cand_01",
        "mutation_type": "feature_engineering",
        "hypothesis_statement": "Unauthorized proposal"
    }).to_string());
    assert!(res1.is_err(), "Critic role must be refused propose_hypothesis");
    assert!(res1.unwrap_err().contains("not authorized"));

    // 2. screen_candidate_fidelity must be refused for critic
    let res2 = critic_orch.dispatch_tool_call("screen_candidate_fidelity", &json!({
        "candidate_id": "cand_01",
        "dataset_id": "ds_01"
    }).to_string());
    assert!(res2.is_err(), "Critic role must be refused screen_candidate_fidelity");
    assert!(res2.unwrap_err().contains("not authorized"));

    // 3. query_pareto_frontier must be refused for critic
    let res3 = critic_orch.dispatch_tool_call("query_pareto_frontier", "{}");
    assert!(res3.is_err(), "Critic role must be refused query_pareto_frontier");
    assert!(res3.unwrap_err().contains("not authorized"));
}
