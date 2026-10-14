use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::params;
use serde_json::json;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::{
        context::AgentContextManager,
        gateway::{ChatMessage, FunctionCall, OutboundFilter, ScriptedProviderFixture, ToolCall, ToolDefinition, ToolFunction},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
        roles::{get_consolidated_default_roles, get_role, get_standard_roles},
        task_contract::{
            check_user_approval, get_task_assignment, get_task_completion,
            record_task_assignment, record_task_completion, register_user_approval,
            TaskAssignment, TaskCompletionReceipt,
        },
        workspace::BranchWorkspaceManager,
    },
    core::{Operation, Policy, Request},
    execution,
    search::{evaluate_comparative_strategies, get_experiment_dag_node, init_search, SearchSpec},
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
    b.insert("cost_millicents".into(), limit * 50);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy { max_tasks: 1024, budget: b, capacity: c }
}

fn test_context(base: &str, code: &str) -> mlrescurix_harness::contracts::Context {
    mlrescurix_harness::contracts::Context {
        role_id: "lead".into(),
        role_version: "1".into(),
        prompt_hash: "prompt-hash".into(),
        model: None,
        phase: "research".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "inspect".into(),
        stop_conditions: vec!["budget".into()],
    }
}

fn setup_p5_project(root: &Path, project: &str, task_id: &str) -> AgentOrchestrator {
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
            intent: "P5 collaborative research test".into(),
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
            base_hash: "base-p5".into(),
            code_hash: "code-p5".into(),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("task-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.into(),
            branch_id: branch.clone(),
            context: test_context("base-p5", "code-p5"),
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
            expires_ms: now + 300_000,
            expected_generation: 0,
        },
    }).unwrap();

    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    AgentOrchestrator::new(
        root,
        OrchestratorConfig {
            project_id: project.into(),
            task_id: task_id.into(),
            role_id: "lead".into(),
            branch_id: branch,
            model: "mock-model".into(),
            max_turns: 10,
        },
        gateway,
    ).unwrap()
}

fn create_test_project(root: &Path, project: &str) {
    let r = Runtime::open(root).unwrap();
    let _ = r.command(Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P5 test project".into(),
            finalization_id: format!("final-{project}"),
            policy: test_policy(100),
        },
    }).unwrap();
}

#[test]
fn test_p5_1_task_assignment_and_completion_schemas_and_persistence() {
    let tmp = tempfile::tempdir().unwrap();
    let root_buf = tmp.path().join("state");
    let root = root_buf.as_path();
    Runtime::init(root, test_policy(100)).unwrap();
    execution::migrate(root).unwrap();
    create_test_project(root, "proj-p5");

    // 1. TaskAssignment validation
    let mut assignment = TaskAssignment::new("proj-p5", "task-p5-1", "experiment_engineer", "Optimize boosting hyperparameters", "branch-task-1");
    assignment.hypothesis = Some("Increasing max_depth to 12 will improve AUC".into());
    assignment.immutable_inputs = vec!["train.csv".into()];
    assignment.allowed_tools = vec!["workspace_fs".into(), "sandbox_exec".into()];
    assignment.expected_artifacts = vec!["candidate.py".into(), "manifest.json".into()];
    assignment.acceptance_criteria = vec!["AUC >= 0.85".into()];
    assignment.dependency_ids = vec!["task-prior".into()];
    assert!(assignment.validate().is_ok());

    // Schema version check
    let mut invalid_assignment = assignment.clone();
    invalid_assignment.schema_version = "v1".into();
    assert!(invalid_assignment.validate().is_err());

    // Empty goal check
    let mut invalid_goal = assignment.clone();
    invalid_goal.goal = "   ".into();
    assert!(invalid_goal.validate().is_err());

    // Zero limit check
    let mut invalid_limits = assignment.clone();
    invalid_limits.limits.max_turns = 0;
    assert!(invalid_limits.validate().is_err());

    // Persist and retrieve
    record_task_assignment(root, &assignment).unwrap();
    let retrieved = get_task_assignment(root, "proj-p5", "task-p5-1").unwrap().expect("assignment should exist");
    assert_eq!(retrieved.task_id, "task-p5-1");
    assert_eq!(retrieved.goal, "Optimize boosting hyperparameters");
    assert_eq!(retrieved.branch_ownership, "branch-task-1");
    assert_eq!(retrieved.hypothesis, Some("Increasing max_depth to 12 will improve AUC".into()));

    // 2. TaskCompletionReceipt validation
    let mut completion = TaskCompletionReceipt::new("proj-p5", "task-p5-1", "succeeded", "Increasing max_depth improved AUC by 0.03 without overfitting");
    completion.experiment_id = Some("exp-p5-boost".into());
    completion.candidate_id = Some("cand-p5-boost".into());
    completion.metric_delta = Some(json!({"auc": 0.03}));
    completion.protocol = Some("stratified_5fold".into());
    completion.artifacts_produced = vec!["candidate.py".into()];
    assert!(completion.validate().is_ok());

    // Invalid verdict check
    let mut invalid_verdict = completion.clone();
    invalid_verdict.verdict = "maybe".into();
    assert!(invalid_verdict.validate().is_err());

    // Empty learnings check
    let mut invalid_learnings = completion.clone();
    invalid_learnings.bounded_learnings = "".into();
    assert!(invalid_learnings.validate().is_err());

    // Persist and retrieve
    record_task_completion(root, &completion).unwrap();
    let retrieved_comp = get_task_completion(root, "proj-p5", "task-p5-1").unwrap().expect("completion should exist");
    assert_eq!(retrieved_comp.verdict, "succeeded");
    assert_eq!(retrieved_comp.candidate_id, Some("cand-p5-boost".into()));
    assert_eq!(retrieved_comp.metric_delta, Some(json!({"auc": 0.03})));
}

#[test]
fn test_p5_1_consolidated_roles_and_aliases() {
    let consolidated = get_consolidated_default_roles();
    assert_eq!(consolidated.len(), 5, "Expected 5 consolidated default roles");
    let ids: Vec<&str> = consolidated.iter().map(|r| r.role_id.as_str()).collect();
    assert!(ids.contains(&"lead"));
    assert!(ids.contains(&"data_diagnostics"));
    assert!(ids.contains(&"experiment_engineer"));
    assert!(ids.contains(&"verifier"));
    assert!(ids.contains(&"critic"));

    // Aliases normalization
    let diag = get_role("data/diagnostics").expect("data/diagnostics alias should resolve");
    assert_eq!(diag.role_id, "data_diagnostics");

    let eng = get_role("experiment/engineer").expect("experiment/engineer alias should resolve");
    assert_eq!(eng.role_id, "experiment_engineer");

    // Backward compatibility: all 8 standard roles remain intact
    let standard = get_standard_roles();
    assert_eq!(standard.len(), 8, "Expected 8 standard roles in registry");
}

#[test]
fn test_p5_1_branch_workspace_isolation_and_durable_lock_ownership() {
    let tmp = tempfile::tempdir().unwrap();
    let root_buf = tmp.path().join("state");
    let root = root_buf.as_path();
    Runtime::init(root, test_policy(100)).unwrap();
    execution::migrate(root).unwrap();

    let project = "proj-branch-test";
    let branch_a = "branch-worker-a";
    let branch_b = "branch-worker-b";

    create_test_project(root, project);
    BranchWorkspaceManager::init_branch(root, project, branch_a, "main").unwrap();
    BranchWorkspaceManager::init_branch(root, project, branch_b, "main").unwrap();

    // Worker A writes file in branch A
    let sha_a = BranchWorkspaceManager::write_file_with_expected_hash(
        root, project, branch_a, "model.py", "class WorkerA: pass\n", None,
    ).unwrap();
    assert!(!sha_a.is_empty());

    // Worker B lists files in branch B - must NOT see Worker A's model.py
    let files_b = BranchWorkspaceManager::list_files(root, project, branch_b).unwrap();
    let has_model_in_b = files_b.iter().any(|(path, _, _)| path == "model.py");
    assert!(!has_model_in_b, "Branch B must be isolated from Branch A mutations");

    // Acquire branch lock for Worker A
    let lock_a = BranchWorkspaceManager::acquire_lock(root, project, branch_a).unwrap();

    // Conflicting acquisition of branch lock on branch A by Worker C must fail
    let lock_c_attempt = BranchWorkspaceManager::acquire_lock(root, project, branch_a);
    assert!(lock_c_attempt.is_err(), "Concurrent lock on same branch must be rejected");

    // Worker B can lock branch B independently
    let lock_b = BranchWorkspaceManager::acquire_lock(root, project, branch_b).unwrap();
    drop(lock_b);
    drop(lock_a);
    assert!(BranchWorkspaceManager::acquire_lock(root, project, branch_a).is_ok());
}

#[test]
fn test_p5_2_experiment_dag_operations_and_parent_evidence() {
    let tmp = tempfile::tempdir().unwrap();
    let root_buf = tmp.path().join("state");
    let root = root_buf.as_path();
    Runtime::init(root, test_policy(100)).unwrap();
    execution::migrate(root).unwrap();

    let project = "proj-dag";
    let session_id = "session-dag-1";
    create_test_project(root, project);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    db.execute("INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, 'ds1', '/tmp/ds1.csv', 1000, 'sha-ds1', 0)", [project]).unwrap();
    db.execute("INSERT INTO search_sessions(project, id, dataset_id, spec, status, created_ms, updated_ms) VALUES(?1, ?2, 'ds1', '{}', 'evaluating', 0, 0)", params![project, session_id]).unwrap();

    // Insert baseline experiment
    let exp_base = "exp-baseline-001";
    db.execute(
        "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
         VALUES(?1, ?2, ?3, 'tree', '{\"max_depth\": 5}', 2026, 1.0, 'baseline', NULL, 1000)",
        params![project, session_id, exp_base],
    ).unwrap();
    db.execute(
        "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
         VALUES(?1, ?2, 'evaluated', 0.812, '{\"auc\": 0.812}', '{}', 1, 150.0, 'model-base-sha', 'pred-base-sha', 'art.json', 1100)",
        params![project, exp_base],
    ).unwrap();

    // Insert child experiment with parent evidence linkage
    let exp_child = "exp-child-002";
    db.execute(
        "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
         VALUES(?1, ?2, ?3, 'tree', '{\"max_depth\": 10}', 2026, 1.0, 'full_fidelity', ?4, 2000)",
        params![project, session_id, exp_child, exp_base],
    ).unwrap();
    db.execute(
        "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
         VALUES(?1, ?2, 'evaluated', 0.845, '{\"auc\": 0.845}', '{}', 1, 280.0, 'model-child-sha', 'pred-child-sha', 'art2.json', 2100)",
        params![project, exp_child],
    ).unwrap();

    // Query child node and verify parent evidence traversal
    let child_node = get_experiment_dag_node(root, project, exp_child).unwrap();
    assert_eq!(child_node["experiment_id"], exp_child);
    assert_eq!(child_node["primary_metric"], 0.845);
    assert_eq!(child_node["parent_id"], exp_base);

    let parent_ev = &child_node["parent_evidence"];
    assert!(!parent_ev.is_null());
    assert_eq!(parent_ev["parent_id"], exp_base);
    assert_eq!(parent_ev["primary_metric"], 0.812);
    assert_eq!(parent_ev["model_sha256"], "model-base-sha");
    assert_eq!(parent_ev["predictions_sha256"], "pred-base-sha");

    // Query parent node and verify children list
    let parent_node = get_experiment_dag_node(root, project, exp_base).unwrap();
    let children = parent_node["children"].as_array().unwrap();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0], exp_child);
}

#[test]
fn test_p5_2_bounded_best_first_and_comparative_search_strategy() {
    let tmp = tempfile::tempdir().unwrap();
    let root_buf = tmp.path().join("state");
    let root = root_buf.as_path();
    Runtime::init(root, test_policy(100)).unwrap();
    execution::migrate(root).unwrap();

    let project = "proj-search";
    let session_id = "session-search-1";
    create_test_project(root, project);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    db.execute("INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, 'ds1', '/tmp/ds1.csv', 1000, 'sha-ds1', 0)", [project]).unwrap();

    let spec = SearchSpec {
        session_id: session_id.into(),
        project_id: project.into(),
        dataset_id: "ds1".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "roc_auc".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["tree".into(), "linear".into()],
        max_trials: 10,
        max_screen_trials: 6,
        max_full_trials: 4,
        search_wall_ms: 300_000,
        job_timeout_ms: 30_000,
        finalization_allowance_ms: 60_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };
    init_search(root, spec).unwrap();

    // Populate multi-fidelity trials: screening (0.5) and full fidelity (1.0)
    for i in 1..=4 {
        let fid = if i <= 2 { 0.5 } else { 1.0 };
        let exp_id = format!("trial-{i}");
        db.execute(
            "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
             VALUES(?1, ?2, ?3, 'tree', '{\"max_depth\": 5}', 2026, ?4, 'screening', NULL, ?5)",
            params![project, session_id, exp_id, fid, i * 1000],
        ).unwrap();
        db.execute(
            "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
             VALUES(?1, ?2, 'evaluated', ?3, '{\"roc_auc\": ?3}', '{}', 1, 100.0, 'sha-m', 'sha-p', 'p', ?4)",
            params![project, exp_id, 0.80 + (i as f64 * 0.02), i * 1000 + 10],
        ).unwrap();
    }

    let report = evaluate_comparative_strategies(root, project, session_id).unwrap();
    assert_eq!(report.session_id, session_id);
    assert_eq!(report.metric, "roc_auc");
    assert_eq!(report.primary_strategy.strategy_name, "bounded_multi_fidelity_best_first");
    assert_eq!(report.primary_strategy.trials_evaluated, 4);
    assert!(report.primary_strategy.validation_queries_used <= 50, "Validation queries must be capped at 50");
    assert!(report.primary_strategy.reward > 0.0);
    assert_eq!(report.comparative_strategies.len(), 2);
    assert_eq!(report.comparative_strategies[0].strategy_name, "uniform_random_full_fidelity");
    assert_eq!(report.comparative_strategies[1].strategy_name, "mcts_ucb1_exploratory");
    assert!(report.comparison_verdict.contains("bounded_multi_fidelity_best_first"));
}

#[test]
fn test_p5_3_request_accounting_and_context_bounds() {
    let mut ctx = AgentContextManager::new("proj-bounds", "task-bounds", "lead", "You are the Lead scientific agent.");

    // Economical default is 16,384 tokens
    assert_eq!(ctx.max_context_tokens, 16_384);
    assert_eq!(ctx.reserve_tokens, 4_096);

    // Hard ceiling <= 64K (65,536 tokens)
    assert!(ctx.set_context_window(65_536, 8_192).is_ok());
    assert_eq!(ctx.max_context_tokens, 65_536);

    // Over 64K is deterministically rejected
    assert!(ctx.set_context_window(70_000, 8_192).is_err());

    // Reserve >= max is rejected
    assert!(ctx.set_context_window(16_384, 16_384).is_err());

    // Check essential state fits refusal
    let tools = vec![
        ToolDefinition {
            tool_type: "function".into(),
            function: ToolFunction {
                name: "test_tool".into(),
                description: "A tool with small schema".into(),
                parameters: json!({"type": "object"}),
            },
        }
    ];
    // Small window cannot fit tools + reserves + system prompt
    ctx.set_context_window(4_100, 4_096).unwrap();
    let fit_res = ctx.check_essential_state_fits(&tools);
    assert!(fit_res.is_err(), "Should refuse dispatch when essential state cannot fit within context window");
    assert!(fit_res.unwrap_err().contains("cannot fit within configured context window"));
}

#[test]
fn test_p5_3_outbound_filter_and_tool_pairing_validation() {
    // 1. OutboundFilter tabular row redaction
    let raw_csv_output = "col1,col2,col3\n1.2,3.4,5.6\n7.8,9.0,1.1\n2.2,3.3,4.4\n5.5,6.6,7.7\n8.8,9.9,0.0\n1.1,2.2,3.3\n";
    let sanitized = OutboundFilter::sanitize_content(raw_csv_output);
    assert!(sanitized.contains("col1,col2,col3"));
    assert!(sanitized.contains("[REDACTED_ROW_DATA:"));
    assert!(sanitized.contains("raw tabular rows omitted; use aggregate diagnostics"));

    // 2. Tool pairing validation
    let mut ctx = AgentContextManager::new("p", "t", "lead", "system");

    // Case A: Valid assistant tool call and matching tool message
    ctx.add_message(ChatMessage {
        role: "assistant".into(),
        content: None,
        tool_calls: Some(vec![
            ToolCall {
                id: "call-1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "workspace_fs".into(),
                    arguments: "{\"action\": \"list\"}".into(),
                },
            }
        ]),
        tool_call_id: None,
        name: None,
    });
    ctx.add_message(ChatMessage::tool("{\"files\": []}", "call-1", "workspace_fs"));
    assert!(ctx.validate_tool_pairings().is_ok());

    // Case B: Assistant tool call with invalid JSON arguments
    let mut broken_ctx = AgentContextManager::new("p", "t", "lead", "system");
    broken_ctx.add_message(ChatMessage {
        role: "assistant".into(),
        content: None,
        tool_calls: Some(vec![
            ToolCall {
                id: "call-broken".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "workspace_fs".into(),
                    arguments: "{broken_json: true".into(),
                },
            }
        ]),
        tool_call_id: None,
        name: None,
    });
    assert!(broken_ctx.validate_tool_pairings().is_err());

    // Case C: Unmatched/unparented tool response
    let mut unparented_ctx = AgentContextManager::new("p", "t", "lead", "system");
    unparented_ctx.add_message(ChatMessage::tool("{}", "call-ghost", "workspace_fs"));
    assert!(unparented_ctx.validate_tool_pairings().is_err());
}

#[test]
fn test_p5_4_milestone_and_budget_pressure_compaction_preserves_steering_and_evidence() {
    let mut ctx = AgentContextManager::new("proj-compact", "task-compact", "lead", "System prompt");
    ctx.current_hypothesis = Some("Hypothesis: tuning learning_rate yields higher AUC".into());
    ctx.current_steering = Some("Focus on gradient boosting rather than SVM".into());
    ctx.constraints = Some("max_memory_mb <= 2048".into());
    ctx.final_test_identity = Some("test-gate-001".into());
    ctx.approvals = vec!["approval-user-123".into()];
    ctx.set_context_window(4_096, 1_024).unwrap();

    // Add multiple long messages to induce budget pressure
    for i in 0..15 {
        ctx.add_message(ChatMessage::user(format!("Iteration message turn {i} with extensive details {}", "x".repeat(1000))));
        ctx.add_message(ChatMessage::assistant(Some(format!("Turn response {i}")), None));
    }

    assert!(ctx.estimate_tokens() > 3000);
    let compacted = ctx.maybe_compact();
    assert!(compacted, "Compaction should be triggered under context pressure");

    // Verify compaction summary preserves essential state
    let compaction_message = &ctx.messages[1];
    let content = compaction_message.content.as_deref().unwrap();
    assert!(content.contains("Active Hypothesis: Hypothesis: tuning learning_rate yields higher AUC"));
    assert!(content.contains("Current Steering: Focus on gradient boosting rather than SVM"));
    assert!(content.contains("Constraints: max_memory_mb <= 2048"));
    assert!(content.contains("Final Test Identity: test-gate-001"));
    assert!(content.contains("Registered Approvals: approval-user-123"));
}

#[test]
fn test_p5_5_human_and_critic_steering_at_turn_boundaries() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut orch = setup_p5_project(root, "proj-steer", "task-steer");

    // Apply human steering at turn boundary
    orch.apply_steering("Prioritize linear model calibration", "human:operator").unwrap();
    assert_eq!(orch.context.current_steering.as_deref(), Some("Prioritize linear model calibration"));
    assert_eq!(orch.context.handoff_version, 2);

    // Apply critic steering
    orch.apply_critic_steering(
        "Leakage detected in fold 3: feature normalization crossed split boundaries",
        &json!({"verdict": "rejected", "audit_id": "audit-456"}),
    ).unwrap();
    assert!(orch.context.current_steering.as_ref().unwrap().contains("Leakage detected in fold 3"));
    assert_eq!(orch.context.handoff_version, 3);

    // Verify steering message injected at turn boundary
    let last_msg = orch.context.messages.last().unwrap();
    assert!(last_msg.content.as_ref().unwrap().contains("[STEERING DIRECTIVE - Turn Boundary (Epoch v3) by critic:agent]"));
}

#[test]
fn test_p5_5_explicit_user_approval_enforcement() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut orch = setup_p5_project(root, "proj-approval", "task-approval");
    let cand_id = "cand-001";

    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    // Insert approved critic audit
    db.execute(
        "INSERT INTO critic_audits(project, candidate_id, auditor_role, status, rationale, source_hash, protocol, constraints_verified, created_ms)
         VALUES(?1, ?2, 'critic', 'approved', 'Verified cleanly', 'sha123', 'kfold', 1, 1000)",
        params!["proj-approval", cand_id],
    ).unwrap();

    // 1. Attempt finalization without user approval: MUST fail
    let finalize_args = json!({
        "candidate_id": cand_id,
        "rationale": "Attempting finalization without approval",
        "test_access_id": "test-gate-unapproved",
    }).to_string();
    let res = orch.dispatch_tool_call("request_finalization", &finalize_args);
    assert!(res.is_err());
    assert!(res.unwrap_err().contains("lacks verified user approval in approvals ledger"));

    // 2. Register user approval via task contract
    register_user_approval(root, "proj-approval", "appr-001", "deployment_handoff", cand_id, "operator-alice", true, 300_000).unwrap();
    assert!(check_user_approval(root, "proj-approval", "deployment_handoff", cand_id).unwrap());
}

#[test]
fn test_p5_6_acceptance_gate_full_closure() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Step 1: Initialize project and orchestrator
    let mut orch = setup_p5_project(root, "proj-gate", "task-gate");

    // Record an achievement with durable evidence SHA
    let evidence_content = b"Validated fold metrics: fold_1=0.82, fold_2=0.84, fold_3=0.83";
    let sha = format!("{:x}", Sha256::digest(evidence_content));
    let evidence_dir = root.join(".m2/evidence");
    fs::create_dir_all(&evidence_dir).unwrap();
    fs::write(evidence_dir.join(&sha), evidence_content).unwrap();

    orch.context.record_achievement(
        root,
        "metric_verified",
        "Fold metrics successfully evaluated and verified",
        Some(sha.clone()),
        json!({"mean_auc": 0.83}),
    ).unwrap();

    // Apply human steering
    orch.apply_steering("Evaluate ensemble with tree baseline", "human:operator").unwrap();

    // Record checkpoint
    let chk_id = orch.context.record_checkpoint(root).unwrap();
    assert!(!chk_id.is_empty());

    // Step 2: Simulate process crash/restart and restore
    let restored_ctx = AgentContextManager::restore(root, "proj-gate", "task-gate").unwrap();
    assert_eq!(restored_ctx.role_id, "lead");
    assert_eq!(restored_ctx.achievements.len(), 1);
    assert_eq!(restored_ctx.achievements[0].evidence_sha256.as_deref(), Some(sha.as_str()));
    assert_eq!(restored_ctx.current_steering.as_deref(), Some("Evaluate ensemble with tree baseline"));
    assert_eq!(restored_ctx.handoff_version, 2);

    // Step 3: Verify evidence integrity check during restore detects tampering
    fs::write(evidence_dir.join(&sha), b"corrupted evidence payload").unwrap();
    let tamper_restore = AgentContextManager::restore(root, "proj-gate", "task-gate");
    assert!(tamper_restore.is_err(), "Restoring corrupted evidence must fail");
    assert!(tamper_restore.unwrap_err().contains("Durable evidence hash mismatch"));

    // Fix evidence back
    fs::write(evidence_dir.join(&sha), evidence_content).unwrap();

    // Step 4: Verify budget queries and tracking
    let budget_info = AgentContextManager::query_authoritative_budget(root, "proj-gate").unwrap();
    assert_eq!(budget_info["project_id"], "proj-gate");
    assert!(budget_info["remaining"]["wall_ms_effective"].as_i64().unwrap() > 0);
    assert!(budget_info["finalization_reserve"]["wall_ms"].as_i64().unwrap() > 0);

    // Step 5: Verify branch workspace isolation
    let branch_lead = "branch-lead-1";
    let branch_spec = "branch-specialist-1";
    BranchWorkspaceManager::init_branch(root, "proj-gate", branch_lead, "main").unwrap();
    BranchWorkspaceManager::init_branch(root, "proj-gate", branch_spec, "main").unwrap();

    BranchWorkspaceManager::write_file_with_expected_hash(
        root, "proj-gate", branch_lead, "lead_script.py", "print('lead')", None,
    ).unwrap();
    let spec_files = BranchWorkspaceManager::list_files(root, "proj-gate", branch_spec).unwrap();
    assert!(!spec_files.iter().any(|(p, _, _)| p == "lead_script.py"), "Specialist branch must be isolated from lead branch");
}
