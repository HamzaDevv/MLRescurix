use std::{
    collections::BTreeMap,
    fs,
    sync::Arc,
};
use serde_json::json;
use mlrescurix_harness::{
    agents::{
        context::AgentContextManager,
        gateway::{
            ChatMessage, ChatCompletionResponse, FunctionCall,
            OutboundFilter, ScriptedProviderFixture, TokenUsage, ToolCall,
        },
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
        roles::{get_role, get_standard_roles},
        sandbox::{SandboxDetector, SandboxLimits, SandboxRunner, SandboxStatus},
        tools::{ToolDescriptor, ToolRegistry},
        workspace::BranchWorkspaceManager,
    },
    contracts::Context,
    core::*,
    execution,
    search::{self, SearchSpec},
    Runtime,
};

fn m1policy(limit: i64, workers: i64) -> Policy {
    Policy {
        budget: BTreeMap::from([
            ("calls".into(), limit),
            ("tokens".into(), limit * 1000),
            ("wall_ms".into(), limit * 10000),
        ]),
        capacity: BTreeMap::from([("workers".into(), workers)]),
        max_tasks: 1024,
    }
}

fn context(base: &str, code: &str) -> Context {
    Context {
        role_id: "researcher".into(),
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

fn req(project: &str, cmd: &str, op: Operation) -> Request {
    Request {
        command_id: cmd.into(),
        project_id: project.into(),
        expected_revision: None,
        operation: op,
    }
}

fn send(r: &Runtime, p: &str, c: &str, o: Operation) -> mlrescurix_harness::Receipt {
    r.command(req(p, c, o)).unwrap()
}

fn create_m1(r: &Runtime, p: &str, limit: i64) {
    assert_eq!(
        send(r, p, "project", Operation::CreateProject {
            intent: "M4 agent testing".into(),
            finalization_id: format!("final-{p}"),
            policy: m1policy(limit, 8),
        }).status,
        "applied"
    );
}

fn branch_task(r: &Runtime, p: &str, branch: &str, task: &str) {
    let _ = send(r, p, &format!("b{branch}"), Operation::CreateBranch {
        branch_id: branch.into(),
        base_hash: format!("base-{branch}"),
        code_hash: format!("code-{branch}"),
    });
    let _ = send(r, p, &format!("t{task}"), Operation::CreateTask {
        task_id: task.into(),
        branch_id: branch.into(),
        context: context(&format!("base-{branch}"), &format!("code-{branch}")),
    });
}

fn execution_task(r: &Runtime, p: &str, t: &str) {
    branch_task(r, p, &format!("b{t}"), t);
}

#[test]
fn m4_role_registry_defines_all_eight_specialist_roles() {
    let roles = get_standard_roles();
    assert_eq!(roles.len(), 8);

    let role_ids: Vec<String> = roles.into_iter().map(|r| r.role_id).collect();
    assert!(role_ids.contains(&"lead".to_string()));
    assert!(role_ids.contains(&"auditor".to_string()));
    assert!(role_ids.contains(&"feature_engineer".to_string()));
    assert!(role_ids.contains(&"model_specialist".to_string()));
    assert!(role_ids.contains(&"search_engineer".to_string()));
    assert!(role_ids.contains(&"diagnostic_engineer".to_string()));
    assert!(role_ids.contains(&"verifier".to_string()));
    assert!(role_ids.contains(&"critic".to_string()));

    let critic = get_role("critic").unwrap();
    assert!(critic.capabilities.contains(&"inspect_evidence".to_string()));
    assert!(critic.capabilities.contains(&"approve_or_reject_finalization".to_string()));
}

#[test]
fn m4_sandbox_detector_and_honest_refusal_when_unavailable() {
    let d = tempfile::tempdir().unwrap();
    let workspace = d.path().join("workspace");
    fs::create_dir_all(&workspace).unwrap();
    let dummy_script = workspace.join("test.py");
    fs::write(&dummy_script, "print('hello')").unwrap();

    let status = SandboxDetector::detect();
    match status {
        SandboxStatus::Available { engine, version } => {
            assert_eq!(engine, "docker");
            assert!(!version.is_empty());
        }
        SandboxStatus::Unavailable { reason } => {
            // When Docker daemon is inactive, execute MUST refuse rather than substitute host Python
            let limits = SandboxLimits::default();
            let res = SandboxRunner::execute(&workspace, "test.py", &[], &limits, None);
            assert!(res.is_err());
            let err = res.unwrap_err();
            assert!(err.contains("Host Python substitution for untrusted code is prohibited"));
            assert!(err.contains(&reason));
        }
    }
}

#[test]
fn m4_path_traversal_and_symlink_rejection_in_workspace() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(10, 2)).unwrap();
    create_m1(&r, "p-sec", 10);

    BranchWorkspaceManager::init_branch(&root, "p-sec", "branch-sec", "main").unwrap();

    // Traversal attack
    let err = BranchWorkspaceManager::write_file(&root, "p-sec", "branch-sec", "../../../escape.txt", "malicious").unwrap_err();
    assert!(err.contains("escapes branch workspace boundary"));

    // Reading traversal
    let err_read = BranchWorkspaceManager::read_file(&root, "p-sec", "branch-sec", "../../etc/passwd").unwrap_err();
    assert!(err_read.contains("escapes branch workspace boundary"));
}

#[test]
fn m4_agent_authors_tests_and_reuses_diagnostic_tool_and_adapts_decision() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-diag", 20);
    execution_task(&r, "p-diag", "t-diag");

    BranchWorkspaceManager::init_branch(&root, "p-diag", "feat-diag", "main").unwrap();

    // 1. Diagnostic specialist authors a Python diagnostic tool inside its branch
    let tool_script = r#"
import json
import sys

def main():
    input_file = sys.argv[2]
    with open(input_file) as f:
        data = json.load(f)
    features = data.get("columns", [])
    # Calculate simulated missingness
    missing_report = {col: 0.45 if col == "f2" else 0.0 for col in features}
    print(json.dumps({"status": "success", "missing_report": missing_report, "high_missing_column": "f2"}))

if __name__ == "__main__":
    main()
"#;
    BranchWorkspaceManager::write_file(&root, "p-diag", "feat-diag", "diagnostics/missingness.py", tool_script).unwrap();

    // 2. Compute branch head hash and register versioned descriptor
    let source_hash = BranchWorkspaceManager::compute_branch_hash(&root, "p-diag", "feat-diag").unwrap();
    let desc = ToolDescriptor {
        name: "missingness_checker".into(),
        version: "v1".into(),
        description: "Checks missingness fractions across tabular features".into(),
        input_schema: json!({"type": "object", "properties": {"columns": {"type": "array"}}}),
        output_schema: json!({"type": "object"}),
        entrypoint: "diagnostics/missingness.py".into(),
        source_hash,
        environment_id: "scientific".into(),
        permitted_artifacts: vec![],
        max_wall_ms: 10_000,
        provenance: json!({"author_role": "diagnostic_engineer", "task_id": "t-diag"}),
    };
    ToolRegistry::register_tool(&root, "p-diag", desc).unwrap();

    // 3. Agent reuses tool in a subsequent turn
    let input = json!({"columns": ["f1", "f2", "f3"]});
    let receipt = ToolRegistry::invoke_tool(&root, "p-diag", "feat-diag", "missingness_checker", "v1", &input).unwrap();

    assert_eq!(receipt.status, "succeeded");
    assert_eq!(receipt.exit_code, 0);
    let output_data = receipt.output_data.expect("Output data must be parsed JSON");
    assert_eq!(output_data["high_missing_column"], "f2");

    // 4. Agent adapts decision based on result: record finding that f2 requires median imputation
    let mut ctx = AgentContextManager::new("p-diag", "t-diag", "diagnostic_engineer", "system prompt");
    let finding = ctx.record_finding(
        &root,
        "hyp-missingness-f2",
        "supported",
        "Feature f2 has 45% missingness; median imputation required before fitting linear model",
        output_data["missing_report"].clone(),
        vec!["diagnostics/missingness.py".into()],
    ).unwrap();

    assert_eq!(finding.verdict, "supported");
    assert!(finding.rationale.contains("Feature f2 has 45% missingness"));

    // Verify finding persisted in SQLite
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let verdict: String = db.query_row("SELECT verdict FROM scientific_findings WHERE project='p-diag' AND hypothesis_id='hyp-missingness-f2'", [], |r| r.get(0)).unwrap();
    assert_eq!(verdict, "supported");
}

#[test]
fn m4_cross_branch_immutable_snapshot_reuse() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-iso", 20);

    // Branch A creates utility
    BranchWorkspaceManager::init_branch(&root, "p-iso", "branch-a", "main").unwrap();
    BranchWorkspaceManager::write_file(&root, "p-iso", "branch-a", "utils/clean.py", "def clean(): return 1").unwrap();

    // Branch A exports immutable snapshot
    let (snap_hash, snap_dir) = BranchWorkspaceManager::export_immutable_snapshot(&root, "p-iso", "branch-a").unwrap();
    assert!(snap_dir.exists());

    // Branch B imports snapshot
    BranchWorkspaceManager::init_branch(&root, "p-iso", "branch-b", "main").unwrap();
    BranchWorkspaceManager::import_snapshot(&root, "p-iso", "branch-b", &snap_hash).unwrap();

    let content_b = BranchWorkspaceManager::read_file(&root, "p-iso", "branch-b", "utils/clean.py").unwrap();
    assert_eq!(content_b, "def clean(): return 1");

    // Branch A modifies its file
    BranchWorkspaceManager::write_file(&root, "p-iso", "branch-a", "utils/clean.py", "def clean(): return 999").unwrap();

    // Branch B remains completely isolated and unchanged!
    let content_b_after = BranchWorkspaceManager::read_file(&root, "p-iso", "branch-b", "utils/clean.py").unwrap();
    assert_eq!(content_b_after, "def clean(): return 1");
}

#[test]
fn m4_achievement_checkpointing_and_context_compaction() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-ctx", 20);
    execution_task(&r, "p-ctx", "t-ctx");

    let mut ctx = AgentContextManager::new("p-ctx", "t-ctx", "lead", "You are the research lead.");
    ctx.max_context_tokens = 500;
    ctx.reserve_tokens = 100;

    // Add achievements
    ctx.record_achievement(&root, "data_audited", "Dataset verified with 100 rows and 5 features", None, json!({"rows": 100})).unwrap();
    ctx.record_achievement(&root, "baseline_evaluated", "Baseline linear model achieved balanced_accuracy=0.78", None, json!({"balanced_accuracy": 0.78})).unwrap();

    // Record an earlier failed hypothesis
    ctx.record_finding(&root, "hyp-polynomial-features", "rejected", "Degree 3 polynomial caused extreme overfitting and degraded validation score", json!({"score": 0.51}), vec![]).unwrap();

    // Add many verbose messages to trigger token limit
    for i in 0..20 {
        ctx.add_message(ChatMessage::user(format!("Turn {i}: Intermediate diagnostic verbose logs with multiple trace lines...")));
        ctx.add_message(ChatMessage::assistant(Some(format!("Turn {i}: Response reasoning steps...")), None));
    }

    assert!(ctx.estimate_tokens() > 400);

    // Compaction boundary
    let compacted = ctx.maybe_compact();
    assert!(compacted);

    // System prompt preserved
    assert_eq!(ctx.messages[0].role, "system");
    // Summary of achievements and findings preserved
    let summary_msg = &ctx.messages[1].content.as_ref().unwrap();
    assert!(summary_msg.contains("Context Compaction & Authoritative State Summary"));
    assert!(summary_msg.contains("data_audited"));
    assert!(summary_msg.contains("baseline_evaluated"));
    assert!(summary_msg.contains("hyp-polynomial-features"));

    // Test retrieval of earlier failed approach
    let failed = ctx.retrieve_historical_findings("polynomial");
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].verdict, "rejected");

    // Apply steering at turn boundary
    ctx.apply_steering(&root, "Focus strictly on tree depth tuning rather than feature expansion").unwrap();
    assert_eq!(ctx.handoff_version, 2);
    let last_msg = ctx.messages.last().unwrap();
    assert!(last_msg.content.as_ref().unwrap().contains("Focus strictly on tree depth tuning"));
}

#[test]
fn m4_outbound_data_redaction_filter() {
    let raw_csv_output = "col1,col2,col3\n1.2,3.4,5.6\n7.8,9.0,1.1\n2.2,3.3,4.4\n5.5,6.6,7.7\n8.8,9.9,0.0\n1.1,2.2,3.3\n";
    let sanitized = OutboundFilter::sanitize_content(raw_csv_output);

    // Kept header and first few lines, redacted the rest
    assert!(sanitized.contains("col1,col2,col3"));
    assert!(sanitized.contains("[REDACTED_ROW_DATA:"));
    assert!(sanitized.contains("raw tabular rows omitted; use aggregate diagnostics"));
}

#[test]
fn m4_research_closure_and_critic_finalization() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-close", 20);
    execution_task(&r, "p-close", "t-close");

    let csv = d.path().join("data.csv");
    fs::write(&csv, "f1,target\n1,0\n2,1\n3,0\n4,1\n").unwrap();
    execution::register_dataset(&root, "p-close", "data-1", &csv).unwrap();

    let fixture = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: critic requests finalization
        ChatCompletionResponse {
            id: "call-finalize".into(),
            model: "test-model".into(),
            content: Some("Critic audit passed. Requesting finalization for apex candidate linear_v1.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call-1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "request_finalization".into(),
                        arguments: json!({"candidate_id": "linear_v1", "rationale": "Validated across 3 seeds without leakage"}).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 },
            latency_ms: 10,
        },
        // Turn 2: final stop
        ChatCompletionResponse {
            id: "call-stop".into(),
            model: "test-model".into(),
            content: Some("[FINAL] Research complete and finalized.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 40, completion_tokens: 10, total_tokens: 50 },
            latency_ms: 10,
        },
    ]));

    let config = OrchestratorConfig {
        project_id: "p-close".into(),
        task_id: "t-close".into(),
        role_id: "critic".into(),
        branch_id: "branch-critic".into(),
        model: "test-model".into(),
        max_turns: 5,
    };

    let mut orch = AgentOrchestrator::new(&root, config, fixture).unwrap();

    // Turn 1: finalization admitted
    let res1 = orch.step().unwrap();
    assert_eq!(res1.tool_calls_executed, 1);
    assert!(orch.is_research_closed().unwrap());

    // Verify project state in SQLite transitioned to finalizing
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let state: String = db.query_row("SELECT state FROM projects WHERE id='p-close'", [], |r| r.get(0)).unwrap();
    assert_eq!(state, "finalizing");

    // Turn 2: agent stops
    let res2 = orch.step().unwrap();
    assert!(res2.stopped);

    // Any attempt to initiate new research or search is permanently rejected
    let search_spec = SearchSpec {
        session_id: "sess-after-close".into(),
        project_id: "p-close".into(),
        dataset_id: "data-1".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "balanced_accuracy".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into()],
        max_trials: 4,
        max_screen_trials: 2,
        max_full_trials: 1,
        search_wall_ms: 60_000,
        job_timeout_ms: 10_000,
        finalization_allowance_ms: 5_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };
    let err = search::init_search(&root, search_spec).unwrap_err();
    assert!(err.contains("research admission is closed"));
}

#[test]
fn m4_ten_projects_and_three_branches_fairness() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(1000, 32)).unwrap();

    for i in 0..10 {
        let p = format!("p{i}");
        create_m1(&r, &p, 100);
        for b in 0..3 {
            let branch = format!("branch-{b}");
            BranchWorkspaceManager::init_branch(&root, &p, &branch, "main").unwrap();
            BranchWorkspaceManager::write_file(&root, &p, &branch, "code.py", &format!("# Project {p} branch {branch}")).unwrap();
        }
    }

    // Verify all 30 branches exist and are isolated
    for i in 0..10 {
        let p = format!("p{i}");
        for b in 0..3 {
            let branch = format!("branch-{b}");
            let code = BranchWorkspaceManager::read_file(&root, &p, &branch, "code.py").unwrap();
            assert_eq!(code, format!("# Project {p} branch {branch}"));
        }
    }
}
