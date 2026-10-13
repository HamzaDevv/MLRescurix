use std::{
    collections::BTreeMap,
    fs,
    sync::Arc,
};
use sha2::{Digest, Sha256};
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
    execution::{self, JobSpec, Probe, Supervisor},
    search::{self, SearchSpec},
    Runtime,
};

fn chrono_free_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

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

fn claim_task(r: &Runtime, p: &str, t: &str) {
    let now = chrono_free_now();
    let _ = send(r, p, &format!("claim{t}"), Operation::Claim {
        task_id: t.into(),
        owner: "local-worker".into(),
        lease_id: format!("lease{t}"),
        now_ms: now,
        expires_ms: now + 3_600_000,
        expected_generation: 1,
    });
}

fn execution_task(r: &Runtime, p: &str, t: &str) {
    branch_task(r, p, &format!("b{t}"), t);
    claim_task(r, p, t);
}

fn job(p: &str, id: &str, t: &str, probe: Probe, timeout: u64) -> JobSpec {
    JobSpec {
        job_id: id.into(),
        project_id: p.into(),
        task_id: t.into(),
        generation: 2,
        timeout_ms: timeout,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        probe,
    }
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
    ToolRegistry::register_tool(&root, "p-diag", desc.clone()).unwrap();

    // Re-registering the identical descriptor is idempotent. A changed version
    // descriptor must never overwrite the originally registered tool.
    ToolRegistry::register_tool(&root, "p-diag", desc.clone()).unwrap();
    let mut conflicting = desc.clone();
    conflicting.environment_id = "different-environment".into();
    let err = ToolRegistry::register_tool(&root, "p-diag", conflicting).unwrap_err();
    assert!(err.contains("Tool versions are immutable"));
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let stored: String = db.query_row(
        "SELECT descriptor_json FROM agent_tools WHERE project='p-diag' AND tool_name='missingness_checker' AND version='v1'",
        [], |row| row.get(0),
    ).unwrap();
    assert_eq!(serde_json::from_str::<ToolDescriptor>(&stored).unwrap(), desc);

    // 3. Agent reuses tool in a subsequent turn
    let input = json!({"columns": ["f1", "f2", "f3"]});
    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        let err = ToolRegistry::invoke_tool(&root, "p-diag", "feat-diag", "missingness_checker", "v1", &input).unwrap_err();
        assert!(err.contains("Host Python substitution for untrusted code is prohibited"));
        assert!(!BranchWorkspaceManager::get_branch_dir(&root, "p-diag", "feat-diag").join(".tool_input.json").exists());
        eprintln!("UNVERIFIED: diagnostic execution/adaptation requires an available container runtime; refusal and registration checks passed.");
        return;
    }
    let receipt = ToolRegistry::invoke_tool(&root, "p-diag", "feat-diag", "missingness_checker", "v1", &input).unwrap();

    assert_eq!(receipt.status, "succeeded", "Tool execution failed with exit_code {}: stderr='{}', stdout='{}'", receipt.exit_code, receipt.stderr, receipt.stdout);
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
    branch_task(&r, "p-close", "bt-close", "t-close");

    let csv = d.path().join("data.csv");
    fs::write(&csv, "f1,target\n1,0\n2,1\n3,0\n4,1\n").unwrap();
    execution::register_dataset(&root, "p-close", "data-1", &csv).unwrap();

    // 1. Critic specialist submits audit
    let critic_fixture = Arc::new(ScriptedProviderFixture::new(vec![
        ChatCompletionResponse {
            id: "call-critic".into(),
            model: "test-model".into(),
            content: Some("Critic audit passed. Submitting audit for linear_v1.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call-c1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "submit_critic_audit".into(),
                        arguments: json!({
                            "candidate_id": "linear_v1",
                            "verdict": "approved",
                            "evidence_summary": "Independent audit: validated across 3 seeds without leakage",
                            "source_hash": "hash_linear_v1",
                            "protocol": "stratified_5fold",
                            "protocol_compliance": true,
                        }).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 },
            latency_ms: 10,
        },
        ChatCompletionResponse {
            id: "call-c-stop".into(),
            model: "test-model".into(),
            content: Some("[FINAL] Audit submitted.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 40, completion_tokens: 10, total_tokens: 50 },
            latency_ms: 10,
        },
    ]));

    let critic_config = OrchestratorConfig {
        project_id: "p-close".into(),
        task_id: "t-close".into(),
        role_id: "critic".into(),
        branch_id: "branch-critic".into(),
        model: "test-model".into(),
        max_turns: 5,
    };
    let mut critic_orch = AgentOrchestrator::new(&root, critic_config, critic_fixture).unwrap();
    let c_res1 = critic_orch.step().unwrap();
    assert_eq!(c_res1.tool_calls_executed, 1);
    let c_res2 = critic_orch.step().unwrap();
    assert!(c_res2.stopped);

    // Verify critic audit is recorded in database
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let audit_count: i64 = db.query_row(
        "SELECT count(*) FROM critic_audits WHERE project='p-close' AND candidate_id='linear_v1' AND status='approved'",
        [], |r| r.get(0),
    ).unwrap();
    assert_eq!(audit_count, 1);

    // 2. Negative gate: Lead attempts finalization before user approval is recorded
    let lead_fail_fixture = Arc::new(ScriptedProviderFixture::new(vec![
        ChatCompletionResponse {
            id: "call-finalize-fail".into(),
            model: "test-model".into(),
            content: Some("Lead requesting finalization without user approval.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call-l1".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "request_finalization".into(),
                        arguments: json!({"candidate_id": "linear_v1", "rationale": "Attempt before user approval"}).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 50, completion_tokens: 20, total_tokens: 70 },
            latency_ms: 10,
        },
        ChatCompletionResponse {
            id: "call-stop".into(),
            model: "test-model".into(),
            content: Some("[FINAL] Stopped.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 30, completion_tokens: 10, total_tokens: 40 },
            latency_ms: 10,
        },
    ]));
    let lead_config = OrchestratorConfig {
        project_id: "p-close".into(),
        task_id: "t-close".into(),
        role_id: "lead".into(),
        branch_id: "branch-lead".into(),
        model: "test-model".into(),
        max_turns: 5,
    };
    let mut lead_orch_fail = AgentOrchestrator::new(&root, lead_config.clone(), lead_fail_fixture).unwrap();
    let lf_res = lead_orch_fail.step().unwrap();
    assert_eq!(lf_res.tool_calls_executed, 1);
    assert!(!lead_orch_fail.is_research_closed().unwrap());
    let tool_msg = lead_orch_fail.context.messages.iter().rev().find(|m| m.role == "tool").unwrap();
    assert!(tool_msg.content.as_deref().unwrap().contains("lacks verified user approval"));

    // 3. User records authenticated approval
    db.execute(
        "INSERT INTO approvals(project, candidate_id, approved, user_identity, rationale, created_ms)
         VALUES(?1, ?2, 1, 'lead-engineer', 'Verified offline scorecard and verified constraints', ?3)",
        rusqlite::params!["p-close", "linear_v1", 123456789],
    ).unwrap();

    // 4. Legacy metadata approvals cannot authorize content-bound finalization.
    let lead_succ_fixture = Arc::new(ScriptedProviderFixture::new(vec![
        ChatCompletionResponse {
            id: "call-finalize-succ".into(),
            model: "test-model".into(),
            content: Some("Requesting finalization with approved critic audit and user receipt.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call-l2".into(),
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
        ChatCompletionResponse {
            id: "call-stop-succ".into(),
            model: "test-model".into(),
            content: Some("[FINAL] Research complete and finalized.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 40, completion_tokens: 10, total_tokens: 50 },
            latency_ms: 10,
        },
    ]));
    let mut lead_orch_succ = AgentOrchestrator::new(&root, lead_config, lead_succ_fixture).unwrap();
    let ls_res1 = lead_orch_succ.step().unwrap();
    assert_eq!(ls_res1.tool_calls_executed, 1);
    assert!(!lead_orch_succ.is_research_closed().unwrap());
    let refusal=lead_orch_succ.context.messages.iter().rev().find(|m|m.role=="tool").unwrap();
    assert!(refusal.content.as_deref().unwrap().contains("lacks a verified full-pipeline receipt"));

    // Fault injection for the independent research-closure regression below.
    // This is not evidence of successful refit, sealed-test evaluation or delivery.
    db.execute("UPDATE projects SET state='finalizing' WHERE id='p-close'",[]).unwrap();
    let state: String = db.query_row("SELECT state FROM projects WHERE id='p-close'", [], |r| r.get(0)).unwrap();
    assert_eq!(state, "finalizing");

    // Turn 2: agent stops
    let ls_res2 = lead_orch_succ.step().unwrap();
    assert!(ls_res2.stopped);

    // 5. Negative gate: Duplicate finalization is permanently rejected
    let err_dup = lead_orch_succ.dispatch_tool_call(
        "request_finalization",
        &json!({"candidate_id": "linear_v1", "rationale": "duplicate finalization attempt"}).to_string(),
    ).unwrap_err();
    assert!(err_dup.contains("Research admission is permanently closed"));

    // 6. Any attempt to initiate new research or search is permanently rejected
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

    // Setup 10 projects with 3 branches each
    for i in 0..10 {
        let p = format!("p{i}");
        create_m1(&r, &p, 100);
        for b in 0..3 {
            let branch = format!("branch-{b}");
            BranchWorkspaceManager::init_branch(&root, &p, &branch, "main").unwrap();
            BranchWorkspaceManager::write_file(&root, &p, &branch, "code.py", &format!("# Project {p} branch {branch}")).unwrap();
            let task_id = format!("t-{p}-{b}");
            branch_task(&r, &p, &branch, &task_id);
            claim_task(&r, &p, &task_id);
        }
    }

    // Submit jobs across the 10 projects and 3 branches
    for i in 0..10 {
        let p = format!("p{i}");
        for b in 0..3 {
            let task_id = format!("t-{p}-{b}");
            let job_id = format!("j-{p}-{b}");
            let spec = job(&p, &job_id, &task_id, Probe::Sleep { millis: 20 }, 5000);
            execution::submit(&root, spec).unwrap();
        }
    }

    // Submit an extra job in p0 to test cancellation
    let cancel_task = "t-p0-cancel";
    branch_task(&r, "p0", "branch-0", cancel_task);
    claim_task(&r, "p0", cancel_task);
    let cancel_spec = job("p0", "j-p0-cancel", cancel_task, Probe::Sleep { millis: 500 }, 5000);
    execution::submit(&root, cancel_spec).unwrap();
    let cancel_res = execution::cancel(&root, "p0", "j-p0-cancel").unwrap();
    assert_eq!(cancel_res.status, "cancelled");

    // Launch a 4-worker supervisor
    let bin = std::path::Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 4, bin).unwrap();

    // Verify another supervisor cannot take the same state lock
    assert!(Supervisor::start_with_executable(&root, 4, bin).is_err());

    let mut max_running = 0;
    let mut completed_count = 0;
    for _ in 0..300 {
        supervisor.tick().unwrap();
        let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
        let running: i64 = db.query_row(
            "SELECT count(*) FROM jobs WHERE status IN ('running', 'terminating')",
            [], |x| x.get(0),
        ).unwrap();
        max_running = max_running.max(running);
        let done: i64 = db.query_row(
            "SELECT count(*) FROM jobs WHERE status = 'succeeded'",
            [], |x| x.get(0),
        ).unwrap();
        completed_count = done;
        if done == 30 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(15));
    }

    // Bounded worker allocation assertion: max concurrent running jobs must never exceed the 4 configured workers
    assert!(max_running <= 4, "Concurrent running workers ({max_running}) exceeded ceiling 4");
    assert_eq!(completed_count, 30, "All 30 jobs across 10 projects must complete successfully");

    // Fairness assertion: verify all 10 projects had jobs executed and completed
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    for i in 0..10 {
        let p = format!("p{i}");
        let p_done: i64 = db.query_row(
            "SELECT count(*) FROM jobs WHERE project=?1 AND status='succeeded'",
            [&p], |x| x.get(0),
        ).unwrap();
        assert_eq!(p_done, 3, "Project {p} did not complete all 3 branch jobs (fairness violated)");
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

#[test]
fn m4_reconstructable_checkpoint_and_evidence_integrity() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-rest", 20);
    execution_task(&r, "p-rest", "t-rest");

    let mut ctx = AgentContextManager::new("p-rest", "t-rest", "lead", "Lead system prompt");
    ctx.current_hypothesis = Some("Linear baseline beats null model".into());
    ctx.handoff_version = 3;
    ctx.record_achievement(&root, "data_audited", "Dataset audited cleanly", None, json!({"rows": 50})).unwrap();
    ctx.record_finding(&root, "hyp-1", "supported", "Baseline validated", json!({"score": 0.85}), vec![]).unwrap();
    ctx.record_checkpoint(&root).unwrap();

    // Restore from state
    let restored = AgentContextManager::restore(&root, "p-rest", "t-rest").unwrap();
    assert_eq!(restored.role_id, "lead");
    assert_eq!(restored.current_hypothesis.as_deref(), Some("Linear baseline beats null model"));
    assert_eq!(restored.achievements.len(), 1);
    assert_eq!(restored.findings.len(), 1);
    assert_eq!(restored.handoff_version, 3);

    // Negative check: corrupt evidence file SHA-256
    let ev_content = b"evidence payload for verification";
    let ev_sha = format!("{:x}", Sha256::digest(ev_content));
    let ev_dir = root.join(".m2/evidence");
    fs::create_dir_all(&ev_dir).unwrap();
    let ev_file = ev_dir.join(&ev_sha);
    fs::write(&ev_file, ev_content).unwrap();

    // Register reference in M0 refs table
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO refs(project, id, category, digest) VALUES('p-rest', ?1, 'evidence', ?2)",
        rusqlite::params![&ev_sha, &ev_sha],
    ).unwrap();

    // Record achievement with this evidence
    ctx.record_achievement(&root, "ev_verified", "Verified evidence test", Some(ev_file.to_str().unwrap().to_string()), json!({})).unwrap();
    ctx.record_checkpoint(&root).unwrap();

    // Tamper with evidence file content
    fs::write(&ev_file, b"corrupted evidence payload").unwrap();
    let restore_err = AgentContextManager::restore(&root, "p-rest", "t-rest").unwrap_err();
    assert!(restore_err.contains("Durable evidence hash mismatch") || restore_err.contains("mismatch"));
    fs::remove_file(&ev_file).unwrap();
    let missing_err = AgentContextManager::restore(&root, "p-rest", "t-rest").unwrap_err();
    assert!(missing_err.contains("missing from managed storage"));
}

#[test]
fn m4_role_rbac_enforcement() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-rbac", 20);
    execution_task(&r, "p-rbac", "t-rbac");

    BranchWorkspaceManager::init_branch(&root, "p-rbac", "branch-verifier", "main").unwrap();

    let dummy_fixture = Arc::new(ScriptedProviderFixture::new(vec![]));
    let config = OrchestratorConfig {
        project_id: "p-rbac".into(),
        task_id: "t-rbac".into(),
        role_id: "verifier".into(),
        branch_id: "branch-verifier".into(),
        model: "test-model".into(),
        max_turns: 5,
    };
    let mut orch = AgentOrchestrator::new(&root, config, dummy_fixture).unwrap();

    // Verifier attempts to invoke request_finalization -> rejected by RBAC
    let err_fin = orch.dispatch_tool_call("request_finalization", &json!({"candidate_id": "c1", "rationale": "r"}).to_string()).unwrap_err();
    assert!(err_fin.contains("Role 'verifier' is not authorized to invoke tool 'request_finalization'"));

    // Verifier attempts to invoke delegate -> rejected by RBAC
    let err_del = orch.dispatch_tool_call("delegate", &json!({"role_id": "model_specialist", "intent": "train"}).to_string()).unwrap_err();
    assert!(err_del.contains("Role 'verifier' is not authorized to invoke tool 'delegate'"));

    // Auditor attempts to invoke register_project_tool -> rejected by RBAC
    let mut auditor_orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-rbac".into(),
        task_id: "t-rbac".into(),
        role_id: "auditor".into(),
        branch_id: "branch-verifier".into(),
        model: "test-model".into(),
        max_turns: 5,
    }, Arc::new(ScriptedProviderFixture::new(vec![]))).unwrap();
    let err_tool = auditor_orch.dispatch_tool_call("register_project_tool", &json!({"name": "t", "version": "v1", "description": "d", "entrypoint": "e"}).to_string()).unwrap_err();
    assert!(err_tool.contains("Role 'auditor' is not authorized to invoke tool 'register_project_tool'"));
}

#[test]
fn m4_conservative_ceiling_settlement_on_provider_failure() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-fail", 20);
    branch_task(&r, "p-fail", "bt-fail", "t-fail");

    // Fixture with empty responses causes chat_completion to return Err
    let failing_fixture = Arc::new(ScriptedProviderFixture::new(vec![]));
    let config = OrchestratorConfig {
        project_id: "p-fail".into(),
        task_id: "t-fail".into(),
        role_id: "lead".into(),
        branch_id: "branch-fail".into(),
        model: "test-model".into(),
        max_turns: 5,
    };
    let mut orch = AgentOrchestrator::new(&root, config, failing_fixture).unwrap();

    let step_res = orch.step();
    assert!(step_res.is_err());

    // Verify conservative ceiling settlement in provider_accounting and charges ledgers
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let accounting_row: (String, i64, i64) = db.query_row(
        "SELECT status, cost_millicents, prompt_tokens + completion_tokens FROM provider_accounting WHERE project='p-fail' ORDER BY created_ms DESC LIMIT 1",
        [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).unwrap();
    assert_eq!(accounting_row.0, "outcome_unknown_ceiling_estimate");
    assert_eq!(accounting_row.1, 0, "Unconfigured monetary cost must not be fabricated");
    assert!(accounting_row.2 > 4000, "Conservative token ceiling must be charged");

    // Verify charge in M0 charges ledger
    let charge_count: i64 = db.query_row(
        "SELECT count(*) FROM charges WHERE project='p-fail'",
        [], |r| r.get(0),
    ).unwrap();
    assert!(charge_count > 0, "Failed provider call must be settled in M0 charges ledger");
}

#[test]
fn m4_authoritative_budget_query_and_remaining_calculation() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(50, 4)).unwrap();
    create_m1(&r, "p-bud", 50);

    let budget_info = AgentContextManager::query_authoritative_budget(&root, "p-bud").unwrap();
    assert_eq!(budget_info["project_id"], "p-bud");
    assert_eq!(budget_info["budget"]["calls"], 50);
    assert_eq!(budget_info["budget"]["tokens"], 50_000);
    assert_eq!(budget_info["budget"]["wall_ms"], 500_000);
    // Finalization reserve is 20% of wall_ms
    assert_eq!(budget_info["finalization_reserve_wall_ms"], 100_000);
    // Remaining available
    assert_eq!(budget_info["remaining"]["tokens"], 50_000);
    assert_eq!(budget_info["remaining"]["wall_ms_effective"], 400_000);
}

#[test]
fn m4_schema_validation_pre_and_post_execution() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-schema", 20);

    let desc = ToolDescriptor {
        name: "typed_tool".into(),
        version: "v1".into(),
        description: "Tool with strict schema".into(),
        input_schema: json!({
            "type": "object",
            "required": ["threshold", "tags"],
            "properties": {
                "threshold": {"type": "number"},
                "tags": {"type": "array"}
            }
        }),
        output_schema: json!({
            "type": "object",
            "required": ["score"],
            "properties": {
                "score": {"type": "number"}
            }
        }),
        entrypoint: "tool.py".into(),
        source_hash: "dummy_hash".into(),
        environment_id: "scientific".into(),
        permitted_artifacts: vec![],
        max_wall_ms: 10_000,
        provenance: json!({}),
    };

    ToolRegistry::register_tool(&root, "p-schema", desc).unwrap();

    let schema = json!({
        "type": "object",
        "required": ["threshold", "tags"],
        "properties": {
            "threshold": {"type": "number"},
            "tags": {"type": "array"}
        }
    });

    // 1. Valid input matches schema
    let valid_input = json!({"threshold": 0.75, "tags": ["a", "b"]});
    assert!(ToolRegistry::validate_json_schema(&valid_input, &schema).is_ok());

    // 2. Missing required field rejected
    let invalid_missing = json!({"threshold": 0.75});
    let err_missing = ToolRegistry::validate_json_schema(&invalid_missing, &schema).unwrap_err();
    assert!(err_missing.contains("missing required field 'tags'"));

    // 3. Wrong type rejected
    let type_schema = json!({
        "type": "object",
        "properties": {"threshold": {"type": "number"}}
    });
    let invalid_type = json!({"threshold": "not_a_number", "tags": []});
    let err_type = ToolRegistry::validate_json_schema(&invalid_type, &type_schema).unwrap_err();
    assert!(err_type.contains("expected type 'number'"));
}

#[test]
fn m4_exhausted_budget_blocks_provider_before_dispatch() {
    struct NeverCall;
    impl mlrescurix_harness::agents::ProviderGateway for NeverCall {
        fn chat_completion(&self, _: &mlrescurix_harness::agents::ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            panic!("Provider must not be invoked without budget admission");
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(0, 2)).unwrap();
    create_m1(&runtime, "p-empty", 0);
    branch_task(&runtime, "p-empty", "branch", "task");
    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-empty".into(), task_id: "task".into(), role_id: "lead".into(),
        branch_id: "branch".into(), model: "fixture".into(), max_turns: 1,
    }, Arc::new(NeverCall)).unwrap();
    let error = orch.step().unwrap_err();
    assert!(error.contains("budget exhausted"), "{error}");
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let calls: i64 = db.query_row("SELECT count(*) FROM provider_accounting", [], |row| row.get(0)).unwrap();
    assert_eq!(calls, 0);
}

#[test]
#[cfg(unix)]
fn m4_workspace_symlink_cannot_read_or_overwrite_host_file() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&runtime, "p-link", 20);
    let branch = BranchWorkspaceManager::init_branch(&root, "p-link", "branch", "main").unwrap();
    let sentinel = dir.path().join("sentinel");
    fs::write(&sentinel, "private").unwrap();
    std::os::unix::fs::symlink(&sentinel, branch.join("link")).unwrap();
    assert!(BranchWorkspaceManager::read_file(&root, "p-link", "branch", "link").is_err());
    assert!(BranchWorkspaceManager::write_file(&root, "p-link", "branch", "link", "overwrite").is_err());
    assert!(BranchWorkspaceManager::list_files(&root, "p-link", "branch").is_err());
    assert_eq!(fs::read_to_string(sentinel).unwrap(), "private");
}

#[test]
fn m5_delegation_preserves_prior_tasks_and_reports_provider_failure() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 8)).unwrap();
    create_m1(&runtime, "p", 100);
    branch_task(&runtime, "p", "main", "parent");
    BranchWorkspaceManager::init_branch(&root, "p", "main", "root").unwrap();
    let gateway = Arc::new(ScriptedProviderFixture::new(vec![]));
    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p".into(), task_id: "parent".into(), role_id: "lead".into(),
        branch_id: "main".into(), model: "fixture".into(), max_turns: 3,
    }, gateway).unwrap();
    let args = json!({"role_id":"model_specialist","intent":"investigate"}).to_string();
    let first: serde_json::Value = serde_json::from_str(&orch.dispatch_tool_call("delegate", &args).unwrap()).unwrap();
    let second: serde_json::Value = serde_json::from_str(&orch.dispatch_tool_call("delegate", &args).unwrap()).unwrap();
    assert_eq!(first["status"], "failed");
    assert!(first["error"].is_string());
    assert_ne!(first["sub_task_id"], second["sub_task_id"]);
    let db = rusqlite::Connection::open(&root.join("state.sqlite")).unwrap();
    let count: i64 = db.query_row("SELECT count(*) FROM tasks WHERE project='p'", [], |r| r.get(0)).unwrap();
    assert_eq!(count, 3);
}

#[test]
fn m6_provider_turn_crash_reconciliation_all_four_states() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&runtime, "p-rec", 100);
    branch_task(&runtime, "p-rec", "main", "t-rec");
    let now = chrono_free_now();
    assert_eq!(runtime.command(req("p-rec", "claim-rec", Operation::Claim {
        task_id: "t-rec".into(),
        owner: "agent-t-rec".into(),
        lease_id: "lease-rec-1".into(),
        now_ms: now,
        expires_ms: now + 300_000,
        expected_generation: 1,
    })).unwrap().status, "applied");

    // 1. Crash in `awaiting_response` with a dispatched M0 action
    let act1 = "act-awaiting-1";
    assert_eq!(runtime.command(req("p-rec", "prep-1", Operation::Prepare {
        task_id: "t-rec".into(), generation: 2, lease_id: "lease-rec-1".into(),
        action_id: act1.into(), input: "in1".into(),
        budget: BTreeMap::from([("calls".into(), 1), ("tokens".into(), 5000), ("wall_ms".into(), 60000)]),
        capacity: BTreeMap::from([("workers".into(), 1)]),
        deadline_ms: now + 60_000, now_ms: now,
    })).unwrap().status, "applied");
    assert_eq!(runtime.command(req("p-rec", "disp-1", Operation::Dispatch {
        task_id: "t-rec".into(), generation: 2, lease_id: "lease-rec-1".into(),
        action_id: act1.into(), attempt_id: act1.into(), external_request_id: None, now_ms: now,
    })).unwrap().status, "applied");
    execution::migrate(&root).unwrap();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,updated_ms) VALUES('p-rec','t-rec',1,?1,'awaiting_response',?2)",
        rusqlite::params![act1, now],
    ).unwrap();

    // Gateway that panics if invoked during reconciliation
    struct NoReplayGateway;
    impl mlrescurix_harness::agents::ProviderGateway for NoReplayGateway {
        fn chat_completion(&self, _: &mlrescurix_harness::agents::ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            panic!("Reconciliation must never replay an uncertain provider call");
        }
    }

    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-rec".into(), task_id: "t-rec".into(), role_id: "feature_engineer".into(),
        branch_id: "main".into(), model: "fixture".into(), max_turns: 10,
    }, Arc::new(NoReplayGateway)).unwrap();

    // Blind step() must refuse before reconciliation
    let blind_err = orch.step().unwrap_err();
    assert!(blind_err.contains("requires reconciliation"));

    let rec1 = orch.reconcile().unwrap();
    assert_eq!(rec1["reconciled_turns"][0]["prior_status"], "awaiting_response");
    assert_eq!(rec1["reconciled_turns"][0]["resolution"], "settled_at_admitted_ceiling_no_replay");

    // 2. Crash in `outcome_unknown`
    let act2 = "act-unknown-2";
    assert_eq!(runtime.command(req("p-rec", "prep-2", Operation::Prepare {
        task_id: "t-rec".into(), generation: 2, lease_id: "lease-rec-1".into(),
        action_id: act2.into(), input: "in2".into(),
        budget: BTreeMap::from([("calls".into(), 1), ("tokens".into(), 5000), ("wall_ms".into(), 60000)]),
        capacity: BTreeMap::from([("workers".into(), 1)]),
        deadline_ms: now + 60_000, now_ms: now,
    })).unwrap().status, "applied");
    assert_eq!(runtime.command(req("p-rec", "disp-2", Operation::Dispatch {
        task_id: "t-rec".into(), generation: 2, lease_id: "lease-rec-1".into(),
        action_id: act2.into(), attempt_id: act2.into(), external_request_id: None, now_ms: now,
    })).unwrap().status, "applied");
    assert_eq!(runtime.command(req("p-rec", "unk-2", Operation::MarkUnknown {
        action_id: act2.into(), reason: "network drop".into(),
    })).unwrap().status, "applied");
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,updated_ms) VALUES('p-rec','t-rec',2,?1,'outcome_unknown',?2)",
        rusqlite::params![act2, now],
    ).unwrap();
    let rec2 = orch.reconcile().unwrap();
    assert_eq!(rec2["reconciled_turns"][0]["prior_status"], "outcome_unknown");

    // 3. Crash in `response_received` (cached response with tool calls, before M0 commit or tool execution)
    let act3 = "act-resp-3";
    assert_eq!(runtime.command(req("p-rec", "prep-3", Operation::Prepare {
        task_id: "t-rec".into(), generation: 2, lease_id: "lease-rec-1".into(),
        action_id: act3.into(), input: "in3".into(),
        budget: BTreeMap::from([("calls".into(), 1), ("tokens".into(), 5000), ("wall_ms".into(), 60000)]),
        capacity: BTreeMap::from([("workers".into(), 1)]),
        deadline_ms: now + 60_000, now_ms: now,
    })).unwrap().status, "applied");
    assert_eq!(runtime.command(req("p-rec", "disp-3", Operation::Dispatch {
        task_id: "t-rec".into(), generation: 2, lease_id: "lease-rec-1".into(),
        action_id: act3.into(), attempt_id: act3.into(), external_request_id: None, now_ms: now,
    })).unwrap().status, "applied");
    let cached_resp_3 = ChatCompletionResponse {
        id: "resp-3".into(),
        model: "fixture".into(),
        content: Some("Writing feature file".into()),
        tool_calls: vec![ToolCall {
            id: "call-write-3".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "file_write".into(),
                arguments: json!({"path": "from_rec3.txt", "content": "reconciled-3"}).to_string(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 12,
    };
    orch.context.record_checkpoint(&root).unwrap();
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,updated_ms) VALUES('p-rec','t-rec',3,?1,'response_received',?2,?3)",
        rusqlite::params![act3, serde_json::to_string(&cached_resp_3).unwrap(), now],
    ).unwrap();
    let rec3 = orch.reconcile().unwrap();
    assert_eq!(rec3["reconciled_turns"][0]["prior_status"], "response_received");
    assert_eq!(rec3["reconciled_turns"][0]["tools_executed"], 1);
    assert_eq!(BranchWorkspaceManager::read_file(&root, "p-rec", "main", "from_rec3.txt").unwrap(), "reconciled-3");

    // 4. Crash in `executing_tools` after tool call 1 of 2 already executed and checkpointed
    // Verify tool call 1 is NOT re-executed (e.g., if tool 1 wrote a file that was then modified or checked, or we check tools_skipped_already_completed == 1)
    let act4 = "act-exec-4";
    // A tool execution follows an already-settled authoritative provider action.
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-rec' AND id=?2",rusqlite::params![act4,act3]).unwrap();
    let cached_resp_4 = ChatCompletionResponse {
        id: "resp-4".into(),
        model: "fixture".into(),
        content: Some("Two tool calls".into()),
        tool_calls: vec![
            ToolCall {
                id: "call-4-first".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "file_write".into(),
                    arguments: json!({"path": "tool1.txt", "content": "would-overwrite-if-replayed"}).to_string(),
                },
            },
            ToolCall {
                id: "call-4-second".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "file_write".into(),
                    arguments: json!({"path": "tool2.txt", "content": "second-tool-executed"}).to_string(),
                },
            },
        ],
        usage: TokenUsage { prompt_tokens: 120, completion_tokens: 60, total_tokens: 180 },
        latency_ms: 15,
    };
    // Simulate that tool 1 already ran, wrote "original-from-first-run", and recorded its tool message in the checkpoint
    BranchWorkspaceManager::write_file(&root, "p-rec", "main", "tool1.txt", "original-from-first-run").unwrap();
    orch.context.add_message(ChatMessage::assistant(cached_resp_4.content.clone(), Some(cached_resp_4.tool_calls.clone())));
    orch.context.add_message(ChatMessage::tool(json!({"status": "written", "path": "tool1.txt"}).to_string(), "call-4-first", "file_write"));
    orch.context.record_checkpoint(&root).unwrap();
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,updated_ms) VALUES('p-rec','t-rec',4,?1,'executing_tools',?2,?3)",
        rusqlite::params![act4, serde_json::to_string(&cached_resp_4).unwrap(), now],
    ).unwrap();

    // The second write happened, but its checkpoint was lost in the crash.
    BranchWorkspaceManager::write_file(&root, "p-rec", "main", "tool2.txt", "original-before-crash").unwrap();
    let err = orch.reconcile().unwrap_err();
    assert!(err.contains("tool outcome unknown"), "{err}");
    assert_eq!(BranchWorkspaceManager::read_file(&root, "p-rec", "main", "tool1.txt").unwrap(), "original-from-first-run");
    assert_eq!(BranchWorkspaceManager::read_file(&root, "p-rec", "main", "tool2.txt").unwrap(), "original-before-crash");
    // Modern protocol: effect was published, then both outcome and context checkpoint
    // were lost. The durable intent proves the exact file bytes without replay.
    let args = &cached_resp_4.tool_calls[1].function.arguments;
    mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project:"p-rec",task:"t-rec",branch:"main",turn:4,call_id:"call-4-second",name:"file_write",arguments:args,
        },true,|| panic!("file reconciliation must not use generic dispatch")).unwrap();
    db.execute("UPDATE tool_outcomes SET outcome=NULL,outcome_sha256=NULL WHERE project='p-rec' AND turn=4",[]).unwrap();
    db.execute("UPDATE provider_turn_receipts SET tool_protocol=1 WHERE project='p-rec' AND turn=4",[]).unwrap();
    // Force recovery to claim a new lease and settle from original action evidence.
    db.execute("UPDATE tasks SET data=json_set(data,'$.expires_ms',0) WHERE project='p-rec' AND id='t-rec'",[]).unwrap();
    let recovered = orch.reconcile().unwrap();
    assert_eq!(recovered["reconciled_turns"][0]["tools_executed"],1);
    assert_eq!(BranchWorkspaceManager::read_file(&root,"p-rec","main","tool2.txt").unwrap(),"second-tool-executed");
    assert!(orch.reconcile().unwrap()["reconciled_turns"].as_array().unwrap().is_empty());
    // A persisted outcome survives subsequent workspace changes; it is delivered
    // from its receipt, without clobbering the new content.
    BranchWorkspaceManager::write_file(&root,"p-rec","main","tool2.txt","later-content").unwrap();
    mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project:"p-rec",task:"t-rec",branch:"main",turn:4,call_id:"call-4-second",name:"file_write",arguments:args,
        },false,||panic!("stored result must never replay")).unwrap();
    assert_eq!(BranchWorkspaceManager::read_file(&root,"p-rec","main","tool2.txt").unwrap(),"later-content");
    db.execute("UPDATE tool_outcomes SET outcome=NULL,outcome_sha256=NULL WHERE project='p-rec' AND turn=4",[]).unwrap();
    let conflict = mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project:"p-rec",task:"t-rec",branch:"main",turn:4,call_id:"call-4-second",name:"file_write",arguments:args,
        },false,||panic!("unknown effect must not replay")).unwrap_err();
    assert!(conflict.contains("differs from durable before and after"));
    // Legacy missing action evidence still fails before result restoration.
    db.execute("DELETE FROM tool_outcomes WHERE project='p-rec' AND turn=4",[]).unwrap();
    db.execute("UPDATE provider_turn_receipts SET status='executing_tools' WHERE project='p-rec' AND turn=4",[]).unwrap();
    db.execute("DELETE FROM actions WHERE project='p-rec' AND id=?1",[act4]).unwrap();
    assert!(orch.reconcile().unwrap_err().contains("authoritative action receipt"));
}

#[test]
fn m6_specialist_scheduling_crash_recovery_and_handoff() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&runtime, "p-spec", 100);
    branch_task(&runtime, "p-spec", "main", "lead-task");

    let lead_gw = Arc::new(ScriptedProviderFixture::new(vec![]));
    let orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-spec".into(), task_id: "lead-task".into(), role_id: "lead".into(),
        branch_id: "main".into(), model: "fixture".into(), max_turns: 4,
    }, lead_gw).unwrap();

    let s1 = orch.schedule_specialist("feature_engineer", "Create engineered feature spec", Some("H1: ratio features")).unwrap();
    let s1_id = s1["sub_task_id"].as_str().unwrap().to_string();
    let s1_branch = s1["assigned_branch"].as_str().unwrap().to_string();

    // Create a genuine admitted provider action before recording its cached response.
    let now=chrono_free_now();
    let claim=runtime.command(req("p-spec","claim-recovery-spec",Operation::Claim{
        task_id:s1_id.clone(),owner:format!("agent-{s1_id}"),lease_id:"spec-recovery-lease".into(),
        now_ms:now,expires_ms:now+300_000,expected_generation:1,
    })).unwrap();assert_eq!(claim.status,"applied","{}",claim.result);
    let generation=claim.result["generation"].as_u64().unwrap();
    let prepared=runtime.command(req("p-spec","prepare-recovery-spec",Operation::Prepare{
        task_id:s1_id.clone(),generation,lease_id:"spec-recovery-lease".into(),action_id:"act-s1-1".into(),input:"specialist fixture request".into(),
        budget:BTreeMap::from([("calls".into(),1),("tokens".into(),5000),("wall_ms".into(),60000)]),
        capacity:BTreeMap::from([("workers".into(),1)]),deadline_ms:now+60000,now_ms:now,
    })).unwrap();assert_eq!(prepared.status,"applied","{}",prepared.result);
    let dispatched=runtime.command(req("p-spec","dispatch-recovery-spec",Operation::Dispatch{
        task_id:s1_id.clone(),generation,lease_id:"spec-recovery-lease".into(),action_id:"act-s1-1".into(),attempt_id:"spec-recovery-attempt".into(),external_request_id:None,now_ms:now,
    })).unwrap();assert_eq!(dispatched.status,"applied","{}",dispatched.result);

    // Simulate a crash mid-turn (`response_received`) on s1 before running the scheduler
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let cached_s1 = ChatCompletionResponse {
        id: "s1-turn1".into(),
        model: "fixture".into(),
        content: Some("Writing feature note [STOP]".into()),
        tool_calls: vec![ToolCall {
            id: "s1-call-1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "file_write".into(),
                arguments: json!({"path": "fe_spec.txt", "content": "ratio_a_b"}).to_string(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 80, completion_tokens: 40, total_tokens: 120 },
        latency_ms: 10,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,updated_ms) VALUES('p-spec',?1,1,'act-s1-1','response_received',?2,?3)",
        rusqlite::params![s1_id, serde_json::to_string(&cached_s1).unwrap(), chrono_free_now()],
    ).unwrap();

    // Scripted gateway has only 1 response for turn 2 of s1 (turn 1 must be recovered from `response_received` without calling the provider)
    let sched_gw = Arc::new(ScriptedProviderFixture::new(vec![
        ChatCompletionResponse {
            id: "s1-turn2".into(),
            model: "fixture".into(),
            content: Some("Specialist complete [STOP]".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 90, completion_tokens: 20, total_tokens: 110 },
            latency_ms: 8,
        },
    ]));

    let summary = AgentOrchestrator::run_scheduled_specialists(&root, "p-spec", sched_gw, 3).unwrap();
    assert_eq!(summary["specialists_processed"], 1);
    assert_eq!(summary["specialists"][0]["status"], "completed");
    assert_eq!(summary["specialists"][0]["reconciled"][0]["prior_status"], "response_received");
    assert_eq!(BranchWorkspaceManager::read_file(&root, "p-spec", &s1_branch, "fe_spec.txt").unwrap(), "ratio_a_b");

    let (task_status, handoff_status): (String, String) = db.query_row(
        "SELECT t.status, h.status FROM agent_tasks t JOIN handoffs h ON t.project=h.project AND t.task_id=h.task_id WHERE t.project='p-spec' AND t.task_id=?1",
        [&s1_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).unwrap();
    assert_eq!(task_status, "completed");
    assert_eq!(handoff_status, "accepted");
}

#[test]
fn m6_outbound_view_policy_approval_content_binding_and_expiry() {
    struct PolicyBoundGateway;
    impl mlrescurix_harness::agents::ProviderGateway for PolicyBoundGateway {
        fn requires_policy_approval(&self) -> bool { true }
        fn chat_completion(&self, _: &mlrescurix_harness::agents::ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                id: "pol-1".into(),
                model: "fixture".into(),
                content: Some("Approved outbound response [STOP]".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 50, completion_tokens: 10, total_tokens: 60 },
                latency_ms: 5,
            })
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 2)).unwrap();
    create_m1(&runtime, "p-out", 100);
    branch_task(&runtime, "p-out", "main", "t-out");

    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-out".into(), task_id: "t-out".into(), role_id: "lead".into(),
        branch_id: "main".into(), model: "fixture".into(), max_turns: 5,
    }, Arc::new(PolicyBoundGateway)).unwrap();
    orch.context.add_message(ChatMessage::user("Inspect summary metrics"));

    // 1. Unapproved outbound call is rejected
    let err_unapproved = orch.step().unwrap_err();
    assert!(err_unapproved.contains("outbound provider request requires a content-bound policy-approved view"));

    // 2. Stage and approve with expired TTL -> rejected
    let staged = orch.stage_outbound_view("security-reviewer").unwrap();
    let digest = staged["request_sha256"].as_str().unwrap().to_string();
    mlrescurix_harness::agents::approve_outbound_view(&root, "p-out", &digest, "security-reviewer", true, 1).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let err_expired = orch.step().unwrap_err();
    assert!(err_expired.contains("outbound provider request requires a content-bound policy-approved view"));

    // 3. Approve with valid TTL, then mutate context before step -> rejected due to digest mismatch
    mlrescurix_harness::agents::approve_outbound_view(&root, "p-out", &digest, "security-reviewer", true, 600_000).unwrap();
    orch.context.add_message(ChatMessage::user("Mutated prompt content after approval"));
    let err_mutated = orch.step().unwrap_err();
    assert!(err_mutated.contains("outbound provider request requires a content-bound policy-approved view"));

    // 4. Stage and approve exact current request -> succeeds
    let staged2 = orch.stage_outbound_view("security-reviewer").unwrap();
    let digest2 = staged2["request_sha256"].as_str().unwrap().to_string();
    assert_ne!(digest, digest2);
    mlrescurix_harness::agents::approve_outbound_view(&root, "p-out", &digest2, "security-reviewer", true, 600_000).unwrap();
    let res = orch.step().unwrap();
    assert!(res.stopped);
}

#[test]
fn m6_currency_cost_millicents_and_finalization_reserve_enforcement() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let policy = Policy {
        budget: BTreeMap::from([
            ("calls".into(), 10),
            ("tokens".into(), 100_000),
            ("wall_ms".into(), 500_000),
            ("cost_millicents".into(), 10_000),
        ]),
        capacity: BTreeMap::from([("workers".into(), 2)]),
        max_tasks: 64,
    };
    let runtime = Runtime::init(&root, policy.clone()).unwrap();
    runtime.command(req("p-cost", "create-p-cost", Operation::CreateProject {
        intent: "cost-governed-study".into(),
        policy,
        finalization_id: "fin-cost".into(),
    })).unwrap();
    branch_task(&runtime, "p-cost", "main", "t-cost");

    let gw = Arc::new(ScriptedProviderFixture::new(vec![
        ChatCompletionResponse {
            id: "c-1".into(),
            model: "priced-model".into(),
            content: Some("Turn 1 complete".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 1_000, completion_tokens: 500, total_tokens: 1_500 },
            latency_ms: 10,
        },
        ChatCompletionResponse {
            id: "c-2".into(),
            model: "priced-model".into(),
            content: Some("Finalization critic turn complete [STOP]".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 500, completion_tokens: 200, total_tokens: 700 },
            latency_ms: 10,
        },
    ]));

    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-cost".into(), task_id: "t-cost".into(), role_id: "lead".into(),
        branch_id: "main".into(), model: "priced-model".into(), max_turns: 5,
    }, gw.clone()).unwrap();

    // 1. Fails closed when model has no pricing registered in model_pricing
    let unpriced_err = orch.step().unwrap_err();
    assert!(unpriced_err.contains("no registered pricing in model_pricing"), "{unpriced_err}");

    // 2. Register model pricing: 100 millicents / 1k prompt, 200 millicents / 1k completion
    mlrescurix_harness::agents::register_model_pricing(&root, "priced-model", 100, 200).unwrap();
    let turn1 = orch.step().unwrap();
    assert_eq!(turn1.turn_index, 1);

    let budget_after_1 = AgentContextManager::query_authoritative_budget(&root, "p-cost").unwrap();
    // 1000 prompt * 100/1000 + 500 completion * 200/1000 = 100 + 100 = 200 millicents charged
    assert_eq!(budget_after_1["consumption"]["cost_millicents_charged"], 200);
    assert_eq!(budget_after_1["finalization_reserve"]["cost_millicents"], 2_000);
    assert_eq!(budget_after_1["remaining"]["cost_millicents_gross"], 9_800);
    assert_eq!(budget_after_1["remaining"]["cost_millicents_effective"], 7_800);

    // 3. Charge exploratory consumption up to 80% ceiling via authoritative M0 ledger (leaving only the 20% finalization reserve: 2 calls, 2000 cost_millicents)
    let now = chrono_free_now();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let task_json: String = db.query_row("SELECT data FROM tasks WHERE project='p-cost' AND id='t-cost'", [], |r| r.get(0)).unwrap();
    let task: Task = serde_json::from_str(&task_json).unwrap();
    let lease_id = task.lease_id.unwrap();
    let exp_budget = BTreeMap::from([
        ("calls".into(), 7),
        ("tokens".into(), 75_000),
        ("wall_ms".into(), 350_000),
        ("cost_millicents".into(), 7_800),
    ]);
    runtime.command(req("p-cost", "exp-prep", Operation::Prepare {
        task_id: "t-cost".into(),
        generation: task.generation,
        lease_id: lease_id.clone(),
        action_id: "exp-act".into(),
        input: "exp".into(),
        budget: exp_budget.clone(),
        capacity: BTreeMap::from([("workers".into(), 1)]),
        deadline_ms: now + 60_000,
        now_ms: now,
    })).unwrap();
    runtime.command(req("p-cost", "exp-disp", Operation::Dispatch {
        task_id: "t-cost".into(),
        generation: task.generation,
        lease_id: lease_id.clone(),
        action_id: "exp-act".into(),
        attempt_id: "exp-act".into(),
        external_request_id: None,
        now_ms: now,
    })).unwrap();
    runtime.command(req("p-cost", "exp-resp", Operation::Response {
        task_id: "t-cost".into(),
        generation: task.generation,
        lease_id: lease_id.clone(),
        action_id: "exp-act".into(),
        frame: json!({
            "version": 1,
            "kind": "observation",
            "payload": {
                "project_id": "p-cost",
                "task_id": "t-cost",
                "action_id": "exp-act",
                "generation": task.generation,
                "phase": "exploratory_charge",
                "detail": "ok"
            }
        }).to_string(),
        now_ms: now,
    })).unwrap();
    runtime.command(req("p-cost", "exp-commit", Operation::Commit {
        task_id: "t-cost".into(),
        generation: task.generation,
        lease_id,
        action_id: "exp-act".into(),
        actual: exp_budget,
        outcome: "succeeded".into(),
        now_ms: now,
    })).unwrap();

    // Exploratory turn / specialist / pipeline must now be blocked to protect the 20% finalization reserve
    let reserve_err = orch.step().unwrap_err();
    assert!(reserve_err.contains("20% finalization reserve protected"), "{reserve_err}");
    let spec_err = orch.schedule_specialist("auditor", "extra exploratory audit", None).unwrap_err();
    assert!(spec_err.contains("20% finalization reserve protected"), "{spec_err}");

    // 4. Finalization turn (e.g., critic role or [FINALIZE] directive) is allowed to consume from the reserved 20%
    orch.context.add_message(ChatMessage::user("[FINALIZE] Proceed with finalization review"));
    let fin_turn = orch.step().unwrap();
    assert!(fin_turn.stopped);
}
