use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
};
use sha2::{Digest, Sha256};
use serde_json::json;
use mlrescurix_harness::{
    agents::{
        context::AgentContextManager,
        gateway::{
            ChatMessage, ChatCompletionRequest, ChatCompletionResponse, FunctionCall,
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
    JobSpec { resources: Default::default(),
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

#[test]
fn p1_scheduler_job_submission_crash_reconciliation_and_fault_regressions() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(200, 8)).unwrap();
    create_m1(&runtime, "p-sub", 200);
    branch_task(&runtime, "p-sub", "main", "t-sub");
    let now = chrono_free_now();
    assert_eq!(
        runtime.command(req("p-sub", "claim-t-sub", Operation::Claim {
            task_id: "t-sub".into(),
            owner: "agent-t-sub".into(),
            lease_id: "leaset-sub".into(),
            now_ms: now,
            expires_ms: now + 3_600_000,
            expected_generation: 1,
        })).unwrap().status,
        "applied"
    );
    execution::migrate(&root).unwrap();

    // Register test dataset
    let ds_path = dir.path().join("dataset.csv");
    fs::write(&ds_path, "row_id,feature,target\n1,0.1,0\n2,0.9,1\n3,0.5,1\n").unwrap();
    execution::register_dataset(&root, "p-sub", "ds-1", &ds_path).unwrap();

    // Register ready container environment
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO environments(id, python, identity, path, status) VALUES('env-test', 'docker:mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd', 'ident-test', '/test/env', 'ready')",
        [],
    ).unwrap();

    struct NoReplayGateway;
    impl mlrescurix_harness::agents::ProviderGateway for NoReplayGateway {
        fn chat_completion(&self, _: &mlrescurix_harness::agents::ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            panic!("Reconciliation must never replay an uncertain provider call");
        }
    }

    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-sub".into(), task_id: "t-sub".into(), role_id: "feature_engineer".into(),
        branch_id: "main".into(), model: "fixture".into(), max_turns: 10,
    }, Arc::new(NoReplayGateway)).unwrap();

    let now = chrono_free_now();

    // -------------------------------------------------------------
    // Regression 1: Crash before submission (intent saved, job not in jobs table)
    // Missing authoritative evidence must produce explicit uncertain/rejected outcome.
    // -------------------------------------------------------------
    let act1 = "act-turn-1";
    assert_eq!(runtime.command(req("p-sub", "prep-1", Operation::Prepare {
        task_id: "t-sub".into(), generation: 2, lease_id: "leaset-sub".into(),
        action_id: act1.into(), input: "in1".into(),
        budget: BTreeMap::from([("calls".into(), 1), ("tokens".into(), 5000), ("wall_ms".into(), 60000)]),
        capacity: BTreeMap::from([("workers".into(), 1)]),
        deadline_ms: now + 60_000, now_ms: now,
    })).unwrap().status, "applied");
    assert_eq!(runtime.command(req("p-sub", "disp-1", Operation::Dispatch {
        task_id: "t-sub".into(), generation: 2, lease_id: "leaset-sub".into(),
        action_id: act1.into(), attempt_id: act1.into(), external_request_id: None, now_ms: now,
    })).unwrap().status, "applied");

    let protocol_1 = json!({
        "format": "mlrescurix.pipeline.v1",
        "experiment_id": "cand-pre-crash",
        "target_column": "target",
        "folds": 2
    });
    let call_args_1 = json!({
        "job_id": "job-pre-crash",
        "dataset_id": "ds-1",
        "environment_id": "env-test",
        "protocol": protocol_1
    }).to_string();

    let resp_turn_1 = ChatCompletionResponse {
        id: "resp-1".into(),
        model: "fixture".into(),
        content: Some("Scheduling pipeline before crash".into()),
        tool_calls: vec![ToolCall {
            id: "call-1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "schedule_pipeline".into(),
                arguments: call_args_1.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };

    // Simulate crash: intent row was committed to tool_outcomes, but process crashed before execution::submit
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-sub','t-sub',1,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act1, serde_json::to_string(&resp_turn_1).unwrap(), now],
    ).unwrap();

    let planned_spec_1 = JobSpec { resources: Default::default(),
        job_id: "job-pre-crash".into(),
        project_id: "p-sub".into(),
        task_id: "t-sub".into(),
        generation: 2,
        timeout_ms: 45_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("env-test".into()),
        probe: Probe::Scientific {
            operation: "pipeline".into(),
            dataset_id: "ds-1".into(),
            spec: Some(protocol_1.clone()),
            candidate_id: None,
            test_access_id: None,
        },
    };
    let planned_spec_1_json = serde_json::to_string(&planned_spec_1).unwrap();
    let planned_spec_1_sha = format!("{:x}", Sha256::digest(planned_spec_1_json.as_bytes()));

    let intent_1 = json!({
        "kind": "scheduler_job_submission",
        "operation": "schedule_pipeline",
        "project": "p-sub",
        "task": "t-sub",
        "turn": 1,
        "call_id": "call-1",
        "action": act1,
        "job_id": "job-pre-crash",
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "dataset_id": "ds-1",
        "dataset_sha256": "fake-ds-sha",
        "environment_id": "env-test",
        "environment_identity": "ident-test",
        "protocol_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&protocol_1).unwrap().as_bytes())),
        "canonical_arguments": mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_1).unwrap()),
        "canonical_arguments_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_1).unwrap())).unwrap().as_bytes())),
        "planned_spec": planned_spec_1,
        "planned_spec_sha256": planned_spec_1_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-sub','t-sub',1,'call-1',?1,'main','schedule_pipeline',?2,?3)",
        rusqlite::params![act1, call_args_1, intent_1.to_string()],
    ).unwrap();

    // Reconcile turn 1!
    let rec1 = orch.reconcile().unwrap();
    assert_eq!(rec1["reconciled_turns"][0]["prior_status"], "executing_tools");
    assert_eq!(rec1["reconciled_turns"][0]["tools_executed"], 1);

    // Verify outcome recorded was uncertain/rejected:
    let outcome_1: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-sub' AND task='t-sub' AND turn=1 AND call_id='call-1'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_1: serde_json::Value = serde_json::from_str(&outcome_1).unwrap();
    assert_eq!(outcome_val_1["status"], "rejected");
    assert_eq!(outcome_val_1["reason"], "unconfirmed_submission_crash");
    assert_eq!(outcome_val_1["effect_outcome"], "uncertain");

    // Verify NO job was created in jobs table:
    let job_count: i64 = db.query_row("SELECT count(*) FROM jobs WHERE id='job-pre-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(job_count, 0, "Pre-submission crash must not leave phantom job");

    // Turn 1 receipt must be completed:
    let turn1_status: String = db.query_row("SELECT status FROM provider_turn_receipts WHERE project='p-sub' AND turn=1", [], |r| r.get(0)).unwrap();
    assert_eq!(turn1_status, "completed");

    // -------------------------------------------------------------
    // Regression 2: Crash after admission, before outcome persistence
    // Submission succeeded, job is queued in `jobs`, outcome is NULL.
    // Restart must reconcile against authoritative job without duplicate submission.
    // -------------------------------------------------------------
    let act2 = "act-turn-2";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-sub' AND id=?2", rusqlite::params![act2, act1]).unwrap();

    let protocol_2 = json!({
        "format": "mlrescurix.pipeline.v1",
        "experiment_id": "cand-post-crash",
        "target_column": "target",
        "folds": 2
    });
    let call_args_2 = json!({
        "job_id": "job-post-crash",
        "dataset_id": "ds-1",
        "environment_id": "env-test",
        "protocol": protocol_2
    }).to_string();

    let resp_turn_2 = ChatCompletionResponse {
        id: "resp-2".into(),
        model: "fixture".into(),
        content: Some("Scheduling pipeline with crash after admission".into()),
        tool_calls: vec![ToolCall {
            id: "call-2".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "schedule_pipeline".into(),
                arguments: call_args_2.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 110, completion_tokens: 55, total_tokens: 165 },
        latency_ms: 25,
    };

    // Submission succeeded:
    let submitted_view = execution::submit(&root, JobSpec { resources: Default::default(),
        job_id: "job-post-crash".into(),
        project_id: "p-sub".into(),
        task_id: "t-sub".into(),
        generation: 2,
        timeout_ms: 45_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("env-test".into()),
        probe: Probe::Scientific {
            operation: "pipeline".into(),
            dataset_id: "ds-1".into(),
            spec: Some(protocol_2.clone()),
            candidate_id: None,
            test_access_id: None,
        },
    }).unwrap();
    assert_eq!(submitted_view.status, "queued");

    // Process crashed before updating tool_outcomes:
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-sub','t-sub',2,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act2, serde_json::to_string(&resp_turn_2).unwrap(), now],
    ).unwrap();

    let planned_spec_2_json = serde_json::to_string(&submitted_view.spec).unwrap();
    let planned_spec_2_sha = format!("{:x}", Sha256::digest(planned_spec_2_json.as_bytes()));
    let canonical_args_2 = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_2).unwrap());
    let canonical_args_2_sha = format!("{:x}", Sha256::digest(serde_json::to_string(&canonical_args_2).unwrap().as_bytes()));
    let intent_2 = json!({
        "kind": "scheduler_job_submission",
        "operation": "schedule_pipeline",
        "project": "p-sub",
        "task": "t-sub",
        "turn": 2,
        "call_id": "call-2",
        "action": act2,
        "job_id": "job-post-crash",
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "dataset_id": "ds-1",
        "dataset_sha256": "fake-ds-sha",
        "environment_id": "env-test",
        "environment_identity": "ident-test",
        "protocol_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&protocol_2).unwrap().as_bytes())),
        "canonical_arguments": canonical_args_2,
        "canonical_arguments_sha256": canonical_args_2_sha,
        "planned_spec": submitted_view.spec,
        "planned_spec_sha256": planned_spec_2_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-sub','t-sub',2,'call-2',?1,'main','schedule_pipeline',?2,?3)",
        rusqlite::params![act2, call_args_2, intent_2.to_string()],
    ).unwrap();

    // Reconcile turn 2!
    let rec2 = orch.reconcile().unwrap();
    assert_eq!(rec2["reconciled_turns"][0]["prior_status"], "executing_tools");
    assert_eq!(rec2["reconciled_turns"][0]["tools_executed"], 1);

    // Verify outcome was populated from the authoritative job:
    let outcome_2: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-sub' AND task='t-sub' AND turn=2 AND call_id='call-2'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_2: serde_json::Value = serde_json::from_str(&outcome_2).unwrap();
    assert_eq!(outcome_val_2["spec"]["job_id"], "job-post-crash");
    assert_eq!(outcome_val_2["status"], "queued");

    // Exactly one job in jobs table:
    let job_count_2: i64 = db.query_row("SELECT count(*) FROM jobs WHERE id='job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(job_count_2, 1, "Must reconcile against existing job without duplicate submission");

    // -------------------------------------------------------------
    // Regression 3: Crash after outcome persistence, before context checkpoint
    // Outcome is in tool_outcomes, but context checkpoint is missing tool message.
    // Reconcile must deliver the outcome to context without creating a job.
    // -------------------------------------------------------------
    let act3 = "act-turn-3";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-sub' AND id=?2", rusqlite::params![act3, act1]).unwrap();

    let resp_turn_3 = ChatCompletionResponse {
        id: "resp-3".into(),
        model: "fixture".into(),
        content: Some("Scheduling pipeline with crash before checkpoint".into()),
        tool_calls: vec![ToolCall {
            id: "call-3".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "schedule_pipeline".into(),
                arguments: call_args_2.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 110, completion_tokens: 55, total_tokens: 165 },
        latency_ms: 25,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-sub','t-sub',3,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act3, serde_json::to_string(&resp_turn_3).unwrap(), now],
    ).unwrap();

    let outcome_3_str = serde_json::to_string(&submitted_view).unwrap();
    let outcome_3_sha = format!("{:x}", Sha256::digest(outcome_3_str.as_bytes()));
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json,outcome,outcome_sha256) VALUES('p-sub','t-sub',3,'call-3',?1,'main','schedule_pipeline',?2,?3,?4,?5)",
        rusqlite::params![act3, call_args_2, intent_2.to_string(), outcome_3_str, outcome_3_sha],
    ).unwrap();

    // Context does NOT have tool message for call-3 yet (lost in crash).
    let rec3 = orch.reconcile().unwrap();
    assert_eq!(rec3["reconciled_turns"][0]["prior_status"], "executing_tools");
    assert_eq!(rec3["reconciled_turns"][0]["tools_executed"], 1);

    // Verify context now has tool message:
    assert!(orch.context.messages.iter().any(|m| m.role == "tool" && m.tool_call_id.as_deref() == Some("call-3")));

    // -------------------------------------------------------------
    // Regression 4: Duplicate recovery idempotency
    // Calling reconcile again must be a clean no-op (0 tools executed).
    // -------------------------------------------------------------
    let rec_dup = orch.reconcile().unwrap();
    assert!(rec_dup["reconciled_turns"].as_array().unwrap().is_empty());

    // -------------------------------------------------------------
    // Regression 5: Conflicting payloads / Identity reuse rejection
    // Same call_id with different arguments must error.
    // -------------------------------------------------------------
    let conflicting_args = json!({
        "job_id": "job-conflicting",
        "dataset_id": "ds-1",
        "environment_id": "env-test",
        "protocol": protocol_1
    }).to_string();

    let conflict_err = mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project: "p-sub", task: "t-sub", branch: "main",
            turn: 2, call_id: "call-2", name: "schedule_pipeline",
            arguments: &conflicting_args,
        }, false, || panic!("Conflicting payload must not dispatch")
    ).unwrap_err();
    assert!(conflict_err.contains("tool intent is not bound to the authoritative cached response"), "{conflict_err}");

    // If cached response contains the new arguments, but tool_outcomes already has an intent for that call_id:
    let resp_turn_2_conflict = ChatCompletionResponse {
        id: "resp-2-conflict".into(),
        model: "fixture".into(),
        content: Some("Conflicting args test".into()),
        tool_calls: vec![ToolCall {
            id: "call-2".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "schedule_pipeline".into(),
                arguments: conflicting_args.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 10,
    };
    db.execute(
        "UPDATE provider_turn_receipts SET response_json=?1 WHERE project='p-sub' AND turn=2",
        rusqlite::params![serde_json::to_string(&resp_turn_2_conflict).unwrap()],
    ).unwrap();

    let identity_err = mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project: "p-sub", task: "t-sub", branch: "main",
            turn: 2, call_id: "call-2", name: "schedule_pipeline",
            arguments: &conflicting_args,
        }, false, || panic!("Conflicting payload must not dispatch")
    ).unwrap_err();
    assert!(identity_err.contains("durable tool identity conflict"), "{identity_err}");

    // Restore turn 2 cached response
    db.execute(
        "UPDATE provider_turn_receipts SET response_json=?1 WHERE project='p-sub' AND turn=2",
        rusqlite::params![serde_json::to_string(&resp_turn_2).unwrap()],
    ).unwrap();

    // Also test contradictory job spec in jobs table:
    let act5b = "act-turn-5b";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-sub' AND id=?2", rusqlite::params![act5b, act1]).unwrap();

    let resp_turn_5b = ChatCompletionResponse {
        id: "resp-5b".into(),
        model: "fixture".into(),
        content: Some("Contradictory spec test".into()),
        tool_calls: vec![ToolCall {
            id: "call-5b".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "schedule_pipeline".into(),
                arguments: call_args_2.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 10,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-sub','t-sub',5,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act5b, serde_json::to_string(&resp_turn_5b).unwrap(), now],
    ).unwrap();

    let mut intent_5b = intent_2.clone();
    intent_5b["turn"] = json!(5);
    intent_5b["call_id"] = json!("call-5b");
    intent_5b["action"] = json!(act5b);
    intent_5b["planned_spec_sha256"] = json!("contradictory-spec-sha256"); // deliberate mismatch
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-sub','t-sub',5,'call-5b',?1,'main','schedule_pipeline',?2,?3)",
        rusqlite::params![act5b, call_args_2, intent_5b.to_string()],
    ).unwrap();

    let rec5b = orch.reconcile().unwrap();
    assert_eq!(rec5b["reconciled_turns"][0]["turn"], 5);
    let outcome_5b: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-sub' AND task='t-sub' AND turn=5 AND call_id='call-5b'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_5b: serde_json::Value = serde_json::from_str(&outcome_5b).unwrap();
    assert_eq!(outcome_val_5b["status"], "rejected");
    assert_eq!(outcome_val_5b["reason"], "contradictory_job_spec");
    assert_eq!(outcome_val_5b["effect_outcome"], "rejected");

    // -------------------------------------------------------------
    // Regression 6: Stale generation fencing
    // Intent has generation 2, but task advances to generation 3.
    // Recovery must reject with stale_generation_fenced.
    // -------------------------------------------------------------
    let act6 = "act-turn-6";
    let intent_6 = json!({
        "kind": "scheduler_job_submission",
        "operation": "schedule_pipeline",
        "project": "p-sub",
        "task": "t-sub",
        "turn": 6,
        "call_id": "call-6",
        "action": act6,
        "job_id": "job-post-crash",
        "generation": 2, // stale generation
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "dataset_id": "ds-1",
        "dataset_sha256": "fake-ds-sha",
        "environment_id": "env-test",
        "environment_identity": "ident-test",
        "protocol_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&protocol_2).unwrap().as_bytes())),
        "canonical_arguments": canonical_args_2,
        "canonical_arguments_sha256": canonical_args_2_sha,
        "planned_spec": submitted_view.spec,
        "planned_spec_sha256": planned_spec_2_sha,
    });
    let resp_turn_6 = ChatCompletionResponse {
        id: "resp-6".into(),
        model: "fixture".into(),
        content: Some("Stale generation test".into()),
        tool_calls: vec![ToolCall {
            id: "call-6".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "schedule_pipeline".into(),
                arguments: call_args_2.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 10,
    };
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-sub' AND id=?2", rusqlite::params![act6, act1]).unwrap();
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-sub','t-sub',6,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act6, serde_json::to_string(&resp_turn_6).unwrap(), now],
    ).unwrap();
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-sub','t-sub',6,'call-6',?1,'main','schedule_pipeline',?2,?3)",
        rusqlite::params![act6, call_args_2, intent_6.to_string()],
    ).unwrap();

    // Bump task generation in tasks table:
    db.execute("UPDATE tasks SET data=json_set(data, '$.generation', 3) WHERE project='p-sub' AND id='t-sub'", []).unwrap();

    let rec6 = orch.reconcile().unwrap();
    assert_eq!(rec6["reconciled_turns"][0]["turn"], 6);
    let outcome_6: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-sub' AND task='t-sub' AND turn=6 AND call_id='call-6'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_6: serde_json::Value = serde_json::from_str(&outcome_6).unwrap();
    assert_eq!(outcome_val_6["status"], "rejected");
    assert_eq!(outcome_val_6["reason"], "stale_generation_fenced");
    assert_eq!(outcome_val_6["effect_outcome"], "uncertain_fenced");

    // -------------------------------------------------------------
    // Regression 7: Verify one authoritative job/reservation and charge settlement
    // Restore generation to 2 so supervisor can process queued job-post-crash.
    // -------------------------------------------------------------
    db.execute("UPDATE tasks SET data=json_set(data, '$.generation', 2) WHERE project='p-sub' AND id='t-sub'", []).unwrap();

    // Verify exactly one job exists for job-post-crash
    let pre_jobs: i64 = db.query_row("SELECT count(*) FROM jobs WHERE id='job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(pre_jobs, 1);

    // Verify no actions or charges yet for job.job-post-crash
    let pre_actions: i64 = db.query_row("SELECT count(*) FROM actions WHERE id='job.job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(pre_actions, 0);

    // Supervisor ticks: admits job, creates exactly 1 action, reserves resources
    let binary = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    supervisor.tick().unwrap();

    let post_admit_actions: i64 = db.query_row("SELECT count(*) FROM actions WHERE id='job.job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(post_admit_actions, 1, "Exactly one action created for admitted job");

    let post_admit_reservations: i64 = db.query_row("SELECT count(*) FROM reservations WHERE action='job.job-post-crash' AND held=1", [], |r| r.get(0)).unwrap();
    assert_eq!(post_admit_reservations, 1, "Exactly one reservation held for admitted job");

    // Wait for worker to finish and supervisor to reap
    for _ in 0..250 {
        supervisor.tick().unwrap();
        let j = execution::inspect(&root, "p-sub", "job-post-crash").unwrap();
        if !matches!(j.status.as_str(), "queued" | "running" | "recovering" | "terminating") {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    drop(supervisor);

    // Verify charge settlement: exactly 1 charge row, held reservation released
    let post_finish_charges: i64 = db.query_row("SELECT count(*) FROM charges WHERE action='job.job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(post_finish_charges, 1, "Exactly one charge settled for completed job");

    let held_reservations: i64 = db.query_row("SELECT count(*) FROM reservations WHERE action='job.job-post-crash' AND held=1", [], |r| r.get(0)).unwrap();
    assert_eq!(held_reservations, 0, "Held reservation must be released upon settlement");

    // Reconcile again: must remain authoritative and not create new jobs or duplicate charges
    let final_jobs: i64 = db.query_row("SELECT count(*) FROM jobs WHERE id='job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(final_jobs, 1);
    let final_charges: i64 = db.query_row("SELECT count(*) FROM charges WHERE action='job.job-post-crash'", [], |r| r.get(0)).unwrap();
    assert_eq!(final_charges, 1);
}

#[test]
fn p1_delegation_and_registration_crash_reconciliation_and_fault_regressions() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(200, 8)).unwrap();
    create_m1(&runtime, "p-del", 200);
    branch_task(&runtime, "p-del", "main", "t-del");
    let now = chrono_free_now();
    assert_eq!(
        runtime.command(req("p-del", "claim-t-del", Operation::Claim {
            task_id: "t-del".into(),
            owner: "agent-t-del".into(),
            lease_id: "leaset-del".into(),
            now_ms: now,
            expires_ms: now + 3_600_000,
            expected_generation: 1,
        })).unwrap().status,
        "applied"
    );
    execution::migrate(&root).unwrap();

    struct NoReplayGateway;
    impl mlrescurix_harness::agents::ProviderGateway for NoReplayGateway {
        fn chat_completion(&self, _: &mlrescurix_harness::agents::ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            panic!("Reconciliation must never replay an uncertain provider call");
        }
    }

    let mut orch = AgentOrchestrator::new(&root, OrchestratorConfig {
        project_id: "p-del".into(), task_id: "t-del".into(), role_id: "lead".into(),
        branch_id: "main".into(), model: "fixture".into(), max_turns: 20,
    }, Arc::new(NoReplayGateway)).unwrap();

    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    // -------------------------------------------------------------
    // Part 1: Specialist Delegation
    // -------------------------------------------------------------

    // Regression 1: Crash before child task admission (intent saved, child task not in agent_tasks/tasks)
    // Missing authoritative evidence must produce explicit uncertain/rejected outcome with "unconfirmed_delegation_crash".
    let act1 = "act-del-1";
    assert_eq!(runtime.command(req("p-del", "prep-1", Operation::Prepare {
        task_id: "t-del".into(), generation: 2, lease_id: "leaset-del".into(),
        action_id: act1.into(), input: "in1".into(),
        budget: BTreeMap::from([("calls".into(), 1), ("tokens".into(), 5000), ("wall_ms".into(), 60000)]),
        capacity: BTreeMap::from([("workers".into(), 1)]),
        deadline_ms: now + 60_000, now_ms: now,
    })).unwrap().status, "applied");
    assert_eq!(runtime.command(req("p-del", "disp-1", Operation::Dispatch {
        task_id: "t-del".into(), generation: 2, lease_id: "leaset-del".into(),
        action_id: act1.into(), attempt_id: act1.into(), external_request_id: None, now_ms: now,
    })).unwrap().status, "applied");

    let call_args_1 = json!({
        "role_id": "auditor",
        "intent": "Audit dataset schema and distributions",
        "hypothesis": "Feature missingness affects baseline",
        "sub_task_id": "t-child-pre",
    }).to_string();

    let resp_turn_1 = ChatCompletionResponse {
        id: "resp-del-1".into(),
        model: "fixture".into(),
        content: Some("Delegating to auditor before crash".into()),
        tool_calls: vec![ToolCall {
            id: "call-del-1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "delegate_task".into(),
                arguments: call_args_1.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };

    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',1,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act1, serde_json::to_string(&resp_turn_1).unwrap(), now],
    ).unwrap();

    let canonical_args_1 = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_1).unwrap());
    let canonical_args_1_sha = format!("{:x}", Sha256::digest(serde_json::to_string(&canonical_args_1).unwrap().as_bytes()));

    let intent_1 = json!({
        "kind": "specialist_delegation",
        "operation": "delegate_task",
        "project": "p-del",
        "task": "t-del",
        "turn": 1,
        "call_id": "call-del-1",
        "action": act1,
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "target_role": "auditor",
        "target_task_id": "t-child-pre",
        "canonical_arguments": canonical_args_1,
        "canonical_arguments_sha256": canonical_args_1_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',1,'call-del-1',?1,'main','delegate_task',?2,?3)",
        rusqlite::params![act1, call_args_1, intent_1.to_string()],
    ).unwrap();

    // Reconcile turn 1: process crashed before child task admission
    let rec1 = orch.reconcile().unwrap();
    assert_eq!(rec1["reconciled_turns"][0]["prior_status"], "executing_tools");
    assert_eq!(rec1["reconciled_turns"][0]["tools_executed"], 1);

    let outcome_1: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=1 AND call_id='call-del-1'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_1: serde_json::Value = serde_json::from_str(&outcome_1).unwrap();
    assert_eq!(outcome_val_1["status"], "rejected");
    assert_eq!(outcome_val_1["reason"], "unconfirmed_delegation_crash");
    assert_eq!(outcome_val_1["effect_outcome"], "uncertain");

    // Verify NO child task in agent_tasks or tasks
    let agent_task_count_1: i64 = db.query_row("SELECT count(*) FROM agent_tasks WHERE project='p-del' AND task_id='t-child-pre'", [], |r| r.get(0)).unwrap();
    assert_eq!(agent_task_count_1, 0, "Pre-delegation crash must not leave phantom child task");
    let task_count_1: i64 = db.query_row("SELECT count(*) FROM tasks WHERE project='p-del' AND id='t-child-pre'", [], |r| r.get(0)).unwrap();
    assert_eq!(task_count_1, 0);

    // Turn 1 receipt marked completed:
    let turn1_status: String = db.query_row("SELECT status FROM provider_turn_receipts WHERE project='p-del' AND turn=1", [], |r| r.get(0)).unwrap();
    assert_eq!(turn1_status, "completed");

    // -------------------------------------------------------------
    // Regression 2: Crash after child task admission, before outcome persistence
    // Child task exists in agent_tasks, outcome is NULL in tool_outcomes.
    // Reconcile must recover authoritative child task view without duplicate child tasks or double billing.
    // -------------------------------------------------------------
    let act2 = "act-del-2";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act2, act1]).unwrap();

    let call_args_2 = json!({
        "role_id": "auditor",
        "intent": "Audit dataset missingness",
        "hypothesis": "Nulls in colA affect score",
        "sub_task_id": "t-child-post",
    }).to_string();

    let resp_turn_2 = ChatCompletionResponse {
        id: "resp-del-2".into(),
        model: "fixture".into(),
        content: Some("Delegating to auditor with crash after admission".into()),
        tool_calls: vec![ToolCall {
            id: "call-del-2".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "delegate_task".into(),
                arguments: call_args_2.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 110, completion_tokens: 55, total_tokens: 165 },
        latency_ms: 25,
    };

    // Child task was admitted prior to crash:
    db.execute(
        "INSERT INTO agent_tasks(project, task_id, role_id, parent_task_id, status, hypothesis, intent, assigned_branch, created_ms, updated_ms)
         VALUES('p-del', 't-child-post', 'auditor', 't-del', 'paused', 'Nulls in colA affect score', 'Audit dataset missingness', 'branch-child-post', ?1, ?1)",
        rusqlite::params![now],
    ).unwrap();

    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',2,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act2, serde_json::to_string(&resp_turn_2).unwrap(), now],
    ).unwrap();

    let canonical_args_2 = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_2).unwrap());
    let canonical_args_2_sha = format!("{:x}", Sha256::digest(serde_json::to_string(&canonical_args_2).unwrap().as_bytes()));

    let intent_2 = json!({
        "kind": "specialist_delegation",
        "operation": "delegate_task",
        "project": "p-del",
        "task": "t-del",
        "turn": 2,
        "call_id": "call-del-2",
        "action": act2,
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "target_role": "auditor",
        "target_task_id": "t-child-post",
        "canonical_arguments": canonical_args_2,
        "canonical_arguments_sha256": canonical_args_2_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',2,'call-del-2',?1,'main','delegate_task',?2,?3)",
        rusqlite::params![act2, call_args_2, intent_2.to_string()],
    ).unwrap();

    // Reconcile turn 2:
    let rec2 = orch.reconcile().unwrap();
    assert_eq!(rec2["reconciled_turns"][0]["prior_status"], "executing_tools");
    assert_eq!(rec2["reconciled_turns"][0]["tools_executed"], 1);

    let outcome_2: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=2 AND call_id='call-del-2'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_2: serde_json::Value = serde_json::from_str(&outcome_2).unwrap();
    assert_eq!(outcome_val_2["status"], "scheduled");
    assert_eq!(outcome_val_2["sub_task_id"], "t-child-post");
    assert_eq!(outcome_val_2["role"], "auditor");
    assert_eq!(outcome_val_2["reconciled"], true);

    // Exactly one child task in agent_tasks:
    let count_post: i64 = db.query_row("SELECT count(*) FROM agent_tasks WHERE project='p-del' AND task_id='t-child-post'", [], |r| r.get(0)).unwrap();
    assert_eq!(count_post, 1, "Must reconcile against existing child task without duplicate");

    // -------------------------------------------------------------
    // Regression 3: Crash after outcome persistence, before context checkpoint
    // -------------------------------------------------------------
    let act3 = "act-del-3";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act3, act1]).unwrap();

    let resp_turn_3 = ChatCompletionResponse {
        id: "resp-del-3".into(),
        model: "fixture".into(),
        content: Some("Delegating with crash before checkpoint".into()),
        tool_calls: vec![ToolCall {
            id: "call-del-3".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "delegate_task".into(),
                arguments: call_args_2.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 110, completion_tokens: 55, total_tokens: 165 },
        latency_ms: 25,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',3,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act3, serde_json::to_string(&resp_turn_3).unwrap(), now],
    ).unwrap();

    let outcome_3_str = outcome_2.clone();
    let outcome_3_sha = format!("{:x}", Sha256::digest(outcome_3_str.as_bytes()));
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json,outcome,outcome_sha256) VALUES('p-del','t-del',3,'call-del-3',?1,'main','delegate_task',?2,?3,?4,?5)",
        rusqlite::params![act3, call_args_2, intent_2.to_string(), outcome_3_str, outcome_3_sha],
    ).unwrap();

    let rec3 = orch.reconcile().unwrap();
    assert_eq!(rec3["reconciled_turns"][0]["turn"], 3);
    assert_eq!(rec3["reconciled_turns"][0]["tools_executed"], 1);

    // -------------------------------------------------------------
    // Regression 4: Conflicting arguments rejection
    // -------------------------------------------------------------
    let conflicting_call_args = json!({
        "role_id": "auditor",
        "intent": "CONTRADICTORY INTENT",
        "hypothesis": "Contradictory hypothesis",
        "sub_task_id": "t-child-post",
    }).to_string();

    let conflict_err = mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project: "p-del", task: "t-del", branch: "main",
            turn: 2, call_id: "call-del-2", name: "delegate_task",
            arguments: &conflicting_call_args,
        },
        false,
        || panic!("must not execute on conflicting identity"),
    ).unwrap_err();
    assert!(conflict_err.contains("tool intent is not bound to the authoritative cached response"), "{conflict_err}");

    // If cached response contains the new arguments, but tool_outcomes already has an intent for that call_id:
    let resp_turn_2_conflict = ChatCompletionResponse {
        id: "resp-del-2-conflict".into(),
        model: "fixture".into(),
        content: Some("Conflicting args test".into()),
        tool_calls: vec![ToolCall {
            id: "call-del-2".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "delegate_task".into(),
                arguments: conflicting_call_args.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 10,
    };
    db.execute(
        "UPDATE provider_turn_receipts SET response_json=?1 WHERE project='p-del' AND turn=2",
        rusqlite::params![serde_json::to_string(&resp_turn_2_conflict).unwrap()],
    ).unwrap();

    let identity_err = mlrescurix_harness::agents::recovery::execute(&root,
        mlrescurix_harness::agents::recovery::ToolIdentity {
            project: "p-del", task: "t-del", branch: "main",
            turn: 2, call_id: "call-del-2", name: "delegate_task",
            arguments: &conflicting_call_args,
        },
        false,
        || panic!("must not execute on conflicting identity"),
    ).unwrap_err();
    assert!(identity_err.contains("durable tool identity conflict"), "{identity_err}");

    // Restore turn 2 cached response
    db.execute(
        "UPDATE provider_turn_receipts SET response_json=?1 WHERE project='p-del' AND turn=2",
        rusqlite::params![serde_json::to_string(&resp_turn_2).unwrap()],
    ).unwrap();

    // -------------------------------------------------------------
    // Regression 5: Target task identity reuse rejection
    // -------------------------------------------------------------
    let act5 = "act-del-5";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act5, act1]).unwrap();

    // Trying to build intent with an existing sub_task_id ("t-child-post" already in agent_tasks)
    let tx5 = db.unchecked_transaction().unwrap();
    let reuse_err = mlrescurix_harness::agents::recovery::build_delegation_intent(
        &tx5,
        &mlrescurix_harness::agents::recovery::ToolIdentity {
            project: "p-del", task: "t-del", branch: "main",
            turn: 5, call_id: "call-del-5", name: "delegate_task",
            arguments: &call_args_2,
        },
        &serde_json::from_str::<serde_json::Value>(&call_args_2).unwrap(),
        act5,
    ).unwrap_err();
    assert!(reuse_err.contains("target task identity reuse"));
    tx5.rollback().unwrap();

    // Also test crash recovery: if child task in agent_tasks has a different role/parent
    let act5b = "act-del-5b";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act5b, act1]).unwrap();

    let call_args_5b = json!({
        "role_id": "model_specialist", // different role than what is in agent_tasks
        "intent": "Audit dataset missingness",
        "sub_task_id": "t-child-post",
    }).to_string();

    let resp_turn_5b = ChatCompletionResponse {
        id: "resp-del-5b".into(),
        model: "fixture".into(),
        content: Some("Delegating with role mismatch".into()),
        tool_calls: vec![ToolCall {
            id: "call-del-5b".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "delegate_task".into(),
                arguments: call_args_5b.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 110, completion_tokens: 55, total_tokens: 165 },
        latency_ms: 25,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',5,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act5b, serde_json::to_string(&resp_turn_5b).unwrap(), now],
    ).unwrap();

    let mut intent_5b = intent_2.clone();
    intent_5b["turn"] = json!(5);
    intent_5b["call_id"] = json!("call-del-5b");
    intent_5b["action"] = json!(act5b);
    intent_5b["target_role"] = json!("model_specialist");
    intent_5b["canonical_arguments"] = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_5b).unwrap());
    intent_5b["canonical_arguments_sha256"] = json!(format!("{:x}", Sha256::digest(serde_json::to_string(&intent_5b["canonical_arguments"]).unwrap().as_bytes())));

    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',5,'call-del-5b',?1,'main','delegate_task',?2,?3)",
        rusqlite::params![act5b, call_args_5b, intent_5b.to_string()],
    ).unwrap();

    let rec5b = orch.reconcile().unwrap();
    assert_eq!(rec5b["reconciled_turns"][0]["turn"], 5);
    let outcome_5b: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=5 AND call_id='call-del-5b'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_5b: serde_json::Value = serde_json::from_str(&outcome_5b).unwrap();
    assert_eq!(outcome_val_5b["status"], "rejected");
    assert_eq!(outcome_val_5b["reason"], "target_task_identity_reuse");
    assert_eq!(outcome_val_5b["effect_outcome"], "rejected");

    // -------------------------------------------------------------
    // Regression 6: Stale generation fencing for delegation
    // -------------------------------------------------------------
    let act6 = "act-del-6";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act6, act1]).unwrap();

    let call_args_6 = json!({
        "role_id": "auditor",
        "intent": "Audit dataset missingness",
        "sub_task_id": "t-child-post",
    }).to_string();

    let resp_turn_6 = ChatCompletionResponse {
        id: "resp-del-6".into(),
        model: "fixture".into(),
        content: Some("Stale gen test".into()),
        tool_calls: vec![ToolCall {
            id: "call-del-6".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "delegate_task".into(),
                arguments: call_args_6.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 110, completion_tokens: 55, total_tokens: 165 },
        latency_ms: 25,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',6,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act6, serde_json::to_string(&resp_turn_6).unwrap(), now],
    ).unwrap();

    let mut intent_6 = intent_2.clone();
    intent_6["turn"] = json!(6);
    intent_6["call_id"] = json!("call-del-6");
    intent_6["action"] = json!(act6);
    intent_6["generation"] = json!(2); // Intent recorded at generation 2
    intent_6["canonical_arguments"] = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_6).unwrap());
    intent_6["canonical_arguments_sha256"] = json!(format!("{:x}", Sha256::digest(serde_json::to_string(&intent_6["canonical_arguments"]).unwrap().as_bytes())));

    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',6,'call-del-6',?1,'main','delegate_task',?2,?3)",
        rusqlite::params![act6, call_args_6, intent_6.to_string()],
    ).unwrap();

    // Advance task generation to 3
    db.execute("UPDATE tasks SET data=json_set(data, '$.generation', 3) WHERE project='p-del' AND id='t-del'", []).unwrap();

    let rec6 = orch.reconcile().unwrap();
    assert_eq!(rec6["reconciled_turns"][0]["turn"], 6);
    let outcome_6: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=6 AND call_id='call-del-6'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_6: serde_json::Value = serde_json::from_str(&outcome_6).unwrap();
    assert_eq!(outcome_val_6["status"], "rejected");
    assert_eq!(outcome_val_6["reason"], "stale_generation_fenced");
    assert_eq!(outcome_val_6["effect_outcome"], "uncertain_fenced");

    // Restore generation to 2
    db.execute("UPDATE tasks SET data=json_set(data, '$.generation', 2) WHERE project='p-del' AND id='t-del'", []).unwrap();

    // -------------------------------------------------------------
    // Part 2: Dataset Registration
    // -------------------------------------------------------------

    // Regression 7: Crash before dataset registration
    let act7 = "act-reg-7";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act7, act1]).unwrap();

    let sample_csv_path = dir.path().join("sample.csv");
    fs::write(&sample_csv_path, "col1,col2,target\n1,2,0\n3,4,1\n").unwrap();
    let sample_csv_bytes = fs::read(&sample_csv_path).unwrap();
    let sample_csv_sha = format!("{:x}", Sha256::digest(&sample_csv_bytes));

    let call_args_7 = json!({
        "dataset_id": "ds-crash-pre",
        "path": sample_csv_path.to_str().unwrap(),
    }).to_string();

    let resp_turn_7 = ChatCompletionResponse {
        id: "resp-reg-7".into(),
        model: "fixture".into(),
        content: Some("Registering dataset before crash".into()),
        tool_calls: vec![ToolCall {
            id: "call-reg-7".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "register_dataset".into(),
                arguments: call_args_7.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',7,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act7, serde_json::to_string(&resp_turn_7).unwrap(), now],
    ).unwrap();

    let canonical_args_7 = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_7).unwrap());
    let canonical_args_7_sha = format!("{:x}", Sha256::digest(serde_json::to_string(&canonical_args_7).unwrap().as_bytes()));

    let intent_7 = json!({
        "kind": "dataset_registration",
        "operation": "register_dataset",
        "project": "p-del",
        "task": "t-del",
        "turn": 7,
        "call_id": "call-reg-7",
        "action": act7,
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "target_dataset_id": "ds-crash-pre",
        "source_path": sample_csv_path.to_str().unwrap(),
        "source_sha256": sample_csv_sha,
        "source_bytes": sample_csv_bytes.len(),
        "canonical_arguments": canonical_args_7,
        "canonical_arguments_sha256": canonical_args_7_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',7,'call-reg-7',?1,'main','register_dataset',?2,?3)",
        rusqlite::params![act7, call_args_7, intent_7.to_string()],
    ).unwrap();

    let rec7 = orch.reconcile().unwrap();
    assert_eq!(rec7["reconciled_turns"][0]["turn"], 7);
    let outcome_7: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=7 AND call_id='call-reg-7'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_7: serde_json::Value = serde_json::from_str(&outcome_7).unwrap();
    assert_eq!(outcome_val_7["status"], "rejected");
    assert_eq!(outcome_val_7["reason"], "unconfirmed_registration_crash");
    assert_eq!(outcome_val_7["effect_outcome"], "uncertain");

    let ds_count_7: i64 = db.query_row("SELECT count(*) FROM scientific_datasets WHERE project='p-del' AND id='ds-crash-pre'", [], |r| r.get(0)).unwrap();
    assert_eq!(ds_count_7, 0, "Pre-registration crash must not leave phantom dataset");

    // Regression 8: Crash after dataset registration, before tool outcome persistence
    let act8 = "act-reg-8";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act8, act1]).unwrap();

    let reg_result = execution::register_dataset(&root, "p-del", "ds-crash-post", &sample_csv_path).unwrap();
    assert_eq!(reg_result["reused"], false);

    let call_args_8 = json!({
        "dataset_id": "ds-crash-post",
        "path": sample_csv_path.to_str().unwrap(),
    }).to_string();

    let resp_turn_8 = ChatCompletionResponse {
        id: "resp-reg-8".into(),
        model: "fixture".into(),
        content: Some("Registering dataset with crash after registration".into()),
        tool_calls: vec![ToolCall {
            id: "call-reg-8".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "register_dataset".into(),
                arguments: call_args_8.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',8,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act8, serde_json::to_string(&resp_turn_8).unwrap(), now],
    ).unwrap();

    let canonical_args_8 = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_8).unwrap());
    let canonical_args_8_sha = format!("{:x}", Sha256::digest(serde_json::to_string(&canonical_args_8).unwrap().as_bytes()));

    let intent_8 = json!({
        "kind": "dataset_registration",
        "operation": "register_dataset",
        "project": "p-del",
        "task": "t-del",
        "turn": 8,
        "call_id": "call-reg-8",
        "action": act8,
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "target_dataset_id": "ds-crash-post",
        "source_path": sample_csv_path.to_str().unwrap(),
        "source_sha256": sample_csv_sha,
        "source_bytes": sample_csv_bytes.len(),
        "canonical_arguments": canonical_args_8,
        "canonical_arguments_sha256": canonical_args_8_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',8,'call-reg-8',?1,'main','register_dataset',?2,?3)",
        rusqlite::params![act8, call_args_8, intent_8.to_string()],
    ).unwrap();

    let rec8 = orch.reconcile().unwrap();
    assert_eq!(rec8["reconciled_turns"][0]["turn"], 8);
    let outcome_8: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=8 AND call_id='call-reg-8'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_8: serde_json::Value = serde_json::from_str(&outcome_8).unwrap();
    assert_eq!(outcome_val_8["dataset_id"], "ds-crash-post");
    assert_eq!(outcome_val_8["reused"], true);
    assert_eq!(outcome_val_8["sha256"], sample_csv_sha);

    let ds_count_8: i64 = db.query_row("SELECT count(*) FROM scientific_datasets WHERE project='p-del' AND id='ds-crash-post'", [], |r| r.get(0)).unwrap();
    assert_eq!(ds_count_8, 1, "Must reconcile against authoritative dataset without duplicate");

    // Regression 9: Contradictory dataset content rejection
    let act9 = "act-reg-9";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act9, act1]).unwrap();

    let mut intent_9 = intent_8.clone();
    intent_9["turn"] = json!(9);
    intent_9["call_id"] = json!("call-reg-9");
    intent_9["action"] = json!(act9);
    intent_9["source_sha256"] = json!("contradictory-dataset-sha256");

    let resp_turn_9 = ChatCompletionResponse {
        id: "resp-reg-9".into(),
        model: "fixture".into(),
        content: Some("Contradictory dataset test".into()),
        tool_calls: vec![ToolCall {
            id: "call-reg-9".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "register_dataset".into(),
                arguments: call_args_8.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',9,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act9, serde_json::to_string(&resp_turn_9).unwrap(), now],
    ).unwrap();
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',9,'call-reg-9',?1,'main','register_dataset',?2,?3)",
        rusqlite::params![act9, call_args_8, intent_9.to_string()],
    ).unwrap();

    let rec9 = orch.reconcile().unwrap();
    assert_eq!(rec9["reconciled_turns"][0]["turn"], 9);
    let outcome_9: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=9 AND call_id='call-reg-9'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_9: serde_json::Value = serde_json::from_str(&outcome_9).unwrap();
    assert_eq!(outcome_val_9["status"], "rejected");
    assert_eq!(outcome_val_9["reason"], "contradictory_dataset_content");
    assert_eq!(outcome_val_9["effect_outcome"], "rejected");

    // -------------------------------------------------------------
    // Part 3: Delivery Publication
    // -------------------------------------------------------------

    // Setup candidate critic audit and approval for delivery publication
    let cand_id = "cand-pub-1";
    db.execute(
        "INSERT INTO critic_audits(project, candidate_id, auditor_role, status, rationale, source_hash, protocol, constraints_verified, created_ms)
         VALUES('p-del', ?1, 'critic', 'approved', 'Verified under declared protocol', 'src-hash', 'standard_holdout', 1, ?2)",
        rusqlite::params![cand_id, now],
    ).unwrap();
    db.execute(
        "INSERT INTO approvals(project, candidate_id, approved, user_identity, rationale, created_ms)
         VALUES('p-del', ?1, 1, 'user', 'Approved for finalization', ?2)",
        rusqlite::params![cand_id, now],
    ).unwrap();

    // Regression 10: Crash before finalization job admission
    let act10 = "act-pub-10";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act10, act1]).unwrap();

    let call_args_10 = json!({
        "candidate_id": cand_id,
        "rationale": "Final freeze justification",
        "approval_token": "token-user",
    }).to_string();

    let resp_turn_10 = ChatCompletionResponse {
        id: "resp-pub-10".into(),
        model: "fixture".into(),
        content: Some("Requesting finalization before crash".into()),
        tool_calls: vec![ToolCall {
            id: "call-pub-10".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "request_finalization".into(),
                arguments: call_args_10.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',10,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act10, serde_json::to_string(&resp_turn_10).unwrap(), now],
    ).unwrap();

    let canonical_args_10 = mlrescurix_harness::agents::recovery::canonicalize_value(&serde_json::from_str::<serde_json::Value>(&call_args_10).unwrap());
    let canonical_args_10_sha = format!("{:x}", Sha256::digest(serde_json::to_string(&canonical_args_10).unwrap().as_bytes()));

    let intent_10 = json!({
        "kind": "delivery_publication",
        "operation": "request_finalization",
        "project": "p-del",
        "task": "t-del",
        "turn": 10,
        "call_id": "call-pub-10",
        "action": act10,
        "generation": 2,
        "intent_revision": 1,
        "policy_sha256": format!("{:x}", Sha256::digest(serde_json::to_string(&m1policy(200, 8)).unwrap().as_bytes())),
        "candidate_id": cand_id,
        "target_bundle_id": cand_id,
        "finalize_job_id": format!("finalize-{cand_id}"),
        "canonical_arguments": canonical_args_10,
        "canonical_arguments_sha256": canonical_args_10_sha,
    });
    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',10,'call-pub-10',?1,'main','request_finalization',?2,?3)",
        rusqlite::params![act10, call_args_10, intent_10.to_string()],
    ).unwrap();

    let rec10 = orch.reconcile().unwrap();
    assert_eq!(rec10["reconciled_turns"][0]["turn"], 10);
    let outcome_10: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=10 AND call_id='call-pub-10'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_10: serde_json::Value = serde_json::from_str(&outcome_10).unwrap();
    assert_eq!(outcome_val_10["status"], "rejected");
    assert_eq!(outcome_val_10["reason"], "unconfirmed_publication_crash");
    assert_eq!(outcome_val_10["effect_outcome"], "uncertain");

    // Regression 11: Crash after finalization job admission, before tool outcome persistence
    let act11 = "act-pub-11";
    db.execute("INSERT INTO actions(project,id,task,data) SELECT project,?1,task,json_set(data,'$.action_id',?1,'$.receipt.action_id',?1) FROM actions WHERE project='p-del' AND id=?2", rusqlite::params![act11, act1]).unwrap();

    // Finalize job was admitted to jobs table
    let finalize_spec = JobSpec { resources: Default::default(),
        job_id: format!("finalize-{cand_id}"),
        project_id: "p-del".into(),
        task_id: "t-del".into(),
        generation: 2,
        timeout_ms: 60_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("env-test".into()),
        probe: Probe::Scientific {
            operation: "finalize".into(),
            dataset_id: "ds-crash-post".into(),
            spec: None,
            candidate_id: Some(cand_id.into()),
            test_access_id: Some("final-test-1".into()),
        },
    };
    db.execute(
        "INSERT INTO jobs(id, project, task, generation, spec, status, submitted_ms) VALUES(?1, 'p-del', 't-del', 2, ?2, 'queued', ?3)",
        rusqlite::params![format!("finalize-{cand_id}"), serde_json::to_string(&finalize_spec).unwrap(), now],
    ).unwrap();

    let resp_turn_11 = ChatCompletionResponse {
        id: "resp-pub-11".into(),
        model: "fixture".into(),
        content: Some("Requesting finalization with crash after job admission".into()),
        tool_calls: vec![ToolCall {
            id: "call-pub-11".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "request_finalization".into(),
                arguments: call_args_10.clone(),
            },
        }],
        usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        latency_ms: 20,
    };
    db.execute(
        "INSERT INTO provider_turn_receipts(project,task,turn,action,status,response_json,tool_protocol,updated_ms) VALUES('p-del','t-del',11,?1,'executing_tools',?2,1,?3)",
        rusqlite::params![act11, serde_json::to_string(&resp_turn_11).unwrap(), now],
    ).unwrap();

    let mut intent_11 = intent_10.clone();
    intent_11["turn"] = json!(11);
    intent_11["call_id"] = json!("call-pub-11");
    intent_11["action"] = json!(act11);

    db.execute(
        "INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES('p-del','t-del',11,'call-pub-11',?1,'main','request_finalization',?2,?3)",
        rusqlite::params![act11, call_args_10, intent_11.to_string()],
    ).unwrap();

    let rec11 = orch.reconcile().unwrap();
    assert_eq!(rec11["reconciled_turns"][0]["turn"], 11);
    let outcome_11: String = db.query_row(
        "SELECT outcome FROM tool_outcomes WHERE project='p-del' AND task='t-del' AND turn=11 AND call_id='call-pub-11'",
        [], |r| r.get(0),
    ).unwrap();
    let outcome_val_11: serde_json::Value = serde_json::from_str(&outcome_11).unwrap();
    assert_eq!(outcome_val_11["status"], "finalization_queued");
    assert_eq!(outcome_val_11["candidate_id"], cand_id);
    assert_eq!(outcome_val_11["finalize_job_id"], format!("finalize-{cand_id}"));
    assert_eq!(outcome_val_11["reconciled"], true);

    let job_count_11: i64 = db.query_row("SELECT count(*) FROM jobs WHERE id=?1", [format!("finalize-{cand_id}")], |r| r.get(0)).unwrap();
    assert_eq!(job_count_11, 1, "Must reconcile against authoritative finalize job without duplicate");
}

#[test]
fn test_dispatch_gateway_outbound_policy_approval_and_accounting() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 2)).unwrap();
    create_m1(&runtime, "p-gw", 100);
    branch_task(&runtime, "p-gw", "main", "t-gw");

    let req = ChatCompletionRequest {
        model: "deepseek/deepseek-v4.1-flash-1017".into(),
        messages: vec![ChatMessage::user("Analyze dataset structure")],
        tools: vec![],
        temperature: Some(0.1),
        max_tokens: Some(1024),
        provider: Some(json!({"only":["inference-net/fp8"],"order":["inference-net/fp8"],"allow_fallbacks":false})),
        reasoning: None,
    };

    let ledger = dir.path().join("campaign_budget_ledger.json");
    std::fs::write(&ledger, json!({"campaign_ceiling_usd":"0.30", "total_billed_usd":"0.050724020",
        "total_reserved_usd":"0.01", "total_uncertain_usd":"0.00",
        "active_reservations":{"campaign-1:turn:1":{"run_id":"campaign-1", "amount_usd":"0.01"}}}).to_string()).unwrap();
    struct PolicyGateway(std::path::PathBuf);
    impl mlrescurix_harness::agents::ProviderGateway for PolicyGateway {
        fn requires_policy_approval(&self) -> bool { true }
        fn campaign_budget_context(&self) -> mlrescurix_harness::Result<(std::path::PathBuf, String)> {
            Ok((self.0.clone(), "campaign-1".into()))
        }
        fn chat_completion(&self, _: &ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            Ok(ChatCompletionResponse {
                id: "resp-gw-1".into(),
                model: "deepseek/deepseek-v4.1-flash-1017".into(),
                content: Some("{\"action_type\": \"ANALYZE\", \"parameters\": {}}".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 150, completion_tokens: 30, total_tokens: 180 },
                latency_ms: 12,
            })
        }
    }

    mlrescurix_harness::agents::register_model_pricing(&root, "deepseek/deepseek-v4.1-flash-1017", 8, 50).unwrap();

    let err_unapproved = mlrescurix_harness::agents::dispatch_gateway(
        &root, "p-gw", "t-gw", "scientific-agent", &req, &PolicyGateway(ledger.clone()), true
    ).unwrap_err();
    assert!(err_unapproved.contains("outbound provider request requires a content-bound policy-approved view"));

    let staged = mlrescurix_harness::agents::stage_outbound_request(&root, "p-gw", &req, "controller").unwrap();
    let digest = staged["request_sha256"].as_str().unwrap();
    let inspection = mlrescurix_harness::agents::gateway::inspect_outbound_view(&root, "p-gw", digest).unwrap();
    assert_eq!(inspection["approved"], false);
    assert_eq!(inspection["view"]["messages"][0]["content"], "Analyze dataset structure");

    assert!(mlrescurix_harness::agents::approve_outbound_view(&root, "p-gw", digest, "controller", true, 600_000).is_err());
    mlrescurix_harness::agents::approve_outbound_view(&root, "p-gw", digest, "authorized-operator", true, 600_000).unwrap();

    let res = mlrescurix_harness::agents::dispatch_gateway_with_id(
        &root, "p-gw", "t-gw", "scientific-agent", "campaign-1:turn:1", &req, &PolicyGateway(ledger.clone()), true
    ).unwrap();
    assert_eq!(res["status"], "completed");
    assert_eq!(res["id"], "resp-gw-1");
    assert_eq!(res["usage"]["prompt_tokens"], 150);
    assert_eq!(res["usage"]["completion_tokens"], 30);
    assert_eq!(res["cost_millicents"], 4);
    assert_eq!(res["cost_is_estimate"], true);
    assert_eq!(res["replayed"], false);

    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let recorded_tokens: u32 = db.query_row(
        "SELECT prompt_tokens FROM provider_accounting WHERE project='p-gw' AND model='deepseek/deepseek-v4.1-flash-1017'",
        [], |r| r.get(0)
    ).unwrap();
    assert_eq!(recorded_tokens, 150);

    let replay = mlrescurix_harness::agents::dispatch_gateway_with_id(
        &root, "p-gw", "t-gw", "scientific-agent", "campaign-1:turn:1", &req,
        &ScriptedProviderFixture::new(vec![]), true
    ).unwrap();
    assert_eq!(replay["replayed"], true);
    assert_eq!(replay["id"], "resp-gw-1");
    let accounting_rows: i64 = db.query_row(
        "SELECT count(*) FROM provider_accounting WHERE project='p-gw'", [], |row| row.get(0)
    ).unwrap();
    assert_eq!(accounting_rows, 1, "stable request identity must not create duplicate charges");
}

#[test]
fn provider_gateway_requires_prices_before_any_transport() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct CountingGateway(AtomicBool);
    impl mlrescurix_harness::agents::ProviderGateway for CountingGateway {
        fn chat_completion(&self, _: &ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            self.0.store(true, Ordering::SeqCst);
            Err("must not be called".into())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 2)).unwrap();
    create_m1(&runtime, "p-unpriced", 100);
    branch_task(&runtime, "p-unpriced", "main", "t-unpriced");
    let gateway = CountingGateway(AtomicBool::new(false));
    let req = ChatCompletionRequest {
        model: "unpriced-model".into(), messages: vec![ChatMessage::user("safe prompt")],
        tools: vec![], temperature: None, max_tokens: Some(100), provider: None, reasoning: None,
    };
    let err = mlrescurix_harness::agents::dispatch_gateway_with_id(
        &root, "p-unpriced", "t-unpriced", "scientific-agent", "session-x:turn:1", &req, &gateway, false
    ).unwrap_err();
    assert!(err.contains("no registered pricing"));
    assert!(!gateway.0.load(Ordering::SeqCst));
}

#[test]
fn gateway_concurrent_duplicate_and_ambiguous_receipts_never_redispatch() {
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    struct CountingFixture(AtomicUsize);
    impl mlrescurix_harness::agents::ProviderGateway for CountingFixture {
        fn is_offline_fixture(&self) -> bool { true }
        fn chat_completion(&self, _: &ChatCompletionRequest) -> mlrescurix_harness::Result<ChatCompletionResponse> {
            self.0.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(30));
            Ok(ChatCompletionResponse {id:"one".into(), model:"model".into(), content:Some("response".into()),
                tool_calls:vec![], usage:TokenUsage{prompt_tokens:1,completion_tokens:1,total_tokens:2},latency_ms:1})
        }
    }
    let dir = tempfile::tempdir().unwrap(); let root = dir.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100, 2)).unwrap();
    create_m1(&runtime, "concurrent", 100); branch_task(&runtime, "concurrent", "main", "t");
    mlrescurix_harness::agents::register_model_pricing(&root, "model", 8, 50).unwrap();
    let req = ChatCompletionRequest {model:"model".into(),messages:vec![ChatMessage::user("offline")],tools:vec![],
        temperature:None,max_tokens:Some(100),provider:None,reasoning:None};
    let provider = Arc::new(CountingFixture(AtomicUsize::new(0)));
    std::thread::scope(|scope| {
        let mut handles = vec![];
        for _ in 0..8 {
            let root = &root; let req = &req; let provider = provider.clone();
            handles.push(scope.spawn(move || mlrescurix_harness::agents::dispatch_gateway_with_id(
                root,"concurrent","t","agent","stable",req,provider.as_ref(),false)));
        }
        for handle in handles { let result = handle.join().unwrap();
            if let Err(error) = result { assert!(error.contains("ambiguous prior outcome"), "{error}"); }
        }
    });
    assert_eq!(provider.0.load(Ordering::SeqCst),1);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    for status in ["awaiting_response", "outcome_unknown"] {
        db.execute("UPDATE gateway_dispatch_receipts SET status=?1,response_json=NULL WHERE request_id='stable'",[status]).unwrap();
        assert!(mlrescurix_harness::agents::dispatch_gateway_with_id(&root,"concurrent","t","agent","stable",
            &req,provider.as_ref(),false).unwrap_err().contains("ambiguous prior outcome"));
    }
    assert_eq!(provider.0.load(Ordering::SeqCst),1);
}
