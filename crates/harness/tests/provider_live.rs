use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
};
use mlrescurix_harness::{
    agents::{
        AgentOrchestrator, ChatCompletionRequest, ChatMessage, OpenAICompatibleAdapter,
        OrchestratorConfig, ProviderGateway,
    },
    contracts::Context,
    core::*,
    Runtime,
};

fn m1policy(limit: i64, workers: i64) -> Policy {
    Policy {
        budget: BTreeMap::from([
            ("calls".into(), limit),
            ("tokens".into(), limit * 2048),
            ("wall_ms".into(), limit * 30_000),
        ]),
        capacity: BTreeMap::from([("workers".into(), workers)]),
        max_tasks: 1024,
    }
}

fn create_m1(r: &Runtime, p: &str, limit: i64) {
    let req = Request {
        command_id: format!("create-{p}"),
        project_id: p.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "Live provider test".into(),
            finalization_id: format!("final-{p}"),
            policy: m1policy(limit, 4),
        },
    };
    assert_eq!(r.command(req).unwrap().status, "applied");
}

fn branch_task(r: &Runtime, p: &str, branch: &str, task: &str) {
    let r1 = Request {
        command_id: format!("b-{branch}"),
        project_id: p.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: branch.into(),
            base_hash: "base-live".into(),
            code_hash: "code-live".into(),
        },
    };
    r.command(r1).unwrap();

    let r2 = Request {
        command_id: format!("t-{task}"),
        project_id: p.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task.into(),
            branch_id: branch.into(),
            context: Context {
                role_id: "lead".into(),
                role_version: "1".into(),
                prompt_hash: "prompt-live".into(),
                model: Some("deepseek/deepseek-chat".into()),
                phase: "research".into(),
                hypothesis: None,
                base_hash: "base-live".into(),
                code_hash: "code-live".into(),
                evidence: vec![],
                artifacts: vec![],
                pending_questions: vec![],
                calls: vec![],
                jobs: vec![],
                next_action: "query".into(),
                stop_conditions: vec!["budget".into()],
            },
        },
    };
    r.command(r2).unwrap();
}

fn load_api_key() -> Option<String> {
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY") {
        if !key.trim().is_empty() {
            return Some(key.trim().to_string());
        }
    }
    if let Ok(key) = fs::read_to_string(".config/openrouter.key") {
        if !key.trim().is_empty() {
            return Some(key.trim().to_string());
        }
    }
    let home = std::env::var("HOME").ok()?;
    let path = Path::new(&home).join(".config/mlrescurix/openrouter.key");
    if let Ok(key) = fs::read_to_string(path) {
        if !key.trim().is_empty() {
            return Some(key.trim().to_string());
        }
    }
    None
}

#[test]
#[ignore = "Spends provider funds; run explicitly only with current user authorization and a replacement credential"]
fn test_live_openrouter_deepseek_call_and_accounting() {
    let api_key = match load_api_key() {
        Some(k) => k,
        None => {
            eprintln!("Skipping live provider test: no API key found in .config/openrouter.key");
            return;
        }
    };

    let model = "deepseek/deepseek-v4.1-flash-1017";
    let endpoint = "https://openrouter.ai/api/v1/chat/completions";
    let adapter = OpenAICompatibleAdapter::new(endpoint.to_string(), api_key);

    // 1. Direct gateway invocation test with conservative tokens
    let req = ChatCompletionRequest {
        model: model.into(),
        messages: vec![ChatMessage::user("Respond with the exact word: ACKNOWLEDGED")],
        tools: vec![],
        temperature: Some(0.0),
        max_tokens: Some(50),
        provider: Some(serde_json::json!({
            "only": ["deepinfra/fp8/flex"],
            "order": ["deepinfra/fp8/flex"],
            "allow_fallbacks": false,
        })),
        reasoning: None,
    };

    let resp = adapter.chat_completion(&req).expect("Live provider request must succeed");
    eprintln!("Live model resp: {:?}", resp);
    assert!(!resp.id.is_empty(), "Response ID must not be empty");
    assert!(resp.content.is_some(), "Response must have text content");
    let content = resp.content.unwrap();
    eprintln!("Live model response: {content}");
    assert!(resp.usage.prompt_tokens > 0, "Prompt tokens must be > 0");
    assert!(resp.usage.completion_tokens > 0, "Completion tokens must be > 0");
    assert!(resp.latency_ms > 0, "Latency must be measured");

    // 2. Full AgentOrchestrator turn execution with authoritative M0 accounting
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(10, 2)).unwrap();
    create_m1(&r, "p-live", 10);
    branch_task(&r, "p-live", "branch-live", "task-live");

    mlrescurix_harness::agents::gateway::register_model_pricing(&root, model, 16, 48).unwrap();

    let config = OrchestratorConfig {
        project_id: "p-live".into(),
        task_id: "task-live".into(),
        role_id: "lead".into(),
        branch_id: "branch-live".into(),
        model: model.into(),
        max_turns: 2,
    };

    let mut orch = AgentOrchestrator::new(&root, config, Arc::new(adapter)).unwrap();
    let staged = orch.stage_outbound_view("policy-auditor").unwrap();
    let digest = staged["request_sha256"].as_str().unwrap().to_string();
    mlrescurix_harness::agents::gateway::approve_outbound_view(&root, "p-live", &digest, "policy-auditor", true, 600_000).unwrap();
    let turn_res = orch.step().expect("Orchestrator step must succeed with live provider");
    assert_eq!(turn_res.turn_index, 1);
    assert!(turn_res.assistant_content.is_some());

    // 3. Verify SQLite accounting records
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let (acct_model, prompt_tok, comp_tok, status): (String, i64, i64, String) = db.query_row(
        "SELECT model, prompt_tokens, completion_tokens, status FROM provider_accounting WHERE project='p-live' ORDER BY created_ms DESC LIMIT 1",
        [],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
    ).unwrap();

    assert_eq!(status, "priced");
    assert!(prompt_tok > 0, "Prompt tokens must be accounted");
    assert!(comp_tok > 0, "Completion tokens must be accounted");
    eprintln!("Live provider accounting verified: model={acct_model}, prompt_tokens={prompt_tok}, completion_tokens={comp_tok}");

    // 4. Verify M0 charges ledger record
    let charge_count: i64 = db.query_row(
        "SELECT count(*) FROM charges WHERE project='p-live'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(charge_count > 0, "M0 charges ledger must record the settled action");
}
