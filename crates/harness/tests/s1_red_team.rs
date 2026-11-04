//! Boundary regressions discovered while independently reviewing S1.
use std::{fs, sync::Arc};
use mlrescurix_harness::{
    agents::gateway::{ChatCompletionResponse, FunctionCall, ScriptedProviderFixture, TokenUsage, ToolCall},
    campaign::{generate_default_solution_template, run_autonomous_campaign, run_offline_fixture_campaign, CampaignProblemSpec},
    trusted_evaluation::Metric,
};

fn problem(root: &std::path::Path) -> CampaignProblemSpec {
    let dataset_path = root.join("data.csv");
    let mut data = String::from("x,target\n");
    for i in 0..30 { data.push_str(&format!("{},{}\n", i as f64 / 30.0, i as f64 / 15.0)); }
    fs::write(&dataset_path, data).unwrap();
    CampaignProblemSpec {
        project_id: "review_boundary".into(), dataset_path, target_column: "target".into(),
        task_type: "regression".into(), metric: Metric::Mae, budget_millicents: 150_000,
        model_name: "deepseek/deepseek-chat".into(), endpoint: None, api_key: None,
        lockbox_fraction: Some(0.2), seed: Some(2026), max_turns: Some(1),
        split_strategy: None, time_column: None, group_column: None,
    }
}

#[test]
fn production_requires_real_isolation_before_state_creation() {
    let tmp = tempfile::tempdir().unwrap();
    let spec = problem(tmp.path());
    let state = tmp.path().join("state");
    let err = run_autonomous_campaign(&state, &spec, None).unwrap_err();
    assert!(err.contains("isolated worker execution is not yet implemented"), "{err}");
    assert!(!state.exists());
}

#[test]
fn challenger_must_return_an_unambiguous_explicit_verdict() {
    for content in [Some(""), None, Some("I cannot inspect the evidence"), Some("NOT APPROVED"), Some("APPROVED or CHALLENGED"), Some("APPROVED: no, actually CHALLENGED")] {
        let tmp = tempfile::tempdir().unwrap();
        let spec = problem(tmp.path());
        let (custom_solution_py, program_manifest) = generate_default_solution_template(
            "regression",
            &["x".into()],
            "target",
            "cand_review_boundary",
        ).unwrap();
        let responses = vec![
            ChatCompletionResponse {
                id: "turn_spec_author".into(),
                model: spec.model_name.clone(),
                content: Some("Authoring solution and manifest".into()),
                tool_calls: vec![
                    ToolCall {
                        id: "call_ws_solution".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "workspace_fs".into(),
                            arguments: serde_json::json!({
                                "action": "write",
                                "path": "solution.py",
                                "content": custom_solution_py,
                            }).to_string(),
                        },
                    },
                    ToolCall {
                        id: "call_ws_program".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "workspace_fs".into(),
                            arguments: serde_json::json!({
                                "action": "write",
                                "path": "program.json",
                                "content": serde_json::to_string_pretty(&program_manifest).unwrap(),
                            }).to_string(),
                        },
                    },
                ],
                usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
                latency_ms: 0,
            },
            ChatCompletionResponse {
                id: "turn_boss_delegate".into(),
                model: spec.model_name.clone(),
                content: Some("DELEGATE_CHALLENGER: audit required".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
                latency_ms: 0,
            },
            ChatCompletionResponse {
                id: "turn_challenger".into(),
                model: spec.model_name.clone(),
                content: content.map(str::to_string),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
                latency_ms: 0,
            },
        ];
        let gateway = Arc::new(ScriptedProviderFixture::new(responses));
        let err = run_offline_fixture_campaign(&tmp.path().join("state"), &spec, Some(gateway)).unwrap_err();
        assert!(err.contains("challenger verdict"), "{content:?}: {err}");
        let db = rusqlite::Connection::open(tmp.path().join("state/controller/state.sqlite")).unwrap();
        let count: i64 = db.query_row("SELECT count(*) FROM assessment_lockbox_consumptions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
    }
}

#[test]
fn offline_fixture_rejects_live_credentials_and_remote_gateways_before_state_creation() {
    let tmp = tempfile::tempdir().unwrap();
    let mut spec = problem(tmp.path());
    let state = tmp.path().join("state");
    spec.api_key = Some("fixture-placeholder-never-transmitted".into());
    let err = run_offline_fixture_campaign(&state, &spec, None).unwrap_err();
    assert!(err.contains("rejects live provider"));
    spec.api_key = None;
    let remote = mlrescurix_harness::agents::gateway::OpenAICompatibleAdapter::new(
        "https://example.invalid/v1".into(), "placeholder".into(),
    );
    let err = run_offline_fixture_campaign(&state, &spec, Some(Arc::new(remote))).unwrap_err();
    assert!(err.contains("rejects live provider"));
    assert!(!state.exists());
}

#[test]
fn malformed_csv_width_is_rejected_without_panic() {
    for row in ["1,2,3\n", "1\n"] {
        let tmp = tempfile::tempdir().unwrap();
        let spec = problem(tmp.path());
        fs::write(&spec.dataset_path, format!("x,target\n{row}")).unwrap();
        let err = run_offline_fixture_campaign(&tmp.path().join("state"), &spec, None).unwrap_err();
        assert!(err.contains("fields; expected 2"), "{err}");
    }
}
