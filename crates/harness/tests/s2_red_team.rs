//! Independent S2 boundary regressions; trusted host fixtures only.
use std::{fs, sync::Arc};
use mlrescurix_harness::{
    agents::gateway::{ChatCompletionResponse, FunctionCall, ScriptedProviderFixture, TokenUsage, ToolCall},
    campaign::{generate_default_solution_template, parse_structured_challenger_result, run_offline_fixture_campaign, CampaignProblemSpec},
    trusted_evaluation::Metric,
};

fn problem(root: &std::path::Path) -> CampaignProblemSpec {
    let dataset_path = root.join("data.csv");
    let mut csv = String::from("x,target\n");
    for i in 0..30 { csv.push_str(&format!("{i},{}\n", i * 2)); }
    fs::write(&dataset_path, csv).unwrap();
    CampaignProblemSpec {
        project_id: "s2_review".into(), dataset_path, target_column: "target".into(),
        task_type: "regression".into(), metric: Metric::Mae, budget_millicents: 200_000,
        model_name: "deepseek/deepseek-chat".into(), endpoint: None, api_key: None,
        lockbox_fraction: Some(0.2), seed: Some(2026), max_turns: Some(4),
        split_strategy: None, time_column: None, group_column: None,
    }
}

fn authoring_turns(model: &str) -> Vec<ChatCompletionResponse> {
    let (custom_solution_py, program_manifest) = generate_default_solution_template(
        "regression",
        &["x".into()],
        "target",
        "cand_s2_red_team",
    ).unwrap();
    vec![
        ChatCompletionResponse {
            id: "turn_spec_author".into(),
            model: model.into(),
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
            id: "turn_spec_finalize".into(),
            model: model.into(),
            content: Some("[FINAL] Solution authored.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 0,
        },
    ]
}

fn gateway(contents: Vec<Option<&str>>) -> Arc<ScriptedProviderFixture> {
    Arc::new(ScriptedProviderFixture::new(contents.into_iter().enumerate().map(|(i, c)| ChatCompletionResponse {
        id: format!("review_{i}"), model: "deepseek/deepseek-chat".into(), content: c.map(str::to_string),
        tool_calls: vec![], usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 }, latency_ms: 0,
    }).collect()))
}

#[test]
fn specialist_text_cannot_select_an_unauthored_reference() {
    for content in ["[FINAL]", "reference", "[FINAL] Reference diagnostic."] {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = problem(tmp.path());
        spec.max_turns = Some(1);
        let state = tmp.path().join("state");
        let err = run_offline_fixture_campaign(&state, &spec, Some(gateway(vec![Some(content)]))).unwrap_err();
        assert!(err.contains("specialist failed to author"), "unexpected error: {err}");
        let conn = rusqlite::Connection::open(state.join("controller/state.sqlite")).unwrap();
        let count: i64 = conn.query_row("SELECT count(*) FROM assessment_lockbox_consumptions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
    }
}

#[test]
fn owner_missing_ambiguous_or_negative_decision_cannot_finalize() {
    for content in [None, Some(""), Some("I need more evidence"), Some("NOT FINALIZE_AND_STOP: do not release"), Some("FINALIZE_AND_STOP: release without audit"),
                    Some("FINALIZE_AND_STOP: maybe DELEGATE_CHALLENGER"), Some("DELEGATE_CHALLENGER: not FINALIZE_AND_STOP")] {
        let tmp = tempfile::tempdir().unwrap();
        let spec = problem(tmp.path());
        let state = tmp.path().join("state");
        let mut responses = authoring_turns(&spec.model_name);
        responses.push(ChatCompletionResponse {
            id: "turn_boss".into(),
            model: spec.model_name.clone(),
            content: content.map(str::to_string),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 0,
        });
        let gateway = Arc::new(ScriptedProviderFixture::new(responses));
        let err = run_offline_fixture_campaign(&state, &spec, Some(gateway)).unwrap_err();
        assert!(!err.contains("specialist failed to author"), "{content:?}: reached wrong boundary: {err}");
        assert!(err.contains("owner") || err.contains("finaliz") || err.contains("decision"), "{content:?}: {err}");
        let conn = rusqlite::Connection::open(state.join("controller/state.sqlite")).unwrap();
        let count: i64 = conn.query_row("SELECT count(*) FROM assessment_lockbox_consumptions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0, "invalid owner reply consumed lockbox: {content:?}");
    }
}

#[test]
fn post_audit_owner_reply_requires_explicit_finalization() {
    for content in [None, Some(""), Some("Keep investigating"), Some("DELEGATE_CHALLENGER: audit again")] {
        let tmp = tempfile::tempdir().unwrap();
        let spec = problem(tmp.path());
        let state = tmp.path().join("state");
        let mut responses = authoring_turns(&spec.model_name);
        responses.push(ChatCompletionResponse {
            id: "turn_boss_delegate".into(),
            model: spec.model_name.clone(),
            content: Some("DELEGATE_CHALLENGER: independent audit required".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 0,
        });
        responses.push(ChatCompletionResponse {
            id: "turn_challenger".into(),
            model: spec.model_name.clone(),
            content: Some("APPROVED: trusted synthetic diagnostic only".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 0,
        });
        responses.push(ChatCompletionResponse {
            id: "turn_boss_post_audit".into(),
            model: spec.model_name.clone(),
            content: content.map(str::to_string),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 0,
        });
        let gateway = Arc::new(ScriptedProviderFixture::new(responses));
        let err = run_offline_fixture_campaign(&state, &spec, Some(gateway)).unwrap_err();
        assert!(!err.contains("specialist failed to author"), "{content:?}: reached wrong boundary: {err}");
        assert!(err.contains("owner") || err.contains("finalization"), "{content:?}: {err}");
        let conn = rusqlite::Connection::open(state.join("controller/state.sqlite")).unwrap();
        let count: i64 = conn.query_row("SELECT count(*) FROM assessment_lockbox_consumptions", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 0);
    }
}

#[test]
fn challenger_rejects_contradictory_bindings_and_malformed_audits() {
    for content in [
        r#"{"verdict":"APPROVED"}"#,
        r#"{"verdict":"APPROVED","rationale":"ok","candidate_id":"other"}"#,
        r#"{"verdict":"APPROVED","rationale":"ok","protocol_sha256":"other"}"#,
        r#"{"verdict":"APPROVED","rationale":"ok","task_id":"other"}"#,
        r#"{"verdict":"APPROVED","rationale":"ok","validation_score":9000}"#,
        r#"{"verdict":"APPROVED","rationale":"ok","schema_version":"unknown"}"#,
        r#"{"verdict":"APPROVED","rationale":"ok","findings":[12]}"#,
    ] {
        assert!(parse_structured_challenger_result(content, "task", "candidate", "protocol", 1.0).is_err(),
            "invalid audit accepted: {content}");
    }
}
