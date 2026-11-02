//! Independent S2 boundary regressions; trusted host fixtures only.
use std::{fs, sync::Arc};
use mlrescurix_harness::{
    agents::gateway::{ChatCompletionResponse, ScriptedProviderFixture, TokenUsage},
    campaign::{parse_structured_challenger_result, run_offline_fixture_campaign, CampaignProblemSpec},
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
    }
}

fn gateway(contents: Vec<Option<&str>>) -> Arc<ScriptedProviderFixture> {
    Arc::new(ScriptedProviderFixture::new(contents.into_iter().enumerate().map(|(i, c)| ChatCompletionResponse {
        id: format!("review_{i}"), model: "deepseek/deepseek-chat".into(), content: c.map(str::to_string),
        tool_calls: vec![], usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 }, latency_ms: 0,
    }).collect()))
}

#[test]
fn owner_missing_ambiguous_or_negative_decision_cannot_finalize() {
    for content in [None, Some(""), Some("I need more evidence"), Some("NOT FINALIZE_AND_STOP: do not release"), Some("FINALIZE_AND_STOP: release without audit"),
                    Some("FINALIZE_AND_STOP: maybe DELEGATE_CHALLENGER"), Some("DELEGATE_CHALLENGER: not FINALIZE_AND_STOP")] {
        let tmp = tempfile::tempdir().unwrap();
        let spec = problem(tmp.path());
        let state = tmp.path().join("state");
        let result = run_offline_fixture_campaign(&state, &spec,
            Some(gateway(vec![Some("[FINAL] Reference diagnostic."), content])));
        assert!(result.is_err(), "invalid owner reply finalized: {content:?}");
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
        let result = run_offline_fixture_campaign(&state, &spec, Some(gateway(vec![
            Some("[FINAL] Reference diagnostic."), Some("DELEGATE_CHALLENGER: independent audit required"),
            Some("APPROVED: trusted synthetic diagnostic only"), content,
        ])));
        assert!(result.is_err(), "post-audit invalid reply finalized: {content:?}");
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
