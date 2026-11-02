//! Boss-and-Specialist Vertical Slice & Contract Isolation Tests (Evidence A, B, C)
//!
//! Validates:
//! Evidence A:
//! 1. End-to-end provider-fixture campaign on synthetic data with real model fit, trusted score,
//!    owner observation, specialist completion, challenger result and checkpoint/handoff.
//! 2. A second configuration/question (classification / Accuracy) changes behavior without editing orchestration code.
//! Evidence B:
//! Serialized context isolation: challenger's inputs exclude foreign transcripts/advocacy
//! while retaining necessary scoped facts (protocol SHA, metric, validation score, feature columns).
//! Evidence C:
//! Resumption and lease fencing:
//! 1. Restart at completion boundary preserves incumbent and returns existing verified outcome without double-consuming lockbox.
//! 2. Active owner lease prevents conflicting concurrent writers.
//! Challenger Rejection:
//! When challenger challenges candidate, Boss halts closed and rejects candidate before lockbox consumption.

use std::{
    fs,
    path::Path,
    sync::Arc,
};

use rusqlite::Connection;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::gateway::{
        ChatCompletionResponse, FunctionCall,
        ScriptedProviderFixture, TokenUsage, ToolCall,
    },
    campaign::{run_offline_fixture_campaign as run_autonomous_campaign, CampaignOutcome, CampaignProblemSpec},
    trusted_evaluation::Metric,
};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_synthetic_regression_dataset(path: &Path, num_rows: usize) -> (String, Vec<(u64, f64)>) {
    let mut csv = String::from("x1,x2,target\n");
    let mut labels = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let x1 = (i as f64 - (num_rows as f64 / 2.0)) / 10.0;
        let x2 = (i as f64) * 0.15;
        let y = 2.5 * x1 * x1 - 1.2 * x2.sin() + 0.5;
        let y_round = (y * 10000.0).round() / 10000.0;
        csv.push_str(&format!("{x1:.4},{x2:.4},{y_round:.4}\n"));
        labels.push((i as u64, y_round));
    }
    fs::write(path, &csv).unwrap();
    let bytes = fs::read(path).unwrap();
    (sha256_bytes(&bytes), labels)
}

fn create_synthetic_classification_dataset(path: &Path, num_rows: usize) -> (String, Vec<(u64, f64)>) {
    let mut csv = String::from("feat_a,feat_b,target\n");
    let mut labels = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let fa = (i as f64 - (num_rows as f64 / 2.0)) / 5.0;
        let fb = (i as f64) * 0.2 - 2.0;
        let y = if fa + fb > 0.0 { 1.0 } else { 0.0 };
        csv.push_str(&format!("{fa:.4},{fb:.4},{y:.1}\n"));
        labels.push((i as u64, y));
    }
    fs::write(path, &csv).unwrap();
    let bytes = fs::read(path).unwrap();
    (sha256_bytes(&bytes), labels)
}

#[test]
fn test_evidence_a_boss_specialist_slice_end_to_end() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("regression_data.csv");
    let (ds_hash, _labels) = create_synthetic_regression_dataset(&dataset_path, 30);

    let model_name = "deepseek/deepseek-chat";

    let custom_solution_py = r#"
import sys, json, math, csv, argparse

def basis(x1, x2):
    return [1.0, x1, x2, x1 * x2, x1 * x1]

def fit(train_csv, weights_out):
    rows = []
    with open(train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((basis(float(r["x1"]), float(r["x2"])), float(r["target"])))
    dim = 5
    w = [0.0] * dim
    lr = 0.04
    for _ in range(500):
        grad = [0.0] * dim
        for phi, y in rows:
            pred = sum(w[j] * phi[j] for j in range(dim))
            err = pred - y
            for j in range(dim):
                grad[j] += err * phi[j]
        for j in range(dim):
            w[j] -= lr * (grad[j] / len(rows))
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump({"weights": w, "arch": "LinearSpecialist"}, f, indent=2)

def infer(feat_csv, weights_in, preds_out):
    with open(weights_in, "r", encoding="utf-8") as f:
        w = json.load(f)["weights"]
    preds = []
    with open(feat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            phi = basis(float(r["x1"]), float(r["x2"]))
            val = sum(w[j] * phi[j] for j in range(len(w)))
            preds.append({"row_id": row_id, "prediction": round(val, 6)})
    with open(preds_out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f_p = sub.add_parser("fit")
    f_p.add_argument("--train-data", required=True)
    f_p.add_argument("--weights-out", required=True)
    i_p = sub.add_parser("infer")
    i_p.add_argument("--features-in", required=True)
    i_p.add_argument("--weights-in", required=True)
    i_p.add_argument("--predictions-out", required=True)
    args = p.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
"#;

    let program_manifest = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "entry_point": "solution.py",
        "architecture": "LinearSpecialist"
    });

    let scripted_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1 (Specialist): author solution.py
        ChatCompletionResponse {
            id: "turn1-write-solution".into(),
            model: model_name.into(),
            content: Some("Writing solution.py".into()),
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
                }
            ],
            usage: TokenUsage { prompt_tokens: 250, completion_tokens: 120, total_tokens: 370 },
            latency_ms: 20,
        },
        // Turn 2 (Specialist): author program.json
        ChatCompletionResponse {
            id: "turn2-write-program".into(),
            model: model_name.into(),
            content: Some("Writing program.json".into()),
            tool_calls: vec![
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
                }
            ],
            usage: TokenUsage { prompt_tokens: 180, completion_tokens: 40, total_tokens: 220 },
            latency_ms: 15,
        },
        // Turn 3 (Specialist): finalize authoring
        ChatCompletionResponse {
            id: "turn3-finalize".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution authored.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 90, completion_tokens: 15, total_tokens: 105 },
            latency_ms: 10,
        },
        // Turn 4 (Boss): observe evidence and delegate to challenger
        ChatCompletionResponse {
            id: "turn4-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Validation score is competitive; audit for variance and protocol leak.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 25, total_tokens: 175 },
            latency_ms: 12,
        },
        // Turn 5 (Challenger): audit candidate evidence with fresh context
        ChatCompletionResponse {
            id: "turn5-challenger-audit".into(),
            model: model_name.into(),
            content: Some("APPROVED: Validation score is within expected range and protocol split boundaries are strictly respected.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 300, completion_tokens: 45, total_tokens: 345 },
            latency_ms: 25,
        },
        // Turn 6 (Boss): observe challenger approval and finalize
        ChatCompletionResponse {
            id: "turn6-boss-finalize".into(),
            model: model_name.into(),
            content: Some("FINALIZE_AND_STOP: Candidate independently audited and approved.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 180, completion_tokens: 20, total_tokens: 200 },
            latency_ms: 10,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_slice_e2e".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 200_000,
        model_name: model_name.into(),
        endpoint: Some("https://openrouter.ai/api/v1".into()),
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(4),
    };

    let outcome: CampaignOutcome = run_autonomous_campaign(
        root,
        &problem_spec,
        Some(scripted_gateway),
    ).unwrap();

    assert_eq!(outcome.project_id, "proj_slice_e2e");
    assert_eq!(outcome.dataset_sha256, ds_hash);
    assert!(!outcome.candidate_id.is_empty());
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.lockbox_score.is_finite());

    // Delivery bundle handoff assertions
    assert!(outcome.bundle_dir.is_dir());
    assert!(outcome.bundle_dir.join("inference.py").is_file());
    assert!(outcome.bundle_dir.join("serve.py").is_file());
    assert!(outcome.bundle_dir.join("frozen_model.joblib").is_file());
    assert!(outcome.bundle_dir.join("model_weights.json").is_file());
    assert!(outcome.bundle_dir.join("manifest.json").is_file());
    assert!(outcome.bundle_dir.join("scorecard.json").is_file());
    assert!(outcome.bundle_verification.manifest_verified);
    assert!(outcome.bundle_verification.contracts_verified);

    // Verify database state: task assignments and completions for both specialist and challenger
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    let spec_completed: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM task_completions WHERE project='proj_slice_e2e' AND task_id LIKE 'task_authoring%' AND verdict='succeeded')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(spec_completed, "specialist completion record missing");

    let challenger_completed: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM task_completions WHERE project='proj_slice_e2e' AND task_id LIKE 'task_challenger%' AND verdict='succeeded')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(challenger_completed, "challenger completion record missing");

    // Verify lease was cleanly released
    let lease_status: String = conn.query_row(
        "SELECT status FROM campaign_leases WHERE project='proj_slice_e2e'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(lease_status, "released");
}

#[test]
fn test_evidence_a_second_configuration_changes_behavior() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("classification_data.csv");
    let (ds_hash, _labels) = create_synthetic_classification_dataset(&dataset_path, 32);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_slice_classification".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "classification".into(),
        metric: Metric::Accuracy,
        budget_millicents: 150_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.25),
        seed: Some(42),
        max_turns: Some(3),
    };

    let outcome: CampaignOutcome = run_autonomous_campaign(root, &problem_spec, None).unwrap();

    assert_eq!(outcome.project_id, "proj_slice_classification");
    assert_eq!(outcome.dataset_sha256, ds_hash);
    assert!(outcome.validation_score >= 0.0 && outcome.validation_score <= 1.0);
    assert!(outcome.lockbox_score >= 0.0 && outcome.lockbox_score <= 1.0);

    // Verify scorecard reflects classification task type and accuracy metric
    let scorecard_path = outcome.bundle_dir.join("scorecard.json");
    let scorecard: serde_json::Value = serde_json::from_slice(&fs::read(&scorecard_path).unwrap()).unwrap();
    assert_eq!(scorecard["task_type"], "classification");
    assert_eq!(scorecard["metric"], "accuracy");

    // Verify input_schema reflects classification
    let in_schema_path = outcome.bundle_dir.join("input_schema.json");
    let in_schema: serde_json::Value = serde_json::from_slice(&fs::read(&in_schema_path).unwrap()).unwrap();
    assert_eq!(in_schema["task"], "classification");
}

#[test]
fn test_evidence_b_serialized_context_isolation_and_evidence_binding() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("context_iso_data.csv");
    let (ds_hash, _labels) = create_synthetic_regression_dataset(&dataset_path, 26);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_context_iso".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
    };

    let outcome = run_autonomous_campaign(root, &problem_spec, None).unwrap();

    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    // 1. Challenger task assignment contains ONLY protocol and primary evidence
    let challenger_inputs_str: String = conn.query_row(
        "SELECT immutable_inputs FROM task_assignments WHERE project='proj_context_iso' AND role_id='challenger'",
        [],
        |r| r.get(0),
    ).unwrap();
    let challenger_inputs: serde_json::Value = serde_json::from_str(&challenger_inputs_str).unwrap();

    // Verify scoped facts are present
    assert_eq!(challenger_inputs["candidate_id"], outcome.candidate_id);
    assert_eq!(challenger_inputs["dataset_sha256"], ds_hash);
    assert_eq!(challenger_inputs["protocol_sha256"], outcome.protocol_sha256);
    assert!(challenger_inputs["validation_score"].as_f64().is_some());
    assert_eq!(challenger_inputs["metric"], "Mae");

    // Verify proposer/owner advocacy is strictly absent from challenger inputs
    assert!(challenger_inputs.get("proposer_advocacy").is_none());
    assert!(challenger_inputs.get("model_code").is_none());
    assert!(challenger_inputs.get("worker_dialogue").is_none());
    assert!(challenger_inputs.get("internal_cot").is_none());

    // 2. Task completion receipt binds real task, candidate, protocol, and metric
    let completion_candidate: String = conn.query_row(
        "SELECT candidate_id FROM task_completions WHERE project='proj_context_iso' AND task_id LIKE 'task_challenger%'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(completion_candidate, outcome.candidate_id);

    let completion_protocol: String = conn.query_row(
        "SELECT protocol FROM task_completions WHERE project='proj_context_iso' AND task_id LIKE 'task_challenger%'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(completion_protocol, outcome.protocol_sha256);
}

#[test]
fn test_evidence_c_restart_boundary_preserves_outcome_and_prevents_double_lockbox() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("restart_data.csv");
    let (ds_hash, _labels) = create_synthetic_regression_dataset(&dataset_path, 28);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_restart_test".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 150_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
    };

    // First execution: completes campaign from scratch
    let outcome1 = run_autonomous_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(outcome1.dataset_sha256, ds_hash);

    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    let initial_consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_restart_test'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(initial_consumptions, 1, "exactly 1 lockbox consumption expected after first run");

    // Second execution (Restart at completion boundary):
    // Must return existing verified outcome immediately without re-running or double-consuming holdout
    let outcome2 = run_autonomous_campaign(root, &problem_spec, None).unwrap();

    assert_eq!(outcome1.candidate_id, outcome2.candidate_id);
    assert_eq!(outcome1.weights_sha256, outcome2.weights_sha256);
    assert!((outcome1.validation_score - outcome2.validation_score).abs() < 1e-9);
    assert!((outcome1.lockbox_score - outcome2.lockbox_score).abs() < 1e-9);
    assert_eq!(outcome1.lockbox_receipt_sha256, outcome2.lockbox_receipt_sha256);

    // Verify lockbox was NOT double consumed
    let final_consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_restart_test'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(final_consumptions, 1, "lockbox was double consumed on restart!");

    // Third check: active owner lease fencing
    // Insert an artificial active lease and verify that concurrent execution fails closed
    conn.execute(
        "UPDATE campaign_leases SET status = 'active' WHERE project='proj_restart_test'",
        [],
    ).unwrap();

    let conflict_err = run_autonomous_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        conflict_err.contains("campaign lease conflict"),
        "expected lease conflict error, got: {conflict_err}"
    );
}

#[test]
fn test_challenger_rejection_halts_boss_and_rejects_candidate() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("rejection_data.csv");
    let (_ds_hash, _labels) = create_synthetic_regression_dataset(&dataset_path, 25);

    let model_name = "deepseek/deepseek-chat";

    // Scripted gateway where challenger rejects the candidate
    let rejection_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1 (Specialist): finalize default
        ChatCompletionResponse {
            id: "turn1-spec".into(),
            model: model_name.into(),
            content: Some("[FINAL] Use default solution.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 10,
        },
        // Turn 2 (Boss): observe evidence and delegate to challenger
        ChatCompletionResponse {
            id: "turn2-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Audit candidate validation evidence.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 20, total_tokens: 170 },
            latency_ms: 10,
        },
        // Turn 3 (Challenger): challenges candidate due to suspect validation
        ChatCompletionResponse {
            id: "turn3-challenger-reject".into(),
            model: model_name.into(),
            content: Some("CHALLENGED: Unrealistic validation score variance detected; fails adversarial audit.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 40, total_tokens: 240 },
            latency_ms: 15,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_challenger_reject".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: model_name.into(),
        endpoint: Some("https://openrouter.ai/api/v1".into()),
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
    };

    let err = run_autonomous_campaign(root, &problem_spec, Some(rejection_gateway)).unwrap_err();

    assert!(
        err.contains("challenged by independent scientific challenger"),
        "expected challenger rejection error, got: {err}"
    );

    // Verify lockbox was NOT consumed
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_challenger_reject'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions, 0, "lockbox should NOT have been consumed after challenger rejection");
}

#[test]
fn test_evidence_c_restart_recovers_when_bundle_incomplete_after_lockbox_consumption() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("restart_crash_data.csv");
    let (ds_hash, _labels) = create_synthetic_regression_dataset(&dataset_path, 30);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_restart_interrupted".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 150_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
    };

    // First execution: completes campaign
    let outcome1 = run_autonomous_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(outcome1.dataset_sha256, ds_hash);

    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    let consumptions_before: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_restart_interrupted'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions_before, 1);

    // Simulate crash between lockbox consumption and bundle sealing:
    // Remove manifest.json from the delivery bundle directory so it is incomplete
    let manifest_path = outcome1.bundle_dir.join("manifest.json");
    assert!(manifest_path.is_file());
    fs::remove_file(&manifest_path).unwrap();
    assert!(!manifest_path.is_file());

    // Resumption: Must NOT fail with "assessment lockbox has already been consumed".
    // Must gracefully recover using stored lockbox receipt, regenerate/verify bundle, and succeed!
    let outcome2 = run_autonomous_campaign(root, &problem_spec, None).unwrap();

    assert_eq!(outcome1.candidate_id, outcome2.candidate_id);
    assert_eq!(outcome1.weights_sha256, outcome2.weights_sha256);
    assert!((outcome1.validation_score - outcome2.validation_score).abs() < 1e-9);
    assert!((outcome1.lockbox_score - outcome2.lockbox_score).abs() < 1e-9);
    assert_eq!(outcome1.lockbox_receipt_sha256, outcome2.lockbox_receipt_sha256);

    // Verify lockbox consumption count is STILL strictly 1 (no double consumption!)
    let consumptions_after: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_restart_interrupted'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions_after, 1, "lockbox was double-consumed on crash recovery!");

    // Verify manifest was regenerated and bundle is verified
    assert!(manifest_path.is_file());
    assert!(outcome2.bundle_verification.manifest_verified);
    assert!(outcome2.bundle_verification.contracts_verified);
}

#[test]
fn test_evidence_c_restart_after_challenger_approval_does_not_re_step_challenger() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("restart_challenger_data.csv");
    let (_ds_hash, _labels) = create_synthetic_regression_dataset(&dataset_path, 28);

    let model_name = "deepseek/deepseek-chat";

    let scripted_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Specialist finalize
        ChatCompletionResponse {
            id: "turn1-spec".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution ready.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 10,
        },
        // Turn 2: Boss ReAct Turn 1 observes evidence and delegates to challenger
        ChatCompletionResponse {
            id: "turn2-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Validation score acceptable; audit for variance and protocol leak.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 25, total_tokens: 175 },
            latency_ms: 12,
        },
        // Turn 3: Challenger approves
        ChatCompletionResponse {
            id: "turn3-challenger".into(),
            model: model_name.into(),
            content: Some("APPROVED: Validation score and protocol independently verified.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 30, total_tokens: 230 },
            latency_ms: 15,
        },
        // Turn 4: Boss ReAct Turn 2 observes challenger audit and approves finalization
        ChatCompletionResponse {
            id: "turn4-boss-finalize".into(),
            model: model_name.into(),
            content: Some("FINALIZE_AND_STOP: Candidate independently audited and approved.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 180, completion_tokens: 20, total_tokens: 200 },
            latency_ms: 10,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_challenger_replay_guard".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 150_000,
        model_name: model_name.into(),
        endpoint: Some("https://openrouter.ai/api/v1".into()),
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
    };

    let outcome1 = run_autonomous_campaign(root, &problem_spec, Some(scripted_gateway)).unwrap();

    // Now resume with an EMPTY scripted gateway (0 responses remaining)
    // If orchestrator attempts to step the challenger again, ScriptedProviderFixture will panic/error!
    let empty_gateway = Arc::new(ScriptedProviderFixture::new(vec![]));

    let outcome2 = run_autonomous_campaign(root, &problem_spec, Some(empty_gateway)).unwrap();
    assert_eq!(outcome1.candidate_id, outcome2.candidate_id);
    assert!((outcome1.validation_score - outcome2.validation_score).abs() < 1e-9);
    assert!((outcome1.lockbox_score - outcome2.lockbox_score).abs() < 1e-9);
}

#[test]
fn test_evidence_d_csv_header_whitespace_resilience() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("whitespace_data.csv");
    // Header with intentional whitespace around column names
    let csv = "  x1  ,  x2  ,  target  \n0.1,0.2,1.5\n0.3,0.4,2.5\n0.5,0.6,3.5\n0.7,0.8,4.5\n0.9,1.0,5.5\n1.1,1.2,6.5\n";
    fs::write(&dataset_path, csv).unwrap();

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_whitespace_test".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.25),
        seed: Some(2026),
        max_turns: Some(2),
    };

    let outcome = run_autonomous_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(outcome.project_id, "proj_whitespace_test");
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.lockbox_score.is_finite());
}

#[test]
fn test_evidence_d_action_parity_embedded_assets_json() {
    use mlrescurix_harness::assets::ACTIONS_V3_JSON;
    let schema: serde_json::Value = serde_json::from_str(ACTIONS_V3_JSON).expect("ACTIONS_V3_JSON must be valid JSON");
    assert_eq!(schema["schema_version"], "mlrescurix.actions.v1");

    let actions = schema["actions"].as_array().expect("actions must be an array");
    assert_eq!(actions.len(), 10, "exactly 10 actions must be defined");

    let action_names: std::collections::HashSet<&str> = actions
        .iter()
        .filter_map(|a| a["name"].as_str())
        .collect();

    let expected: std::collections::HashSet<&str> = [
        "HYPOTHESIS_TRANSFORM",
        "EVALUATE_PORTFOLIO",
        "BLEND_AND_AUDIT",
        "RESEARCH_TOPIC",
        "WRITE_PYTHON",
        "RUN_PYTHON",
        "ANALYZE",
        "HANDOFF",
        "STOP_EARLY",
        "FINALIZE_AND_STOP",
    ].into_iter().collect();

    assert_eq!(action_names, expected);
    for a in actions {
        assert_eq!(a["implemented"], true);
    }
}

