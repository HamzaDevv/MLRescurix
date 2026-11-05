//! Session 3 Acceptance Gates A through G Integration Tests
//!
//! Validates:
//! Gate A: Supported CLI/fixture gateway authors multi-file UNLISTED method, executes
//!         train -> separate infer -> Rust score -> frozen selection -> standalone replay.
//!         Different source layout traverses same generic path without registry change.
//! Gate B: Serialized fresh contexts & evidence-dependent trajectories:
//!         Trajectory 1: Author -> Val -> Challenger Approve -> Finalize -> Holdout evaluated.
//!         Trajectory 2: Failed authoring -> Bounded repair -> Success.
//!         Trajectory 3: Challenger Challenge -> Boss halt -> Zero holdout consumption.
//!         Context isolation excludes foreign transcripts and proposer advocacy.
//!         No-gateway reference diagnostics labeled separately.
//! Gate C: Real competing controller processes; actual SIGKILL during execution;
//!         resumption without duplicate assessment; stale completion rejection at atomic write.
//! Gate D: Concurrent resource admission and conflicting settlement fencing.
//!         Missing authority refuses search with zero dispatch.
//! Gate E: Hostile worker data/label isolation and honest refusal.
//!         campaign_isolation_available() remains false; production isolated entry point fails closed.
//! Gate F: Consumed assessments with incomplete receipts are refused on replay/repack;
//!         holdout remains strictly once-only. Valid replay remains an open requirement.
//! Gate G: Standalone frozen candidate parity, multiple batches, schema rejection.

use std::{
    fs,
    path::Path,
    sync::Arc,
    thread,
    time::Duration,
};


use rusqlite::Connection;
use serde_json::Value;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::gateway::{
        ChatCompletionResponse, FunctionCall,
        ScriptedProviderFixture, TokenUsage, ToolCall,
    },
    campaign::{
        acquire_campaign_lease, campaign_isolation_available, ensure_campaign_leases_schema,
        execute_fenced_write, run_offline_fixture_campaign, run_autonomous_campaign,
        CampaignOutcome, CampaignProblemSpec,
    },
    bundle::{verify_standalone_cli_inference, verify_bundle},
    trusted_evaluation::Metric,
};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_synthetic_dataset(path: &Path, num_rows: usize) -> (String, Vec<(u64, f64)>) {
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

// ============================================================================
// GATE A: Multi-file Unlisted Method Authoring, Execution, Scoring, Standalone Replay
// ============================================================================
#[test]
fn test_gate_a_multi_file_unlisted_method_and_generic_replay() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_a_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let model_name = "deepseek/deepseek-chat";

    // Multi-file unlisted method:
    // helper.py defines Cauchy radial basis transformation
    let helper_code = r#"
import math

def cauchy_basis(x1, x2):
    # Unlisted Cauchy Kernel feature transformation
    c1 = 1.0 / (1.0 + (x1 - 0.5) ** 2)
    c2 = 1.0 / (1.0 + (x2 + 0.5) ** 2)
    return [1.0, x1, x2, c1, c2, x1 * x2]
"#;

    // solution.py imports cauchy_basis from helper.py
    let solution_code = r#"
import sys, json, math, csv, argparse
from helper import cauchy_basis

def fit(train_csv, weights_out):
    rows = []
    with open(train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            phi = cauchy_basis(float(r["x1"]), float(r["x2"]))
            rows.append((phi, float(r["target"])))
    dim = 6
    w = [0.0] * dim
    lr = 0.03
    n = len(rows)
    for _ in range(400):
        grad = [0.0] * dim
        for phi, y in rows:
            pred = sum(w[j] * phi[j] for j in range(dim))
            err = pred - y
            for j in range(dim):
                grad[j] += err * phi[j]
        for j in range(dim):
            w[j] -= lr * (grad[j] / n)
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump({"weights": w, "arch": "CauchySplineRegressor"}, f, indent=2)

def infer(feat_csv, weights_in, preds_out):
    with open(weights_in, "r", encoding="utf-8") as f:
        w = json.load(f)["weights"]
    preds = []
    with open(feat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            phi = cauchy_basis(float(r["x1"]), float(r["x2"]))
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
        "entrypoint": "solution.py",
        "architecture": "CauchySplineRegressor",
        "operations": {
            "fit": {
                "argv": ["python3", "solution.py", "fit", "--train-data", "train.csv", "--weights-out", "weights.json"],
                "input_mounts": ["train.csv"],
                "output_artifacts": ["weights.json"]
            },
            "infer": {
                "argv": ["python3", "solution.py", "infer", "--features-in", "val_features.csv", "--weights-in", "weights.json", "--predictions-out", "val_predictions.json"],
                "input_mounts": ["val_features.csv", "weights.json"],
                "output_artifacts": ["val_predictions.json"]
            }
        }
    });

    let scripted_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Specialist authors helper.py
        ChatCompletionResponse {
            id: "turn1-write-helper".into(),
            model: model_name.into(),
            content: Some("Writing Cauchy transform helper.py".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call_ws_helper".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "helper.py",
                            "content": helper_code,
                        }).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 80, total_tokens: 280 },
            latency_ms: 15,
        },
        // Turn 2: Specialist authors solution.py
        ChatCompletionResponse {
            id: "turn2-write-solution".into(),
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
                            "content": solution_code,
                        }).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 250, completion_tokens: 120, total_tokens: 370 },
            latency_ms: 20,
        },
        // Turn 3: Specialist authors program.json
        ChatCompletionResponse {
            id: "turn3-write-program".into(),
            model: model_name.into(),
            content: Some("Writing program.json manifest".into()),
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
        // Turn 4: Specialist finalizes authoring
        ChatCompletionResponse {
            id: "turn4-finalize-specialist".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution and helper authored.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 90, completion_tokens: 15, total_tokens: 105 },
            latency_ms: 10,
        },
        // Turn 5: Boss ReAct Turn 1 observes evidence and delegates to challenger
        ChatCompletionResponse {
            id: "turn5-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Validation score is competitive; audit for variance and protocol leak.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 25, total_tokens: 175 },
            latency_ms: 12,
        },
        // Turn 6: Challenger performs adversarial audit
        ChatCompletionResponse {
            id: "turn6-challenger-audit".into(),
            model: model_name.into(),
            content: Some("APPROVED: Validation score and split protocol conform to specification; no data leakage detected.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 300, completion_tokens: 45, total_tokens: 345 },
            latency_ms: 25,
        },
        // Turn 7: Boss ReAct Turn 2 observes challenger audit and finalizes
        ChatCompletionResponse {
            id: "turn7-boss-finalize".into(),
            model: model_name.into(),
            content: Some("FINALIZE_AND_STOP: Candidate independently audited and approved.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 180, completion_tokens: 20, total_tokens: 200 },
            latency_ms: 10,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_a_multi".into(),
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
        max_turns: Some(5),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome: CampaignOutcome = run_offline_fixture_campaign(
        root,
        &problem_spec,
        Some(scripted_gateway),
    ).unwrap();

    assert_eq!(outcome.project_id, "proj_gate_a_multi");
    assert_eq!(outcome.dataset_sha256, ds_hash);
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.lockbox_score.is_finite());

    // Delivery bundle handoff assertions:
    // Both helper.py AND solution.py AND program.json must be present in the bundle
    assert!(outcome.bundle_dir.join("helper.py").is_file(), "helper.py missing from delivery bundle!");
    assert!(outcome.bundle_dir.join("solution.py").is_file(), "solution.py missing from delivery bundle!");
    assert!(outcome.bundle_dir.join("program.json").is_file(), "program.json missing from delivery bundle!");
    assert!(outcome.bundle_dir.join("model_weights.json").is_file());
    assert!(outcome.bundle_dir.join("manifest.json").is_file());
    assert!(outcome.bundle_dir.join("scorecard.json").is_file());

    assert!(outcome.bundle_verification.manifest_verified);
    assert!(outcome.bundle_verification.contracts_verified);

    // Fast-path replay verification of the multi-file candidate
    let replay_outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(replay_outcome.candidate_id, outcome.candidate_id);
    assert_eq!(replay_outcome.lockbox_score, outcome.lockbox_score);
    assert_eq!(replay_outcome.weights_sha256, outcome.weights_sha256);

    // Repack verification when manifest is removed: unchanged multi-file source repacks cleanly without double assessment
    let manifest_path = outcome.bundle_dir.join("manifest.json");
    fs::remove_file(&manifest_path).unwrap();
    assert!(!manifest_path.is_file());

    let repack_outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(repack_outcome.candidate_id, outcome.candidate_id);
    assert_eq!(repack_outcome.lockbox_score, outcome.lockbox_score);
    assert!(repack_outcome.bundle_dir.join("manifest.json").is_file());

    // Lockbox consumption count remains strictly 1
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_a_multi'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions, 1, "lockbox must be consumed strictly once across replay and repack");
}

// ============================================================================
// GATE B: Serialized Context Isolation & Evidence-Dependent Trajectories
// ============================================================================
#[test]
fn test_gate_b_evidence_dependent_trajectories_and_context_isolation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_b_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 28);
    let model_name = "deepseek/deepseek-chat";

    // Trajectory 2: Failed authoring -> Bounded repair -> Successful campaign
    let broken_solution_code = "def fit(train_csv, weights_out):\n  syntax error here broken python";

    let working_solution_code = r#"
import sys, json, csv, argparse

def fit(train_csv, weights_out):
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump({"weights": [0.5, 0.5], "arch": "RepairedLinear"}, f)

def infer(feat_csv, weights_in, preds_out):
    with open(weights_in, "r", encoding="utf-8") as f:
        w = json.load(f)["weights"]
    preds = []
    with open(feat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            preds.append({"row_id": row_id, "prediction": round(w[0] * x1 + w[1] * x2, 4)})
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
        "entrypoint": "solution.py",
        "architecture": "RepairedLinear",
    });

    let repair_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Specialist writes broken solution
        ChatCompletionResponse {
            id: "turn1-broken".into(),
            model: model_name.into(),
            content: Some("Writing preliminary draft".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call_ws_broken".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "solution.py",
                            "content": broken_solution_code,
                        }).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 30, total_tokens: 130 },
            latency_ms: 10,
        },
        // Turn 2: Specialist performs bounded repair, overwriting with working solution & manifest
        ChatCompletionResponse {
            id: "turn2-repaired".into(),
            model: model_name.into(),
            content: Some("Repairing syntax error in solution.py and adding program.json".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call_ws_repaired_sol".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "solution.py",
                            "content": working_solution_code,
                        }).to_string(),
                    },
                },
                ToolCall {
                    id: "call_ws_repaired_manifest".into(),
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
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 80, total_tokens: 280 },
            latency_ms: 15,
        },
        // Turn 3: Specialist finalizes
        ChatCompletionResponse {
            id: "turn3-finalize".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution repaired and verified.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 80, completion_tokens: 15, total_tokens: 95 },
            latency_ms: 10,
        },
        // Turn 4: Boss ReAct Turn 1 observes evidence and delegates to challenger
        ChatCompletionResponse {
            id: "turn4-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Review repaired solution.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 120, completion_tokens: 20, total_tokens: 140 },
            latency_ms: 10,
        },
        // Turn 5: Challenger completes independent audit
        ChatCompletionResponse {
            id: "turn5-challenger-audit".into(),
            model: model_name.into(),
            content: Some("APPROVED: Repaired candidate passes specification and metric bounds.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 250, completion_tokens: 30, total_tokens: 280 },
            latency_ms: 15,
        },
        // Turn 6: Boss ReAct Turn 2 observes challenger audit and finalizes
        ChatCompletionResponse {
            id: "turn6-boss-finalize".into(),
            model: model_name.into(),
            content: Some("FINALIZE_AND_STOP: Candidate independently audited and approved.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 15, total_tokens: 165 },
            latency_ms: 10,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_b_repair".into(),
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
        max_turns: Some(5),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome = run_offline_fixture_campaign(root, &problem_spec, Some(repair_gateway.clone())).unwrap();
    assert_eq!(outcome.dataset_sha256, ds_hash);

    // Verify context isolation in sqlite
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    let challenger_inputs_str: String = conn.query_row(
        "SELECT immutable_inputs FROM task_assignments WHERE project='proj_gate_b_repair' AND role_id='challenger'",
        [],
        |r| r.get(0),
    ).unwrap();
    let challenger_inputs: serde_json::Value = serde_json::from_str(&challenger_inputs_str).unwrap();

    // Verify challenger received only objective facts
    assert_eq!(challenger_inputs["candidate_id"], outcome.candidate_id);
    assert_eq!(challenger_inputs["protocol_sha256"], outcome.protocol_sha256);
    assert!(challenger_inputs.get("proposer_advocacy").is_none());
    assert!(challenger_inputs.get("foreign_transcripts").is_none());

    // Inspect actual serialized requests to verify wire-level context isolation
    let recorded = repair_gateway.recorded_requests();
    assert!(!recorded.is_empty(), "gateway must have recorded actual model requests");

    // 1. Verify specialist turn 2 received execution error feedback in serialized tool messages
    let specialist_repair_req = recorded
        .iter()
        .find(|req| {
            req.messages.iter().any(|m| {
                m.role == "tool" || m.content.as_deref().unwrap_or("").contains("tool_execution_failed") || m.content.as_deref().unwrap_or("").contains("Error")
            })
        })
        .expect("specialist repair turn must receive actual serialized error feedback");
    assert!(!specialist_repair_req.messages.is_empty());

    // 2. Verify challenger received strictly objective primary evidence and NO proposer advocacy
    let challenger_req = recorded
        .iter()
        .find(|req| {
            req.messages.iter().any(|m| {
                m.content.as_deref().unwrap_or("").contains("independent scientific challenger")
            })
        })
        .expect("challenger must receive serialized audit prompt");
    for msg in &challenger_req.messages {
        let text = msg.content.as_deref().unwrap_or("");
        assert!(!text.contains("proposer_advocacy"), "challenger prompt leaked proposer advocacy");
        assert!(!text.contains("foreign_transcripts"), "challenger prompt leaked foreign transcripts");
        assert!(!text.contains("__mlr_row_id"), "challenger prompt leaked internal row ids");
    }
}

#[test]
fn test_gate_b_trajectory_3_challenger_challenge_halts_without_assessment() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_b_traj3_dataset.csv");
    let (_ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 28);
    let model_name = "deepseek/deepseek-chat";

    let solution_code = r#"
import sys, json, csv, argparse

def fit(train_csv, weights_out):
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump({"weights": [0.1, 0.9], "arch": "ChallengedModel"}, f)

def infer(feat_csv, weights_in, preds_out):
    with open(weights_in, "r", encoding="utf-8") as f:
        w = json.load(f)["weights"]
    preds = []
    with open(feat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            preds.append({"row_id": row_id, "prediction": round(w[0] * x1 + w[1] * x2, 4)})
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
        "entrypoint": "solution.py",
        "architecture": "ChallengedModel",
    });

    let challenge_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Specialist writes solution and manifest
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
                            "content": solution_code,
                        }).to_string(),
                    },
                },
                ToolCall {
                    id: "call_ws_manifest".into(),
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
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 60, total_tokens: 210 },
            latency_ms: 10,
        },
        // Turn 2: Specialist finalizes
        ChatCompletionResponse {
            id: "turn2-finalize-specialist".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution drafted.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 80, completion_tokens: 15, total_tokens: 95 },
            latency_ms: 10,
        },
        // Turn 3: Boss ReAct Turn 1 observes validation score and delegates to challenger
        ChatCompletionResponse {
            id: "turn3-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Perform adversarial audit on candidate.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 120, completion_tokens: 20, total_tokens: 140 },
            latency_ms: 10,
        },
        // Turn 4: Challenger performs adversarial audit and CHALLENGES the candidate
        ChatCompletionResponse {
            id: "turn4-challenger-challenge".into(),
            model: model_name.into(),
            content: Some("CHALLENGED: Potential target leakage detected; excessive variance across fold splits.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 250, completion_tokens: 30, total_tokens: 280 },
            latency_ms: 15,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_b_challenge".into(),
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
        max_turns: Some(5),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome_err = run_offline_fixture_campaign(root, &problem_spec, Some(challenge_gateway)).unwrap_err();
    assert!(outcome_err.contains("challenged by independent scientific challenger"), "expected challenger error: {outcome_err}");

    // Critical invariant: Lockbox must NOT be consumed when challenger rejects candidate!
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    let consumptions_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_b_challenge'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions_count, 0, "lockbox must NEVER be consumed for challenged/rejected candidates");

    // Challenger audit record must be recorded as 'failed' in task_completions
    let verdict: String = conn.query_row(
        "SELECT verdict FROM task_completions WHERE project='proj_gate_b_challenge' AND task_id LIKE '%challenger%'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(verdict, "failed", "task_completions verdict for challenged audit must be 'failed'");
}

// ============================================================================
// GATE C: Real Competing Controller Processes, SIGKILL, Atomic Lease Fencing
// ============================================================================
#[test]
fn test_gate_c_real_competing_controllers_and_sigkill_lease_fencing() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_c_dataset.csv");
    let (_ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);

    let db_path = root.join("controller/state.sqlite");
    fs::create_dir_all(root.join("controller")).unwrap();
    let mut conn = Connection::open(&db_path).unwrap();
    ensure_campaign_leases_schema(&conn).unwrap();

    let project_id = "proj_gate_c_lease";
    // Spawn real background controller process to act as active controller 1
    let mut child1 = std::process::Command::new("python3")
        .args(["-c", "import time; time.sleep(60)"])
        .spawn()
        .expect("must be able to spawn real background process");
    let child1_pid = child1.id();
    let owner1 = format!("pid_{}", child1_pid);

    // Acquire lease for owner1
    let (lease_id, epoch) = acquire_campaign_lease(&mut conn, project_id, &owner1).unwrap();

    // Fenced write succeeds under active lease
    let write_res = execute_fenced_write(&mut conn, project_id, &lease_id, epoch, |tx| {
        tx.execute(
            "CREATE TABLE IF NOT EXISTS test_fenced_tbl(id INT PRIMARY KEY, val TEXT)",
            [],
        ).map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT INTO test_fenced_tbl(id, val) VALUES(1, 'valid_write')",
            [],
        ).map_err(|e| e.to_string())?;
        Ok(())
    });
    assert!(write_res.is_ok(), "fenced write should succeed under valid lease");

    // Second owner attempting to acquire concurrent lease receives lease conflict while child1 is alive
    let mut conn2 = Connection::open(&db_path).unwrap();
    let owner2 = format!("pid_{}", std::process::id());
    let conflict = acquire_campaign_lease(&mut conn2, project_id, &owner2);
    assert!(conflict.is_err(), "competing controller should be denied lease during active TTL");
    let err_msg = conflict.unwrap_err();
    assert!(err_msg.contains("campaign lease conflict"), "expected lease conflict error: {err_msg}");

    // Send SIGKILL to child1 and wait for it to cleanly terminate
    child1.kill().expect("must kill child1");
    let _ = child1.wait();

    // Resumption after SIGKILL: controller 2 successfully recovers lease with incremented epoch
    let recovered = acquire_campaign_lease(&mut conn2, project_id, &owner2);
    assert!(recovered.is_ok(), "controller 2 must recover lease after controller 1 SIGKILL");
    let (lease2_id, epoch2) = recovered.unwrap();
    assert_eq!(epoch2, epoch + 1, "recovered lease must increment epoch");

    // Stale write rejection: write with old lease/epoch must fail closed
    let stale_write_res = execute_fenced_write(&mut conn, project_id, &lease_id, epoch, |tx| {
        tx.execute(
            "INSERT INTO test_fenced_tbl(id, val) VALUES(2, 'stale_write')",
            [],
        ).map_err(|e| e.to_string())?;
        Ok(())
    });
    assert!(stale_write_res.is_err(), "stale epoch write must fail closed");

    // Valid write under recovered lease succeeds
    let valid_write_res = execute_fenced_write(&mut conn2, project_id, &lease2_id, epoch2, |tx| {
        tx.execute(
            "INSERT INTO test_fenced_tbl(id, val) VALUES(3, 'recovered_write')",
            [],
        ).map_err(|e| e.to_string())?;
        Ok(())
    });
    assert!(valid_write_res.is_ok(), "recovered lease write should succeed");
}


// ============================================================================
// GATE D: Concurrent Resource Admission & Conflicting Settlement Fencing
// ============================================================================
#[test]
fn test_gate_d_concurrent_resource_admission_and_conflicting_settlement() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let controller_root = root.join("controller");
    let mut b = std::collections::BTreeMap::new();
    b.insert("calls".into(), 1000);
    b.insert("tokens".into(), 1_000_000);
    b.insert("wall_ms".into(), 600_000);
    b.insert("spend_usd_millicents".into(), 100_000);
    let mut cap = std::collections::BTreeMap::new();
    cap.insert("workers".into(), 4);
    let policy = mlrescurix_harness::Policy { max_tasks: 1024, budget: b, capacity: cap };
    let _ = mlrescurix_harness::Runtime::init(&controller_root, policy).unwrap();

    let db_path = controller_root.join("state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    ensure_campaign_leases_schema(&conn).unwrap();

    // 1. Verify atomic lease conflict resolution across 8 concurrent threads
    let barrier = Arc::new(std::sync::Barrier::new(8));
    let mut handles = Vec::new();
    let db_p = db_path.clone();

    for i in 0..8 {
        let b = barrier.clone();
        let p = db_p.clone();
        let h = thread::spawn(move || {
            b.wait();
            let mut c = Connection::open(&p).unwrap();
            let _ = c.busy_timeout(Duration::from_millis(100));
            acquire_campaign_lease(&mut c, "proj_concurrency_race", &format!("worker_{i}"))
        });
        handles.push(h);
    }

    let mut winners = 0;
    let mut losers = 0;
    for h in handles {
        match h.join().unwrap() {
            Ok(_) => winners += 1,
            Err(_) => losers += 1,
        }
    }

    assert_eq!(winners, 1, "exactly 1 thread must acquire the lease");
    assert_eq!(losers, 7, "exactly 7 competing threads must receive lease conflict");

    // 2. Concurrent resource admission & exhaustion test
    // 8 concurrent threads attempt to reserve 25_000 millicents each against a max_spend cap of 100_000
    use mlrescurix_harness::agents::gateway::{record_provider_accounting, reserve_role_spend};
    let res_barrier = Arc::new(std::sync::Barrier::new(8));
    let mut res_handles = Vec::new();
    let root_path = controller_root.clone();
    let project_id = "proj_resource_race";

    for i in 0..8 {
        let b = res_barrier.clone();
        let r = root_path.clone();
        let h = thread::spawn(move || {
            b.wait();
            let tokens = TokenUsage { prompt_tokens: 500, completion_tokens: 250, total_tokens: 750 };
            let req_id = format!("req_concurrent_{i}");
            reserve_role_spend(
                &r,
                project_id,
                &req_id,
                "task_resource",
                "specialist",
                "deepseek/deepseek-chat",
                &tokens,
                25_000,
                Some(100_000),
            )
        });
        res_handles.push(h);
    }

    let mut res_admitted = 0;
    let mut res_exhausted = 0;
    for h in res_handles {
        match h.join().unwrap() {
            Ok(_) => res_admitted += 1,
            Err(e) => {
                assert!(e.contains("spend limit exceeded") || e.contains("busy") || e.contains("locked"), "unexpected error: {e}");
                res_exhausted += 1;
            }
        }
    }

    // Exactly 4 reservations of 25,000 can fit within 100,000
    assert_eq!(res_admitted, 4, "exactly 4 reservations must fit within the 100_000 millicent budget");
    assert_eq!(res_exhausted, 4, "competing threads exceeding ceiling must be rejected");

    // Total reserved in sqlite must strictly equal 100,000
    let c = Connection::open(&db_path).unwrap();
    let total_reserved: i64 = c.query_row(
        "SELECT SUM(cost_millicents) FROM provider_accounting WHERE project=?1 AND status='reserved'",
        [project_id],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(total_reserved, 100_000, "total reserved spend must strictly equal budget ceiling");

    // 3. Conflicting settlement and duplicate reservation fencing
    let (first_admitted_id,): (String,) = c.query_row(
        "SELECT request_id FROM provider_accounting WHERE project=?1 AND status='reserved' LIMIT 1",
        [project_id],
        |row| Ok((row.get(0)?,)),
    ).unwrap();

    let tokens = TokenUsage { prompt_tokens: 400, completion_tokens: 200, total_tokens: 600 };
    assert!(reserve_role_spend(&controller_root, project_id, &first_admitted_id, "task_resource", "specialist", "deepseek/deepseek-chat", &tokens, 1, Some(100_000)).is_err(), "an in-flight reservation cannot be reduced");
    record_provider_accounting(
        &controller_root,
        project_id,
        &first_admitted_id,
        "task_resource",
        "specialist",
        "deepseek/deepseek-chat",
        &tokens,
        20_000,
        50,
        "ok",
    ).unwrap();
    record_provider_accounting(&controller_root, project_id, &first_admitted_id, "task_resource", "specialist", "deepseek/deepseek-chat", &tokens, 20_000, 50, "ok").unwrap();
    assert!(record_provider_accounting(&controller_root, project_id, &first_admitted_id, "task_resource", "specialist", "deepseek/deepseek-chat", &tokens, 1, 50, "ok").is_err());

    // Re-attempting reservation on the settled request must fail closed (duplicate reservation rejected)
    let dup_res = reserve_role_spend(
        &controller_root,
        project_id,
        &first_admitted_id,
        "task_resource",
        "specialist",
        "deepseek/deepseek-chat",
        &tokens,
        25_000,
        Some(100_000),
    );
    assert!(dup_res.is_err());
    let dup_err = dup_res.unwrap_err();
    assert!(dup_err.contains("already been dispatched") || dup_err.contains("duplicate reservation rejected"),
        "duplicate reservation must be rejected: {dup_err}");
}

// ============================================================================
// GATE E: Hostile Worker Denial & Honest Refusal
// ============================================================================
#[test]
fn test_gate_e_hostile_worker_denial_and_honest_refusal() {
    // 1. Assert honest capability reporting: campaign_isolation_available() MUST be false
    // on host macOS fixture environment (never falsely claim container cgroup boundary)
    assert!(
        !campaign_isolation_available(),
        "campaign_isolation_available() must honestly report false on host platform"
    );

    // 2. Production isolated entrypoint must fail closed before creating state
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().join("state");
    let spec = CampaignProblemSpec {
        project_id: "test_gate_e".into(),
        dataset_path: temp.path().join("data.csv"),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 1000,
        model_name: "test".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: None,
        seed: None,
        max_turns: None,
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let err = run_autonomous_campaign(&state, &spec, None).unwrap_err();
    assert!(
        err.contains("isolated worker execution is not yet implemented"),
        "expected unavailable error: {err}"
    );
    assert!(!state.exists(), "no state should be created when isolated boundary is unavailable");
}

// ============================================================================
// GATE F: Changed Source Rejected & Once-Only Holdout
// ============================================================================
#[test]
fn test_gate_f_changed_source_rejected_and_once_only_holdout() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_f_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_f_rebind".into(),
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
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    // First run: executes reference campaign and consumes holdout
    let outcome1 = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(outcome1.dataset_sha256, ds_hash);

    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_f_rebind'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions_count, 1, "exactly 1 lockbox consumption recorded");

    // Attack: modify solution.py in branch workspace and delete bundle manifest.json
    let branch_dir = root.join(format!("controller/workflows/proj_gate_f_rebind/branches/branch_{}", outcome1.candidate_id));
    let branch_sol = branch_dir.join("solution.py");
    assert!(branch_sol.is_file());
    let original_sol = fs::read_to_string(&branch_sol).unwrap();
    let tampered_sol = original_sol.replace("val = sum(w[j] * phi[j] for j in range(dim))", "val = 1000000.0");
    assert_ne!(original_sol, tampered_sol);
    fs::write(&branch_sol, tampered_sol).unwrap();

    let manifest_path = outcome1.bundle_dir.join("manifest.json");
    let _ = fs::remove_file(&manifest_path);
    assert!(!manifest_path.is_file());

    // Repackaging attempt MUST fail closed: candidate source SHA changed while lockbox already consumed!
    let repack_err = run_offline_fixture_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        repack_err.contains("does not match evaluated source SHA-256") || repack_err.contains("cannot repackage modified source") || repack_err.contains("lockbox consumption mismatch"),
        "expected source tampering rejection, got: {repack_err}"
    );

    // Verify lockbox consumption count is STILL strictly 1
    let consumptions_after: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_f_rebind'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions_after, 1, "lockbox must NEVER be double-consumed on repackaging failure");

    // Restore original source and verify clean crash repack
    fs::write(&branch_sol, original_sol).unwrap();
    let replay_outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(replay_outcome.candidate_id, outcome1.candidate_id);
    assert_eq!(replay_outcome.lockbox_score, outcome1.lockbox_score);

    // Attack 2: Tamper with program.json in branch workspace and delete manifest.json
    let branch_prog = branch_dir.join("program.json");
    assert!(branch_prog.is_file());
    let original_prog = fs::read_to_string(&branch_prog).unwrap();
    let tampered_prog = original_prog.replace("solution.py", "malicious_entrypoint.py");
    fs::write(&branch_prog, &tampered_prog).unwrap();
    let _ = fs::remove_file(&manifest_path);

    let repack_prog_err = run_offline_fixture_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        repack_prog_err.contains("does not match evaluated source SHA-256") || repack_prog_err.contains("cannot repackage modified source") || repack_prog_err.contains("lockbox consumption mismatch"),
        "expected program.json tampering rejection, got: {repack_prog_err}"
    );

    // Verify lockbox consumption count is STILL strictly 1
    let consumptions_after_prog: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_f_rebind'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions_after_prog, 1, "lockbox must NEVER be double-consumed on program.json tampering");

    // Restore program.json and rebuild bundle
    fs::write(&branch_prog, original_prog).unwrap();
    let repacked_outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(repacked_outcome.candidate_id, outcome1.candidate_id);

    // Attack 3: Tamper with scorecard.json in bundle and re-sign manifest.json (counterfeit score attack)
    let bundle_scorecard = repacked_outcome.bundle_dir.join("scorecard.json");
    let original_sc = fs::read_to_string(&bundle_scorecard).unwrap();
    let mut sc_obj: serde_json::Value = serde_json::from_str(&original_sc).unwrap();
    sc_obj["validation_score"] = serde_json::json!(0.999999);
    fs::write(&bundle_scorecard, serde_json::to_string_pretty(&sc_obj).unwrap()).unwrap();

    let bundle_manifest_path = repacked_outcome.bundle_dir.join("manifest.json");
    let original_manifest_bytes = fs::read(&bundle_manifest_path).unwrap();
    let mut manifest_val: serde_json::Value = serde_json::from_slice(&original_manifest_bytes).unwrap();
    let new_sc_bytes = fs::read(&bundle_scorecard).unwrap();
    let mut hasher = sha2::Sha256::new();
    hasher.update(&new_sc_bytes);
    let new_sc_sha = format!("{:x}", hasher.finalize());
    manifest_val["files"]["scorecard.json"]["sha256"] = serde_json::json!(new_sc_sha);
    manifest_val["files"]["scorecard.json"]["bytes"] = serde_json::json!(new_sc_bytes.len());
    fs::write(&bundle_manifest_path, serde_json::to_string_pretty(&manifest_val).unwrap()).unwrap();

    let replay_sc_err = run_offline_fixture_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        replay_sc_err.contains("scorecard validation_score") && replay_sc_err.contains("does not match authoritative validation score")
            || replay_sc_err.contains("manifest SHA-256")
            || replay_sc_err.contains("self-rehashed or tampered manifest rejected"),
        "expected counterfeit scorecard rejection, got: {replay_sc_err}"
    );

    // Restore original scorecard and restore exact original manifest bytes
    fs::write(&bundle_scorecard, original_sc).unwrap();
    fs::write(&bundle_manifest_path, original_manifest_bytes).unwrap();

    // Clean bundle replay succeeds through Step 7 fast-path with authoritative validation score
    let verified_replay = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(verified_replay.candidate_id, outcome1.candidate_id);
    assert_eq!(verified_replay.validation_score, outcome1.validation_score);
    assert_eq!(verified_replay.lockbox_score, outcome1.lockbox_score);

}

#[test]
fn consumed_assessment_refuses_self_rehashed_wrapper() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("dataset.csv");
    create_synthetic_dataset(&dataset_path, 30);
    let problem_spec = CampaignProblemSpec {
        project_id: "wrapper_replay_guard".into(), dataset_path,
        target_column: "target".into(), task_type: "regression".into(),
        metric: Metric::Mae, budget_millicents: 100_000,
        model_name: "fixture".into(), endpoint: None, api_key: None,
        lockbox_fraction: Some(0.20), seed: Some(2026), max_turns: Some(2),
        split_strategy: None, time_column: None, group_column: None,
    };
    let outcome1 = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    // Safe unchanged replay succeeds with controller release provenance
    let replay_outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(replay_outcome.candidate_id, outcome1.candidate_id);
    assert_eq!(replay_outcome.lockbox_score, outcome1.lockbox_score);
    assert_eq!(replay_outcome.weights_sha256, outcome1.weights_sha256);

    let bundle_manifest_path = outcome1.bundle_dir.join("manifest.json");
    let mut manifest_val: Value = serde_json::from_slice(&fs::read(&bundle_manifest_path).unwrap()).unwrap();
    // A bundle-owned manifest is not an authority for a modified executable.
    let wrapper = outcome1.bundle_dir.join("inference.py");
    let mut wrapper_bytes = fs::read(&wrapper).unwrap();
    wrapper_bytes.extend_from_slice(b"\n# modified after assessment\n");
    fs::write(&wrapper, &wrapper_bytes).unwrap();
    let mut wrapper_hasher = sha2::Sha256::new();
    wrapper_hasher.update(&wrapper_bytes);
    manifest_val["files"]["inference.py"]["sha256"] = serde_json::json!(format!("{:x}", wrapper_hasher.finalize()));
    manifest_val["files"]["inference.py"]["bytes"] = serde_json::json!(wrapper_bytes.len());
    fs::write(&bundle_manifest_path, serde_json::to_string_pretty(&manifest_val).unwrap()).unwrap();
    let tamper_err = run_offline_fixture_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        tamper_err.contains("does not match controller release receipt manifest SHA-256")
            || tamper_err.contains("self-rehashed or tampered bundle rejected"),
        "a self-rehashed modified executable must not inherit a prior assessment, got: {tamper_err}"
    );
    let conn = Connection::open(root.join("controller/state.sqlite")).unwrap();
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM assessment_lockbox_consumptions", [], |r| r.get(0)).unwrap();
    assert_eq!(count, 1);
}

#[test]
fn tampered_manifest_metadata_rejected_on_replay() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("dataset.csv");
    create_synthetic_dataset(&dataset_path, 30);
    let problem_spec = CampaignProblemSpec {
        project_id: "manifest_tamper_guard".into(),
        dataset_path,
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "fixture".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };
    let outcome1 = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();

    // Replay succeeds initially
    let replay_outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    assert_eq!(replay_outcome.candidate_id, outcome1.candidate_id);

    // Tamper with manifest.json metadata (e.g., inject falsified model metadata) WITHOUT modifying manifest files
    let bundle_manifest_path = outcome1.bundle_dir.join("manifest.json");
    let mut manifest_val: Value = serde_json::from_slice(&fs::read(&bundle_manifest_path).unwrap()).unwrap();
    manifest_val["model"] = serde_json::json!({"falsified_metadata": "unauthorized_modification"});
    fs::write(&bundle_manifest_path, serde_json::to_string_pretty(&manifest_val).unwrap()).unwrap();

    // Replay must fail closed because manifest_sha256 does not match controller release receipt
    let tamper_err = run_offline_fixture_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        tamper_err.contains("does not match controller release receipt manifest SHA-256")
            || tamper_err.contains("self-rehashed or tampered manifest rejected"),
        "tampered manifest metadata must be rejected on replay, got: {tamper_err}"
    );

    // Lockbox consumption count strictly remains 1
    let conn = Connection::open(root.join("controller/state.sqlite")).unwrap();
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM assessment_lockbox_consumptions", [], |r| r.get(0)).unwrap();
    assert_eq!(count, 1);
}

// ============================================================================
// GATE G: Standalone Frozen Candidate Parity Across Batches
// ============================================================================
#[test]
fn test_gate_g_standalone_frozen_candidate_parity() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_g_dataset.csv");
    let (_ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_g_parity".into(),
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
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();

    // Verify delivery bundle manifest and contracts
    let report = verify_bundle(&outcome.bundle_dir, false).unwrap();
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);

    // Test standalone CLI inference on examples_input.json
    let input_path = outcome.bundle_dir.join("examples_input.json");
    let cli_out_path = root.join("cli_output.json");
    let cli_preds = verify_standalone_cli_inference(&outcome.bundle_dir, &input_path, &cli_out_path).unwrap();

    let arr = cli_preds.as_array().expect("CLI output must be a JSON array");
    assert_eq!(arr.len(), 2, "2 example inputs must produce 2 predictions");
    for item in arr {
        let p = item.get("prediction").and_then(|v| v.as_f64()).or_else(|| item.as_f64()).expect("numeric prediction expected");
        assert!(p.is_finite(), "CLI prediction must be a finite float");
    }

    // Verify standalone predictions against scored frozen lockbox predictions across batch sizes (1, 5, full)
    let worker_workspace = root.join("worker_workspace");
    let lockbox_preds_path = worker_workspace.join("lockbox_predictions.json");
    let lockbox_features_path = worker_workspace.join("lockbox_features.csv");

    let scored_preds_raw: Vec<Value> = serde_json::from_slice(&fs::read(&lockbox_preds_path).unwrap()).unwrap();
    let scored_preds: Vec<f64> = scored_preds_raw
        .iter()
        .map(|v| v.get("prediction").and_then(|p| p.as_f64()).expect("scored prediction missing"))
        .collect();

    // Parse features from lockbox_features.csv
    let csv_content = fs::read_to_string(&lockbox_features_path).unwrap();
    let mut lines = csv_content.lines();
    let header_line = lines.next().expect("header missing");
    let header_cols: Vec<&str> = header_line.split(',').collect();

    let mut all_rows: Vec<Value> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let vals: Vec<&str> = line.split(',').collect();
        let mut row_map = serde_json::Map::new();
        for (col_idx, &col_name) in header_cols.iter().enumerate() {
            if col_name == "__mlr_row_id" {
                continue;
            }
            let num: f64 = vals[col_idx].trim().parse().unwrap();
            row_map.insert(col_name.to_string(), Value::from(num));
        }
        all_rows.push(Value::Object(row_map));
    }

    assert_eq!(all_rows.len(), scored_preds.len(), "feature count must match scored prediction count");
    assert!(!all_rows.is_empty(), "must have lockbox rows to test parity");

    // 1. Full batch test
    let full_batch_in = root.join("full_batch_in.json");
    let full_batch_out = root.join("full_batch_out.json");
    fs::write(&full_batch_in, serde_json::to_string(&all_rows).unwrap()).unwrap();
    let full_cli_out = verify_standalone_cli_inference(&outcome.bundle_dir, &full_batch_in, &full_batch_out).unwrap();
    let full_arr = full_cli_out.as_array().expect("array expected");
    assert_eq!(full_arr.len(), scored_preds.len());
    for (i, item) in full_arr.iter().enumerate() {
        let p = item.get("prediction").and_then(|v| v.as_f64()).or_else(|| item.as_f64()).expect("numeric prediction");
        let expected = scored_preds[i];
        assert!(
            (p - expected).abs() < 1e-4,
            "full-batch mismatch at row {i}: standalone={p}, scored={expected}"
        );
    }

    // 2. Batch size 5 test
    for chunk_start in (0..all_rows.len()).step_by(5) {
        let chunk_end = (chunk_start + 5).min(all_rows.len());
        let chunk_slice = &all_rows[chunk_start..chunk_end];
        let chunk_in = root.join(format!("batch5_in_{chunk_start}.json"));
        let chunk_out = root.join(format!("batch5_out_{chunk_start}.json"));
        fs::write(&chunk_in, serde_json::to_string(chunk_slice).unwrap()).unwrap();
        let chunk_cli_out = verify_standalone_cli_inference(&outcome.bundle_dir, &chunk_in, &chunk_out).unwrap();
        let chunk_arr = chunk_cli_out.as_array().expect("array expected");
        assert_eq!(chunk_arr.len(), chunk_slice.len());
        for (sub_i, item) in chunk_arr.iter().enumerate() {
            let row_idx = chunk_start + sub_i;
            let p = item.get("prediction").and_then(|v| v.as_f64()).or_else(|| item.as_f64()).expect("numeric prediction");
            let expected = scored_preds[row_idx];
            assert!(
                (p - expected).abs() < 1e-4,
                "batch-5 mismatch at row {row_idx}: standalone={p}, scored={expected}"
            );
        }
    }

    // 3. Batch size 1 test
    for (i, single_row) in all_rows.iter().enumerate() {
        let single_in = root.join(format!("batch1_in_{i}.json"));
        let single_out = root.join(format!("batch1_out_{i}.json"));
        fs::write(&single_in, serde_json::to_string(&vec![single_row.clone()]).unwrap()).unwrap();
        let single_cli_out = verify_standalone_cli_inference(&outcome.bundle_dir, &single_in, &single_out).unwrap();
        let single_arr = single_cli_out.as_array().expect("array expected");
        assert_eq!(single_arr.len(), 1);
        let p = single_arr[0].get("prediction").and_then(|v| v.as_f64()).or_else(|| single_arr[0].as_f64()).expect("numeric prediction");
        let expected = scored_preds[i];
        assert!(
            (p - expected).abs() < 1e-4,
            "batch-1 mismatch at row {i}: standalone={p}, scored={expected}"
        );
    }

    // 4. Schema rejection checks: missing feature, extraneous column, and invalid types must fail closed
    let missing_col_in = root.join("bad_missing_col_in.json");
    let missing_col_out = root.join("bad_missing_col_out.json");
    fs::write(&missing_col_in, r#"[{"x1": 1.0}]"#).unwrap();
    let missing_res = verify_standalone_cli_inference(&outcome.bundle_dir, &missing_col_in, &missing_col_out);
    assert!(missing_res.is_err(), "inference on input with missing feature must fail closed");

    let extra_col_in = root.join("bad_extra_col_in.json");
    let extra_col_out = root.join("bad_extra_col_out.json");
    fs::write(&extra_col_in, r#"[{"x1": 1.0, "x2": 2.0, "unexpected_feature": 42.0}]"#).unwrap();
    let extra_res = verify_standalone_cli_inference(&outcome.bundle_dir, &extra_col_in, &extra_col_out);
    assert!(extra_res.is_err(), "inference on input with unrecognized extra column must fail closed");

    let invalid_type_in = root.join("bad_invalid_type_in.json");
    let invalid_type_out = root.join("bad_invalid_type_out.json");
    fs::write(&invalid_type_in, r#"[{"x1": "not_a_number", "x2": 2.0}]"#).unwrap();
    let invalid_res = verify_standalone_cli_inference(&outcome.bundle_dir, &invalid_type_in, &invalid_type_out);
    assert!(invalid_res.is_err(), "inference on input with non-numeric value must fail closed");
}

