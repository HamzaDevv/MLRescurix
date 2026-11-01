//! Authoritative End-to-End Vertical Slice Test: Governed Local Source-Execution
//!
//! Validates:
//! 1. Controller-owned policy and budget admission (provider pricing namespacing & spend reservation).
//! 2. Authoritative frozen custom protocol with deterministic train/val splits and carved lockbox.
//! 3. Controller-owned snapshot generation with strict physical label isolation (held out from worker).
//! 4. Fresh-context agent program authoring with unfamiliar custom model (Spline Regressor with Cauchy Loss).
//! 5. Process isolation: Execution of agent program via local subprocess invocation (fit and infer phases).
//! 6. Controller-owned independent inference scoring against private held-out labels.
//! 7. Usable model delivery bundle with actual inference parity:
//!    - Standalone CLI inference (`--input` / `--output`),
//!    - Loopback HTTP serving recipe (`/health` and `/predict`),
//!    - Missing / extra column rejection.
//! 8. Authoritative final lockbox evaluation, atomic one-shot consumption, and crash recovery.
//! 9. Adversarial checks: group overlap rejection, temporal lookahead rejection, duplicate spend rejection.

use mlrescurix_harness::{
    agents::{
        gateway::{
            register_model_pricing, register_namespaced_model_pricing, reserve_role_spend,
            resolve_namespaced_model_pricing, ChatCompletionResponse, FunctionCall,
            ScriptedProviderFixture, TokenUsage, ToolCall,
        },
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
        sandbox::{SandboxDetector, SandboxLimits, SandboxRunner, SandboxStatus},
        workspace::BranchWorkspaceManager,
    },
    bundle::{
        verify_bundle, verify_standalone_cli_inference, verify_standalone_serving_recipe,
        BundleManifest, BundleManifestFileEntry,
    },
    contracts::{Context, MetricSpec},
    core::{Operation, Request},
    custom_protocol::{
        self, carve_lockbox, ensure_custom_protocol_schema, freeze,
        get_assessment_consumption_receipt, CustomProtocol,
    },
    custom_snapshot::{evaluate_predictions, prepare_registered, verify, SnapshotIdentity},
    trusted_evaluation::{
        evaluate_with_spec, evaluate_with_spec_and_evidence,
        EvaluationIdentity, Metric, PredictionBatch, ReferenceEvidence,
    },
    Policy, Runtime,
};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path, process::Command, sync::Arc};

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

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> String {
    let bytes = fs::read(path).expect("failed to read file for hash");
    sha256_bytes(&bytes)
}

#[test]
fn governed_local_source_execution_vertical_slice() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // Architectural isolation: Controller root vs Worker workspace
    let controller_root = root.join("controller");
    let worker_workspace = root.join("worker_workspace");
    fs::create_dir_all(&worker_workspace).unwrap();

    let _r = Runtime::init(&controller_root, m1policy(100, 8)).unwrap();
    mlrescurix_harness::execution::migrate(&controller_root).unwrap();

    // =========================================================================
    // Stage 1: Synthetic Dataset & Ground Truth Generation
    // Target y = 2.5 * x1^2 - 1.2 * sin(x2) + 0.5 (non-linear continuous regression)
    // =========================================================================
    let raw_csv = controller_root.join("raw_dataset.csv");
    let mut csv_content = String::from("x1,x2,target\n");
    let total_rows = 30;
    let mut ground_truth_labels = Vec::with_capacity(total_rows);

    for i in 0..total_rows {
        let x1 = (i as f64 - 15.0) / 10.0;
        let x2 = (i as f64) * 0.2;
        let y = 2.5 * x1 * x1 - 1.2 * x2.sin() + 0.5;
        let y_str = format!("{y:.4}");
        let y_val: f64 = y_str.parse().unwrap();
        csv_content.push_str(&format!("{x1:.4},{x2:.4},{y_str}\n"));
        ground_truth_labels.push((i as u64, y_val));
    }
    fs::write(&raw_csv, &csv_content).unwrap();
    let raw_csv_bytes = fs::read(&raw_csv).unwrap();
    let dataset_sha256 = sha256_bytes(&raw_csv_bytes);

    // =========================================================================
    // Stage 2: Controller Budget Admission & Provider Accounting
    // =========================================================================
    let project_id = "governed_project";
    let dataset_id = "ds_spline_cauchy";

    // Initialize controller database with full migrations
    mlrescurix_harness::execution::migrate(&controller_root).unwrap();
    let mut conn = Connection::open(controller_root.join("state.sqlite")).unwrap();
    ensure_custom_protocol_schema(&conn).unwrap();

    // Create M1 Project in runtime
    _r.command(Request {
        command_id: "init_project".into(),
        project_id: project_id.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "Governed local source execution vertical slice".into(),
            finalization_id: format!("final-{project_id}"),
            policy: m1policy(100, 8),
        },
    }).unwrap();

    // Register namespaced model pricing with locked pricing file
    let model_name = "deepseek/deepseek-chat";
    let endpoint = "https://openrouter.ai/api/v1";
    register_namespaced_model_pricing(
        &controller_root,
        endpoint,
        model_name,
        2,
        8,
    ).unwrap();
    register_model_pricing(&controller_root, model_name, 2, 8).unwrap();

    // Authoritative pricing lookup without unnamespaced fallback
    let pricing = resolve_namespaced_model_pricing(&controller_root, endpoint, model_name).unwrap().unwrap();
    assert_eq!(pricing.0, 2);
    assert_eq!(pricing.1, 8);

    // Policy and budget admission: reserve role spend for worker task
    let est_tokens = TokenUsage {
        prompt_tokens: 500,
        completion_tokens: 200,
        total_tokens: 700,
    };
    let est_cost = (est_tokens.prompt_tokens as i64 * 2) + (est_tokens.completion_tokens as i64 * 8); // 2600 millicents ($0.026)
    reserve_role_spend(
        &controller_root,
        project_id,
        "req_worker_01",
        "task_authoring_01",
        "model_specialist",
        model_name,
        &est_tokens,
        est_cost,
        Some(100_000), // $1.00 budget ceiling
    ).unwrap();

    // Create branch and task in M0 state machine for orchestrator
    _r.command(Request {
        command_id: "create_branch_01".into(),
        project_id: project_id.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: "branch_spline_01".into(),
            base_hash: "base-spline".into(),
            code_hash: "code-spline".into(),
        },
    }).unwrap();
    _r.command(Request {
        command_id: "create_task_01".into(),
        project_id: project_id.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: "task_authoring_01".into(),
            branch_id: "branch_spline_01".into(),
            context: Context {
                role_id: "model_specialist".into(),
                role_version: "1".into(),
                prompt_hash: "prompt-spline".into(),
                model: Some(model_name.into()),
                phase: "implementation".into(),
                hypothesis: Some("Spline regressor with Cauchy loss".into()),
                base_hash: "base-spline".into(),
                code_hash: "code-spline".into(),
                evidence: vec![],
                artifacts: vec![],
                pending_questions: vec![],
                calls: vec![],
                jobs: vec![],
                next_action: "implement".into(),
                stop_conditions: vec!["budget".into()],
            },
        },
    }).unwrap();

    // Register dataset in authoritative scientific_datasets
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            project_id,
            dataset_id,
            raw_csv.to_string_lossy(),
            raw_csv_bytes.len() as i64,
            dataset_sha256,
            1000
        ],
    ).unwrap();

    // =========================================================================
    // Stage 3: Split Protocol with Deterministic Lockbox & Freezing
    // Carve 20% lockbox (6 rows); 24 dev rows split into 18 train, 6 validation
    // =========================================================================
    let all_row_ids: Vec<u64> = (0..total_rows as u64).collect();
    let (dev_rows, lockbox_rows) = carve_lockbox(&all_row_ids, 0.20, 2026).unwrap();
    assert_eq!(lockbox_rows.len(), 6);
    assert_eq!(dev_rows.len(), 24);

    let train_rows = &dev_rows[0..16];
    let val_rows = &dev_rows[16..22];
    let test_rows = &dev_rows[22..24];
    let mut protocol = CustomProtocol::single_split(
        dataset_sha256.clone(),
        "target".into(),
        Metric::Mae,
        train_rows.to_vec(),
        val_rows.to_vec(),
        test_rows.to_vec(),
    );
    protocol.lockbox_rows = lockbox_rows.clone();

    let protocol_sha256 = freeze(&mut conn, project_id, dataset_id, protocol).unwrap();

    // =========================================================================
    // Stage 4: Controller-Owned Snapshot Preparation & Strict Label Isolation
    // Worker gets training.csv and validation_features.csv (labels held out in controller root)
    // =========================================================================
    let snapshot_root = controller_root.join("snapshots");
    fs::create_dir_all(&snapshot_root).unwrap();

    // Program source code (agent authored)
    let solution_code = r#"
import sys
import json
import math
import csv

def basis(x1, x2):
    # Non-linear basis: [1, x1, x1^2, x2, sin(x2)]
    return [1.0, x1, x1 * x1, x2, math.sin(x2)]

def fit(train_csv_path, weights_out_path):
    rows = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            y = float(r["target"])
            rows.append((x1, x2, y))

    gamma = 2.0
    w = [0.0] * 5
    lr = 0.05
    n = len(rows)

    for _ in range(1000):
        grad = [0.0] * 5
        for x1, x2, y in rows:
            phi = basis(x1, x2)
            y_pred = sum(w[j] * phi[j] for j in range(5))
            r = y_pred - y
            # Gradient of Cauchy loss L(r) = (gamma^2 / 2) * ln(1 + (r/gamma)^2)
            w_cauchy = r / (1.0 + (r / gamma) ** 2)
            for j in range(5):
                grad[j] += w_cauchy * phi[j]
        for j in range(5):
            w[j] -= lr * (grad[j] / n)

    model_data = {
        "model_architecture": "AsymmetricPseudoHuberSplineRegressor",
        "loss": "CauchyLoss",
        "gamma": gamma,
        "weights": w,
    }
    with open(weights_out_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

def infer(features_csv_path, weights_in_path, predictions_out_path):
    with open(weights_in_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    w = model_data["weights"]

    preds = []
    with open(features_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row_id = int(r["__mlr_row_id"])
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            phi = basis(x1, x2)
            y_pred = sum(w[j] * phi[j] for j in range(5))
            preds.append({"row_id": row_id, "prediction": y_pred})

    with open(predictions_out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fit":
        train_path = sys.argv[sys.argv.index("--train-data") + 1]
        out_path = sys.argv[sys.argv.index("--weights-out") + 1]
        fit(train_path, out_path)
    elif mode == "infer":
        feat_path = sys.argv[sys.argv.index("--features-in") + 1]
        w_path = sys.argv[sys.argv.index("--weights-in") + 1]
        out_path = sys.argv[sys.argv.index("--predictions-out") + 1]
        infer(feat_path, w_path, out_path)
"#;

    let source_sha256 = sha256_bytes(solution_code.as_bytes());

    let snapshot_id = SnapshotIdentity {
        project_id: project_id.into(),
        job_id: "job_spline_01".into(),
        candidate_id: "cand_spline_01".into(),
        dataset_id: dataset_id.into(),
        dataset_sha256: dataset_sha256.clone(),
        protocol_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: "env_local_python".into(),
    };

    let snapshot_dir = prepare_registered(&conn, &snapshot_root, dataset_id, &snapshot_id).unwrap();
    verify(&snapshot_root, &snapshot_id).unwrap();

    let train_csv = snapshot_dir.join("training.csv");
    let val_features_csv = snapshot_dir.join("validation_features.csv");
    let private_labels_json = snapshot_dir.join("labels.json");

    assert!(train_csv.is_file());
    assert!(val_features_csv.is_file());
    assert!(private_labels_json.is_file());

    // Physical isolation: Copy ONLY features and train data into worker workspace
    // labels.json stays strictly inside controller snapshot_dir
    let worker_train_csv = worker_workspace.join("training.csv");
    let worker_val_csv = worker_workspace.join("validation_features.csv");
    fs::copy(&train_csv, &worker_train_csv).unwrap();
    fs::copy(&val_features_csv, &worker_val_csv).unwrap();

    assert!(!worker_workspace.join("labels.json").exists(), "private labels leaked into worker workspace!");

    // Verify label isolation: validation_features.csv MUST NOT contain the target column
    let val_csv_content = fs::read_to_string(&worker_val_csv).unwrap();
    let first_val_line = val_csv_content.lines().next().unwrap();
    assert!(!first_val_line.contains("target"), "target column leaked into validation features!");
    assert!(first_val_line.contains("__mlr_row_id"));
    assert!(first_val_line.contains("x1"));
    assert!(first_val_line.contains("x2"));

    // =========================================================================
    // Stage 5: Fresh-Context Agent Program Authoring via Production AgentOrchestrator
    // =========================================================================
    let program_manifest = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "entrypoint": "solution.py",
        "operations": {
            "fit": {
                "argv": ["python3", "solution.py", "fit", "--train-data", "training.csv", "--weights-out", "weights.json"],
                "input_mounts": ["training.csv"],
                "output_artifacts": ["weights.json"]
            },
            "infer": {
                "argv": ["python3", "solution.py", "infer", "--features-in", "validation_features.csv", "--weights-in", "weights.json", "--predictions-out", "predictions.json"],
                "input_mounts": ["validation_features.csv", "weights.json"],
                "output_artifacts": ["predictions.json"]
            }
        }
    });

    let fixture = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Agent authors solution.py using Category A trusted tool workspace_fs
        ChatCompletionResponse {
            id: "call-modeler-turn1".into(),
            model: model_name.into(),
            content: Some("I will implement the Spline Regressor with Cauchy Loss in solution.py.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "tool_call_ws_solution".into(),
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
            usage: TokenUsage { prompt_tokens: 300, completion_tokens: 150, total_tokens: 450 },
            latency_ms: 20,
        },
        // Turn 2: Agent authors program.json manifest
        ChatCompletionResponse {
            id: "call-modeler-turn2".into(),
            model: model_name.into(),
            content: Some("Authored program manifest. Next I will finalize.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "tool_call_ws_manifest".into(),
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
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 50, total_tokens: 250 },
            latency_ms: 15,
        },
        // Turn 3: Agent signals completion
        ChatCompletionResponse {
            id: "call-modeler-turn3".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution and program manifest authored successfully.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 30, total_tokens: 180 },
            latency_ms: 10,
        },
    ]));

    let orch_config = OrchestratorConfig {
        project_id: project_id.into(),
        task_id: "task_authoring_01".into(),
        role_id: "model_specialist".into(),
        branch_id: "branch_spline_01".into(),
        model: model_name.into(),
        max_turns: 4,
    };
    let mut orchestrator = AgentOrchestrator::new(
        &controller_root,
        orch_config,
        fixture,
    ).unwrap();

    let t1_res = orchestrator.step().unwrap();
    assert_eq!(t1_res.tool_calls_executed, 1);
    assert!(!t1_res.stopped);

    let t2_res = orchestrator.step().unwrap();
    assert_eq!(t2_res.tool_calls_executed, 1);
    assert!(!t2_res.stopped);

    let t3_res = orchestrator.step().unwrap();
    assert_eq!(t3_res.tool_calls_executed, 0);
    assert!(t3_res.stopped);

    // Verify branch workspace contains agent-authored code
    let branch_dir = BranchWorkspaceManager::get_branch_dir(
        &controller_root,
        project_id,
        "branch_spline_01",
    );
    let agent_solution_py = branch_dir.join("solution.py");
    let agent_program_json = branch_dir.join("program.json");
    assert!(agent_solution_py.is_file(), "solution.py not written by agent orchestrator");
    assert!(agent_program_json.is_file(), "program.json not written by agent orchestrator");

    // Honest failure check: when container isolation is unavailable, sandbox_exec refuses host substitution
    match SandboxDetector::detect() {
        SandboxStatus::Available { .. } => {}
        SandboxStatus::Unavailable { ref reason } => {
            let refusal = SandboxRunner::execute(
                &branch_dir,
                "solution.py",
                &["--help".into()],
                &SandboxLimits::default(),
                None,
            ).unwrap_err();
            assert!(refusal.contains("Host Python substitution for untrusted code is prohibited"));
            assert!(refusal.contains(reason));
        }
    }

    // Materialize agent-authored code into worker workspace
    let solution_py = worker_workspace.join("solution.py");
    fs::copy(&agent_solution_py, &solution_py).unwrap();
    fs::copy(&agent_program_json, worker_workspace.join("program.json")).unwrap();

    // =========================================================================
    // Stage 6: Process Isolation: Execution of Agent Program (Fit Phase)
    // =========================================================================
    let weights_file = worker_workspace.join("weights.json");
    let fit_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "fit",
            "--train-data",
            worker_train_csv.to_str().unwrap(),
            "--weights-out",
            weights_file.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute fit command");

    assert!(fit_output.status.success(), "fit process failed: {:?}", String::from_utf8_lossy(&fit_output.stderr));
    assert!(weights_file.is_file(), "weights.json not generated");
    let weights_sha256 = sha256_file(&weights_file);

    // =========================================================================
    // Stage 7: Process Isolation: Execution of Agent Program (Infer Phase)
    // =========================================================================
    let predictions_file = worker_workspace.join("predictions.json");
    let infer_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "infer",
            "--features-in",
            worker_val_csv.to_str().unwrap(),
            "--weights-in",
            weights_file.to_str().unwrap(),
            "--predictions-out",
            predictions_file.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute infer command");

    assert!(infer_output.status.success(), "infer process failed: {:?}", String::from_utf8_lossy(&infer_output.stderr));
    assert!(predictions_file.is_file(), "predictions.json not generated");

    // =========================================================================
    // Stage 8: Controller-Owned Independent Scoring
    // =========================================================================
    let preds_json: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&predictions_file).unwrap()).unwrap();
    assert_eq!(preds_json.len(), 6, "expected 6 validation predictions");

    let eval_identity = EvaluationIdentity {
        project_id: project_id.into(),
        job_id: snapshot_id.job_id.clone(),
        candidate_id: snapshot_id.candidate_id.clone(),
        dataset_id: dataset_id.into(),
        dataset_sha256: dataset_sha256.clone(),
        split_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: snapshot_id.environment_id.clone(),
        weights_sha256: weights_sha256.clone(),
    };

    let mut pred_pairs = Vec::new();
    for p in &preds_json {
        let row_id = p["row_id"].as_u64().unwrap();
        let val = p["prediction"].as_f64().unwrap();
        pred_pairs.push((row_id, val));
    }
    let prediction_batch = PredictionBatch::scalar(eval_identity.clone(), pred_pairs);

    // Independent evaluation against private controller snapshot
    let receipt = evaluate_predictions(
        &conn,
        &snapshot_root,
        dataset_id,
        &snapshot_id,
        &eval_identity,
        &prediction_batch,
    ).unwrap();

    assert_eq!(receipt.metric, Metric::Mae);
    assert_eq!(receipt.row_count, 6);
    assert!(receipt.score < 0.60, "expected good fit, got MAE = {}", receipt.score);

    // Spec-based evaluation check with Parameterized Cauchy Loss (gamma = 2.0)
    let spec_cauchy = MetricSpec {
        name: "cauchy".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: Some(2.0),
        decision_threshold: None,
        variant: None,
    };
    let private_labels_raw: Vec<(u64, f64)> = serde_json::from_slice(&fs::read(&private_labels_json).unwrap()).unwrap();
    let cauchy_receipt = evaluate_with_spec(&eval_identity, &spec_cauchy, &private_labels_raw, &prediction_batch).unwrap();
    assert!(cauchy_receipt.score > 0.0);
    assert_eq!(cauchy_receipt.assurance, "provisional_spec_checked");

    // Self-supplied reference evidence without authoritative fixture store remains provisional
    let fixture_sha = sha256_bytes(format!("{}:{}", receipt.labels_sha256, receipt.predictions_sha256).as_bytes());
    let caller_evidence = ReferenceEvidence {
        reference_identity: "caller_claimed_reference".into(),
        fixture_sha256: fixture_sha.clone(),
        expected_result: receipt.score,
        verified_receipt_sha256: "1".repeat(64),
        metric_version: "v1.0".into(),
        tolerance: 1e-6,
    };
    let spec_mae = MetricSpec {
        name: "mae".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let uncertified_receipt = evaluate_with_spec_and_evidence(
        &eval_identity,
        &spec_mae,
        &private_labels_raw,
        &prediction_batch,
        Some(&caller_evidence),
    ).unwrap();
    assert_eq!(uncertified_receipt.assurance, "provisional_spec_checked");
    assert!(uncertified_receipt.reference_evidence.is_none());

    // =========================================================================
    // Stage 9: Usable Model Delivery Bundle & Standalone Parity Verification
    // =========================================================================
    let bundle_dir = root.join("delivery_bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    // Copy model weights into bundle
    fs::copy(&weights_file, bundle_dir.join("frozen_model.joblib")).unwrap();
    let frozen_model_sha = sha256_file(&bundle_dir.join("frozen_model.joblib"));

    fs::write(bundle_dir.join("input_schema.json"), r#"{"type":"object","properties":{"x1":{"type":"number"},"x2":{"type":"number"}}}"#).unwrap();
    fs::write(bundle_dir.join("output_schema.json"), r#"{"type":"object","properties":{"prediction":{"type":"number"}}}"#).unwrap();
    let scorecard = serde_json::json!({
        "candidate_id": snapshot_id.candidate_id,
        "score": receipt.score,
        "metric": "mae"
    });
    fs::write(bundle_dir.join("scorecard.json"), serde_json::to_string(&scorecard).unwrap()).unwrap();
    let verification = serde_json::json!({
        "verified": true,
        "verifier": "mlrescurix_controller",
        "frozen_model_sha256": frozen_model_sha
    });
    fs::write(bundle_dir.join("verification.json"), serde_json::to_string(&verification).unwrap()).unwrap();
    fs::write(bundle_dir.join("final.json"), r#"{"candidate":"cand_spline_01","decision":"accepted"}"#).unwrap();
    fs::write(bundle_dir.join("inference.lock"), "# clean dependencies\n").unwrap();
    fs::write(bundle_dir.join("training.lock"), "# clean dependencies\n").unwrap();
    fs::write(bundle_dir.join("README.md"), "# Governed Spline Model Bundle\n").unwrap();

    // Production-ready standalone inference.py with schema validation and CLI mode
    let bundle_inference_py = r#"
import sys
import json
import math
import argparse
from pathlib import Path

def basis(x1, x2):
    return [1.0, float(x1), float(x1) * float(x1), float(x2), math.sin(float(x2))]

def load_weights(bundle_dir=None):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)
    weight_file = bundle_dir / "frozen_model.joblib"
    if not weight_file.is_file():
        weight_file = bundle_dir / "weights.json"
    with open(weight_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["weights"]

def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError(f"expected object record, got {type(record)}")
    allowed_cols = {"x1", "x2", "__mlr_row_id", "__row_id__"}
    for k in record.keys():
        if k not in allowed_cols:
            raise ValueError(f"extra unknown column '{k}' rejected by schema")
    if "x1" not in record:
        raise ValueError("missing required column 'x1'")
    if "x2" not in record:
        raise ValueError("missing required column 'x2'")

def predict(inputs, bundle_dir=None):
    weights = load_weights(bundle_dir)
    is_single = isinstance(inputs, dict)
    records = [inputs] if is_single else inputs
    if not isinstance(records, list):
        raise ValueError("inputs must be a list of records or single record")

    results = []
    for r in records:
        validate_record(r)
        x1 = float(r["x1"])
        x2 = float(r["x2"])
        phi = basis(x1, x2)
        y_pred = sum(weights[j] * phi[j] for j in range(5))
        pred_item = {"prediction": round(y_pred, 4)}
        if "__mlr_row_id" in r:
            pred_item["row_id"] = int(r["__mlr_row_id"])
        elif "row_id" in r:
            pred_item["row_id"] = int(r["row_id"])
        results.append(pred_item)

    return results[0] if is_single else results

def main():
    parser = argparse.ArgumentParser(description="Standalone Inference CLI")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        inputs = json.load(f)

    preds = predict(inputs, bundle_dir=args.bundle_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

if __name__ == "__main__":
    main()
"#;
    fs::write(bundle_dir.join("inference.py"), bundle_inference_py).unwrap();

    // Production-ready standalone serve.py loopback server recipe
    let bundle_serve_py = r#"
import sys
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

bundle_path = Path(__file__).resolve().parent

class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "serving_status": "healthy"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                inputs = json.loads(body)

                sys.path.insert(0, str(bundle_path))
                import inference
                preds = inference.predict(inputs, bundle_dir=bundle_path)
                self._send_json(200, {"status": "ok", "predictions": preds})
            except Exception as e:
                self._send_json(400, {"status": "error", "error": str(e)})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass

def main():
    global bundle_path
    parser = argparse.ArgumentParser(description="Standalone Model Serving Recipe")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    if args.bundle_dir:
        bundle_path = Path(args.bundle_dir).resolve()

    server = HTTPServer(("127.0.0.1", args.port), InferenceHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
"#;
    fs::write(bundle_dir.join("serve.py"), bundle_serve_py).unwrap();

    // Generate real examples_input.json and examples_expected.json matching inference parity
    let example_inputs = serde_json::json!([
        {"x1": 0.5, "x2": 1.0},
        {"x1": -0.2, "x2": 2.5}
    ]);
    fs::write(bundle_dir.join("examples_input.json"), serde_json::to_string(&example_inputs).unwrap()).unwrap();

    // Compute expected predictions using standalone CLI inference
    let cli_output_temp = root.join("cli_parity_output.json");
    let cli_preds = verify_standalone_cli_inference(
        &bundle_dir,
        &bundle_dir.join("examples_input.json"),
        &cli_output_temp,
    ).unwrap();
    assert!(cli_preds.is_array());
    fs::copy(&cli_output_temp, bundle_dir.join("examples_expected.json")).unwrap();

    // Compile bytecode (.pyc) to verify strict manifest bytecode inclusion rule
    let pycache_dir = bundle_dir.join("__pycache__");
    fs::create_dir_all(&pycache_dir).unwrap();
    let pyc_path = pycache_dir.join("inference.cpython-313.pyc");
    fs::write(&pyc_path, b"\x00\x00\x00\x00bytecode_sample").unwrap();

    // Construct BundleManifest covering 100% of files
    let mut bundle_files = std::collections::HashMap::new();
    let mut file_list = Vec::new();
    fn walk_files(base: &Path, curr: &Path, out: &mut Vec<String>) {
        for entry in fs::read_dir(curr).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            if p.is_dir() {
                walk_files(base, &p, out);
            } else if p.is_file() {
                let rel = p.strip_prefix(base).unwrap().to_string_lossy().to_string();
                out.push(rel);
            }
        }
    }
    walk_files(&bundle_dir, &bundle_dir, &mut file_list);

    for rel in file_list {
        let p = bundle_dir.join(&rel);
        let bytes = fs::read(&p).unwrap();
        bundle_files.insert(rel, BundleManifestFileEntry {
            bytes: bytes.len() as u64,
            sha256: sha256_bytes(&bytes),
        });
    }

    let manifest = BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: project_id.into(),
        candidate_id: snapshot_id.candidate_id.clone(),
        files: bundle_files,
        model: None,
        pipeline: None,
        contracts: None,
        dependencies: None,
        scorecard: None,
        lineage: None,
        resource_measurements: None,
    };
    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    // 9a. Authoritative bundle verification: 100% file coverage, no untracked files, no symlinks
    let report = verify_bundle(&bundle_dir, false).unwrap();
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
    assert_eq!(report.format, "mlrescurix.bundle.v2");

    // 9b. Standalone CLI inference parity verification
    let cli_verified = verify_standalone_cli_inference(
        &bundle_dir,
        &bundle_dir.join("examples_input.json"),
        &root.join("cli_verify_run.json"),
    ).unwrap();
    assert_eq!(cli_verified, cli_preds);

    // 9c. Standalone Serving Recipe verification (/health, /predict, and bad column rejection)
    let serving_report = verify_standalone_serving_recipe(&bundle_dir, 19876).unwrap();
    assert_eq!(serving_report["health_verified"], true);
    assert_eq!(serving_report["prediction_verified"], true);
    assert_eq!(serving_report["schema_rejection_verified"], true);

    // =========================================================================
    // Stage 10: Final Lockbox Assessment, Atomic Consumption & Crash Resilience
    // =========================================================================
    let lockbox_features_csv = controller_root.join("lockbox_features.csv");
    let mut lockbox_csv_content = String::from("__mlr_row_id,x1,x2\n");
    for &row_id in &lockbox_rows {
        let (id, _) = ground_truth_labels[row_id as usize];
        let x1 = (id as f64 - 15.0) / 10.0;
        let x2 = (id as f64) * 0.2;
        lockbox_csv_content.push_str(&format!("{id},{x1:.4},{x2:.4}\n"));
    }
    fs::write(&lockbox_features_csv, &lockbox_csv_content).unwrap();

    // Run inference on lockbox features
    let lockbox_preds_file = worker_workspace.join("lockbox_predictions.json");
    let infer_lockbox_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "infer",
            "--features-in",
            lockbox_features_csv.to_str().unwrap(),
            "--weights-in",
            weights_file.to_str().unwrap(),
            "--predictions-out",
            lockbox_preds_file.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute lockbox inference");

    assert!(infer_lockbox_output.status.success());
    let lockbox_preds_raw: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&lockbox_preds_file).unwrap()).unwrap();
    let mut lockbox_pred_pairs = Vec::new();
    for p in &lockbox_preds_raw {
        let row_id = p["row_id"].as_u64().unwrap();
        let val = p["prediction"].as_f64().unwrap();
        lockbox_pred_pairs.push((row_id, val));
    }

    let assessment_identity = EvaluationIdentity {
        project_id: project_id.into(),
        job_id: "job_final_assessment".into(),
        candidate_id: snapshot_id.candidate_id.clone(),
        dataset_id: dataset_id.into(),
        dataset_sha256: dataset_sha256.clone(),
        split_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: snapshot_id.environment_id.clone(),
        weights_sha256: weights_sha256.clone(),
    };
    let lockbox_pred_batch = PredictionBatch::scalar(assessment_identity.clone(), lockbox_pred_pairs);

    // 10a. Authoritative assessment evaluation reading ground truth directly from registered dataset CSV
    let lockbox_receipt = custom_protocol::evaluate_and_consume_assessment_authoritative(
        &mut conn,
        project_id,
        dataset_id,
        &assessment_identity,
        &lockbox_pred_batch,
    ).unwrap();

    assert_eq!(lockbox_receipt.metric, Metric::Mae);
    assert_eq!(lockbox_receipt.row_count, 6);
    assert!(lockbox_receipt.score < 0.60);

    // 10b. Second assessment MUST fail (one-shot lockbox consumption fence)
    let re_eval_err = custom_protocol::evaluate_and_consume_assessment_authoritative(
        &mut conn,
        project_id,
        dataset_id,
        &assessment_identity,
        &lockbox_pred_batch,
    ).unwrap_err();
    assert!(re_eval_err.contains("already been consumed"));

    // 10c. Interrupted Transaction Crash Simulation & Rollback Resilience
    // Simulate a process crash during an uncommitted database mutation
    {
        let db_path = controller_root.join("state.sqlite");
        let mut crash_conn = Connection::open(&db_path).unwrap();
        crash_conn.execute("INSERT OR IGNORE INTO projects VALUES('temp_p', 1, 1, 1, 'test', 'final-crash', 'active', '{}')", []).unwrap();
        let tx = crash_conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).unwrap();
        // Insert a speculative uncommitted record
        let dummy_sha = "c".repeat(64);
        tx.execute(
            "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES('temp_p', 'crash_ds', '/tmp/crash.csv', 10, ?1, 0)",
            rusqlite::params![dummy_sha],
        ).unwrap();
        // Transaction is dropped without commit (simulating SIGKILL / sudden process exit)
        drop(tx);
        drop(crash_conn);
    }

    // Reopen database after simulated crash
    drop(conn);
    let recovered_conn = Connection::open(controller_root.join("state.sqlite")).unwrap();

    // Verify uncommitted crash write was cleanly rolled back
    let uncommitted_exists: bool = recovered_conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM scientific_datasets WHERE project='temp_p' AND id='crash_ds')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(!uncommitted_exists, "uncommitted transaction leaked into database after crash!");

    // 10d. Durable Receipt Recovery after crash/restart
    let recovered_receipt = get_assessment_consumption_receipt(&recovered_conn, project_id, dataset_id).unwrap().unwrap();
    assert_eq!(recovered_receipt.score, lockbox_receipt.score);
    assert_eq!(recovered_receipt.metric, lockbox_receipt.metric);
    assert_eq!(recovered_receipt.predictions_sha256, lockbox_receipt.predictions_sha256);
    assert_eq!(recovered_receipt.identity, lockbox_receipt.identity);
    assert_eq!(recovered_receipt.row_count, lockbox_receipt.row_count);
}

#[test]
fn adversarial_group_and_temporal_leakage_rejection() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let raw_csv = root.join("group_time_dataset.csv");
    let mut csv_content = String::from("group_id,timestamp,x,target\n");
    // Row 0: grp_A, time 100
    // Row 1: grp_A, time 200
    // Row 2: grp_B, time 300
    // Row 3: grp_C, time 400
    // Row 4: grp_D, time 500
    // Row 5: grp_D, time 600
    csv_content.push_str("grp_A,100,1.0,10.0\n");
    csv_content.push_str("grp_A,200,2.0,20.0\n");
    csv_content.push_str("grp_B,300,3.0,30.0\n");
    csv_content.push_str("grp_C,400,4.0,40.0\n");
    csv_content.push_str("grp_D,500,5.0,50.0\n");
    csv_content.push_str("grp_D,600,6.0,60.0\n");
    fs::write(&raw_csv, &csv_content).unwrap();

    let dataset_bytes = fs::read(&raw_csv).unwrap();
    let dataset_sha = sha256_bytes(&dataset_bytes);

    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE scientific_datasets(project TEXT NOT NULL, id TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, registered_ms INTEGER NOT NULL, PRIMARY KEY(project, id));"
    ).unwrap();
    conn.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();

    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES('p', 'd', ?1, ?2, ?3, 0)",
        rusqlite::params![raw_csv.to_string_lossy(), dataset_bytes.len() as i64, dataset_sha],
    ).unwrap();

    // Base protocol: train [0, 1], val [2], test [3], lockbox [4, 5]
    // 1. Group-based protocol: grp_A in train, grp_B in val, grp_C in test, grp_D in lockbox -> succeeds
    let mut valid_p = CustomProtocol::single_split(
        dataset_sha.clone(),
        "target".into(),
        Metric::Mae,
        vec![0, 1],
        vec![2],
        vec![3],
    );
    valid_p.lockbox_rows = vec![4, 5];
    valid_p.group_column = Some("group_id".into());
    valid_p.strategy = Some("group_k_fold".into());

    let digest = freeze(&mut conn, "p", "d", valid_p).unwrap();
    assert!(!digest.is_empty());

    // 2. Leaking group between train and validation: train [0, 2], val [1] -> grp_A in both!
    let mut leak_p = CustomProtocol::single_split(
        dataset_sha.clone(),
        "target".into(),
        Metric::Mae,
        vec![0, 2],
        vec![1],
        vec![3],
    );
    leak_p.lockbox_rows = vec![4, 5];
    leak_p.group_column = Some("group_id".into());
    leak_p.strategy = Some("group_k_fold".into());

    let mut conn2 = Connection::open_in_memory().unwrap();
    conn2.execute_batch("CREATE TABLE scientific_datasets(project TEXT NOT NULL, id TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, registered_ms INTEGER NOT NULL, PRIMARY KEY(project, id));").unwrap();
    conn2.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    conn2.execute("INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES('p', 'd', ?1, ?2, ?3, 0)", rusqlite::params![raw_csv.to_string_lossy(), dataset_bytes.len() as i64, dataset_sha]).unwrap();

    let err_leak = freeze(&mut conn2, "p", "d", leak_p).unwrap_err();
    assert!(err_leak.contains("group leakage detected"), "expected group leakage error, got: {err_leak}");

    // 3. Temporal lookahead rejection: time_column specified, validation (timestamp 100) earlier than train max (timestamp 200)
    let mut time_leak_p = CustomProtocol::single_split(
        dataset_sha.clone(),
        "target".into(),
        Metric::Mae,
        vec![1], // row 1: time 200
        vec![0], // row 0: time 100 (earlier than train!)
        vec![3],
    );
    time_leak_p.lockbox_rows = vec![4, 5];
    time_leak_p.time_column = Some("timestamp".into());
    time_leak_p.strategy = Some("time_series_split".into());

    let mut conn3 = Connection::open_in_memory().unwrap();
    conn3.execute_batch("CREATE TABLE scientific_datasets(project TEXT NOT NULL, id TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, registered_ms INTEGER NOT NULL, PRIMARY KEY(project, id));").unwrap();
    conn3.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    conn3.execute("INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES('p', 'd', ?1, ?2, ?3, 0)", rusqlite::params![raw_csv.to_string_lossy(), dataset_bytes.len() as i64, dataset_sha]).unwrap();

    let err_time = freeze(&mut conn3, "p", "d", time_leak_p).unwrap_err();
    assert!(err_time.contains("temporal lookahead detected"), "expected temporal lookahead error, got: {err_time}");
}

#[test]
fn adversarial_duplicate_spend_and_pricing_protection() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("controller");

    let _r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    mlrescurix_harness::execution::migrate(&root).unwrap();

    // 1. Pricing namespacing: Register pricing under openrouter endpoint
    let endpoint_or = "https://openrouter.ai/api/v1";
    let endpoint_local = "http://localhost:11434/v1";
    let model = "custom-deepseek";

    register_namespaced_model_pricing(
        &root,
        endpoint_or,
        model,
        5,
        15,
    ).unwrap();

    // Lookups under openrouter resolve correctly
    let or_rate = resolve_namespaced_model_pricing(&root, endpoint_or, model).unwrap().unwrap();
    assert_eq!(or_rate.0, 5);
    assert_eq!(or_rate.1, 15);

    // Lookups under localhost fail closed (do NOT silently fall back to openrouter rate)
    let local_lookup = resolve_namespaced_model_pricing(&root, endpoint_local, model).unwrap();
    assert!(local_lookup.is_none());

    // 2. Spend reservation and replay protection
    let tokens = TokenUsage {
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
    };
    reserve_role_spend(
        &root,
        "proj_spend",
        "req_spend_01",
        "task_01",
        "worker",
        model,
        &tokens,
        1250,
        Some(10_000),
    ).unwrap();

    // Simulate settling the request as 'ok' in provider_accounting
    let conn = Connection::open(root.join("state.sqlite")).unwrap();
    conn.execute(
        "UPDATE provider_accounting SET status='ok', cost_millicents=1250 WHERE project='proj_spend' AND request_id='req_spend_01'",
        [],
    ).unwrap();

    // Replay reservation on already-settled 'ok' request MUST fail
    let replay_err = reserve_role_spend(
        &root,
        "proj_spend",
        "req_spend_01",
        "task_01",
        "worker",
        model,
        &tokens,
        1250,
        Some(10_000),
    ).unwrap_err();
    assert!(replay_err.contains("already been dispatched with status 'ok'"));

    // Settled row must not have been modified
    let status_after: String = conn.query_row(
        "SELECT status FROM provider_accounting WHERE project='proj_spend' AND request_id='req_spend_01'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(status_after, "ok");
}

#[test]
fn adversarial_predispatch_failure_vs_uncertain_outcome() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("controller");

    let _r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    mlrescurix_harness::execution::migrate(&root).unwrap();

    let endpoint = "https://openrouter.ai/api/v1";
    let model = "test-model";
    register_namespaced_model_pricing(&root, endpoint, model, 10, 20).unwrap();
    register_model_pricing(&root, model, 10, 20).unwrap();

    let project_id = "proj_dispatch_test";
    let task_id = "task_dispatch_01";
    let req_id_predispatch = "req_predispatch_01";

    let req_payload = mlrescurix_harness::agents::ChatCompletionRequest {
        model: model.into(),
        messages: vec![mlrescurix_harness::agents::ChatMessage::user("Test prompt")],
        tools: vec![],
        temperature: Some(0.0),
        max_tokens: Some(128),
        provider: None,
        reasoning: None,
    };

    let policy_low_budget = mlrescurix_harness::contracts::AgentRoutingPolicy {
        schema_version: "mlrescurix.agent_routing.v1".into(),
        default_worker_profile_ref: "p_model".into(),
        providers: BTreeMap::from([
            ("prov".into(), mlrescurix_harness::contracts::ProviderProfile {
                adapter_protocol: "openai_compatible".into(),
                endpoint: endpoint.into(),
                secret_ref: None,
                permitted_data_disclosure: "redacted_only".into(),
                timeout_seconds: 30,
                max_retries: 0,
            }),
        ]),
        models: BTreeMap::from([
            ("p_model".into(), mlrescurix_harness::contracts::ModelProfile {
                provider_ref: "prov".into(),
                model_id: model.into(),
                context_window: 4096,
                max_output_tokens: 128,
                prompt_millicents_per_1k: 10,
                completion_millicents_per_1k: 20,
                supports_tools: false,
            }),
        ]),
        roles: BTreeMap::from([
            ("worker".into(), mlrescurix_harness::contracts::RoleRouting {
                model_profile_ref: "p_model".into(),
                escalation_profile_ref: None,
                max_spend_millicents: Some(1), // Extremely low ceiling: 1 millicent ($0.00001)
                temperature: Some(0.0),
            }),
        ]),
    };

    let fixture_ok = Arc::new(ScriptedProviderFixture::new(vec![
        ChatCompletionResponse {
            id: "resp_1".into(),
            model: model.into(),
            content: Some("Success".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 50, completion_tokens: 50, total_tokens: 100 },
            latency_ms: 10,
        },
    ]));

    // 1. Pre-dispatch failure: role spend ceiling exceeded
    let err_predispatch = mlrescurix_harness::agents::dispatch_gateway_with_routing(
        &root,
        project_id,
        task_id,
        "worker",
        req_id_predispatch,
        &req_payload,
        &policy_low_budget,
        fixture_ok.as_ref(),
        false,
    ).unwrap_err();
    assert!(err_predispatch.contains("spend limit exceeded"), "Unexpected err_predispatch: {err_predispatch}");

    // Verify pre-dispatch failure did NOT record an outcome_unknown receipt
    let db = Connection::open(root.join("state.sqlite")).unwrap();
    let receipt_status: Option<String> = db.query_row(
        "SELECT status FROM gateway_dispatch_receipts WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, req_id_predispatch],
        |r| r.get(0),
    ).ok();
    assert_eq!(receipt_status, None, "pre-dispatch failure must not record awaiting_response or outcome_unknown receipt");

    // Verify provider_accounting has no stranded reserved rows
    let accounting_status: Option<String> = db.query_row(
        "SELECT status FROM provider_accounting WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, mlrescurix_harness::agents::gateway::accounting_request_key(project_id, task_id, req_id_predispatch)],
        |r| r.get(0),
    ).ok();
    assert_eq!(accounting_status, None, "pre-dispatch failure must clean up reserved accounting row");

    // Retry with SAME request ID after increasing budget succeeds without ambiguous outcome refusal!
    let mut policy_ample_budget = policy_low_budget.clone();
    policy_ample_budget.roles.get_mut("worker").unwrap().max_spend_millicents = Some(100_000);

    let retry_res = mlrescurix_harness::agents::dispatch_gateway_with_routing(
        &root,
        project_id,
        task_id,
        "worker",
        req_id_predispatch,
        &req_payload,
        &policy_ample_budget,
        fixture_ok.as_ref(),
        false,
    ).unwrap();
    assert_eq!(retry_res["status"], "completed");

    // 2. Uncertain remote outcome: simulated remote failure/timeout during transport
    let req_id_remote = "req_remote_uncertain_01";
    let fixture_failing = Arc::new(ScriptedProviderFixture::new(vec![])); // Empty fixture -> returns error on completion

    let err_remote = mlrescurix_harness::agents::dispatch_gateway_with_routing(
        &root,
        project_id,
        task_id,
        "worker",
        req_id_remote,
        &req_payload,
        &policy_ample_budget,
        fixture_failing.as_ref(),
        false,
    ).unwrap_err();
    assert!(err_remote.contains("no remaining queued responses") || err_remote.to_lowercase().contains("provider"));

    // Verify remote error marks gateway_dispatch_receipts as outcome_unknown
    let remote_receipt_status: String = db.query_row(
        "SELECT status FROM gateway_dispatch_receipts WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, req_id_remote],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(remote_receipt_status, "outcome_unknown");

    // Verify provider_accounting updates reservation to outcome_unknown preserving durable spend
    let remote_accounting_status: String = db.query_row(
        "SELECT status FROM provider_accounting WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, mlrescurix_harness::agents::gateway::accounting_request_key(project_id, task_id, req_id_remote)],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(remote_accounting_status, "outcome_unknown");

    // Blind replay with SAME request ID on uncertain outcome MUST BE REFUSED
    let replay_err = mlrescurix_harness::agents::dispatch_gateway_with_routing(
        &root,
        project_id,
        task_id,
        "worker",
        req_id_remote,
        &req_payload,
        &policy_ample_budget,
        fixture_ok.as_ref(),
        false,
    ).unwrap_err();
    assert!(replay_err.contains("ambiguous prior outcome"));
}

#[test]
fn adversarial_concurrent_pricing_file_contention() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().to_path_buf();

    let endpoint = "https://openrouter.ai/api/v1";
    let num_threads = 20;
    let mut handles = Vec::new();

    for i in 0..num_threads {
        let r = root.clone();
        let ep = endpoint.to_string();
        handles.push(std::thread::spawn(move || {
            let model = format!("concurrent_model_{i}");
            register_namespaced_model_pricing(&r, &ep, &model, 10 + i as i64, 20 + i as i64).unwrap();
            let resolved = resolve_namespaced_model_pricing(&r, &ep, &model).unwrap().unwrap();
            assert_eq!(resolved.0, 10 + i as i64);
            assert_eq!(resolved.1, 20 + i as i64);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify pricing.json is intact, valid JSON, containing all 20 models
    let pricing_path = root.join("pricing.json");
    assert!(pricing_path.is_file());
    let bytes = fs::read(&pricing_path).unwrap();
    let val: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let obj = val.as_object().unwrap();

    for i in 0..num_threads {
        let key = format!("{}::concurrent_model_{i}", mlrescurix_harness::agents::gateway::normalize_endpoint(endpoint));
        assert!(obj.contains_key(&key), "missing key '{key}' in pricing table");
    }
}

#[test]
fn adversarial_assessment_split_hash_and_consumption_fences() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let raw_csv = root.join("dataset.csv");
    fs::write(&raw_csv, "x1,target\n1.0,10.0\n2.0,20.0\n3.0,30.0\n4.0,40.0\n5.0,50.0\n").unwrap();
    let csv_bytes = fs::read(&raw_csv).unwrap();
    let dataset_sha = sha256_bytes(&csv_bytes);

    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE scientific_datasets(project TEXT NOT NULL, id TEXT NOT NULL, path TEXT NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, registered_ms INTEGER NOT NULL, PRIMARY KEY(project, id));"
    ).unwrap();
    conn.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();

    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES('proj_adv', 'ds_adv', ?1, ?2, ?3, 0)",
        rusqlite::params![raw_csv.to_string_lossy(), csv_bytes.len() as i64, dataset_sha],
    ).unwrap();

    let mut protocol = CustomProtocol::single_split(
        dataset_sha.clone(),
        "target".into(),
        Metric::Mae,
        vec![0, 1],
        vec![2],
        vec![3],
    );
    protocol.lockbox_rows = vec![4];
    let protocol_sha = freeze(&mut conn, "proj_adv", "ds_adv", protocol).unwrap();

    let eval_identity_forged_split = EvaluationIdentity {
        project_id: "proj_adv".into(),
        job_id: "job_01".into(),
        candidate_id: "cand_01".into(),
        dataset_id: "ds_adv".into(),
        dataset_sha256: dataset_sha.clone(),
        split_sha256: "0".repeat(64), // Forged split hash!
        source_sha256: "a".repeat(64),
        environment_id: "env_test".into(),
        weights_sha256: "b".repeat(64),
    };
    let pred_batch = PredictionBatch::scalar(eval_identity_forged_split.clone(), vec![(4, 50.0)]);

    // 1. Forged split hash fails closed
    let err_forged = custom_protocol::evaluate_and_consume_assessment_authoritative(
        &mut conn,
        "proj_adv",
        "ds_adv",
        &eval_identity_forged_split,
        &pred_batch,
    ).unwrap_err();
    assert!(err_forged.contains("split_sha256") && err_forged.contains("does not match registered protocol"));

    // 2. Legitimate first assessment evaluation succeeds
    let mut legitimate_identity = eval_identity_forged_split.clone();
    legitimate_identity.split_sha256 = protocol_sha.clone();
    let legitimate_pred_batch = PredictionBatch::scalar(legitimate_identity.clone(), vec![(4, 50.0)]);

    let receipt = custom_protocol::evaluate_and_consume_assessment_authoritative(
        &mut conn,
        "proj_adv",
        "ds_adv",
        &legitimate_identity,
        &legitimate_pred_batch,
    ).unwrap();
    assert_eq!(receipt.score, 0.0);

    // 3. One-shot fence: second evaluation fails closed IMMEDIATELY with already consumed
    let err_second = custom_protocol::evaluate_and_consume_assessment_authoritative(
        &mut conn,
        "proj_adv",
        "ds_adv",
        &legitimate_identity,
        &legitimate_pred_batch,
    ).unwrap_err();
    assert!(err_second.contains("already been consumed"));
}
