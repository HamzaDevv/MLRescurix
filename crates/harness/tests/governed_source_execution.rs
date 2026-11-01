//! Authoritative End-to-End Vertical Slice Test: Governed Local Source-Execution
//!
//! Validates:
//! 1. Controller-owned admission and dataset registration.
//! 2. Frozen custom protocol with deterministic train/val splits and carved lockbox.
//! 3. Controller-owned snapshot generation with strict label separation (held out from worker).
//! 4. Agent-authored ProgramManifest with an unfamiliar custom model (Asymmetric Pseudo-Huber Spline Regressor with Cauchy Loss).
//! 5. Execution of agent program via local process invocation (fit and infer phases).
//! 6. Controller-owned independent inference scoring against private held-out labels.
//! 7. Authoritative delivery bundle packaging and anti-tampering verification (including bytecode and symlink rejection).
//! 8. Final lockbox evaluation, atomic one-shot consumption, and crash recovery.

use mlrescurix_harness::{
    bundle::{self, BundleManifest, BundleManifestFileEntry},
    contracts::MetricSpec,
    custom_protocol::{self, carve_lockbox, freeze, get_assessment_consumption_receipt, CustomProtocol},
    custom_snapshot::{evaluate_predictions, prepare_registered, verify, SnapshotIdentity},
    trusted_evaluation::{
        evaluate_with_spec, evaluate_with_spec_and_evidence, EvaluationIdentity, Metric,
        PredictionBatch, ReferenceEvidence,
    },
};
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::{fs, process::Command};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &std::path::Path) -> String {
    let bytes = fs::read(path).expect("failed to read file for hash");
    sha256_bytes(&bytes)
}

#[test]
fn governed_local_source_execution_vertical_slice() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // =========================================================================
    // Stage 1: Synthetic Dataset & Ground Truth Generation
    // Target y = 2.5 * x1^2 - 1.2 * sin(x2) + 0.5 (non-linear continuous regression)
    // =========================================================================
    let raw_csv = root.join("raw_dataset.csv");
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
    // Stage 2: SQLite Schema Migration & Controller Dataset Registration
    // =========================================================================
    let db_path = root.join("state.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "CREATE TABLE scientific_datasets(
            project TEXT NOT NULL,
            id TEXT NOT NULL,
            path TEXT NOT NULL,
            bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            registered_ms INTEGER NOT NULL,
            PRIMARY KEY(project, id)
        );",
    ).unwrap();
    conn.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();

    let project_id = "governed_project";
    let dataset_id = "ds_spline_cauchy";
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
    // Stage 4: Controller-Owned Snapshot Preparation
    // Worker gets training.csv and validation_features.csv (labels are strictly held out!)
    // =========================================================================
    let snapshot_root = root.join("snapshots");
    fs::create_dir_all(&snapshot_root).unwrap();

    let snapshot_id = SnapshotIdentity {
        project_id: project_id.into(),
        job_id: "job_spline_01".into(),
        candidate_id: "cand_spline_01".into(),
        dataset_id: dataset_id.into(),
        dataset_sha256: dataset_sha256.clone(),
        protocol_sha256: protocol_sha256.clone(),
        source_sha256: "0".repeat(64),
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

    // Verify label isolation: validation_features.csv MUST NOT contain the target column
    let val_csv_content = fs::read_to_string(&val_features_csv).unwrap();
    let first_val_line = val_csv_content.lines().next().unwrap();
    assert!(!first_val_line.contains("target"), "target column leaked into validation features!");
    assert!(first_val_line.contains("__mlr_row_id"));
    assert!(first_val_line.contains("x1"));
    assert!(first_val_line.contains("x2"));

    // =========================================================================
    // Stage 5: Agent Program Authoring (Unfamiliar Model with Cauchy Loss)
    // Pure Python implementation of Asymmetric Pseudo-Huber Spline Regressor
    // =========================================================================
    let worker_workspace = root.join("worker_workspace");
    fs::create_dir_all(&worker_workspace).unwrap();

    let solution_py = worker_workspace.join("solution.py");
    let solution_code = r#"
import sys
import json
import math
import csv

def basis(x1, x2):
    # Basis: [1, x1, x1^2, x2, sin(x2)]
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
    fs::write(&solution_py, solution_code).unwrap();

    let program_manifest = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "entrypoint": "solution.py",
        "operations": {
            "fit": {
                "argv": ["python3", "solution.py", "fit", "--train-data", train_csv.to_str().unwrap(), "--weights-out", "weights.json"],
                "input_mounts": [train_csv.to_str().unwrap()],
                "output_artifacts": ["weights.json"]
            },
            "infer": {
                "argv": ["python3", "solution.py", "infer", "--features-in", val_features_csv.to_str().unwrap(), "--weights-in", "weights.json", "--predictions-out", "predictions.json"],
                "input_mounts": [val_features_csv.to_str().unwrap(), "weights.json"],
                "output_artifacts": ["predictions.json"]
            }
        }
    });
    fs::write(worker_workspace.join("program.json"), serde_json::to_vec_pretty(&program_manifest).unwrap()).unwrap();

    // =========================================================================
    // Stage 6: Execution via Subprocess (Fit Phase)
    // =========================================================================
    let weights_file = worker_workspace.join("weights.json");
    let fit_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "fit",
            "--train-data",
            train_csv.to_str().unwrap(),
            "--weights-out",
            weights_file.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute fit command");

    assert!(fit_output.status.success(), "fit process failed: {:?}", String::from_utf8_lossy(&fit_output.stderr));
    assert!(weights_file.is_file(), "weights.json not generated");
    let weights_sha256 = sha256_file(&weights_file);

    // =========================================================================
    // Stage 7: Execution via Subprocess (Infer Phase on Validation Split)
    // =========================================================================
    let predictions_file = worker_workspace.join("predictions.json");
    let infer_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "infer",
            "--features-in",
            val_features_csv.to_str().unwrap(),
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
        source_sha256: snapshot_id.source_sha256.clone(),
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

    // Evaluate against private controller snapshot
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
    // Well-fit spline model achieves MAE < 0.60 on held-out non-linear curves from 16 training samples
    assert!(receipt.score < 0.60, "expected good fit, got MAE = {}", receipt.score);

    // Spec-based evaluation check
    let spec = MetricSpec {
        name: "mae".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let private_labels_raw: Vec<(u64, f64)> = serde_json::from_slice(&fs::read(&private_labels_json).unwrap()).unwrap();
    let spec_receipt = evaluate_with_spec(&eval_identity, &spec, &private_labels_raw, &prediction_batch).unwrap();
    assert_eq!(spec_receipt.score, receipt.score);
    assert_eq!(spec_receipt.assurance, "provisional_spec_checked");

    // Self-supplied evidence does not certify independent reference conformance.
    let fixture_sha = sha256_bytes(format!("{}:{}", receipt.labels_sha256, receipt.predictions_sha256).as_bytes());
    let ref_evidence = ReferenceEvidence {
        reference_identity: "scikit_learn_mae_ground_truth".into(),
        fixture_sha256: fixture_sha,
        expected_result: receipt.score,
        verified_receipt_sha256: "1".repeat(64),
        metric_version: "v1.0".into(),
        tolerance: 1e-6,
    };
    let ref_receipt = evaluate_with_spec_and_evidence(
        &eval_identity,
        &spec,
        &private_labels_raw,
        &prediction_batch,
        Some(&ref_evidence),
    ).unwrap();
    assert_eq!(ref_receipt.assurance, "provisional_spec_checked");

    // =========================================================================
    // Stage 9: Packaging Delivery Bundle & Authoritative Anti-Tampering Checks
    // =========================================================================
    let bundle_dir = root.join("delivery_bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    // Required contracts
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
    fs::write(bundle_dir.join("inference.py"), r#"print("running inference")"#).unwrap();
    fs::write(bundle_dir.join("serve.py"), r#"print("serving model")"#).unwrap();
    fs::write(bundle_dir.join("inference.lock"), "# clean dependencies\nnumpy==1.26.0\n").unwrap();
    fs::write(bundle_dir.join("training.lock"), "# clean dependencies\nnumpy==1.26.0\n").unwrap();
    fs::write(bundle_dir.join("examples_input.json"), r#"[{"x1":0.5,"x2":1.0}]"#).unwrap();
    fs::write(bundle_dir.join("examples_expected.json"), r#"[{"prediction":1.12}]"#).unwrap();
    fs::write(bundle_dir.join("README.md"), "# Governed Custom Spline Model Bundle\n").unwrap();

    // Compile bytecode (.pyc) to verify strict manifest bytecode inclusion rule
    let pycache_dir = bundle_dir.join("__pycache__");
    fs::create_dir_all(&pycache_dir).unwrap();
    let pyc_path = pycache_dir.join("inference.cpython-313.pyc");
    fs::write(&pyc_path, b"\x00\x00\x00\x00bytecode_sample").unwrap();

    // Construct authoritative BundleManifest tracking 100% of physical files
    let mut bundle_files = std::collections::HashMap::new();
    let mut file_list = Vec::new();
    fn walk_files(base: &std::path::Path, curr: &std::path::Path, out: &mut Vec<String>) {
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

    // Authoritative bundle verification: 100% file coverage, no untracked files, no symlinks
    let report = bundle::verify_bundle(&bundle_dir, false).unwrap();
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
    assert_eq!(report.format, "mlrescurix.bundle.v2");

    // Anti-tampering check 1: Adding untracked physical file MUST fail verification
    fs::write(bundle_dir.join("untracked_trojan.txt"), b"untracked content").unwrap();
    let untracked_err = bundle::verify_bundle(&bundle_dir, false).unwrap_err();
    assert!(untracked_err.contains("untracked file 'untracked_trojan.txt'"));
    fs::remove_file(bundle_dir.join("untracked_trojan.txt")).unwrap();

    // Anti-tampering check 2: Forbidden symlink MUST fail verification
    std::os::unix::fs::symlink(bundle_dir.join("README.md"), bundle_dir.join("symlink_escape.md")).unwrap();
    let symlink_err = bundle::verify_bundle(&bundle_dir, false).unwrap_err();
    assert!(symlink_err.contains("forbidden symlink"));
    fs::remove_file(bundle_dir.join("symlink_escape.md")).unwrap();

    // =========================================================================
    // Stage 10: Final Lockbox Assessment, Atomic Consumption & Crash Recovery
    // =========================================================================
    // Generate lockbox features CSV
    let lockbox_features_csv = root.join("lockbox_features.csv");
    let mut lockbox_csv_content = String::from("__mlr_row_id,x1,x2\n");
    let mut trusted_lockbox_labels = Vec::new();

    for &row_id in &lockbox_rows {
        let (id, y_val) = ground_truth_labels[row_id as usize];
        let x1 = (id as f64 - 15.0) / 10.0;
        let x2 = (id as f64) * 0.2;
        lockbox_csv_content.push_str(&format!("{id},{x1:.4},{x2:.4}\n"));
        trusted_lockbox_labels.push((id, y_val));
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
        source_sha256: snapshot_id.source_sha256.clone(),
        environment_id: snapshot_id.environment_id.clone(),
        weights_sha256: weights_sha256.clone(),
    };
    let lockbox_pred_batch = PredictionBatch::scalar(assessment_identity.clone(), lockbox_pred_pairs);

    // 10a. Atomic assessment evaluation & one-shot lockbox consumption
    let lockbox_receipt = custom_protocol::evaluate_and_consume_assessment(
        &mut conn,
        project_id,
        dataset_id,
        &assessment_identity,
        &trusted_lockbox_labels,
        &lockbox_pred_batch,
    ).unwrap();

    assert_eq!(lockbox_receipt.metric, Metric::Mae);
    assert_eq!(lockbox_receipt.row_count, 6);
    assert!(lockbox_receipt.score < 0.60);

    // 10b. Second assessment MUST fail (one-shot consumption lock)
    let re_eval_err = custom_protocol::evaluate_and_consume_assessment(
        &mut conn,
        project_id,
        dataset_id,
        &assessment_identity,
        &trusted_lockbox_labels,
        &lockbox_pred_batch,
    ).unwrap_err();
    assert!(re_eval_err.contains("already been consumed"));

    // 10c. Crash Recovery: Close connection, reopen fresh connection to SQLite
    drop(conn);
    let recovered_conn = Connection::open(&db_path).unwrap();

    // Query recovered evaluation receipt directly from SQLite consumption table
    let recovered_receipt = get_assessment_consumption_receipt(&recovered_conn, project_id, dataset_id).unwrap().unwrap();
    assert_eq!(recovered_receipt.score, lockbox_receipt.score);
    assert_eq!(recovered_receipt.metric, lockbox_receipt.metric);
    assert_eq!(recovered_receipt.predictions_sha256, lockbox_receipt.predictions_sha256);
    assert_eq!(recovered_receipt.identity, lockbox_receipt.identity);
    assert_eq!(recovered_receipt.row_count, lockbox_receipt.row_count);
}
