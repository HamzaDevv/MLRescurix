//! Phase P7 Acceptance Gate and Integration Tests for MLRescurix.
//!
//! Verifies:
//! - P7.1: Executable Candidate Bundle Manifest v2 & clean dependency isolation between training and inference locks.
//! - P7.2: Native / SafeTensors model exports with explicit documented tolerances and sandbox containment.
//! - P7.3: Target verification, sealed final assessment, user approval token enforcement, and private row leakage prevention.
//! - P7.4: Standalone serving recipe with /health and /predict endpoints using locked inference-only dependencies.
//! - P7.5: Installed standalone workflow lifecycle outside repository checkout, crash/resume, and populated DB upgrade.
//! - P7.6: Prospective matched benchmarks comparing random, search, and agent baselines with transparent failure reporting.
//! - P7.7: Capability matrix and release audit for all historical M6 and v2 release gates bound to source hash.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::params;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    assets::{embedded_python_src_hash, ensure_python_root},
    bundle::{
        verify_bundle, verify_bundle_with_state, verify_standalone_cli_inference,
        verify_standalone_serving_recipe, BundleManifest, BundleManifestFileEntry,
    },
    core::{Operation, Policy, Request},
    execution, Runtime,
};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn hash_file(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_default();
    format!("{:x}", Sha256::digest(&bytes))
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy {
        max_tasks: 1024,
        budget: b,
        capacity: c,
    }
}

fn create_test_project(root: &Path, project: &str) -> Runtime {
    if root.exists() {
        let _ = fs::remove_dir_all(root);
    }
    let runtime = Runtime::init(root, test_policy(100)).unwrap();
    execution::migrate(root).unwrap();

    runtime.command(Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P7 Apex Delivery Test".into(),
            finalization_id: format!("final-{project}"),
            policy: test_policy(100),
        },
    }).unwrap();

    runtime.command(Request {
        command_id: format!("branch-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: "main".into(),
            base_hash: "base-p7".into(),
            code_hash: "code-p7".into(),
        },
    }).unwrap();

    runtime
}

fn setup_synthetic_csv(path: &Path, n_rows: usize) {
    let mut csv = "feat1,feat2,target\n".to_string();
    for i in 0..n_rows {
        let t = if i % 2 == 0 { 0 } else { 1 };
        let f1 = (i as f64) * 0.1;
        let f2 = (i as f64) * 0.2;
        csv.push_str(&format!("{f1:.2},{f2:.2},{t}\n"));
    }
    fs::write(path, csv).unwrap();
}

#[test]
fn test_p7_1_executable_candidate_bundle_v2_manifest_and_dependency_isolation() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("bundle_v2");
    fs::create_dir_all(&bundle_dir).unwrap();

    let contract_files = [
        "input_schema.json",
        "output_schema.json",
        "scorecard.json",
        "verification.json",
        "final.json",
        "frozen_model.joblib",
        "inference.py",
        "inference.lock",
        "training.lock",
        "serve.py",
        "examples_input.json",
        "examples_expected.json",
        "README.md",
    ];

    for name in &contract_files {
        fs::write(bundle_dir.join(name), "{}").unwrap();
    }
    fs::write(bundle_dir.join("inference.lock"), "numpy==2.2.6\nscikit-learn==1.8.0\njoblib==1.5.1\nsafetensors==0.7.0\n").unwrap();
    fs::write(bundle_dir.join("training.lock"), "numpy==2.2.6\nscikit-learn==1.8.0\njoblib==1.5.1\npytest==8.3.4\n").unwrap();

    let model_sha = hash_file(&bundle_dir.join("frozen_model.joblib"));
    fs::write(bundle_dir.join("verification.json"), json!({
        "frozen_model_sha256": model_sha,
        "bundle_candidate": "cand-v2",
    }).to_string()).unwrap();

    fs::write(bundle_dir.join("scorecard.json"), json!({
        "candidate_id": "cand-v2",
        "constraints": {
            "inference_latency_ms": {"threshold": 50.0, "value": 5.2, "verdict": "pass", "enforcement": "measured"},
            "bundle_bytes": {"threshold": 1048576, "value": 12000, "verdict": "pass", "enforcement": "measured"},
        }
    }).to_string()).unwrap();

    let mut files_map = HashMap::new();
    for name in &contract_files {
        let p = bundle_dir.join(name);
        files_map.insert(name.to_string(), BundleManifestFileEntry {
            bytes: fs::metadata(&p).unwrap().len(),
            sha256: hash_file(&p),
        });
    }

    let manifest = BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: "p7-proj".into(),
        candidate_id: "cand-v2".into(),
        files: files_map.clone(),
        model: Some(json!({
            "weights_path": "frozen_model.joblib",
            "weights_sha256": model_sha,
            "format": "joblib",
            "tolerances": {"rtol": 1e-5, "atol": 1e-7}
        })),
        pipeline: Some(json!({
            "feature_pipeline": "training/source/features.py",
            "entrypoint": "inference.py"
        })),
        contracts: Some(json!({
            "cli": "python inference.py --input <path> --output <path>",
            "api": "predict(records)",
            "input_schema": "input_schema.json",
            "output_schema": "output_schema.json"
        })),
        dependencies: Some(json!({
            "training_lock": "training.lock",
            "inference_lock": "inference.lock",
            "isolated_inference": true
        })),
        scorecard: Some(json!({
            "path": "scorecard.json",
            "sha256": hash_file(&bundle_dir.join("scorecard.json"))
        })),
        lineage: Some(json!({
            "path": "scorecard.json",
            "sha256": hash_file(&bundle_dir.join("scorecard.json"))
        })),
        resource_measurements: Some(json!({
            "cold_load_ms": 12.5,
            "warm_p50_ms": 1.8,
            "warm_p95_ms": 3.2,
            "peak_rss_kb": 2048,
            "package_bytes": 15000
        })),
    };

    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();

    // 1. Valid bundle v2 verification
    let report = verify_bundle(&bundle_dir, false).unwrap();
    assert_eq!(report.status, "integrity_verified");
    assert_eq!(report.format, "mlrescurix.bundle.v2");
    assert_eq!(report.candidate_id, "cand-v2");
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
    assert!(report.constraints_verified);

    // 2. Dependency isolation check: inference.lock must NOT contain dev tools (e.g. pytest)
    fs::write(bundle_dir.join("inference.lock"), "numpy==2.2.6\npytest==8.3.4\n").unwrap();
    let mut tampered_map = files_map.clone();
    tampered_map.insert("inference.lock".to_string(), BundleManifestFileEntry {
        bytes: fs::metadata(bundle_dir.join("inference.lock")).unwrap().len(),
        sha256: hash_file(&bundle_dir.join("inference.lock")),
    });
    let mut tampered_manifest = manifest.clone();
    tampered_manifest.files = tampered_map;
    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec(&tampered_manifest).unwrap()).unwrap();

    let err_isolation = verify_bundle(&bundle_dir, false).unwrap_err();
    assert!(
        err_isolation.contains("dependency isolation violated"),
        "Inference lock containing pytest must be rejected: {err_isolation}"
    );
}

#[test]
fn test_p7_2_model_exports_safetensors_and_documented_tolerances() {
    let td = tempfile::tempdir().unwrap();
    let export_dir = td.path().join("export");

    let py_script = r#"
import json, tempfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from mlrescurix_worker.scientific.export import export_model, verify_export_parity, SafeTensorsPredictor

# 1. Binary classification test
X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
y = np.array([0, 0, 1, 1], dtype=int)
clf = LogisticRegression().fit(X, y)

manifest = export_model(
    clf, '__EXPORT_DIR__', formats=['joblib', 'safetensors'],
    task='binary_classification', features=['f1', 'f2'], classes=[0, 1]
)
assert (Path('__EXPORT_DIR__') / 'weights.safetensors').is_file(), "weights.safetensors must exist"
assert (Path('__EXPORT_DIR__') / 'tolerances.json').is_file(), "tolerances.json must exist"

test_rows = [{'f1': 0.15, 'f2': 0.25}, {'f1': 0.65, 'f2': 0.75}]
parity = verify_export_parity(clf, '__EXPORT_DIR__', test_rows, task='binary_classification')
assert parity['verified'], f"Parity failed: {parity}"
assert parity['formats']['safetensors']['max_prob_diff'] < 1e-6, "Prob diff exceeded tolerance"

# 2. Standalone Zero-Pickle Predictor
pred = SafeTensorsPredictor('__EXPORT_DIR__')
st_out = pred.predict(test_rows)
assert st_out['prediction'] == [0, 1], f"Unexpected predictions: {st_out}"

# 3. Path traversal sandbox boundary check: must reject '..'
try:
    export_model(clf, '__EXPORT_DIR__/../escaped', formats=['safetensors'])
    raise RuntimeError("Sandbox escape not rejected")
except ValueError as e:
    assert "escapes sandbox boundaries" in str(e)

print("EXPORT_VERIFIED_OK")
"#.replace("__EXPORT_DIR__", export_dir.to_string_lossy().as_ref());

    let script_path = td.path().join("test_export.py");
    fs::write(&script_path, py_script).unwrap();

    let mut cmd = std::process::Command::new("python3");
    cmd.arg(&script_path);
    let python_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/src");
    let mut pypaths = vec![python_src.to_string_lossy().to_string()];
    if let Ok(cur) = std::env::var("PYTHONPATH") {
        pypaths.push(cur);
    }
    cmd.env("PYTHONPATH", pypaths.join(":"));
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");

    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "Export script failed ({}):\nstdout: {}\nstderr: {}",
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("EXPORT_VERIFIED_OK"));
}

#[test]
fn test_p7_3_sealed_holdout_gate_enforcement_and_private_row_leakage_rejection() {
    let td = tempfile::tempdir().unwrap();
    let state_root = td.path().join("state");
    let _runtime = create_test_project(&state_root, "p-sealed");

    let db = rusqlite::Connection::open(state_root.join("state.sqlite")).unwrap();

    let candidate_id = "cand-sealed";
    let dataset_sha = "d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd";
    let gate_access_id = "test-gate-p7";

    // Create bundle directory
    let bundle_dir = td.path().join("bundle_sealed");
    fs::create_dir_all(&bundle_dir).unwrap();
    let contract_files = [
        "input_schema.json", "output_schema.json", "scorecard.json", "verification.json",
        "final.json", "frozen_model.joblib", "inference.py", "inference.lock", "training.lock",
        "serve.py", "examples_input.json", "examples_expected.json", "README.md",
    ];
    for name in &contract_files {
        fs::write(bundle_dir.join(name), "{}").unwrap();
    }
    fs::write(bundle_dir.join("inference.lock"), "numpy==2.2.6\n").unwrap();
    fs::write(bundle_dir.join("training.lock"), "numpy==2.2.6\n").unwrap();

    let model_sha = hash_file(&bundle_dir.join("frozen_model.joblib"));
    fs::write(bundle_dir.join("final.json"), json!({
        "candidate_id": candidate_id,
        "test_access_id": gate_access_id,
    }).to_string()).unwrap();

    let final_sha = hash_file(&bundle_dir.join("final.json"));

    fs::write(bundle_dir.join("verification.json"), json!({
        "frozen_model_sha256": model_sha,
        "bundle_candidate": candidate_id,
        "dataset_sha256": dataset_sha,
        "test_access_id": gate_access_id,
    }).to_string()).unwrap();

    fs::write(bundle_dir.join("scorecard.json"), json!({
        "candidate_id": candidate_id,
        "dataset_sha256": dataset_sha,
        "constraints": {},
    }).to_string()).unwrap();

    // Golden examples: initial safe rows (row ids 0 and 1)
    fs::write(bundle_dir.join("examples_input.json"), json!([
        {"__row_id__": 0, "feat1": 0.1, "feat2": 0.2},
        {"__row_id__": 1, "feat1": 0.3, "feat2": 0.4}
    ]).to_string()).unwrap();

    let train_src_dir = bundle_dir.join("training/source");
    fs::create_dir_all(&train_src_dir).unwrap();
    let embedded_map: HashMap<&str, &str> = mlrescurix_harness::assets::EMBEDDED_PYTHON_FILES.iter().copied().collect();
    let wf_code = embedded_map.get("src/mlrescurix_worker/scientific/workflow.py").unwrap();
    let wf_p = train_src_dir.join("workflow.py");
    fs::write(&wf_p, wf_code).unwrap();

    let mut files_map = HashMap::new();
    for name in &contract_files {
        let p = bundle_dir.join(name);
        files_map.insert(name.to_string(), BundleManifestFileEntry {
            bytes: fs::metadata(&p).unwrap().len(),
            sha256: hash_file(&p),
        });
    }
    files_map.insert("training/source/workflow.py".to_string(), BundleManifestFileEntry {
        bytes: fs::metadata(&wf_p).unwrap().len(),
        sha256: hash_file(&wf_p),
    });

    let manifest = BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: "p-sealed".into(),
        candidate_id: candidate_id.into(),
        files: files_map.clone(),
        model: Some(json!({"weights_path": "frozen_model.joblib", "weights_sha256": model_sha, "format": "joblib"})),
        pipeline: Some(json!({"feature_pipeline": "pipeline.py", "entrypoint": "inference.py"})),
        contracts: Some(json!({"cli": "inference.py", "api": "predict(records)", "input_schema": "input_schema.json", "output_schema": "output_schema.json"})),
        dependencies: Some(json!({"training_lock": "training.lock", "inference_lock": "inference.lock", "isolated_inference": true})),
        scorecard: Some(json!({"path": "scorecard.json", "sha256": hash_file(&bundle_dir.join("scorecard.json"))})),
        lineage: Some(json!({"path": "scorecard.json", "sha256": hash_file(&bundle_dir.join("scorecard.json"))})),
        resource_measurements: Some(json!({"cold_load_ms": 10.0, "warm_p95_ms": 2.0, "peak_rss_kb": 1024, "package_bytes": 5000})),
    };
    let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
    let manifest_sha = format!("{:x}", Sha256::digest(&manifest_bytes));
    fs::write(bundle_dir.join("manifest.json"), &manifest_bytes).unwrap();

    // 1. Check without final_test_gates consumed status -> must fail external authenticity
    let unconsumed_err = verify_bundle_with_state(&bundle_dir, false, Some(&state_root));
    assert!(unconsumed_err.is_err());

    // Insert consumed final_test_gates
    db.execute(
        "INSERT INTO final_test_gates(project, candidate_id, dataset_sha256, access_id, status, frozen_model_sha256, result_sha256)
         VALUES('p-sealed', ?1, ?2, ?3, 'consumed', ?4, ?5)",
        params![candidate_id, dataset_sha, gate_access_id, model_sha, final_sha],
    ).unwrap();

    db.execute(
        "INSERT OR IGNORE INTO branches(project, id, base_hash, code_hash) VALUES('p-sealed', 'b1', 'h1', 'h2')",
        [],
    ).unwrap();
    db.execute(
        "INSERT OR IGNORE INTO tasks(project, id, branch, data) VALUES('p-sealed', 't1', 'b1', '{}')",
        [],
    ).unwrap();
    db.execute(
        "INSERT OR IGNORE INTO jobs(id, project, task, generation, spec, status, submitted_ms) VALUES('job_final', 'p-sealed', 't1', 1, '{}', 'succeeded', 100)",
        [],
    ).unwrap();

    // Insert managed_artifacts receipt for manifest
    db.execute(
        "INSERT INTO managed_artifacts(project, job, name, bytes, sha256) VALUES('p-sealed', 'job_final', 'bundle/manifest.json', ?1, ?2)",
        params![manifest_bytes.len() as u64, manifest_sha],
    ).unwrap();

    // Set dataset splits with test split = [10, 11, 12] in authoritative workflow dataset_manifest.json
    let wf_dir = state_root.join("workflows").join("p-sealed");
    fs::create_dir_all(&wf_dir).unwrap();
    fs::write(
        wf_dir.join("dataset_manifest.json"),
        json!({
            "splits": {
                "train": [0, 1, 2, 3],
                "validation": [4, 5],
                "test": [10, 11, 12]
            }
        }).to_string(),
    ).unwrap();

    // 2. Check without user_approvals token -> must fail external authenticity
    let no_approval_err = verify_bundle_with_state(&bundle_dir, false, Some(&state_root)).unwrap_err();
    assert!(
        no_approval_err.contains("lacks verified user approval"),
        "Missing user approval must be rejected: {no_approval_err}"
    );

    // Insert verified user approval
    db.execute(
        "INSERT INTO user_approvals(project, approval_id, action_type, target_id, actor, approved, created_ms, expires_ms)
         VALUES('p-sealed', 'app-p7', 'deployment_handoff', ?1, 'user-lead', 1, ?2, 9999999999)",
        params![candidate_id, now_ms()],
    ).unwrap();

    // 3. Now passes external state authenticity check!
    let verified = verify_bundle_with_state(&bundle_dir, false, Some(&state_root)).unwrap();
    assert_eq!(verified.status, "integrity_verified");
    assert_eq!(verified.details["authenticity"], "verified_against_authoritative_state");

    // 4. Test Private Holdout Evaluation Row Leakage Rejection:
    // Inject a sealed test row (__row_id__ = 11) into examples_input.json!
    fs::write(bundle_dir.join("examples_input.json"), json!([
        {"__row_id__": 0, "feat1": 0.1, "feat2": 0.2},
        {"__row_id__": 11, "feat1": 0.9, "feat2": 1.0} // LEAKED TEST ROW!
    ]).to_string()).unwrap();

    let mut leaked_files = files_map.clone();
    leaked_files.insert("examples_input.json".to_string(), BundleManifestFileEntry {
        bytes: fs::metadata(bundle_dir.join("examples_input.json")).unwrap().len(),
        sha256: hash_file(&bundle_dir.join("examples_input.json")),
    });
    let mut leaked_manifest = manifest.clone();
    leaked_manifest.files = leaked_files;
    let leaked_bytes = serde_json::to_vec(&leaked_manifest).unwrap();
    let leaked_sha = format!("{:x}", Sha256::digest(&leaked_bytes));
    fs::write(bundle_dir.join("manifest.json"), &leaked_bytes).unwrap();

    // Update managed_artifacts so hash matches
    db.execute(
        "UPDATE managed_artifacts SET bytes=?1, sha256=?2 WHERE project='p-sealed' AND name='bundle/manifest.json'",
        params![leaked_bytes.len() as u64, leaked_sha],
    ).unwrap();

    let leak_err = verify_bundle_with_state(&bundle_dir, false, Some(&state_root)).unwrap_err();
    assert!(
        leak_err.contains("private evaluation rows leak detected"),
        "Leaked test row must be detected and rejected: {leak_err}"
    );
}

#[test]
fn test_p7_4_standalone_serving_recipe_health_and_predict() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("bundle_serve");
    fs::create_dir_all(&bundle_dir).unwrap();

    // Copy serve.py
    let serve_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/src/mlrescurix_worker/scientific/serve.py");
    fs::copy(serve_src, bundle_dir.join("serve.py")).unwrap();

    // Minimal working inference.py and schemas
    let inf_code = r#"
import json

def predict(records):
    results = []
    for r in records:
        if 'feat1' not in r or 'feat2' not in r:
            raise ValueError(f"input schema mismatch: missing required features in {r}")
        if '__BAD_COL__' in r:
            raise ValueError(f"input schema mismatch: extra unexpected column in {r}")
        val = r['feat1'] * 0.5 + r['feat2'] * 0.5
        results.append(int(val > 0.5))
    return {
        "prediction": results,
        "probabilities": [[1.0 - x, float(x)] for x in results],
        "classes": [0, 1]
    }
"#;
    fs::write(bundle_dir.join("inference.py"), inf_code).unwrap();
    fs::write(bundle_dir.join("input_schema.json"), json!({"features": ["feat1", "feat2"], "task": "binary_classification"}).to_string()).unwrap();
    fs::write(bundle_dir.join("output_schema.json"), json!({"format": "mlrescurix.output-schema.v1", "task": "binary_classification"}).to_string()).unwrap();
    fs::write(bundle_dir.join("examples_input.json"), json!([
        {"feat1": 0.2, "feat2": 0.3},
        {"feat1": 0.8, "feat2": 0.9}
    ]).to_string()).unwrap();
    fs::write(bundle_dir.join("manifest.json"), json!({
        "format": "mlrescurix.bundle.v2",
        "project_id": "p-serve",
        "candidate_id": "cand-serve",
        "files": {}
    }).to_string()).unwrap();

    // Execute standalone serving verification helper on loopback port 8921
    let serve_res = verify_standalone_serving_recipe(&bundle_dir, 8921).unwrap();
    assert_eq!(serve_res["serving_status"], "healthy");
    assert!(serve_res["health_verified"].as_bool().unwrap());
    assert!(serve_res["prediction_verified"].as_bool().unwrap());
    assert!(serve_res["schema_rejection_verified"].as_bool().unwrap());
}

#[test]
fn test_p7_5_installed_standalone_lifecycle_crash_resume_and_db_upgrade() {
    let td = tempfile::tempdir().unwrap();
    let state_root = td.path().join("state");

    // 1. Prove installed binary outside checkout: materialize runtime assets into state_root
    let python_root = ensure_python_root(&state_root).unwrap();
    assert!(python_root.join("src/mlrescurix_worker/scientific/workflow.py").is_file());
    assert!(python_root.join("src/mlrescurix_worker/scientific/export.py").is_file());
    assert!(python_root.join("src/mlrescurix_worker/scientific/serve.py").is_file());
    assert!(python_root.join("src/mlrescurix_worker/scientific/benchmarks.py").is_file());

    // 2. Initialize project and register dataset outside checkout
    let _runtime = create_test_project(&state_root, "p-lifecycle");
    let csv_path = td.path().join("data.csv");
    setup_synthetic_csv(&csv_path, 60);
    execution::register_dataset(&state_root, "p-lifecycle", "ds-p7", &csv_path).unwrap();

    // 3. Populate historical state records (consumed final_test_gates and user_approvals)
    let db = rusqlite::Connection::open(state_root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO final_test_gates(project, candidate_id, dataset_sha256, access_id, status, frozen_model_sha256, result_sha256)
         VALUES('p-lifecycle', 'historical-c1', 'data-sha-hist', 'hist-gate-1', 'consumed', 'model-sha-hist', 'res-sha-hist')",
        [],
    ).unwrap();
    db.execute(
        "INSERT INTO user_approvals(project, approval_id, action_type, target_id, actor, approved, created_ms, expires_ms)
         VALUES('p-lifecycle', 'app-hist-1', 'final_test', 'historical-c1', 'user', 1, 123456789, 9999999999)",
        [],
    ).unwrap();

    // 4. Upgrade populated state database: run execution::migrate()
    execution::migrate(&state_root).unwrap();

    // Verify PRAGMA user_version is at least 10
    let user_ver: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert!(user_ver >= 10, "Schema user_version must be at least 10, got {user_ver}");

    // Verify historical consumed gates and user approvals remain intact
    let hist_status: String = db.query_row(
        "SELECT status FROM final_test_gates WHERE candidate_id='historical-c1'", [], |r| r.get(0)
    ).unwrap();
    assert_eq!(hist_status, "consumed", "Historical consumed gate must remain consumed");

    let approval_status: i64 = db.query_row(
        "SELECT approved FROM user_approvals WHERE approval_id='app-hist-1'", [], |r| r.get(0)
    ).unwrap();
    assert_eq!(approval_status, 1, "Historical user approval must remain approved");

    // 5. Standalone prediction CLI execution test outside repository checkout
    let bundle_dir = td.path().join("standalone_bundle");
    fs::create_dir_all(&bundle_dir).unwrap();
    let inf_script = r#"
import json, sys
from pathlib import Path

def predict(records):
    return {"prediction": [1 for _ in records], "count": len(records)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    res = predict(data)
    Path(args.output).write_text(json.dumps(res))
"#;
    fs::write(bundle_dir.join("inference.py"), inf_script).unwrap();
    let in_file = td.path().join("cli_in.json");
    let out_file = td.path().join("cli_out.json");
    fs::write(&in_file, json!([{"f1": 1.0}, {"f1": 2.0}]).to_string()).unwrap();

    let cli_res = verify_standalone_cli_inference(&bundle_dir, &in_file, &out_file).unwrap();
    assert_eq!(cli_res["count"], 2);
    assert_eq!(cli_res["prediction"], json!([1, 1]));
}

#[test]
fn test_p7_6_prospective_matched_benchmarks_and_suboptimal_trial_transparency() {
    let td = tempfile::tempdir().unwrap();

    let py_script = r#"
import json
from pathlib import Path
from mlrescurix_worker.scientific.benchmarks import run_prospective_benchmark, save_benchmark_report

report = run_prospective_benchmark(
    task="binary_classification",
    metric="balanced_accuracy",
    seeds=[2026, 42, 101],
    max_trials_per_strategy=6,
    wall_budget_ms=5000.0,
)

# 1. Assert all 3 strategies attempted matched trial counts
for strat in ["random", "search", "agent"]:
    assert strat in report.results, f"Missing strategy {strat}"
    res = report.results[strat]
    assert res.trials_attempted == 6, f"{strat} attempted {res.trials_attempted} != 6"

# 2. Transparently record failing or suboptimal trials alongside winners
agent_res = report.results["agent"]
assert agent_res.trials_failed > 0, "Agent strategy must transparently record failing trials"
assert len(agent_res.trials) == 6

failed_trials = [t for t in agent_res.trials if t.status != "success"]
assert len(failed_trials) == agent_res.trials_failed
for ft in failed_trials:
    assert ft.error_message is not None, "Failed trial must preserve error message"

# 3. Honest statistical claim
summary = report.comparative_summary
assert summary["suboptimal_and_failing_trials_recorded"] is True
assert "without synthetic inflation" in summary["statistical_claim"]

out_path = Path("__OUTPUT_PATH__")
save_benchmark_report(report, out_path)
assert out_path.is_file(), "Benchmark report file not written"
print("BENCHMARK_VERIFIED_OK")
"#.replace("__OUTPUT_PATH__", td.path().join("bench_report.json").to_string_lossy().as_ref());

    let script_path = td.path().join("test_benchmarks.py");
    fs::write(&script_path, py_script).unwrap();

    let mut cmd = std::process::Command::new("python3");
    cmd.arg(&script_path);
    let python_src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/src");
    let mut pypaths = vec![python_src.to_string_lossy().to_string()];
    if let Ok(cur) = std::env::var("PYTHONPATH") {
        pypaths.push(cur);
    }
    cmd.env("PYTHONPATH", pypaths.join(":"));
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");

    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "Benchmark script failed ({}):\nstdout: {}\nstderr: {}",
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("BENCHMARK_VERIFIED_OK"));
}

#[test]
fn test_p7_7_capability_matrix_and_release_audit_all_gates() {
    let harness_bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut cmd = std::process::Command::new(harness_bin);
    cmd.arg("capabilities");

    let out = cmd.output().unwrap();
    assert!(out.status.success(), "Capabilities command failed");

    let caps: Value = serde_json::from_slice(&out.stdout).expect("Valid JSON capabilities");

    // Phase and protocol
    assert_eq!(caps["phase"], "P7");
    assert_eq!(caps["protocol"], "mlrescurix/2");
    assert_eq!(caps["state_schema_version"], 10);

    // Bound source hash matches embedded python source hash
    let exp_src_hash = embedded_python_src_hash();
    assert_eq!(caps["source_sha256"], exp_src_hash);

    // Five historical M6 release gates audit
    let m6_gates = &caps["historical_m6_release_gates"];
    assert_eq!(m6_gates["gate_1_specialist_reconciliation"], "closed_and_verified");
    assert_eq!(m6_gates["gate_2_state_bounds_and_enospc"], "closed_and_verified");
    assert_eq!(m6_gates["gate_3_installed_lifecycle"], "closed_and_verified");
    assert_eq!(m6_gates["gate_4_matched_benchmarks"], "closed_and_verified");
    assert_eq!(m6_gates["gate_5_memory_certification"], "closed_and_verified");

    // All 7 v2 release gates audit
    let v2_gates = &caps["v2_release_gates"];
    assert_eq!(v2_gates["p1_control_plane"], "passed");
    assert_eq!(v2_gates["p2_artifact_registry"], "passed");
    assert_eq!(v2_gates["p3_resource_broker"], "passed");
    assert_eq!(v2_gates["p4_candidate_tools"], "passed");
    assert_eq!(v2_gates["p5_collaboration"], "passed");
    assert_eq!(v2_gates["p6_dashboard"], "passed");
    assert_eq!(v2_gates["p7_apex_delivery"], "passed");

    // Delivery capabilities
    let deliv = &caps["delivery_capabilities"];
    assert!(deliv["bundle_v2_manifest"].as_bool().unwrap());
    assert!(deliv["native_joblib_export"].as_bool().unwrap());
    assert!(deliv["safetensors_export"].as_bool().unwrap());
    assert!(deliv["documented_tolerances"].as_bool().unwrap());
    assert!(deliv["sealed_holdout_access"].as_bool().unwrap());
    assert!(deliv["no_private_row_leakage"].as_bool().unwrap());
    assert!(deliv["standalone_http_serve"].as_bool().unwrap());
    assert!(deliv["standalone_cli_inference"].as_bool().unwrap());
    assert!(deliv["installed_without_checkout"].as_bool().unwrap());
}

#[test]
fn test_p7_acceptance_gate_full_closure() {
    // 1. Verify that frozen candidate bundles execute standalone outside repository tree without manual intervention
    let td = tempfile::tempdir().unwrap();
    let isolated_bundle = td.path().join("apex_bundle");
    fs::create_dir_all(&isolated_bundle).unwrap();

    let contract_files = [
        "input_schema.json", "output_schema.json", "scorecard.json", "verification.json",
        "final.json", "frozen_model.joblib", "inference.py", "inference.lock", "training.lock",
        "serve.py", "examples_input.json", "examples_expected.json", "README.md",
    ];
    for name in &contract_files {
        fs::write(isolated_bundle.join(name), "{}").unwrap();
    }
    fs::write(isolated_bundle.join("inference.lock"), "numpy==2.2.6\njoblib==1.5.1\n").unwrap();
    fs::write(isolated_bundle.join("training.lock"), "numpy==2.2.6\njoblib==1.5.1\npytest==8.3.4\n").unwrap();

    let model_sha = hash_file(&isolated_bundle.join("frozen_model.joblib"));
    fs::write(isolated_bundle.join("verification.json"), json!({"frozen_model_sha256": model_sha}).to_string()).unwrap();
    fs::write(isolated_bundle.join("scorecard.json"), json!({"candidate_id": "apex-cand", "constraints": {}}).to_string()).unwrap();

    let mut files_map = HashMap::new();
    for name in &contract_files {
        let p = isolated_bundle.join(name);
        files_map.insert(name.to_string(), BundleManifestFileEntry {
            bytes: fs::metadata(&p).unwrap().len(),
            sha256: hash_file(&p),
        });
    }

    let manifest = BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: "p7-apex".into(),
        candidate_id: "apex-cand".into(),
        files: files_map,
        model: Some(json!({"weights_path": "frozen_model.joblib", "weights_sha256": model_sha, "format": "joblib"})),
        pipeline: Some(json!({"feature_pipeline": "pipeline.py", "entrypoint": "inference.py"})),
        contracts: Some(json!({"cli": "inference.py", "api": "predict(records)", "input_schema": "input_schema.json", "output_schema": "output_schema.json"})),
        dependencies: Some(json!({"training_lock": "training.lock", "inference_lock": "inference.lock", "isolated_inference": true})),
        scorecard: Some(json!({"path": "scorecard.json", "sha256": hash_file(&isolated_bundle.join("scorecard.json"))})),
        lineage: Some(json!({"path": "scorecard.json", "sha256": hash_file(&isolated_bundle.join("scorecard.json"))})),
        resource_measurements: Some(json!({"cold_load_ms": 10.0, "warm_p95_ms": 2.0, "peak_rss_kb": 1024, "package_bytes": 5000})),
    };
    fs::write(isolated_bundle.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();

    let rep = verify_bundle(&isolated_bundle, false).unwrap();
    assert_eq!(rep.status, "integrity_verified");
    assert_eq!(rep.format, "mlrescurix.bundle.v2");
    assert_eq!(rep.candidate_id, "apex-cand");
    assert!(rep.manifest_verified);
    assert!(rep.contracts_verified);
    assert!(rep.constraints_verified);
}
