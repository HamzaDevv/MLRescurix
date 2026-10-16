use std::{
    collections::HashMap,
    fs,
    path::Path,
};
use serde_json::json;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    assets::EMBEDDED_PYTHON_FILES,
    bundle::verify_bundle,
};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_valid_p11_bundle(bundle_dir: &Path, project_id: &str, candidate_id: &str) {
    fs::create_dir_all(bundle_dir).unwrap();

    let frozen_model_content = b"DUMMY_FROZEN_JOBLIB_WEIGHTS_FOR_P11_CALIBRATED_MODEL";
    fs::write(bundle_dir.join("frozen_model.joblib"), frozen_model_content).unwrap();
    let frozen_sha = sha256_bytes(frozen_model_content);

    let input_schema = json!({
        "format": "mlrescurix.schema.v2",
        "task": "binary_classification",
        "features": ["feature_num1", "feature_num2"],
        "dtypes": {"feature_num1": "float64", "feature_num2": "float64"}
    });
    fs::write(bundle_dir.join("input_schema.json"), serde_json::to_vec_pretty(&input_schema).unwrap()).unwrap();

    let output_schema = json!({
        "format": "mlrescurix.schema.v2",
        "task": "binary_classification",
        "prediction_type": "integer",
        "classes": [0, 1],
        "decision_threshold": 0.42
    });
    fs::write(bundle_dir.join("output_schema.json"), serde_json::to_vec_pretty(&output_schema).unwrap()).unwrap();

    let model_schema = json!({
        "format": "mlrescurix.schema.v2",
        "task": "binary_classification",
        "model_type": "LogisticRegression",
        "features": ["feature_num1", "feature_num2"],
        "classes": [0, 1],
        "decision_threshold": 0.42,
        "calibration_method": "isotonic",
        "calibration_metrics": {
            "brier_before": 0.1852,
            "brier_after": 0.1421,
            "brier_delta": 0.0431
        },
        "reliability_curves": {
            "n_bins": 10,
            "brier_score": 0.1421
        }
    });
    fs::write(bundle_dir.join("model_schema.json"), serde_json::to_vec_pretty(&model_schema).unwrap()).unwrap();

    let scorecard = json!({
        "candidate_id": candidate_id,
        "project_id": project_id,
        "dataset_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "test_access_id": "gate-p11-apex",
        "primary_metric": 0.892,
        "metrics": {"balanced_accuracy": 0.892},
        "constraints": {
            "inference_latency_ms": {"threshold": 50.0, "value": 2.4, "verdict": "pass", "enforcement": "measured"},
            "bundle_bytes": {"threshold": 10485760, "value": 20480, "verdict": "pass", "enforcement": "measured"}
        }
    });
    fs::write(bundle_dir.join("scorecard.json"), serde_json::to_vec_pretty(&scorecard).unwrap()).unwrap();

    let verification = json!({
        "frozen_model_sha256": frozen_sha,
        "bundle_candidate": candidate_id,
        "dataset_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "test_access_id": "gate-p11-apex",
        "raw_input_api": "predict(records)"
    });
    fs::write(bundle_dir.join("verification.json"), serde_json::to_vec_pretty(&verification).unwrap()).unwrap();

    let final_data = json!({
        "status": "verified_development_only",
        "candidate_id": candidate_id,
        "test_metrics": {"balanced_accuracy": 0.892},
        "test_access_id": "gate-p11-apex",
        "bundle": "bundle"
    });
    fs::write(bundle_dir.join("final.json"), serde_json::to_vec_pretty(&final_data).unwrap()).unwrap();

    let inference_code = r#"
import json
from pathlib import Path

def predict(records):
    return {"prediction": [1] * len(records)}
"#;
    fs::write(bundle_dir.join("inference.py"), inference_code.trim()).unwrap();

    let serve_code = r#"
import http.server

def main():
    pass
"#;
    fs::write(bundle_dir.join("serve.py"), serve_code.trim()).unwrap();

    let inf_lock = "joblib==1.5.1\nnumpy==2.2.6\npandas==2.2.3\nscikit-learn==1.8.0\n";
    fs::write(bundle_dir.join("inference.lock"), inf_lock).unwrap();

    let train_lock = "joblib==1.5.1\nnumpy==2.2.6\npandas==2.2.3\nscikit-learn==1.8.0\n";
    fs::write(bundle_dir.join("training.lock"), train_lock).unwrap();

    let examples_in = json!([{"feature_num1": 0.5, "feature_num2": -1.2}]);
    fs::write(bundle_dir.join("examples_input.json"), serde_json::to_vec_pretty(&examples_in).unwrap()).unwrap();

    let examples_exp = json!({"prediction": [1]});
    fs::write(bundle_dir.join("examples_expected.json"), serde_json::to_vec_pretty(&examples_exp).unwrap()).unwrap();

    let readme = format!("# Bundle {candidate_id}\n\nProject: {project_id}\nDecision Threshold: 0.42\n");
    fs::write(bundle_dir.join("README.md"), readme).unwrap();

    // Collect all physical files and compute SHA256
    let mut files_map = serde_json::Map::new();
    for entry in fs::read_dir(bundle_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            let bytes = fs::read(&path).unwrap();
            let sha = sha256_bytes(&bytes);
            files_map.insert(name, json!({
                "bytes": bytes.len(),
                "sha256": sha
            }));
        }
    }

    let manifest = json!({
        "format": "mlrescurix.bundle.v2",
        "project_id": project_id,
        "candidate_id": candidate_id,
        "model": {
            "weights_path": "frozen_model.joblib",
            "weights_sha256": frozen_sha,
            "format": "joblib",
            "tolerances": {"rtol": 1e-5, "atol": 1e-7},
            "decision_threshold": 0.42,
            "calibration_method": "isotonic",
            "calibration_metrics": {
                "brier_before": 0.1852,
                "brier_after": 0.1421,
                "brier_delta": 0.0431
            },
            "reliability_curves": {
                "n_bins": 10,
                "brier_score": 0.1421
            }
        },
        "pipeline": {
            "feature_pipeline": "inference.py",
            "entrypoint": "inference.py"
        },
        "contracts": {
            "cli": "python3 inference.py --input <path> --output <path>",
            "api": "predict(records)",
            "input_schema": "input_schema.json",
            "output_schema": "output_schema.json",
            "model_schema": "model_schema.json"
        },
        "dependencies": {
            "training_lock": "training.lock",
            "inference_lock": "inference.lock",
            "isolated_inference": true
        },
        "scorecard": {
            "path": "scorecard.json",
            "sha256": files_map["scorecard.json"]["sha256"].as_str().unwrap()
        },
        "lineage": {
            "path": "scorecard.json",
            "sha256": files_map["scorecard.json"]["sha256"].as_str().unwrap()
        },
        "resource_measurements": {
            "cold_load_ms": 10.5,
            "warm_p50_ms": 1.9,
            "warm_p95_ms": 3.2,
            "peak_rss_kb": 22000,
            "package_bytes": 15000
        },
        "files": files_map
    });

    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

#[test]
fn p11_1_embedded_assets_contain_all_training_handoff_primitives() {
    let embedded_map: HashMap<&str, &str> = EMBEDDED_PYTHON_FILES.iter().copied().collect();

    // 1. Verify all required scientific worker modules are compiled in
    let required_modules = [
        "src/mlrescurix_worker/scientific/calibration.py",
        "src/mlrescurix_worker/scientific/features.py",
        "src/mlrescurix_worker/scientific/candidate_synthesis.py",
        "src/mlrescurix_worker/scientific/pareto_ensemble.py",
        "src/mlrescurix_worker/scientific/export.py",
        "src/mlrescurix_worker/scientific/bundle.py",
        "src/mlrescurix_worker/scientific/autonomous_rescue.py",
    ];

    for module in &required_modules {
        assert!(embedded_map.contains_key(module), "Missing embedded module: {module}");
    }

    // 2. Verify calibration primitives
    let calib_src = embedded_map["src/mlrescurix_worker/scientific/calibration.py"];
    assert!(calib_src.contains("class SplitAwareCalibrator"), "calibration.py missing SplitAwareCalibrator");
    assert!(calib_src.contains("def compute_reliability_curve"), "calibration.py missing compute_reliability_curve");
    assert!(calib_src.contains("def optimize_decision_threshold"), "calibration.py missing optimize_decision_threshold");
    assert!(calib_src.contains("def calibrate_probabilities"), "calibration.py missing calibrate_probabilities");

    // 3. Verify feature engineering primitives
    let feat_src = embedded_map["src/mlrescurix_worker/scientific/features.py"];
    assert!(feat_src.contains("class OutOfFoldTargetEncoder"), "features.py missing OutOfFoldTargetEncoder");
    assert!(feat_src.contains("class FrequencyEncoder"), "features.py missing FrequencyEncoder");
    assert!(feat_src.contains("class NumericFeatureCrosses"), "features.py missing NumericFeatureCrosses");
    assert!(feat_src.contains("class CyclicalDatetimeEncoder"), "features.py missing CyclicalDatetimeEncoder");

    // 4. Verify Pareto ensemble synthesis
    let pareto_src = embedded_map["src/mlrescurix_worker/scientific/pareto_ensemble.py"];
    assert!(pareto_src.contains("class ParetoEnsembleSynthesizer"), "pareto_ensemble.py missing ParetoEnsembleSynthesizer");
    assert!(pareto_src.contains("def optimize_bayesian_blend_weights"), "pareto_ensemble.py missing optimize_bayesian_blend_weights");
}

#[test]
fn p11_2_bundle_v2_with_calibration_and_model_schema_passes_verification() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("apex_calibrated_bundle");

    create_valid_p11_bundle(&bundle_dir, "proj-p11-test", "cand-p11-calibrated");

    // Authoritative Rust verification
    let report = verify_bundle(&bundle_dir, false).expect("Bundle verification failed");
    assert_eq!(report.status, "integrity_verified");
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
    assert!(report.constraints_verified);
    assert_eq!(report.candidate_id, "cand-p11-calibrated");
    assert_eq!(report.format, "mlrescurix.bundle.v2");
}

#[test]
fn p11_3_negative_scientific_guard_tampered_model_schema() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("tampered_schema_bundle");

    create_valid_p11_bundle(&bundle_dir, "proj-p11-test", "cand-p11-tampered");

    // Tamper with model_schema.json without updating manifest
    let schema_path = bundle_dir.join("model_schema.json");
    let mut schema_val: serde_json::Value = serde_json::from_slice(&fs::read(&schema_path).unwrap()).unwrap();
    schema_val["decision_threshold"] = json!(0.999);
    fs::write(&schema_path, serde_json::to_vec_pretty(&schema_val).unwrap()).unwrap();

    let err = verify_bundle(&bundle_dir, false).unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("mismatch") || err_str.contains("Tampered") || err_str.contains("SHA256"),
        "Expected anti-tampering failure for model_schema.json, got: {err_str}"
    );
}

#[test]
fn p11_4_negative_scientific_guard_dependency_isolation_violation() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("dependency_violation_bundle");

    create_valid_p11_bundle(&bundle_dir, "proj-p11-test", "cand-p11-dep-violation");

    // Inject forbidden development tool 'pytest' into inference.lock
    let lock_path = bundle_dir.join("inference.lock");
    fs::write(&lock_path, "joblib==1.5.1\npytest==8.3.4\n").unwrap();

    // Re-hash inference.lock in manifest so hash matches, but dependency isolation fails
    let mut manifest: serde_json::Value = serde_json::from_slice(&fs::read(bundle_dir.join("manifest.json")).unwrap()).unwrap();
    let lock_bytes = fs::read(&lock_path).unwrap();
    manifest["files"]["inference.lock"]["bytes"] = json!(lock_bytes.len());
    manifest["files"]["inference.lock"]["sha256"] = json!(sha256_bytes(&lock_bytes));
    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let err = verify_bundle(&bundle_dir, false).unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("dependency isolation violated") && err_str.contains("pytest"),
        "Expected dependency isolation failure, got: {err_str}"
    );
}

#[test]
fn p11_5_negative_scientific_guard_untracked_files_rejected() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("untracked_file_bundle");

    create_valid_p11_bundle(&bundle_dir, "proj-p11-test", "cand-p11-untracked");

    // Add an untracked file to bundle directory
    fs::write(bundle_dir.join("untracked_exploit.py"), "print('untracked')").unwrap();

    let err = verify_bundle(&bundle_dir, false).unwrap_err();
    let err_str = err.to_string();
    assert!(
        err_str.contains("untracked file") && err_str.contains("found in bundle"),
        "Expected untracked file error, got: {err_str}"
    );
}
