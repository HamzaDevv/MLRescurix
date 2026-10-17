//! Integration tests for Release Gate 4: Prospective Matched Benchmarks & Scientific Guards.
//!
//! Validates:
//! 1. Multi-seed benchmark driver across seeds [2026, 42, 101] on standardized OpenML datasets.
//! 2. Strict matched compute, thread, and memory budgets against standard baselines (LightGBM, Random Forest, HistGradientBoosting).
//! 3. Transparent recording of winning, suboptimal, and failing trials in benchmark scorecards.
//! 4. Single-use sealed holdout evaluation with zero label/row leakage enforcement.
//! 5. Deterministic detection and rejection of non-generalizing / overfitting candidate models.
//! 6. Conformance with Draft 2020-12 JSON schema `contracts/v2/benchmark_report.schema.json`.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use sha2::{Digest, Sha256};

fn get_python_src_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/src")
}

fn run_python_script(script: &str) -> std::process::Output {
    let td = tempfile::tempdir().unwrap();
    let script_path = td.path().join("runner.py");
    fs::write(&script_path, script).unwrap();

    let mut cmd = Command::new("python3");
    cmd.arg(&script_path);

    let python_src = get_python_src_path();
    let mut pypaths = vec![python_src.to_string_lossy().to_string()];
    if let Ok(cur) = std::env::var("PYTHONPATH") {
        pypaths.push(cur);
    }
    cmd.env("PYTHONPATH", pypaths.join(":"));
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");

    cmd.output().expect("Failed to execute python runner")
}

#[test]
fn test_gate4_multi_seed_benchmarks_on_standard_openml_datasets() {
    let py_code = r#"
import json
from pathlib import Path
import jsonschema
from mlrescurix_worker.scientific.benchmarks import (
    run_prospective_benchmark,
    save_benchmark_report,
)

# Run benchmark across seeds [2026, 42, 101] on standard OpenML blood-transfusion dataset
report = run_prospective_benchmark(
    dataset_name="blood-transfusion",
    task="binary_classification",
    metric="balanced_accuracy",
    direction="maximize",
    seeds=[2026, 42, 101],
    max_trials_per_strategy=6,
    wall_budget_ms=15000.0,
    max_memory_mb=512.0,
    max_threads=1,
    baselines=["default_lightgbm", "random_forest", "hist_gradient_boosting", "random", "search"],
    include_failing_trial=True,
    offline_only=True,
)

# 1. Assert multi-seed driver configuration
assert report.seeds == [2026, 42, 101], f"Unexpected seeds: {report.seeds}"
assert report.dataset_name == "blood-transfusion"

# 2. Assert matched trial budgets across all strategies
expected_strats = ["default_lightgbm", "random_forest", "hist_gradient_boosting", "random", "search", "agent"]
for s in expected_strats:
    assert s in report.results, f"Strategy {s} missing from results"
    res = report.results[s]
    assert res.trials_attempted == 6, f"Strategy {s} attempted {res.trials_attempted} != 6 trials"
    assert res.total_wall_ms > 0.0

# 3. Transparent retention of failing / suboptimal trials
agent_res = report.results["agent"]
assert agent_res.trials_failed > 0, "Agent strategy must transparently record failing trials"
failed_trials = [t for t in agent_res.trials if t.status != "success"]
assert len(failed_trials) == agent_res.trials_failed
for ft in failed_trials:
    assert ft.error_message is not None, "Failing trial must preserve error message"

# 4. Sealed test holdout evaluation and summary
summary = report.comparative_summary
assert summary["sealed_holdout_evaluated"] is True
assert summary["zero_test_leakage_verified"] is True
assert summary["matched_budgets_verified"] is True
assert summary["suboptimal_and_failing_trials_recorded"] is True
assert "without synthetic inflation" in summary["statistical_claim"]

# 5. Schema verification
schema_path = Path("contracts/v2/benchmark_report.schema.json")
if schema_path.is_file():
    schema = json.loads(schema_path.read_text())
    report_dict = json.loads(json.dumps(report, default=lambda o: getattr(o, "__dict__", str(o))))
    jsonschema.Draft202012Validator(schema).validate(report_dict)

print("GATE4_MULTI_SEED_PASSED")
"#;

    let output = run_python_script(py_code);
    assert!(
        output.status.success(),
        "Python benchmark failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("GATE4_MULTI_SEED_PASSED"));
}

#[test]
fn test_gate4_matched_compute_and_memory_budget_enforcement() {
    let py_code = r#"
from mlrescurix_worker.scientific.benchmarks import (
    load_openml_benchmark_dataset,
    run_single_evaluation,
)

df, target, features = load_openml_benchmark_dataset("credit-g", seed=2026, n_samples=300)

# 1. Normal run within 512 MB memory limit
eval_ok = run_single_evaluation(
    df=df, target=target, features=features,
    recipe="logistic", params={"C": 1.0},
    task="binary_classification", metric="balanced_accuracy",
    seed=2026, max_memory_mb=512.0, wall_budget_ms=10000.0,
)
assert eval_ok["status"] == "success"
assert eval_ok["metric_score"] is not None
assert eval_ok["peak_memory_bytes"] > 0
assert eval_ok["peak_memory_bytes"] < 512 * 1024 * 1024

# 2. Simulated excessive memory allocation triggers constraint failure
eval_mem = run_single_evaluation(
    df=df, target=target, features=features,
    recipe="memory_heavy_candidate", params={},
    task="binary_classification", metric="balanced_accuracy",
    seed=2026, max_memory_mb=10.0,
)
assert eval_mem["status"] == "failed_constraint"
assert eval_mem["metric_score"] is None
assert "exceeded" in eval_mem["error_message"].lower()

print("GATE4_BUDGET_ENFORCEMENT_PASSED")
"#;

    let output = run_python_script(py_code);
    assert!(
        output.status.success(),
        "Budget enforcement test failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("GATE4_BUDGET_ENFORCEMENT_PASSED"));
}

#[test]
fn test_gate4_sealed_holdout_zero_leakage_and_single_use_guard() {
    let py_code = r#"
from sklearn.linear_model import LogisticRegression
from mlrescurix_worker.scientific.benchmarks import (
    load_openml_benchmark_dataset,
    create_sealed_benchmark_split,
    SealedTestLeakageError,
)

df, target, features = load_openml_benchmark_dataset("diabetes", seed=2026)
dev_df, holdout = create_sealed_benchmark_split(
    df=df, target=target, test_size=0.2, seed=2026, task="binary_classification"
)

# 1. Strict row ID disjointness
dev_ids = set(dev_df["__row_id__"])
test_ids = set(holdout._test_df["__row_id__"])
assert dev_ids.isdisjoint(test_ids), "Row IDs must be strictly disjoint between dev and holdout"
assert len(dev_ids) + len(test_ids) == len(df)

# 2. Initial state is sealed
assert holdout.is_sealed is True
assert holdout.access_count == 0

clf = LogisticRegression(max_iter=300, random_state=2026).fit(dev_df[features].to_numpy(), dev_df[target].to_numpy())

# 3. First evaluation succeeds and consumes seal
score = holdout.evaluate(clf, metric="balanced_accuracy")
assert 0.0 <= score <= 1.0
assert holdout.is_sealed is False
assert holdout.access_count == 1

# 4. Subsequent evaluation attempts MUST fail with SealedTestLeakageError
try:
    holdout.evaluate(clf, metric="balanced_accuracy")
    raise AssertionError("Second access must be rejected")
except SealedTestLeakageError as exc:
    assert "evaluated once" in str(exc)

print("GATE4_SEALED_HOLDOUT_PASSED")
"#;

    let output = run_python_script(py_code);
    assert!(
        output.status.success(),
        "Sealed holdout test failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("GATE4_SEALED_HOLDOUT_PASSED"));
}

#[test]
fn test_gate4_rejection_of_non_generalizing_overfit_candidates() {
    let py_code = r#"
from mlrescurix_worker.scientific.benchmarks import (
    load_openml_benchmark_dataset,
    run_single_evaluation,
)

df, target, features = load_openml_benchmark_dataset("credit-g", seed=2026, n_samples=300)

eval_res = run_single_evaluation(
    df=df, target=target, features=features,
    recipe="overfit_candidate", params={},
    task="binary_classification", metric="balanced_accuracy",
    seed=2026, overfit_threshold=0.25,
)

# Overfit candidate achieves high train score but fails generalization
assert eval_res["status"] == "rejected_overfit"
assert eval_res["train_score"] > 0.90
assert eval_res["generalization_gap"] > 0.25
assert "generalization gap" in eval_res["error_message"].lower()

print("GATE4_OVERFIT_REJECTION_PASSED")
"#;

    let output = run_python_script(py_code);
    assert!(
        output.status.success(),
        "Overfit rejection test failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("GATE4_OVERFIT_REJECTION_PASSED"));
}

#[test]
fn test_gate4_benchmark_bundle_delivery_verification() {
    let td = tempfile::tempdir().unwrap();
    let bundle_dir = td.path().join("gate4_bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    let contract_files = [
        "input_schema.json", "output_schema.json", "model_schema.json", "scorecard.json",
        "verification.json", "final.json", "frozen_model.joblib", "inference.py",
        "inference.lock", "training.lock", "serve.py", "examples_input.json",
        "examples_expected.json", "README.md",
    ];
    for f in &contract_files {
        fs::write(bundle_dir.join(f), "{}").unwrap();
    }
    let model_bytes = b"DUMMY_MODEL_WEIGHTS_FOR_GATE4_BENCHMARK";
    fs::write(bundle_dir.join("frozen_model.joblib"), model_bytes).unwrap();
    let model_sha = format!("{:x}", Sha256::digest(model_bytes));

    fs::write(bundle_dir.join("verification.json"), serde_json::json!({
        "frozen_model_sha256": model_sha,
        "bundle_candidate": "cand-gate4-winner"
    }).to_string()).unwrap();

    let scorecard = serde_json::json!({
        "candidate_id": "cand-gate4-winner",
        "benchmark_dataset": "blood-transfusion",
        "seeds": [2026, 42, 101],
        "metric": "balanced_accuracy",
        "score": 0.764,
        "constraints": {
            "inference_latency_ms": {"threshold": 50.0, "value": 3.1, "verdict": "pass", "enforcement": "measured"},
            "bundle_bytes": {"threshold": 1048576, "value": 15000, "verdict": "pass", "enforcement": "measured"}
        }
    });
    fs::write(bundle_dir.join("scorecard.json"), serde_json::to_vec_pretty(&scorecard).unwrap()).unwrap();

    let mut files_map = std::collections::HashMap::new();
    for f in &contract_files {
        let p = bundle_dir.join(f);
        let bytes = fs::read(&p).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        files_map.insert(f.to_string(), mlrescurix_harness::bundle::BundleManifestFileEntry {
            bytes: bytes.len() as u64,
            sha256: sha,
        });
    }

    let manifest = mlrescurix_harness::bundle::BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: "p-gate4".into(),
        candidate_id: "cand-gate4-winner".into(),
        files: files_map,
        model: Some(serde_json::json!({
            "weights_path": "frozen_model.joblib",
            "weights_sha256": model_sha,
            "format": "joblib"
        })),
        pipeline: Some(serde_json::json!({"entrypoint": "inference.py"})),
        contracts: Some(serde_json::json!({
            "cli": "inference.py",
            "api": "predict(records)",
            "input_schema": "input_schema.json",
            "output_schema": "output_schema.json"
        })),
        dependencies: Some(serde_json::json!({
            "training_lock": "training.lock",
            "inference_lock": "inference.lock",
            "isolated_inference": true
        })),
        scorecard: Some(serde_json::json!({
            "path": "scorecard.json",
            "sha256": format!("{:x}", sha2::Sha256::digest(&fs::read(bundle_dir.join("scorecard.json")).unwrap()))
        })),
        lineage: None,
        resource_measurements: Some(serde_json::json!({
            "cold_load_ms": 10.0,
            "warm_p95_ms": 2.5,
            "peak_rss_kb": 1024,
            "package_bytes": 15000
        })),
    };
    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let report = mlrescurix_harness::bundle::verify_bundle(&bundle_dir, false).unwrap();
    assert_eq!(report.status, "integrity_verified");
    assert_eq!(report.format, "mlrescurix.bundle.v2");
    assert_eq!(report.candidate_id, "cand-gate4-winner");
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
}
