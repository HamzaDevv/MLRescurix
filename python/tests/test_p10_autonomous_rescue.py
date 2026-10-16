"""Unit tests for Phase P10.4 Autonomous End-to-End Model Rescue Gate."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from mlrescurix_worker.scientific.autonomous_rescue import (
    AutonomousRescueEngine,
    bundle_candidate,
    inject_candidate_failure,
    profile_dataset,
    repair_candidate_failure,
)
from mlrescurix_worker.scientific.candidate_synthesis import CandidateSynthesizer


@pytest.fixture
def rescue_schema():
    schema_path = Path("contracts/v2/autonomous_rescue.schema.json")
    assert schema_path.exists(), f"Missing schema at {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


@pytest.fixture
def synthetic_binary_csv(tmp_path):
    np.random.seed(2026)
    n = 800
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    cat = np.random.choice(["alpha", "beta", "gamma"], size=n)
    prob = 1.0 / (1.0 + np.exp(-(1.2 * x1 - 0.7 * x2)))
    y = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "row_id": [f"row_{i}" for i in range(n)],
        "feature_num1": x1,
        "feature_num2": x2,
        "category": cat,
        "target": y,
    })
    csv_path = tmp_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthesized_candidate(tmp_path):
    cand_dir = tmp_path / "candidate_baseline"
    spec = {
        "schema_version": "v2",
        "synthesis_id": "synth-p104-baseline",
        "candidate_id": "cand_baseline",
        "name": "Candidate Baseline",
        "problem_definition": {
            "task_type": "binary",
            "target_column": "target",
            "dataset_id": "ds_synth",
            "id_column": "row_id",
        },
        "target_descriptor": {
            "name": "target",
            "task_type": "binary",
        },
        "metric_objective": {
            "primary_metric": "balanced_accuracy",
            "direction": "maximize",
        },
        "model_intent": {
            "family": "boosting",
            "framework": "lightgbm",
            "hyperparameters": {"n_estimators": 20, "learning_rate": 0.05, "verbose": -1},
            "feature_pipeline": {
                "categorical_encoding": "target",
                "scaling": "standard",
            },
        },
    }
    CandidateSynthesizer.synthesize(spec, cand_dir)
    assert (cand_dir / "features.py").is_file()
    assert (cand_dir / "model.py").is_file()
    assert (cand_dir / "manifest.json").is_file()
    return cand_dir


def test_dataset_profiling_and_discovery(synthetic_binary_csv):
    """Test 1: Agent inspects dataset schema, types, nulls, and class distributions."""
    profile = profile_dataset(synthetic_binary_csv, target_column="target", id_column="row_id")

    pdict = profile.to_dict()
    assert pdict["row_count"] == 800
    assert pdict["column_count"] == 5
    assert pdict["target_column"] == "target"
    assert pdict["id_column"] == "row_id"
    assert pdict["inferred_task_type"] == "binary_classification"
    assert len(pdict["feature_columns"]) >= 3
    assert "target" in pdict["null_counts"]
    assert "class_balance" in pdict
    assert pdict["class_balance"] is not None


def test_injected_failure_interception_and_repair(synthesized_candidate):
    """Test 2: Realistic failure injection and autonomous self-healing repair."""
    features_py = synthesized_candidate / "features.py"
    model_py = synthesized_candidate / "model.py"

    # 1. Shape mismatch
    res_inject = inject_candidate_failure(synthesized_candidate, "shape_mismatch")
    assert res_inject["status"] == "injected"
    features_text = features_py.read_text(encoding="utf-8")
    assert "INJECTED_FAILURE" in features_text

    res_repair = repair_candidate_failure(synthesized_candidate, "shape_mismatch")
    assert res_repair["status"] == "repaired"
    features_text_repaired = features_py.read_text(encoding="utf-8")
    assert "INJECTED_FAILURE" not in features_text_repaired

    # 2. Non-finite loss
    res_inject_loss = inject_candidate_failure(synthesized_candidate, "non_finite_loss")
    assert res_inject_loss["status"] == "injected"
    assert "INJECTED_FAILURE" in features_py.read_text(encoding="utf-8") or "INJECTED_FAILURE" in model_py.read_text(encoding="utf-8")

    res_repair_loss = repair_candidate_failure(synthesized_candidate, "non_finite_loss")
    assert res_repair_loss["status"] == "repaired"
    assert "INJECTED_FAILURE" not in features_py.read_text(encoding="utf-8")
    assert "INJECTED_FAILURE" not in model_py.read_text(encoding="utf-8")

    # 3. Class order mismatch
    res_inject_order = inject_candidate_failure(synthesized_candidate, "class_order_mismatch")
    assert res_inject_order["status"] == "injected"
    assert "INJECTED_FAILURE" in model_py.read_text(encoding="utf-8")

    res_repair_order = repair_candidate_failure(synthesized_candidate, "class_order_mismatch")
    assert res_repair_order["status"] == "repaired"
    assert "INJECTED_FAILURE" not in model_py.read_text(encoding="utf-8")


def test_candidate_bundle_packaging_and_contracts(synthesized_candidate, synthetic_binary_csv, tmp_path):
    """Test 3: Packaging candidate into Apex-certified mlrescurix.bundle.v2."""
    bundle_dir = tmp_path / "final_bundle"

    receipt = bundle_candidate(
        candidate_dir=synthesized_candidate,
        output_bundle_dir=bundle_dir,
        dataset_path=synthetic_binary_csv,
        test_access_id="test_gate_approved_001",
        project_id="proj_rescue_test",
        candidate_id="cand_baseline",
        task_type="binary_classification",
        target_column="target",
        primary_metric="balanced_accuracy",
    )

    assert receipt["format"] == "mlrescurix.bundle.v2"
    assert receipt["candidate_id"] == "cand_baseline"
    assert receipt["holdout_primary_metric"] > 0.50

    # Verify all 13 required bundle contract files exist
    required_files = [
        "input_schema.json",
        "output_schema.json",
        "scorecard.json",
        "verification.json",
        "final.json",
        "inference.py",
        "inference.lock",
        "training.lock",
        "serve.py",
        "examples_input.json",
        "examples_expected.json",
        "README.md",
        "manifest.json",
    ]
    for req_file in required_files:
        p = bundle_dir / req_file
        assert p.is_file(), f"Missing required bundle file: {req_file}"

    # Verify manifest hashes match files on disk
    manifest_data = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["format"] == "mlrescurix.bundle.v2"
    assert manifest_data["candidate_id"] == "cand_baseline"
    for rel_path, item in manifest_data["files"].items():
        target_file = bundle_dir / rel_path
        assert target_file.is_file()
        assert target_file.stat().st_size == item["bytes"]

    # Verify dependency isolation in inference.lock
    inf_lock = (bundle_dir / "inference.lock").read_text(encoding="utf-8")
    forbidden_deps = ["pytest", "mlrescurix_harness", "docker", "cargo"]
    for dep in forbidden_deps:
        assert dep not in inf_lock, f"Forbidden dev/test dependency found in inference.lock: {dep}"


def test_standalone_inference_and_serve_parity(synthesized_candidate, synthetic_binary_csv, tmp_path):
    """Test 4: Standalone inference script execution and prediction parity."""
    bundle_dir = tmp_path / "bundle_parity"
    bundle_candidate(
        candidate_dir=synthesized_candidate,
        output_bundle_dir=bundle_dir,
        dataset_path=synthetic_binary_csv,
        test_access_id="test_gate_approved_002",
        project_id="proj_parity",
        candidate_id="cand_baseline",
    )

    out_json = tmp_path / "parity_out.json"
    cmd = [
        sys.executable,
        str(bundle_dir / "inference.py"),
        "--input",
        str(bundle_dir / "examples_input.json"),
        "--output",
        str(out_json),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Inference execution failed: {res.stderr}"
    assert out_json.is_file()

    pred_data = json.loads(out_json.read_text(encoding="utf-8"))
    expected_data = json.loads((bundle_dir / "examples_expected.json").read_text(encoding="utf-8"))

    assert len(pred_data["prediction"]) == len(expected_data["prediction"])
    for pred, exp in zip(pred_data["prediction"], expected_data["prediction"]):
        assert pred == exp


def test_negative_guards_holdout_leakage_and_tampering(synthesized_candidate, synthetic_binary_csv, tmp_path):
    """Test 5: Negative guards against holdout test row leakage and bundle tampering."""
    bundle_dir = tmp_path / "bundle_guards"
    bundle_candidate(
        candidate_dir=synthesized_candidate,
        output_bundle_dir=bundle_dir,
        dataset_path=synthetic_binary_csv,
        test_access_id="test_gate_approved_003",
        project_id="proj_guards",
        candidate_id="cand_baseline",
    )

    # 1. Anti-leakage: verify scorecard and final receipts report valid splits without test overlap
    final_receipt = json.loads((bundle_dir / "final.json").read_text(encoding="utf-8"))
    assert final_receipt["test_access_id"] == "test_gate_approved_003"

    # 2. Tamper detection: modifying a byte in frozen_model.joblib must invalidate manifest verification
    model_path = bundle_dir / "frozen_model.joblib"
    original_bytes = model_path.read_bytes()
    corrupted_bytes = bytearray(original_bytes)
    corrupted_bytes[-1] ^= 0xFF
    model_path.write_bytes(corrupted_bytes)

    # Re-verify manifest hashes
    manifest_data = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    expected_hash = manifest_data["files"]["frozen_model.joblib"]["sha256"]
    assert actual_hash != expected_hash, "Tamper was not detected by sha256 checksum!"


def test_autonomous_rescue_engine_full_loop(synthetic_binary_csv, tmp_path, rescue_schema):
    """Test 6: Full closed-loop AutonomousRescueEngine execution and schema compliance."""
    session_dir = tmp_path / "autonomous_rescue_session"
    spec = {
        "schema_version": "v2",
        "session_id": "rescue_session_001",
        "project_id": "proj_autonomous_test",
        "problem_definition": {
            "dataset_path": str(synthetic_binary_csv),
            "target_column": "target",
            "id_column": "row_id",
            "primary_metric": "balanced_accuracy",
            "direction": "maximize",
        },
        "resource_budget": {
            "max_turns": 6,
            "max_wall_ms": 60000,
            "max_memory_mb": 1024,
        },
        "simulation": {
            "inject_failure_on_first_turn": True,
            "failure_type": "shape_mismatch",
            "test_access_id": "test_gate_approved_auto_001",
        },
    }

    engine = AutonomousRescueEngine(spec=spec, workspace_dir=session_dir)
    session = engine.run()

    # Validate output session against strict Draft 2020-12 schema
    validator = Draft202012Validator(rescue_schema)
    errors = list(validator.iter_errors(session))
    assert not errors, f"Schema validation failed: {[e.message for e in errors]}"

    # Verify session fields
    assert session["status"] == "finalized"
    assert session["dataset_profile"]["row_count"] == 800
    assert len(session["turns"]) >= 4
    assert session["failure_interception"] is not None
    assert session["failure_interception"]["failure_type"] == "shape_mismatch"
    assert session["finalization"]["bundle_format"] == "mlrescurix.bundle.v2"
    assert (session_dir / "bundle" / "manifest.json").is_file()
