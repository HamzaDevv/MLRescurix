"""Comprehensive Test Suite for Session 3: Budgeted Tuning, Delivery & Integrated Readiness.

Verifies all Gate S3 requirements:
1. Budgeted Optuna tuning trial lifecycle (ask, report, prune, complete, best_trial).
2. Durable SQLite persistence and study resumption across process restarts.
3. Multi-objective budget fencing (max_trials, duration, cost, campaign authority).
4. Trial pruning hook using median strategy.
5. Clean-room custom bundle packaging for TinyTorchNeuralCandidate (mlrescurix.bundle.v2).
6. Clean-environment subprocess inference execution with PYTHONPATH unset.
7. Input schema validation and error reporting on missing/invalid features.
8. Numeric dtype tolerance (integer coercion, NaN handling).
9. Telemetry reporting (inference latency in ms, peak RSS memory delta).
10. Adapter independence across 3 diverse engineering fixtures (multimodal, wide, sequential).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.custom.delivery import (
    BundleVerificationError,
    CustomBundlePackager,
)
from mlrescurix_worker.custom.torch_candidate import (
    HAS_TORCH,
    TinyTorchNeuralCandidate,
)
from mlrescurix_worker.custom.tuning import (
    BudgetedOptunaTuner,
    BudgetedTrial,
    TrialPruningHook,
    TrialState,
)
from fixtures.engineering_datasets import (
    fixture_multimodal_numeric_categorical,
    fixture_sequential_temporal,
    fixture_wide_sparse_tabular,
)


# =============================================================================
# 1. Budgeted Optuna Tuning: Trial Lifecycle
# =============================================================================

def test_budgeted_optuna_tuner_trial_lifecycle(tmp_path: Path):
    param_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-1, "log": True},
        "hidden_dim": {"type": "categorical", "choices": [16, 32, 64]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    }

    db_path = tmp_path / "tuning.sqlite"
    tuner = BudgetedOptunaTuner(
        study_name="test_study_v1",
        param_space=param_space,
        direction="maximize",
        max_trials=5,
        storage_path=db_path,
        seed=123,
    )

    # 1. Ask for trial 0
    trial0 = tuner.ask()
    assert trial0 is not None
    assert trial0.trial_id == "trial_test_study_v1_0"
    assert trial0.trial_number == 0
    assert trial0.state == TrialState.RUNNING
    assert "lr" in trial0.params
    assert trial0.params["hidden_dim"] in [16, 32, 64]

    # Report intermediate steps
    trial0.report(0.65, step=1)
    trial0.report(0.72, step=2)
    assert trial0.intermediate_values[1] == 0.65
    assert trial0.intermediate_values[2] == 0.72

    # Finalize trial 0
    tuner.tell(trial0, TrialState.COMPLETE, final_value=0.75, duration_seconds=1.2, cost_usd=0.02)
    assert trial0.state == TrialState.COMPLETE
    assert trial0.final_value == 0.75

    # 2. Ask for trial 1 with better score
    trial1 = tuner.ask()
    assert trial1 is not None
    assert trial1.trial_id == "trial_test_study_v1_1"
    tuner.tell(trial1, TrialState.COMPLETE, final_value=0.88, duration_seconds=0.8, cost_usd=0.015)

    # Best trial check
    best = tuner.best_trial()
    assert best is not None
    assert best.trial_id == "trial_test_study_v1_1"
    assert best.final_value == 0.88

    summary = tuner.summary()
    assert summary["completed_trials"] == 2
    assert summary["best_value"] == 0.88
    assert summary["total_cost_usd"] == pytest.approx(0.035)


# =============================================================================
# 2. Durable SQLite Persistence & Recovery
# =============================================================================

def test_budgeted_optuna_tuner_persistence_and_recovery(tmp_path: Path):
    db_path = tmp_path / "study_store.sqlite"
    param_space = {"alpha": {"type": "float", "low": 0.01, "high": 1.0}}

    tuner1 = BudgetedOptunaTuner(
        study_name="durable_study",
        param_space=param_space,
        direction="minimize",
        max_trials=10,
        storage_path=db_path,
        seed=42,
    )

    t0 = tuner1.ask()
    assert t0 is not None
    tuner1.tell(t0, TrialState.COMPLETE, final_value=0.45, cost_usd=0.05)

    t1 = tuner1.ask()
    assert t1 is not None
    tuner1.tell(t1, TrialState.COMPLETE, final_value=0.32, cost_usd=0.04)

    # Simulate process termination / controller restart
    del tuner1

    # Reconnect with a new tuner instance
    tuner2 = BudgetedOptunaTuner(
        study_name="durable_study",
        param_space=param_space,
        direction="minimize",
        max_trials=10,
        storage_path=db_path,
    )

    assert len(tuner2.trials) == 2
    best = tuner2.best_trial()
    assert best is not None
    assert best.trial_id == "trial_durable_study_1"
    assert best.final_value == 0.32  # Minimize picks lower value

    # Next ask should increment trial_number to 2
    t2 = tuner2.ask()
    assert t2 is not None
    assert t2.trial_number == 2
    assert t2.trial_id == "trial_durable_study_2"


# =============================================================================
# 3. Multi-Objective Budget Fencing
# =============================================================================

def test_budgeted_optuna_tuner_budget_exhaustion(tmp_path: Path):
    param_space = {"x": {"type": "float", "low": 0.0, "high": 1.0}}

    # Case A: Max trials bound
    tuner_trials = BudgetedOptunaTuner(
        study_name="budget_trials",
        param_space=param_space,
        max_trials=2,
    )
    t0 = tuner_trials.ask()
    assert t0 is not None
    tuner_trials.tell(t0, TrialState.COMPLETE, final_value=1.0)
    t1 = tuner_trials.ask()
    assert t1 is not None
    tuner_trials.tell(t1, TrialState.COMPLETE, final_value=2.0)

    # Third ask should be rejected
    exhausted, reason = tuner_trials.is_budget_exhausted()
    assert exhausted is True
    assert "Max trials budget reached" in reason
    assert tuner_trials.ask() is None

    # Case B: Max cost bound
    tuner_cost = BudgetedOptunaTuner(
        study_name="budget_cost",
        param_space=param_space,
        max_cost_usd=0.10,
    )
    tc = tuner_cost.ask()
    assert tc is not None
    tuner_cost.tell(tc, TrialState.COMPLETE, final_value=1.0, cost_usd=0.12)

    exhausted_cost, reason_cost = tuner_cost.is_budget_exhausted()
    assert exhausted_cost is True
    assert "Max cost budget exhausted" in reason_cost
    assert tuner_cost.ask() is None


# =============================================================================
# 4. Trial Pruning Hook with Median Strategy
# =============================================================================

def test_trial_pruning_hook_median_strategy():
    pruner = TrialPruningHook(n_startup_trials=2, n_warmup_steps=1, direction="maximize")

    # Record historical completed trials at step 2: scores 0.70 and 0.80 -> median = 0.75
    pruner.record_step(step=2, value=0.70)
    pruner.record_step(step=2, value=0.80)

    trial_good = BudgetedTrial(
        trial_id="t_good",
        trial_number=3,
        params={},
        _pruner=pruner,
        _direction="maximize",
    )
    trial_good.report(0.78, step=2)
    assert trial_good.should_prune(step=2) is False  # 0.78 > 0.75 -> keep

    trial_poor = BudgetedTrial(
        trial_id="t_poor",
        trial_number=4,
        params={},
        _pruner=pruner,
        _direction="maximize",
    )
    trial_poor.report(0.60, step=2)
    assert trial_poor.should_prune(step=2) is True  # 0.60 < 0.75 -> prune!


# =============================================================================
# 5. Clean-Room Custom Bundle Packaging & Manifest Verification
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural bundle packaging")
def test_custom_bundle_packaging_tiny_torch(tmp_path: Path):
    df, target_col, task_type, metric, direction = fixture_multimodal_numeric_categorical(n_samples=100)
    feature_names = ["num_feature_1", "num_feature_2"]
    X = df[feature_names]
    y = df[target_col]

    model = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01, batch_size=16)
    model.fit(X, y)

    bundle_dir = tmp_path / "torch_bundle"
    packager = CustomBundlePackager()

    receipt = packager.package(
        candidate=model,
        output_bundle_dir=bundle_dir,
        target_column=target_col,
        task_type=task_type,
        primary_metric=metric,
        primary_metric_score=0.82,
        metric_direction=direction,
        feature_names=feature_names,
        raw_example_records=X.head(5).to_dict(orient="records"),
        candidate_id="tiny_torch_candidate_v1",
    )

    assert receipt.candidate_id == "tiny_torch_candidate_v1"
    assert receipt.primary_metric_score == 0.82
    assert receipt.inference_entrypoint == "inference.py"

    # Verify all expected artifacts exist
    for fname in [
        "manifest.json", "model.pt", "input_schema.json", "output_schema.json",
        "inference.py", "inference.lock", "scorecard.json", "README.md",
        "examples_input.json", "examples_expected.json"
    ]:
        assert (bundle_dir / fname).exists(), f"Missing expected bundle artifact: {fname}"

    # Verify manifest.json matches SHA-256 of files
    manifest_data = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["format"] == "mlrescurix.bundle.v2"
    assert manifest_data["model"]["format"] in ("native", "joblib")
    for fname, meta in manifest_data["files"].items():
        actual_sha = hashlib.sha256((bundle_dir / fname).read_bytes()).hexdigest()
        assert meta["sha256"] == actual_sha, f"SHA-256 mismatch for {fname}"
        assert meta["bytes"] == (bundle_dir / fname).stat().st_size


# =============================================================================
# 6. Clean-Environment Subprocess Inference Verification
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for clean subprocess inference")
def test_clean_environment_subprocess_inference(tmp_path: Path):
    df, target_col, task_type, metric, direction = fixture_multimodal_numeric_categorical(n_samples=80)
    feature_names = ["num_feature_1", "num_feature_2"]
    X = df[feature_names]
    y = df[target_col]

    model = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01, batch_size=16)
    model.fit(X, y)

    bundle_dir = tmp_path / "clean_bundle"
    packager = CustomBundlePackager()
    packager.package(
        candidate=model,
        output_bundle_dir=bundle_dir,
        target_column=target_col,
        task_type=task_type,
        primary_metric=metric,
        primary_metric_score=0.85,
        metric_direction=direction,
        feature_names=feature_names,
        raw_example_records=X.head(5).to_dict(orient="records"),
    )

    # Execute inference.py in clean subprocess with PYTHONPATH removed
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    input_file = bundle_dir / "examples_input.json"
    output_file = tmp_path / "clean_output.json"

    proc = subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(input_file), "--output", str(output_file)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, f"Subprocess failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert output_file.exists()

    result = json.loads(output_file.read_text(encoding="utf-8"))
    expected = json.loads((bundle_dir / "examples_expected.json").read_text(encoding="utf-8"))

    assert len(result["prediction"]) == len(expected["prediction"])
    assert result["prediction"] == expected["prediction"]
    assert "probabilities" in result
    assert result["classes"] == [0, 1]


# =============================================================================
# 7. Input Schema Validation & Error Reporting
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_bundle_input_validation_errors(tmp_path: Path):
    df, target_col, task_type, metric, direction = fixture_multimodal_numeric_categorical(n_samples=50)
    feature_names = ["num_feature_1", "num_feature_2"]
    X = df[feature_names]
    y = df[target_col]

    model = TinyTorchNeuralCandidate(hidden_dim=16)
    model.fit(X, y)

    bundle_dir = tmp_path / "val_bundle"
    packager = CustomBundlePackager()
    packager.package(
        candidate=model,
        output_bundle_dir=bundle_dir,
        target_column=target_col,
        task_type=task_type,
        primary_metric=metric,
        primary_metric_score=0.80,
        metric_direction=direction,
        feature_names=feature_names,
        raw_example_records=X.head(5).to_dict(orient="records"),
    )

    # 1. Missing required feature column -> should fail with KeyError
    bad_input_missing = tmp_path / "bad_input_missing.json"
    bad_input_missing.write_text(json.dumps([{"num_feature_1": 1.5}]), encoding="utf-8")
    out_missing = tmp_path / "out_missing.json"

    proc1 = subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(bad_input_missing), "--output", str(out_missing)],
        capture_output=True,
        text=True,
    )
    assert proc1.returncode != 0
    assert "missing required feature 'num_feature_2'" in proc1.stderr

    # 2. Non-numeric value -> should fail with ValueError
    bad_input_type = tmp_path / "bad_input_type.json"
    bad_input_type.write_text(json.dumps([{"num_feature_1": "not_a_number", "num_feature_2": 1.0}]), encoding="utf-8")
    out_type = tmp_path / "out_type.json"

    proc2 = subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(bad_input_type), "--output", str(out_type)],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode != 0
    assert "cannot be coerced to float" in proc2.stderr


# =============================================================================
# 8. Numeric Dtype Tolerance
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_bundle_dtype_tolerance(tmp_path: Path):
    df, target_col, task_type, metric, direction = fixture_multimodal_numeric_categorical(n_samples=50)
    feature_names = ["num_feature_1", "num_feature_2"]
    X = df[feature_names]
    y = df[target_col]

    model = TinyTorchNeuralCandidate(hidden_dim=16)
    model.fit(X, y)

    bundle_dir = tmp_path / "dtype_bundle"
    packager = CustomBundlePackager()
    packager.package(
        candidate=model,
        output_bundle_dir=bundle_dir,
        target_column=target_col,
        task_type=task_type,
        primary_metric=metric,
        primary_metric_score=0.80,
        metric_direction=direction,
        feature_names=feature_names,
        raw_example_records=X.head(5).to_dict(orient="records"),
    )

    # Input with pure integers and string floats -> should coerce smoothly
    tolerated_input = tmp_path / "tolerated_input.json"
    tolerated_input.write_text(json.dumps([
        {"num_feature_1": 1, "num_feature_2": 2},
        {"num_feature_1": "3.5", "num_feature_2": "4.2"},
        {"num_feature_1": None, "num_feature_2": 0},
    ]), encoding="utf-8")
    tolerated_out = tmp_path / "tolerated_out.json"

    proc = subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(tolerated_input), "--output", str(tolerated_out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Dtype coercion failed: {proc.stderr}"
    res = json.loads(tolerated_out.read_text(encoding="utf-8"))
    assert len(res["prediction"]) == 3


# =============================================================================
# 9. Latency and Memory Telemetry Reporting
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required")
def test_bundle_latency_and_memory_reporting(tmp_path: Path):
    df, target_col, task_type, metric, direction = fixture_multimodal_numeric_categorical(n_samples=50)
    feature_names = ["num_feature_1", "num_feature_2"]
    X = df[feature_names]
    y = df[target_col]

    model = TinyTorchNeuralCandidate(hidden_dim=16)
    model.fit(X, y)

    bundle_dir = tmp_path / "telemetry_bundle"
    packager = CustomBundlePackager()
    packager.package(
        candidate=model,
        output_bundle_dir=bundle_dir,
        target_column=target_col,
        task_type=task_type,
        primary_metric=metric,
        primary_metric_score=0.80,
        metric_direction=direction,
        feature_names=feature_names,
        raw_example_records=X.head(5).to_dict(orient="records"),
    )

    input_file = bundle_dir / "examples_input.json"
    output_file = tmp_path / "telemetry_output.json"

    subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(input_file), "--output", str(output_file)],
        check=True,
    )

    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (float, int))
    assert data["latency_ms"] >= 0.0
    assert "peak_memory_bytes" in data
    assert isinstance(data["peak_memory_bytes"], int)
    assert data["peak_memory_bytes"] >= 0


# =============================================================================
# 10. Adapter Independence Across Diverse Engineering Fixtures
# =============================================================================

def test_adapter_independence_across_fixtures(tmp_path: Path):
    fixtures = [
        ("multimodal", fixture_multimodal_numeric_categorical(n_samples=60)),
        ("wide_sparse", fixture_wide_sparse_tabular(n_samples=60, n_features=20)),
        ("sequential", fixture_sequential_temporal(n_steps=60)),
    ]

    for name, (df, target_col, task_type, metric, direction) in fixtures:
        feature_cols = [c for c in df.columns if c != target_col and not c.startswith("step_")]
        # Keep numeric feature columns
        numeric_cols = list(df[feature_cols].select_dtypes(include=[np.number]).columns)
        assert len(numeric_cols) > 0, f"Fixture {name} must have numeric features"

        # 1. Optimize simple hyperparameter on this fixture using BudgetedOptunaTuner
        tuner = BudgetedOptunaTuner(
            study_name=f"tune_{name}",
            param_space={"lr": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True}},
            direction=direction,
            max_trials=3,
            storage_path=tmp_path / f"{name}_tuning.sqlite",
            seed=42,
        )

        def objective(trial: BudgetedTrial) -> float:
            lr = trial.params["lr"]
            # Mock evaluation metric correlated with lr
            if direction == "maximize":
                return 0.80 - abs(lr - 0.01) * 2.0
            return 0.20 + abs(lr - 0.01) * 2.0

        summary = tuner.optimize(objective, n_trials=3)
        assert summary["completed_trials"] == 3
        assert summary["best_value"] is not None

        # 2. Package candidate for this fixture if torch is available
        if HAS_TORCH:
            is_class = "classification" in task_type.lower()
            X = df[numeric_cols]
            y = df[target_col]

            model = TinyTorchNeuralCandidate(
                hidden_dim=16,
                is_classification=is_class,
                batch_size=16,
            )
            model.fit(X, y)

            b_dir = tmp_path / f"bundle_{name}"
            packager = CustomBundlePackager()
            receipt = packager.package(
                candidate=model,
                output_bundle_dir=b_dir,
                target_column=target_col,
                task_type=task_type,
                primary_metric=metric,
                primary_metric_score=summary["best_value"],
                metric_direction=direction,
                feature_names=numeric_cols,
                raw_example_records=X.head(4).to_dict(orient="records"),
                candidate_id=f"cand_{name}",
            )
            assert receipt.candidate_id == f"cand_{name}"
            assert (b_dir / "manifest.json").exists()
            assert (b_dir / "inference.py").exists()


# =============================================================================
# 11. Governed Optuna Tuning with Concurrency Bounding & Campaign Authority
# =============================================================================

def test_optuna_tuning_concurrency_and_campaign_authority(tmp_path: Path):
    from unittest.mock import MagicMock
    from mlrescurix_worker.scientific.dual_gateway import DualTierGateway, RoutePolicy

    param_space = {"param_x": {"type": "float", "low": 0.0, "high": 1.0}}

    # Mock CampaignBudgetAuthority
    mock_authority = MagicMock()
    mock_authority.get_status.return_value = {"remaining_campaign_usd": 0.25}

    tuner = BudgetedOptunaTuner(
        study_name="governed_optuna_study",
        param_space=param_space,
        direction="maximize",
        max_trials=5,
        max_concurrent_trials=1,
        campaign_authority=mock_authority,
        seed=42,
    )

    # 1. Ask trial 0
    t0 = tuner.ask()
    assert t0 is not None
    assert mock_authority.reserve.called
    assert t0.state == TrialState.RUNNING

    # 2. Concurrency bound enforcement: while trial 0 is RUNNING and max_concurrent_trials=1, ask() must return None
    assert tuner.is_concurrency_exhausted()[0] is True
    t_blocked = tuner.ask()
    assert t_blocked is None

    # 3. Complete trial 0 -> settles spend and frees concurrency slot
    tuner.tell(t0, TrialState.COMPLETE, final_value=0.85, cost_usd=0.01)
    assert mock_authority.settle.called
    assert tuner.is_concurrency_exhausted()[0] is False

    # 4. Ask trial 1 now succeeds
    t1 = tuner.ask()
    assert t1 is not None
    tuner.tell(t1, TrialState.COMPLETE, final_value=0.92, cost_usd=0.01)

    assert tuner.best_trial().final_value == 0.92


# =============================================================================
# 12. Governed Dual Gateway with Pre-Dispatch Reservation & Settlement
# =============================================================================

def test_dual_gateway_delegates_only_explicit_primary_route():
    from unittest.mock import MagicMock
    from mlrescurix_worker.scientific.dual_gateway import DualTierGateway, RoutePolicy

    adapter = MagicMock()
    adapter.call_with_request_id.return_value = {"content": "ok"}
    gateway = DualTierGateway(route_policy=RoutePolicy.PRIMARY_ONLY, rust_adapter=adapter)
    messages = [{"role": "user", "content": "hi"}]
    assert gateway.call_with_request_id(messages, 100, "request-1")["content"] == "ok"
    adapter.call_with_request_id.assert_called_once_with(messages, 100, "request-1")
    with pytest.raises(ValueError, match="stable request_id"):
        gateway.call_with_request_id(messages, 100)
    with pytest.raises(RuntimeError, match="fallback is unavailable"):
        gateway.call_with_request_id(messages, 100, "request-2", RoutePolicy.AUTO_FAILOVER)
    gateway.campaign_authority = MagicMock()
    with pytest.raises(RuntimeError, match="campaign settlement is disabled"):
        gateway.call_with_request_id(messages, 100, "request-3")
    adapter.call_with_request_id.assert_called_once()
