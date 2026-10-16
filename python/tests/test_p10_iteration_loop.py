"""Unit tests for Phase P10.3 Multi-Fidelity Iteration & Hypothesis Loop."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from mlrescurix_worker.scientific.iteration_loop import MultiFidelityIterationEngine
from mlrescurix_worker.scientific.candidate_synthesis import CandidateSynthesizer


@pytest.fixture
def hypothesis_schema():
    schema_path = Path("contracts/v2/iteration_hypothesis.schema.json")
    assert schema_path.exists(), f"Missing schema at {schema_path}"
    return json.loads(schema_path.read_text())


@pytest.fixture
def pareto_schema():
    schema_path = Path("contracts/v2/pareto_frontier.schema.json")
    assert schema_path.exists(), f"Missing schema at {schema_path}"
    return json.loads(schema_path.read_text())


@pytest.fixture
def synthetic_binary_csv(tmp_path):
    np.random.seed(2026)
    n = 600
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    prob = 1.0 / (1.0 + np.exp(-(1.5 * x1 - 0.8 * x2)))
    y = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "row_id": [f"r_{i}" for i in range(n)],
        "feat_num1": x1,
        "feat_num2": x2,
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
        "synthesis_id": "synth-p103-test",
        "candidate_id": "cand_baseline",
        "problem_definition": {
            "task_type": "binary",
            "target_column": "target",
            "id_column": "row_id",
        },
        "target_descriptor": {
            "name": "target",
            "task_type": "binary",
            "class_order": [0, 1],
        },
        "metric_objective": {
            "primary_metric": "balanced_accuracy",
            "direction": "maximize",
        },
        "model_intent": {
            "family": "boosting",
            "framework": "lightgbm",
            "hyperparameters": {"n_estimators": 25, "learning_rate": 0.1},
        },
    }
    synthesized = CandidateSynthesizer.synthesize(spec, cand_dir)
    return cand_dir, synthesized.manifest


def test_iteration_hypothesis_schema_validation(hypothesis_schema):
    validator = Draft202012Validator(hypothesis_schema)

    valid_hyp = {
        "schema_version": "v2",
        "hypothesis_id": "hyp-001",
        "candidate_id": "cand-v2",
        "parent_candidate_id": "cand-v1",
        "mutation_type": "feature_engineering",
        "hypothesis_statement": "Adding rolling features improves balanced accuracy by >= 0.01",
        "rationale": "Captures temporal momentum",
        "expected_metric_delta": {
            "metric": "balanced_accuracy",
            "direction": "maximize",
            "target_threshold": 0.85,
            "minimum_improvement": 0.01,
        },
        "multi_fidelity_budget": {
            "proxy_tier": {
                "max_samples": 400,
                "screening_threshold": 0.70,
                "timeout_ms": 15000,
            },
            "full_tier": {
                "folds": 3,
                "seed": 2026,
                "timeout_ms": 45000,
            },
        },
        "outcome": {
            "status": "confirmed",
            "baseline_metric": 0.80,
            "full_metric_observed": 0.86,
            "metric_delta": 0.06,
            "reason": "Met target threshold and exceeded minimum improvement",
            "settled_at_ms": 1729000000,
        },
        "created_ms": 1728990000,
    }
    validator.validate(valid_hyp)

    # Invalid mutation_type
    invalid_hyp = dict(valid_hyp)
    invalid_hyp["mutation_type"] = "invalid_mutation"
    assert not validator.is_valid(invalid_hyp)


def test_pareto_frontier_schema_validation(pareto_schema):
    validator = Draft202012Validator(pareto_schema)

    valid_frontier = {
        "schema_version": "v2",
        "frontier_id": "frontier-proj1-001",
        "project_id": "proj1",
        "primary_metric_name": "balanced_accuracy",
        "optimization_direction": "maximize",
        "records": [
            {
                "candidate_id": "c1",
                "parent_candidate_id": None,
                "validation_score": 0.85,
                "latency_ms": 120.0,
                "peak_memory_bytes": 1000000,
                "domination_status": "non_dominated",
                "dominated_by": [],
            },
            {
                "candidate_id": "c2",
                "parent_candidate_id": "c1",
                "validation_score": 0.80,
                "latency_ms": 200.0,
                "peak_memory_bytes": 1500000,
                "domination_status": "dominated",
                "dominated_by": ["c1"],
            },
        ],
        "frontier_candidate_ids": ["c1"],
        "non_dominated_count": 1,
        "dominated_count": 1,
        "evaluated_at_ms": 1729000000,
    }
    validator.validate(valid_frontier)

    # Invalid domination status
    invalid_frontier = dict(valid_frontier)
    invalid_frontier["records"][0]["domination_status"] = "partially_dominated"
    assert not validator.is_valid(invalid_frontier)


def test_tier_0_proxy_screening_admission_and_rejection(synthetic_binary_csv, synthesized_candidate):
    cand_dir, manifest = synthesized_candidate

    # 1. Realistic screening threshold -> Admitted
    receipt_admitted = MultiFidelityIterationEngine.run_proxy_screen(
        dataset_csv=synthetic_binary_csv,
        manifest=manifest,
        candidate_dir=cand_dir,
        max_samples=250,
        screening_threshold=0.50,
        direction="maximize",
    )
    assert receipt_admitted.status == "admitted"
    assert receipt_admitted.proxy_metric_value is not None
    assert receipt_admitted.proxy_metric_value >= 0.50
    assert receipt_admitted.samples_used <= 250
    assert receipt_admitted.latency_ms > 0

    # 2. Impossibly high threshold -> Rejected early
    receipt_rejected = MultiFidelityIterationEngine.run_proxy_screen(
        dataset_csv=synthetic_binary_csv,
        manifest=manifest,
        candidate_dir=cand_dir,
        max_samples=250,
        screening_threshold=0.999,
        direction="maximize",
    )
    assert receipt_rejected.status == "rejected"
    assert "failed screening threshold" in receipt_rejected.reason.lower()


def test_hypothesis_settlement_all_outcomes():
    hyp_spec = {
        "schema_version": "v2",
        "hypothesis_id": "h-test",
        "candidate_id": "c-test",
        "mutation_type": "hyperparameter_tuning",
        "hypothesis_statement": "Tune max_depth to improve score",
        "expected_metric_delta": {
            "metric": "balanced_accuracy",
            "direction": "maximize",
            "target_threshold": 0.85,
            "minimum_improvement": 0.01,
        },
        "multi_fidelity_budget": {
            "proxy_tier": {"max_samples": 300},
            "full_tier": {"folds": 3},
        },
    }

    # 1. Confirmed: improvement +0.03 >= 0.01 and reaches target 0.85
    settled = MultiFidelityIterationEngine.settle_hypothesis(hyp_spec, observed_metric=0.86, baseline_score=0.83)
    assert settled["outcome"]["status"] == "confirmed"
    assert settled["outcome"]["metric_delta"] == pytest.approx(0.03)

    # 2. Refuted: degraded relative to baseline
    settled = MultiFidelityIterationEngine.settle_hypothesis(hyp_spec, observed_metric=0.80, baseline_score=0.83)
    assert settled["outcome"]["status"] == "refuted"
    assert "degraded" in settled["outcome"]["reason"].lower()

    # 3. Inconclusive: improved by +0.005, but did not reach target threshold 0.85
    settled = MultiFidelityIterationEngine.settle_hypothesis(hyp_spec, observed_metric=0.835, baseline_score=0.83)
    assert settled["outcome"]["status"] == "inconclusive"
    assert "target threshold" in settled["outcome"]["reason"].lower()

    # 3b. Inconclusive: reached target 0.85, but improvement +0.004 < 0.01
    settled = MultiFidelityIterationEngine.settle_hypothesis(hyp_spec, observed_metric=0.852, baseline_score=0.848)
    assert settled["outcome"]["status"] == "inconclusive"
    assert "below required" in settled["outcome"]["reason"].lower()

    # 4. Refuted via proxy screening rejection
    settled = MultiFidelityIterationEngine.settle_hypothesis(
        hyp_spec,
        observed_metric=0.60,
        baseline_score=0.83,
        screening_rejected=True,
        rejection_reason="Failed screening threshold",
    )
    assert settled["outcome"]["status"] == "refuted"
    assert "screening" in settled["outcome"]["reason"].lower()


def test_pareto_frontier_computation_multi_objective():
    records = [
        # c1: Balanced Accuracy 0.85, 100ms, 1MB (Fast baseline)
        {"candidate_id": "c1", "validation_score": 0.85, "latency_ms": 100.0, "peak_memory_bytes": 1000000},
        # c2: Balanced Accuracy 0.92, 250ms, 2MB (High accuracy, trade off latency)
        {"candidate_id": "c2", "validation_score": 0.92, "latency_ms": 250.0, "peak_memory_bytes": 2000000},
        # c3: Balanced Accuracy 0.80, 150ms, 1.5MB (Dominated by c1 on all counts)
        {"candidate_id": "c3", "validation_score": 0.80, "latency_ms": 150.0, "peak_memory_bytes": 1500000},
        # c4: Balanced Accuracy 0.84, 80ms, 0.8MB (Ultra fast/light, trade off score)
        {"candidate_id": "c4", "validation_score": 0.84, "latency_ms": 80.0, "peak_memory_bytes": 800000},
    ]

    frontier = MultiFidelityIterationEngine.compute_pareto_frontier(
        records=records,
        primary_metric_name="balanced_accuracy",
        direction="maximize",
        project_id="test_project",
    )

    # c1, c2, c4 should be on non-dominated frontier; c3 should be dominated by c1
    assert set(frontier["frontier_candidate_ids"]) == {"c1", "c2", "c4"}
    assert frontier["non_dominated_count"] == 3
    assert frontier["dominated_count"] == 1

    c3_record = next(r for r in frontier["records"] if r["candidate_id"] == "c3")
    assert c3_record["domination_status"] == "dominated"
    assert "c1" in c3_record["dominated_by"]


def test_multi_fidelity_iteration_cli_subcommands(tmp_path, synthetic_binary_csv, synthesized_candidate):
    cand_dir, manifest = synthesized_candidate

    # Test pareto-frontier CLI
    pareto_spec = tmp_path / "pareto_spec.json"
    pareto_out = tmp_path / "pareto_out.json"
    pareto_spec.write_text(json.dumps({
        "records": [
            {"candidate_id": "m1", "validation_score": 0.88, "latency_ms": 150.0, "peak_memory_bytes": 1000000},
            {"candidate_id": "m2", "validation_score": 0.82, "latency_ms": 200.0, "peak_memory_bytes": 1500000},
        ],
        "primary_metric_name": "balanced_accuracy",
        "optimization_direction": "maximize",
    }))

    cmd = [
        sys.executable,
        "-m", "mlrescurix_worker.scientific.cli",
        "pareto-frontier",
        "--spec", str(pareto_spec),
        "--output", str(pareto_out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"pareto-frontier CLI failed: {res.stderr}"
    assert pareto_out.exists()
    pareto_data = json.loads(pareto_out.read_text())
    assert pareto_data["frontier_candidate_ids"] == ["m1"]
    assert pareto_data["non_dominated_count"] == 1
    assert pareto_data["dominated_count"] == 1

    # Test iteration-screen CLI
    screen_spec = tmp_path / "screen_spec.json"
    screen_out = tmp_path / "screen_out.json"
    screen_spec.write_text(json.dumps({
        "dataset_path": str(synthetic_binary_csv),
        "candidate_dir": str(cand_dir),
        "manifest": manifest,
        "folds": 2,
        "seed": 2026,
    }))

    cmd2 = [
        sys.executable,
        "-m", "mlrescurix_worker.scientific.cli",
        "iteration-screen",
        "--spec", str(screen_spec),
        "--output", str(screen_out),
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0, f"iteration-screen CLI failed: {res2.stderr}"
    assert screen_out.exists()
    screen_data = json.loads(screen_out.read_text())
    assert screen_data["status"] == "completed"
    assert screen_data["tier_reached"] == 1
    assert "proxy_screening" in screen_data
    assert "full_evaluation" in screen_data
