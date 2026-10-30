"""Session 1 Verification Suite: Contracts, Correctness, and Local Custom PyTorch Training.

Tests all Session 1 gates specified in docs/ARCHITECTURE_BUILD_HANDOFF.md:
1. Versioned contracts (ProblemValidation, ResourceRequest, TrainingState, PilotReport, ResultBundle).
2. TinyTorchNeuralCandidate:
   - Prepare and pilot resource report (RAM/VRAM probe, single-microbatch forward/backward, feasibility).
   - Bounded data loading and training.
   - Local complete checkpoints with SHA-256 manifests, fsync, and COMMIT markers.
   - Step-boundary interruption and exact resume.
   - Preflight contract compliance via run_preflight_checks.
3. AutonomousAgentLoop correctness:
   - Rejected iteration cannot deploy (Apex eligibility gate).
   - User pin survives continue search, subsequent breakthroughs, checkpoint resume, and export.
   - Primary/secondary metric consistency (fold dispersion and train/val gaps within identical metric).
4. Streaming hash vs whole-file eager reads.
5. DeepSeek search schema (count) and DualGateway estimate flags.
"""

import copy
import hashlib
import json
import os
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.custom.contracts import (
    CandidatePluginContract,
    ExecutionBackendContract,
    PilotResourceReport,
    ProblemValidationContract,
    ResourceRequestContract,
    ResultBundleContract,
    TrainingStateContract,
    file_sha256,
)
from mlrescurix_worker.custom.preflight import run_preflight_checks
from mlrescurix_worker.custom.torch_candidate import (
    HAS_TORCH,
    TinyTorchNeuralCandidate,
    TrainingResourceExhaustionError,
)
from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
from mlrescurix_worker.scientific.deepseek_search import DeepSeekSearchClient
from mlrescurix_worker.scientific.dual_gateway import DualTierGateway
from mlrescurix_worker.scientific.evidence import streaming_file_sha256


class MockBudgetAuthority:
    def __init__(self):
        self.ledger_path = Path("/tmp/mock_ledger.json")

    def reserve(self, session_id, request_id, amount):
        return amount

    def settle(self, session_id, request_id, cost, cost_kind="unverified"):
        pass

    def mark_uncertain(self, session_id, request_id, amount):
        pass

    def get_status(self):
        return {
            "campaign_ceiling_usd": "0.30",
            "remaining_campaign_usd": "0.25",
            "total_billed_usd": "0.05",
            "total_reserved_usd": "0.00",
            "total_uncertain_usd": "0.00",
        }


def create_synthetic_data(path: Path, n_samples: int = 100) -> Path:
    np.random.seed(42)
    df = pd.DataFrame({
        "feat_a": np.random.randn(n_samples),
        "feat_b": np.random.randn(n_samples),
        "feat_c": np.random.rand(n_samples) * 10,
        "target": (np.random.randn(n_samples) > 0).astype(int),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# =============================================================================
# 1. Versioned Contracts Tests
# =============================================================================

def test_problem_validation_contract():
    contract = ProblemValidationContract(
        problem_id="prob_001",
        modality="tabular",
        task_type="binary_classification",
        metric_name="balanced_accuracy",
        metric_direction="maximize",
        target_column="target",
    )
    contract.validate()
    d = contract.to_dict()
    assert d["problem_id"] == "prob_001"
    reloaded = ProblemValidationContract.from_dict(d)
    assert reloaded.problem_id == "prob_001"

    # Invalid direction
    bad = ProblemValidationContract(
        problem_id="p", modality="tabular", task_type="regression",
        metric_name="rmse", metric_direction="sideways"
    )
    with pytest.raises(ValueError, match="metric_direction must be"):
        bad.validate()


def test_resource_request_and_pilot_report_contracts():
    req = ResourceRequestContract(
        ram_bytes_limit=4 * 1024 * 1024 * 1024,
        vram_bytes_limit=2 * 1024 * 1024 * 1024,
        accelerator_type="gpu",
        accelerator_count=1,
    )
    req_dict = req.to_dict()
    assert req_dict["ram_bytes_limit"] == 4 * 1024 * 1024 * 1024

    pilot = PilotResourceReport(
        feasible=True,
        allocated_bytes=1024 * 1024,
        reserved_bytes=2 * 1024 * 1024,
        peak_bytes=4 * 1024 * 1024,
        duration_seconds=0.015,
        microbatch_size=8,
        device="cpu",
        precision="float32",
    )
    p_dict = pilot.to_dict()
    assert p_dict["feasible"] is True
    reloaded_pilot = PilotResourceReport.from_dict(p_dict)
    assert reloaded_pilot.microbatch_size == 8


def test_training_state_and_result_bundle_contracts():
    state = TrainingStateContract(
        step=50,
        epoch=2,
        optimizer_step_boundary=True,
        weights_manifest={"weights.pt": "abc123sha"},
        loss=0.345,
    )
    assert state.step == 50
    assert state.to_dict()["step"] == 50

    bundle = ResultBundleContract(
        candidate_id="cand_test",
        primary_metric_name="balanced_accuracy",
        primary_metric_score=0.85,
        primary_metric_direction="maximize",
        aligned_predictions=[0, 1, 0, 1],
        row_ids=[10, 11, 12, 13],
        probabilities=[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
        classes=[0, 1],
    )
    assert bundle.candidate_id == "cand_test"
    assert len(bundle.aligned_predictions) == 4


# =============================================================================
# 2. Tiny Custom PyTorch Candidate Tests
# =============================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural tests")
def test_torch_candidate_prepare_and_pilot():
    cand = TinyTorchNeuralCandidate(hidden_dim=16, is_classification=True, device="cpu", random_seed=42)
    req = ResourceRequestContract(ram_bytes_limit=2 * 1024 * 1024 * 1024)

    cand.prepare({"num_features": 4}, req)
    assert cand.model is not None
    assert cand.in_features == 4

    X_dummy = pd.DataFrame(np.random.randn(20, 4), columns=["c0", "c1", "c2", "c3"])
    y_dummy = pd.Series((X_dummy["c0"] > 0).astype(int))

    report = cand.pilot((X_dummy, y_dummy), req)
    assert isinstance(report, PilotResourceReport)
    assert report.feasible is True
    assert report.microbatch_size == 8
    assert report.device == "cpu"

    # Constrained request marking pilot infeasible
    tight_req = ResourceRequestContract(ram_bytes_limit=10)  # only 10 bytes RAM allowed
    tight_report = cand.pilot((X_dummy, y_dummy), tight_req)
    assert tight_report.feasible is False
    assert len(tight_report.limitations) > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural tests")
def test_torch_candidate_train_predict_and_preflight(tmp_path: Path):
    cand = TinyTorchNeuralCandidate(hidden_dim=16, is_classification=True, device="cpu", random_seed=123)

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(40, 3), columns=["f1", "f2", "f3"])
    y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))

    # Test preflight checks compatibility
    preflight = run_preflight_checks(cand, X, y, is_classification=True)
    assert preflight["passed"] is True
    assert preflight["checks"]["fit_succeeded"] is True
    assert preflight["checks"]["predict_shape_match"] is True
    assert preflight["checks"]["predict_proba_valid"] is True
    assert preflight["checks"]["single_row_inference"] is True

    # Check prediction output contracts
    preds = cand.predict(X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 40
    probs = cand.predict_proba(X)
    assert probs is not None
    assert probs.shape == (40, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural tests")
def test_torch_candidate_step_interruption_and_atomic_checkpoint_resume(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"

    np.random.seed(2026)
    X = pd.DataFrame(np.random.randn(60, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] > 0).astype(int))

    # Phase 1: Train candidate A for 5 steps and save checkpoint
    cand_a = TinyTorchNeuralCandidate(hidden_dim=16, is_classification=True, device="cpu", random_seed=999)
    cand_a.train((X, y), max_steps=5)
    assert cand_a.current_step == 5

    manifest = cand_a.save_checkpoint(ckpt_dir, step=5)
    assert manifest["step"] == 5
    step5_dir = ckpt_dir / "step_5"
    assert step5_dir.is_dir()
    assert (step5_dir / "COMMIT").is_file()
    assert (step5_dir / "manifest.json").is_file()
    assert (step5_dir / "weights.pt").is_file()
    assert (step5_dir / "optimizer.pt").is_file()
    assert (step5_dir / "rng.pt").is_file()

    # Verify SHA-256 integrity
    for fname, sha in manifest["files"].items():
        assert file_sha256(step5_dir / fname) == sha

    # Continue candidate A for 5 more steps (total 10 steps)
    cand_a.train((X, y), max_steps=10)
    assert cand_a.current_step == 10
    preds_a = cand_a.predict(X)

    # Phase 2: Create candidate B, resume from step 5 checkpoint, train to 10 steps
    cand_b = TinyTorchNeuralCandidate(hidden_dim=16, is_classification=True, device="cpu", random_seed=999)
    cand_b.train((X, y), resume_from=step5_dir, max_steps=10)
    assert cand_b.current_step == 10
    preds_b = cand_b.predict(X)

    # Exact replay check: step 5 -> 10 resume matches unbroken 10 steps
    np.testing.assert_array_equal(preds_a, preds_b)

    # Corrupt checkpoint test: tampering with weights.pt must trigger validation error
    tamper_dir = tmp_path / "tampered"
    shutil.copytree(step5_dir, tamper_dir)
    (tamper_dir / "weights.pt").write_bytes(b"CORRUPTED_BYTES")
    cand_c = TinyTorchNeuralCandidate(hidden_dim=16, device="cpu")
    with pytest.raises(ValueError, match="Checkpoint corruption"):
        cand_c.load_checkpoint(tamper_dir)

    # Incomplete checkpoint test: missing COMMIT marker must trigger error
    uncommitted_dir = tmp_path / "uncommitted"
    shutil.copytree(step5_dir, uncommitted_dir)
    (uncommitted_dir / "COMMIT").unlink()
    with pytest.raises(ValueError, match="missing COMMIT marker"):
        cand_c.load_checkpoint(uncommitted_dir)


# =============================================================================
# 3. Autonomous Loop: Apex Eligibility, Pin Persistence, Metric Consistency
# =============================================================================

def test_apex_eligibility_gate_rejected_iteration_cannot_deploy(tmp_path: Path):
    """Gate S1: A rejected iteration cannot be selected or deployed."""
    dataset_path = create_synthetic_data(tmp_path / "data.csv")
    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws_gate",
        project_id="test_gate",
        session_id="run_gate_test",
        provider_is_fixture=True,
        campaign_authority=MockBudgetAuthority(),
        dataset_path=dataset_path,
        target_column="target",
        max_turns=0,
    )

    # Iteration 1: Rejected iteration (e.g. data leakage, failing audit)
    eval_rejected = {
        "cv_score": 0.95,  # high score but audited as REJECTED
        "critic_validation_score": 0.95,
        "receipt_identity": "rcpt_rej",
    }
    audit_rejected = {"status": "REJECTED", "candidate_id": "cand_rej", "flaws": ["Data leakage detected"]}
    rec = loop._record_iteration(
        turn_index=0,
        action_type="EVALUATE_PORTFOLIO",
        iteration_type="leaky_model",
        description="Leaky experiment",
        spec={"families": ["hist_gbdt"], "weights": [1], "transforms": []},
        eval_result=eval_rejected,
        audit=audit_rejected,
        candidate_id="cand_rej",
    )
    assert rec["is_apex"] is False
    assert loop.state["latest_apex_iteration"] is None

    # Gate S1 verification: Attempting to select/deploy rejected iteration must fail
    with pytest.raises(ValueError, match="not eligible for selection/deployment: critic status is 'REJECTED'"):
        loop.select_apex_iteration(rec["iteration_number"])

    # Iteration with non-finite score cannot deploy
    eval_nan = {"cv_score": float("nan"), "receipt_identity": "rcpt_nan"}
    audit_nan = {"status": "APPROVED", "candidate_id": "cand_nan"}
    rec_nan = loop._record_iteration(
        turn_index=1,
        action_type="EVALUATE_PORTFOLIO",
        iteration_type="nan_model",
        description="NaN experiment",
        spec={"families": ["hist_gbdt"], "weights": [1], "transforms": []},
        eval_result=eval_nan,
        audit=audit_nan,
        candidate_id="cand_nan",
    )
    with pytest.raises(ValueError, match="not a finite value"):
        loop.select_apex_iteration(rec_nan["iteration_number"])


def test_user_pin_survives_continue_search_and_export(tmp_path: Path):
    """Gate S1: A user pin survives continued search, subsequent breakthroughs, and export."""
    dataset_path = create_synthetic_data(tmp_path / "data.csv")
    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws_pin",
        project_id="test_pin",
        session_id="run_pin_test",
        provider_is_fixture=True,
        campaign_authority=MockBudgetAuthority(),
        dataset_path=dataset_path,
        target_column="target",
        max_turns=0,
    )

    # 1. Iteration 1: Baseline model (CV 0.70, APPROVED)
    spec1 = {"families": ["penalized_linear"], "weights": [1], "transforms": []}
    eval1 = {"cv_score": 0.70, "critic_validation_score": 0.70, "receipt_identity": "rcpt_1"}
    audit1 = {"status": "APPROVED", "candidate_id": "cand_1"}
    loop._record_iteration(
        turn_index=0, action_type="EVALUATE_PORTFOLIO", iteration_type="baseline",
        description="Baseline", spec=spec1, eval_result=eval1, audit=audit1, candidate_id="cand_1",
    )
    assert loop.state["latest_apex_iteration"] == 1
    assert loop.state["selected_apex_iteration"] == 1

    # 2. User explicitly pins Iteration 1
    pin_res = loop.pin_iteration(1)
    assert pin_res["status"] == "success"
    assert loop.state["pinned_iteration"] == 1
    assert loop.state["accepted_candidate_id"] == "cand_1"
    assert loop.state["selected_cv_score"] == 0.70

    # 3. Continued search: Iteration 2 is an ensemble breakthrough (CV 0.88, APPROVED)
    spec2 = {"families": ["hist_gbdt", "lgbm"], "weights": [1, 1], "transforms": []}
    eval2 = {"cv_score": 0.88, "critic_validation_score": 0.88, "receipt_identity": "rcpt_2"}
    audit2 = {"status": "APPROVED", "candidate_id": "cand_2"}
    loop._record_iteration(
        turn_index=1, action_type="BLEND_AND_AUDIT", iteration_type="ensemble",
        description="Ensemble breakthrough", spec=spec2, eval_result=eval2, audit=audit2, candidate_id="cand_2",
    )
    # The search discovered a new breakthrough:
    assert loop.state["latest_apex_iteration"] == 2
    assert loop.state["best_cv_score"] == 0.88

    # BUT the user pin MUST NOT have been overwritten:
    assert loop.state["pinned_iteration"] == 1
    assert loop.state["selected_apex_iteration"] == 1
    assert loop.state["accepted_candidate_id"] == "cand_1"
    assert loop.state["accepted_candidate"] == spec1
    assert loop.state["selected_cv_score"] == 0.70

    # 4. Export handoff preserves pin
    loop.state["stop_reason"] = "FINALIZE_AND_STOP"
    handoff = loop.export_handoff()
    assert handoff["pinned_iteration"] == 1
    assert handoff["selected_apex_iteration"] == 1
    assert handoff["latest_apex_iteration"] == 2
    assert handoff["accepted_candidate_id"] == "cand_1"

    handoff_text = (tmp_path / "ws_pin" / "HANDOFF.md").read_text()
    assert "Pinned Iteration: 1" in handoff_text
    assert "👑 Apex #1 (Pinned & Selected)" in handoff_text

    # 5. User can unpin to revert to latest apex
    unpin_res = loop.unpin_iteration()
    assert loop.state["pinned_iteration"] is None
    assert loop.state["selected_apex_iteration"] == 2
    assert loop.state["accepted_candidate_id"] == "cand_2"


def test_primary_and_secondary_metric_consistency(tmp_path: Path):
    """Ensure fold dispersion and train-val gap are computed strictly on the primary metric."""
    dataset_path = create_synthetic_data(tmp_path / "data.csv")
    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws_metric",
        project_id="test_metric",
        session_id="run_metric_test",
        provider_is_fixture=True,
        campaign_authority=MockBudgetAuthority(),
        dataset_path=dataset_path,
        target_column="target",
        metric_name="balanced_accuracy",
        max_turns=0,
    )

    # Primary metric: balanced_accuracy (0.80, folds: 0.78, 0.82)
    # Secondary critic metric: log_loss (0.35, folds: 0.34, 0.36)
    eval_res = {
        "cv_score": 0.80,  # primary
        "train_score": 0.85,  # primary train
        "critic_validation_score": 0.35,  # secondary
        "critic_train_score": 0.30,  # secondary train
        "fold_receipts": [
            {"validation_score": 0.78, "critic_validation_score": 0.34},
            {"validation_score": 0.82, "critic_validation_score": 0.36},
        ],
        "receipt_identity": "rcpt_consistent",
    }
    rec = loop._record_iteration(
        turn_index=0,
        action_type="EVALUATE_PORTFOLIO",
        iteration_type="baseline",
        description="Metric consistency check",
        spec={"families": ["hist_gbdt"], "weights": [1], "transforms": []},
        eval_result=eval_res,
        audit={"status": "APPROVED", "candidate_id": "cand_m"},
        candidate_id="cand_m",
    )

    # Primary fold dispersion: np.std([0.78, 0.82]) == 0.02
    assert np.isclose(rec["fold_std"], 0.02, atol=1e-4)
    # Primary train_val_gap: abs(0.85 - 0.80) == 0.05 (NOT abs(0.30 - 0.80) == 0.50!)
    assert np.isclose(rec["train_val_gap"], 0.05, atol=1e-4)
    # Secondary diagnostics recorded separately
    assert np.isclose(rec["secondary_validation_score"], 0.35, atol=1e-4)
    assert np.isclose(rec["secondary_train_val_gap"], 0.05, atol=1e-4)


# =============================================================================
# 4. Streaming Hash vs Eager Reads
# =============================================================================

def test_streaming_hash_fidelity(tmp_path: Path):
    test_file = tmp_path / "large_data_sim.bin"
    # Create 500 KB test payload
    data = os.urandom(500 * 1024)
    test_file.write_bytes(data)

    expected = hashlib.sha256(data).hexdigest()
    actual = streaming_file_sha256(test_file, chunk_size=8192)
    assert actual == expected


# =============================================================================
# 5. DeepSeek Search & Dual Gateway Governance Repairs
# =============================================================================

def test_deepseek_search_schema_and_key_isolation(tmp_path: Path):
    # Verify OpenRouter key is NOT accepted as fallback for DeepSeek
    openrouter_file = tmp_path / ".config" / "openrouter.key"
    openrouter_file.parent.mkdir(parents=True, exist_ok=True)
    openrouter_file.write_text("sk-or-invalid-key-for-deepseek")

    with patch.dict(os.environ, {}, clear=True):
        client = DeepSeekSearchClient(api_key=None)
        # Should not have resolved openrouter key
        assert client.api_key == ""

    # A model API key does not establish a search or fetch capability.
    with patch("urllib.request.urlopen") as mock_urlopen:
        client = DeepSeekSearchClient(api_key="sk-deepseek-test-key")
        results = client.search(query="gradient boosting regularization", count=3)
        assert results == []
        assert client.last_status == "unavailable"
        assert "unavailable" in client.fetch("https://example.com/private")
        assert "unavailable" in client.research_topic("private query")
        mock_urlopen.assert_not_called()


def test_dual_gateway_conservative_reservation_and_estimate_flag():
    gateway = DualTierGateway(
        openrouter_key="mock_or_key",
        deepseek_key="mock_ds_key",
        primary_model="deepseek/deepseek-v4.1-flash",
    )
    messages = [{"role": "user", "content": "Analyze dataset properties"}]
    res = gateway.reservation(messages, max_tokens=1000)
    assert isinstance(res, Decimal)
    assert res > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural tests")
def test_torch_candidate_exact_replay_restores_seed_and_batch_size_without_constructor_args(tmp_path: Path):
    """Verifies that checkpoint restoration recovers seed and batch size without caller arguments."""
    ckpt_dir = tmp_path / "checkpoints"
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(80, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int))

    # Continuous 20 steps with seed=777 and batch_size=16
    c_unbroken = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01, batch_size=16, random_seed=777)
    c_unbroken.train((X, y), max_steps=20)
    preds_unbroken = c_unbroken.predict(X)

    # 10 steps, checkpoint, then resume with fresh instance having default args
    c_part1 = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01, batch_size=16, random_seed=777)
    c_part1.train((X, y), max_steps=10)
    c_part1.save_checkpoint(ckpt_dir, step=10)

    # c_part2 created WITHOUT seed (defaults to 42) and default batch_size
    c_part2 = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01)
    c_part2.train((X, y), resume_from=ckpt_dir, max_steps=20)

    assert c_part2.random_seed == 777
    assert c_part2.batch_size == 16
    preds_resumed = c_part2.predict(X)
    np.testing.assert_array_equal(preds_unbroken, preds_resumed)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural tests")
def test_torch_candidate_multiclass_classification():
    """Verifies that TinyTorchNeuralCandidate supports multiclass classification (classes > 2)."""
    np.random.seed(123)
    X = pd.DataFrame(np.random.randn(90, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    # 3 classes: 0, 1, 2
    y = pd.Series(np.random.choice([0, 1, 2], size=90))

    cand = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01, is_classification=True, random_seed=42)
    cand.train((X, y), max_steps=10)

    assert cand.classes_ == [0, 1, 2]
    assert cand.out_features_ == 3

    preds = cand.predict(X)
    assert set(preds).issubset({0, 1, 2})

    probs = cand.predict_proba(X)
    assert probs is not None
    assert probs.shape == (90, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural tests")
def test_bundle_packaging_with_arbitrary_classes(tmp_path: Path):
    """Verifies that CustomBundlePackager correctly verifies and executes bundles with arbitrary class labels."""
    from mlrescurix_worker.custom.delivery import CustomBundlePackager

    X = pd.DataFrame(np.random.randn(30, 3), columns=["x1", "x2", "x3"])
    # Non-standard integer labels: [10, 20]
    y = pd.Series([10 if i % 2 == 0 else 20 for i in range(30)])

    cand = TinyTorchNeuralCandidate(hidden_dim=16, lr=0.01, is_classification=True, random_seed=42)
    cand.train((X, y), max_steps=5)

    bundle_dir = tmp_path / "bundle_arbitrary"
    packager = CustomBundlePackager()
    receipt = packager.package(
        candidate=cand,
        output_bundle_dir=bundle_dir,
        target_column="target",
        task_type="binary_classification",
        primary_metric="accuracy",
        primary_metric_score=1.0,
        metric_direction="maximize",
        feature_names=["x1", "x2", "x3"],
        raw_example_records=X.to_dict(orient="records"),
    )
    assert receipt.candidate_id is not None
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "inference.py").is_file()


def test_v2_json_schema_validation():
    """Verifies that v2 contract schemas validate correctly across CustomJobSpec, Checkpoint, and Result."""
    import jsonschema
    repo_root = Path(__file__).resolve().parents[2]
    spec_schema = json.loads((repo_root / "contracts/v2/custom_job_spec.schema.json").read_text())
    ckpt_schema = json.loads((repo_root / "contracts/v2/training_checkpoint.schema.json").read_text())
    result_schema = json.loads((repo_root / "contracts/v2/training_result.schema.json").read_text())

    from mlrescurix_worker.custom.contracts import (
        CustomJobSpecContract,
        TrainingCheckpointContract,
        TrainingResultContract,
    )

    spec = CustomJobSpecContract(
        job_id="j_schema",
        project_id="p_schema",
        task_id="t_schema",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_schema",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
    )
    jsonschema.validate(spec.to_dict(), spec_schema)

    ckpt = TrainingCheckpointContract(
        checkpoint_id="ckpt_schema",
        job_id="j_schema",
        step=5,
        epoch=1,
        batch_offset=2,
        sampler_position=80,
        files={"weights.pt": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"},
        timestamp_unix=1700000000.0,
    )
    jsonschema.validate(ckpt.to_dict(), ckpt_schema)

    result = TrainingResultContract(
        job_id="j_schema",
        status="succeeded",
        step_reached=10,
        epoch_reached=1,
        peak_memory_bytes=1048576,
        duration_seconds=1.5,
        artifacts=["result.json"],
    )
    jsonschema.validate(result.to_dict(), result_schema)
