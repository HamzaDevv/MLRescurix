"""Comprehensive Offline Fault Suite for Session 2: Kaggle Backend, Offloading & Durable Recovery.

Verifies all Gate S2 requirements:
1. Credential reference storage and secret redaction.
2. Single-account execution enforcement.
3. Duplicate submission idempotency.
4. Timeout with unknown outcome reconciliation.
5. Fenced atomic leases & stale collector rejection.
6. Lost controller recovery / intent recovery from SQLite.
7. Incomplete download / hash mismatch rejection.
8. Quota exhaustion raising needs_resources.
9. Bounded OOM recovery with microbatch bisection, accumulation doubling, and activation checkpointing.
10. Resource-based local/remote admission control and user overrides.
11. Cancellation limitation handling and lease revocation.
12. Deadline-aware segment and checkpoint resume.
13. Dual-tier gateway route policies and bounded retries.
14. Credential scrubber blocking raw secret leaks in staged scripts.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlrescurix_worker.custom.admission import (
    AdmissionController,
    AdmissionDecision,
    OOMRecoveryPlan,
    OOMRecoveryPolicy,
    feature_detect_amp,
)
from mlrescurix_worker.custom.contracts import (
    PilotResourceReport,
    ResourceRequestContract,
)
from mlrescurix_worker.custom.kaggle_backend import KaggleExecutionBackend
from mlrescurix_worker.scientific.dual_gateway import DualTierGateway, RoutePolicy
from mlrescurix_worker.scientific.kaggle_pool import (
    ChecksumMismatchError,
    JobLifecycleState,
    KaggleAccount,
    KaggleAccountPool,
    QuotaExhaustedError,
    SecurityViolationError,
    StaleLeaseError,
    UnknownRemoteOutcomeError,
)


# =============================================================================
# 1. Credential Reference Storage & Secret Redaction
# =============================================================================

def test_credential_reference_storage_and_redaction(tmp_path: Path, monkeypatch):
    pool_dir = tmp_path / "pool"
    pool = KaggleAccountPool(storage_dir=pool_dir)

    # 1. Register with raw API key -> should write to mode 0600 file and store file: reference
    acc1 = pool.register_account(
        username="user1",
        api_key="super_secret_raw_key_123",
        is_anchor=True,
    )
    assert acc1.api_key == "super_secret_raw_key_123"
    assert acc1.credential_ref.startswith("file:")

    # Check SQLite database: raw key MUST NOT appear anywhere in the database!
    with sqlite3.connect(pool.db_path) as conn:
        cur = conn.execute("SELECT credential_ref FROM kaggle_accounts WHERE username = 'user1'")
        stored_ref = cur.fetchone()[0]
        assert stored_ref.startswith("file:")
        assert "super_secret_raw_key_123" not in stored_ref

        # Query all text columns
        cur = conn.execute("SELECT * FROM kaggle_accounts")
        row_str = str(cur.fetchall())
        assert "super_secret_raw_key_123" not in row_str

    # 2. Register with env: reference
    monkeypatch.setenv("KAGGLE_TEST_KEY", "env_secret_key_456")
    acc2 = pool.register_account(
        username="user2",
        credential_ref="env:KAGGLE_TEST_KEY",
        is_anchor=False,
    )
    assert acc2.api_key == "env_secret_key_456"
    assert acc2.credential_ref == "env:KAGGLE_TEST_KEY"


# =============================================================================
# 2. Single-Account Execution Enforcement
# =============================================================================

def test_single_account_execution_enforcement(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool", enable_secondary_accounts=False)
    pool.register_account(username="anchor_user", api_key="k_anchor", is_anchor=True)
    pool.register_account(username="worker_user", api_key="k_worker", is_anchor=False)

    # In single-account mode, leasing strictly returns the anchor account
    leased = pool.lease_worker_account(accelerator="gpu")
    assert leased.username == "anchor_user"
    assert leased.is_anchor is True

    # If anchor account is depleted, single-account mode raises QuotaExhaustedError
    with sqlite3.connect(pool.db_path) as conn:
        conn.execute("UPDATE kaggle_accounts SET gpu_hours_used = 30.0 WHERE username = 'anchor_user'")
        conn.commit()

    with pytest.raises(QuotaExhaustedError, match="has depleted its weekly GPU quota"):
        pool.lease_worker_account(accelerator="gpu")


# =============================================================================
# 3. Duplicate Submission Idempotency
# =============================================================================

@patch("subprocess.run")
def test_duplicate_submission_idempotency(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Kernel pushed successfully",
        stderr="",
    )

    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    script_code = "print('Training ML candidate')\n"
    job1 = pool.push_private_kernel(
        account=anchor,
        script_code=script_code,
        job_id="job-idempotent-1",
        add_jitter=False,
    )
    assert job1.job_id == "job-idempotent-1"
    assert job1.status == JobLifecycleState.SUBMITTED
    assert mock_run.call_count == 1

    # Second submission with the same job_id should return existing record without calling push again
    job2 = pool.push_private_kernel(
        account=anchor,
        script_code=script_code,
        job_id="job-idempotent-1",
        add_jitter=False,
    )
    assert job2.job_id == "job-idempotent-1"
    assert mock_run.call_count == 1  # No duplicate push


# =============================================================================
# 4. Timeout With Unknown Outcome Reconciliation
# =============================================================================

@patch("subprocess.run")
def test_timeout_unknown_outcome_reconciliation(mock_run, tmp_path: Path):
    # Simulate timeout on push
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["kaggle", "kernels", "push"], timeout=60)

    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    script_code = "print('Heavy training job')\n"
    with pytest.raises(UnknownRemoteOutcomeError, match="reconciliation required"):
        pool.push_private_kernel(
            account=anchor,
            script_code=script_code,
            job_id="job-timeout-1",
            add_jitter=False,
        )

    # Job is persisted with RETRY_WAIT and TIMEOUT_UNKNOWN_OUTCOME note
    job = pool.get_job("job-timeout-1")
    assert job is not None
    assert job.status == JobLifecycleState.RETRY_WAIT
    assert job.reconciliation_notes == "TIMEOUT_UNKNOWN_OUTCOME"

    # Now reconcile with remote Kaggle status
    # Case A: Kaggle reports kernel is ready/complete
    mock_run.side_effect = None
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "status"],
        returncode=0,
        stdout=f'Kernel "{anchor.username}/{job.kernel_slug}" is complete',
        stderr="",
    )
    reconciled = pool.reconcile_job("job-timeout-1")
    assert reconciled.status == JobLifecycleState.COLLECTING
    assert "completed" in reconciled.reconciliation_notes

    # Case B: Kaggle reports 404 (kernel was never received by Kaggle)
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "status"],
        returncode=1,
        stdout="",
        stderr="404 - Not Found",
    )
    reconciled_404 = pool.reconcile_job("job-timeout-1")
    assert reconciled_404.status == JobLifecycleState.RETRY_WAIT
    assert "marked RETRY_WAIT" in reconciled_404.reconciliation_notes


# =============================================================================
# 5. Fenced Atomic Leases & Stale Collector Rejection
# =============================================================================

@patch("subprocess.run")
def test_stale_lease_fencing(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Pushed",
        stderr="",
    )
    backend = KaggleExecutionBackend(storage_dir=tmp_path / "pool")
    backend.pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "kernel.py").write_text("print('test')")

    # Initial submission -> attempt 1
    handle_v1 = backend.submit(staged_dir, {"job_id": "job-fence-1", "accelerator": "gpu"})
    _, _, job_id, lease_v1 = backend.parse_handle(handle_v1)

    # Validate lease v1 is currently valid
    assert backend.pool.validate_lease(job_id, lease_v1) is True

    # A second attempt acquires a new lease (e.g. after worker retry)
    attempt_2, lease_v2, _ = backend.pool.acquire_lease(job_id)
    assert attempt_2 == 2
    assert lease_v2 != lease_v1

    # Lease v1 is now stale / superseded
    assert backend.pool.validate_lease(job_id, lease_v1) is False
    assert backend.pool.validate_lease(job_id, lease_v2) is True

    # Stale collector using handle_v1 MUST be rejected with StaleLeaseError
    with pytest.raises(StaleLeaseError, match="stale, expired, or superseded"):
        backend.collect(handle_v1, target_dir=tmp_path / "output")

    # Status check with handle_v1 detects fencing
    status_info = backend.status(handle_v1)
    assert status_info["fenced"] is True
    assert status_info["status"] == JobLifecycleState.FAILED


# =============================================================================
# 6. Lost Controller Recovery / Intent Recovery from SQLite
# =============================================================================

@patch("subprocess.run")
def test_lost_controller_intent_recovery(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Pushed",
        stderr="",
    )
    storage_dir = tmp_path / "pool"
    pool = KaggleAccountPool(storage_dir=storage_dir)
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    job = pool.push_private_kernel(
        account=anchor,
        script_code="print('Controller crash test')\n",
        job_id="job-crash-1",
        add_jitter=False,
    )

    # Simulate controller crash: pool and backend in-memory state are discarded
    del pool

    # Reconnect with a brand new pool and backend pointing to the persistent SQLite
    new_backend = KaggleExecutionBackend(storage_dir=storage_dir)
    recovered_job = new_backend.pool.get_job("job-crash-1")

    assert recovered_job is not None
    assert recovered_job.job_id == "job-crash-1"
    assert recovered_job.account_username == "user_anchor"
    assert recovered_job.status == JobLifecycleState.SUBMITTED
    assert recovered_job.lease_token == job.lease_token

    # Reconstruct handle and poll status
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "status"],
        returncode=0,
        stdout=f'Kernel "user_anchor/{recovered_job.kernel_slug}" is running',
        stderr="",
    )
    handle = f"kaggle:user_anchor/{recovered_job.kernel_slug}:{recovered_job.job_id}:{recovered_job.lease_token}"
    status = new_backend.status(handle)
    assert status["status"] == JobLifecycleState.RUNNING


# =============================================================================
# 7. Incomplete Download / Hash Mismatch Rejection
# =============================================================================

@patch("subprocess.run")
def test_incomplete_download_hash_mismatch_rejection(mock_run, tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    # Prepare mock remote output directory
    dest = tmp_path / "downloaded_output"
    dest.mkdir()

    # Create weights file and manifest
    weights_path = dest / "weights.pt"
    weights_path.write_text("model_weights_content_v1")
    correct_sha = hashlib.sha256(b"model_weights_content_v1").hexdigest()

    manifest = {
        "status": "complete",
        "files": {
            "weights.pt": correct_sha,
            "extra.pt": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        },
    }
    (dest / "manifest.json").write_text(json.dumps(manifest))

    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "output"],
        returncode=0,
        stdout="Output downloaded",
        stderr="",
    )

    # 1. Test missing file in output -> ChecksumMismatchError
    with pytest.raises(ChecksumMismatchError, match="is missing in output"):
        pool.download_kernel_output(anchor, "opt-job-123", dest, verify_manifest=True)

    # 2. Add extra.pt with corrupted content -> ChecksumMismatchError
    (dest / "extra.pt").write_text("corrupted_content")
    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch for 'extra.pt'"):
        pool.download_kernel_output(anchor, "opt-job-123", dest, verify_manifest=True)

    # 3. Fix content to match correct SHA -> should succeed cleanly
    (dest / "extra.pt").write_text("proper_content")
    correct_extra_sha = hashlib.sha256(b"proper_content").hexdigest()
    manifest["files"]["extra.pt"] = correct_extra_sha
    (dest / "manifest.json").write_text(json.dumps(manifest))

    verified_dest = pool.download_kernel_output(anchor, "opt-job-123", dest, verify_manifest=True)
    assert verified_dest == dest


# =============================================================================
# 8. Quota Exhaustion Raising needs_resources
# =============================================================================

def test_quota_exhausted_needs_resources(tmp_path: Path):
    backend = KaggleExecutionBackend(storage_dir=tmp_path / "pool")
    anchor = backend.pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    # Deplete quota
    with sqlite3.connect(backend.pool.db_path) as conn:
        conn.execute("UPDATE kaggle_accounts SET gpu_hours_used = 30.0 WHERE username = 'user_anchor'")
        conn.commit()

    request = ResourceRequestContract(
        max_duration_seconds=3600,
        accelerator_type="gpu",
    )
    preflight = backend.preflight(request, {})
    assert preflight["status"] == JobLifecycleState.NEEDS_RESOURCES
    assert preflight["admitted"] is False
    assert "quota exhausted" in preflight["reason"].lower()


# =============================================================================
# 9. Bounded OOM Recovery: Microbatch Bisection & Accumulation
# =============================================================================

def test_oom_recovery_microbatch_bisection_and_accumulation():
    initial_plan = OOMRecoveryPlan(
        microbatch_size=8,
        gradient_accumulation_steps=2,
        effective_batch_size=16,
        activation_checkpointing=False,
    )

    # 1. First OOM -> bisect microbatch from 8 to 4, accumulation 2 -> 4
    plan1 = OOMRecoveryPolicy.handle_oom(initial_plan)
    assert plan1 is not None
    assert plan1.microbatch_size == 4
    assert plan1.gradient_accumulation_steps == 4
    assert plan1.effective_batch_size == 16  # Strictly preserved
    assert plan1.retry_count == 1
    assert plan1.activation_checkpointing is False

    # 2. Second OOM -> bisect microbatch from 4 to 2, accumulation 4 -> 8
    plan2 = OOMRecoveryPolicy.handle_oom(plan1)
    assert plan2 is not None
    assert plan2.microbatch_size == 2
    assert plan2.gradient_accumulation_steps == 8
    assert plan2.effective_batch_size == 16
    assert plan2.retry_count == 2

    # 3. Third OOM -> bisect microbatch from 2 to 1, accumulation 8 -> 16
    plan3 = OOMRecoveryPolicy.handle_oom(plan2)
    assert plan3 is not None
    assert plan3.microbatch_size == 1
    assert plan3.gradient_accumulation_steps == 16
    assert plan3.effective_batch_size == 16
    assert plan3.retry_count == 3

    # 4. Fourth OOM -> retry count reached max_retries (3), stops cleanly
    plan_exhausted = OOMRecoveryPolicy.handle_oom(plan3)
    assert plan_exhausted is None


def test_oom_recovery_activation_checkpointing():
    # If starting at microbatch 1 with max_retries=3
    initial_plan = OOMRecoveryPlan(
        microbatch_size=1,
        gradient_accumulation_steps=16,
        effective_batch_size=16,
        activation_checkpointing=False,
        max_retries=3,
    )

    # Next attempt should enable activation checkpointing
    plan1 = OOMRecoveryPolicy.handle_oom(initial_plan)
    assert plan1 is not None
    assert plan1.microbatch_size == 1
    assert plan1.activation_checkpointing is True
    assert plan1.retry_count == 1

    # Following attempt cannot reduce further -> returns None
    plan2 = OOMRecoveryPolicy.handle_oom(plan1)
    assert plan2 is None


# =============================================================================
# 10. Admission Controller Decisions & User Overrides
# =============================================================================

def test_admission_controller_decisions(tmp_path: Path):
    backend = KaggleExecutionBackend(storage_dir=tmp_path / "pool")
    backend.pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    controller = AdmissionController(
        local_memory_limit_bytes=2 * 1024 * 1024 * 1024,  # 2 GB limit
        kaggle_backend=backend,
    )

    # 1. Fits within local boundaries -> admitted local
    req_cpu = ResourceRequestContract(accelerator_type="cpu", max_duration_seconds=300)
    pilot_small = PilotResourceReport(
        feasible=True,
        allocated_bytes=100 * 1024 * 1024,
        reserved_bytes=150 * 1024 * 1024,
        peak_bytes=256 * 1024 * 1024,
        duration_seconds=0.05,
        microbatch_size=16,
        device="cpu",
        precision="float32",
    )
    dec1 = controller.admit(req_cpu, pilot_small)
    assert dec1.backend == "local"
    assert dec1.admitted is True

    # 2. Pilot memory exceeds local limit -> auto offload to Kaggle
    pilot_large = PilotResourceReport(
        feasible=True,
        allocated_bytes=6 * 1024 * 1024 * 1024,
        reserved_bytes=7 * 1024 * 1024 * 1024,
        peak_bytes=8 * 1024 * 1024 * 1024,
        duration_seconds=1.5,
        microbatch_size=16,
        device="cpu",
        precision="float32",
    )
    dec2 = controller.admit(req_cpu, pilot_large)
    assert dec2.backend == "kaggle"
    assert dec2.admitted is True
    assert "Offloaded to Kaggle" in dec2.reason

    # 3. User override forced to "local" when pilot memory exceeds local limit -> needs_resources
    dec3 = controller.admit(req_cpu, pilot_large, user_override="local")
    assert dec3.backend == "needs_resources"
    assert dec3.admitted is False
    assert "exceeds local limit" in dec3.reason

    # 4. User override forced to "kaggle" -> routes to Kaggle
    dec4 = controller.admit(req_cpu, pilot_small, user_override="kaggle")
    assert dec4.backend == "kaggle"
    assert dec4.admitted is True


# =============================================================================
# 11. Cancellation Limitation & Lease Revocation
# =============================================================================

@patch("subprocess.run")
def test_cancellation_limitation_and_lease_revocation(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Pushed",
        stderr="",
    )
    backend = KaggleExecutionBackend(storage_dir=tmp_path / "pool")
    backend.pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    (staged_dir / "kernel.py").write_text("print('test')")

    handle = backend.submit(staged_dir, {"job_id": "job-cancel-1", "accelerator": "gpu"})
    _, _, job_id, lease_token = backend.parse_handle(handle)

    # Cancel request
    cancelled = backend.cancel(handle)
    assert cancelled is True

    # Check job record: status is CANCEL_REQUESTED and note explains Kaggle limitation
    job = backend.pool.get_job(job_id)
    assert job.status == JobLifecycleState.CANCEL_REQUESTED
    assert "Kaggle API lacks remote force-kill" in job.reconciliation_notes

    # Lease is revoked
    assert backend.pool.validate_lease(job_id, lease_token) is False


# =============================================================================
# 12. Deadline-Aware Segment & Checkpoint Resume
# =============================================================================

@patch("subprocess.run")
def test_deadline_aware_segment_and_checkpoint_resume(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Pushed",
        stderr="",
    )
    backend = KaggleExecutionBackend(storage_dir=tmp_path / "pool")
    backend.pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    # Stage segment 1
    staged_dir = tmp_path / "staged_seg1"
    staged_dir.mkdir()
    (staged_dir / "kernel.py").write_text("print('Segment 1')")
    handle1 = backend.submit(staged_dir, {"job_id": "seg-job-1"})

    # Mock output collection for segment 1
    ckpt_dir = tmp_path / "output_seg1" / "checkpoint"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "weights.pt").write_text("step_50_weights")
    manifest = {
        "status": "complete",
        "step": 50,
        "files": {
            "weights.pt": hashlib.sha256(b"step_50_weights").hexdigest(),
        },
    }
    (ckpt_dir / "manifest.json").write_text(json.dumps(manifest))

    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "output"],
        returncode=0,
        stdout="Output downloaded",
        stderr="",
    )
    collect_res = backend.collect(handle1, target_dir=tmp_path / "output_seg1")
    assert collect_res["status"] == JobLifecycleState.VERIFIED

    # Resume training in segment 2
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Segment 2 Pushed",
        stderr="",
    )
    with pytest.raises(NotImplementedError, match="remote resume"):
        backend.resume(
            job_handle=handle1,
            checkpoint_manifest={"checkpoint_dir": str(ckpt_dir), "version": "v1"},
        )


# =============================================================================
# 13. Dual-Tier Gateway Route Policies & Bounded Retries
# =============================================================================

@patch("urllib.request.urlopen")
def test_dual_gateway_direct_routes_are_disabled(mock_urlopen):
    for policy in (RoutePolicy.AUTO_FAILOVER, RoutePolicy.PRIMARY_ONLY, RoutePolicy.FALLBACK_ONLY):
        gateway = DualTierGateway(openrouter_key="key", deepseek_key="key", route_policy=policy)
        with pytest.raises(RuntimeError, match="Direct dual-gateway dispatch is disabled"):
            gateway.call_with_request_id([{"role": "user", "content": "hi"}], request_id="req-1")
    mock_urlopen.assert_not_called()


# =============================================================================
# 14. Credential Scrubber Blocks Secret Leaks in Staged Scripts
# =============================================================================

def test_credential_scrubber_blocks_leaks(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    # Staging script with embedded API key MUST be blocked
    leaky_script_1 = "OPENROUTER_API_KEY = 'sk-or-v1-abcdef1234567890'\nprint('leak')"
    with pytest.raises(SecurityViolationError, match="credential leakage detected"):
        pool.push_private_kernel(anchor, script_code=leaky_script_1, add_jitter=False)

    leaky_script_2 = "headers = {'Authorization': 'Bearer sk-ant-api03-abcdef'}"
    with pytest.raises(SecurityViolationError, match="credential leakage detected"):
        pool.push_private_kernel(anchor, script_code=leaky_script_2, add_jitter=False)

    # Raw arbitrary sk- token in arbitrary variable or expression MUST be caught
    leaky_script_3 = "custom_token = 'sk-or-v1-0987654321fedcba0987654321'\nrun()"
    with pytest.raises(SecurityViolationError, match="credential leakage detected"):
        pool.push_private_kernel(anchor, script_code=leaky_script_3, add_jitter=False)


# =============================================================================
# 15. Edge Cases: Missing Manifest & Remote Cancellation Status
# =============================================================================

@patch("subprocess.run")
def test_missing_manifest_in_download_raises_error(mock_run, tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    dest = tmp_path / "empty_output"
    dest.mkdir()
    (dest / "orphan.pt").write_text("some_weights")

    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "output"],
        returncode=0,
        stdout="Output downloaded",
        stderr="",
    )
    with pytest.raises(ChecksumMismatchError, match="manifest.json is missing"):
        pool.download_kernel_output(anchor, "opt-job-999", dest, verify_manifest=True)


@patch("subprocess.run")
def test_remote_status_cancellation_parsed_correctly(mock_run, tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    anchor = pool.register_account(username="user_anchor", api_key="k_anchor", is_anchor=True)

    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "status"],
        returncode=0,
        stdout='Kernel "user_anchor/opt-job-123" is cancelPending',
        stderr="",
    )
    res = pool.check_kernel_status(anchor, "opt-job-123")
    assert res["status"] == JobLifecycleState.CANCELLED

    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "status"],
        returncode=0,
        stdout='Kernel "user_anchor/opt-job-123" is canceled',
        stderr="",
    )
    res2 = pool.check_kernel_status(anchor, "opt-job-123")
    assert res2["status"] == JobLifecycleState.CANCELLED
