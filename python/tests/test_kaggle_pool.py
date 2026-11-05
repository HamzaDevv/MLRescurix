"""Unit tests for Multi-Account Kaggle Compute Pool & Anti-Ban Architecture."""

import json
import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mlrescurix_worker.scientific.kaggle_pool import (
    AntiBanViolationError,
    KaggleAccount,
    KaggleAccountPool,
    QuotaExhaustedError,
)


def test_register_and_list_accounts(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")

    # Register Anchor account
    anchor = pool.register_account(
        username="user_anchor",
        api_key="key_anchor_123",
        is_anchor=True,
    )
    assert anchor.username == "user_anchor"
    assert anchor.is_anchor is True
    assert anchor.remaining_gpu_hours == 30.0

    # Register Worker account 1
    w1 = pool.register_account(
        username="user_worker_1",
        api_key="key_w1_456",
        is_anchor=False,
    )
    assert w1.username == "user_worker_1"
    assert w1.is_anchor is False

    # Register Worker account 2
    w2 = pool.register_account(
        username="user_worker_2",
        api_key="key_w2_789",
        is_anchor=False,
    )

    accounts = pool.list_accounts()
    assert len(accounts) == 3
    # Anchor should be first
    assert accounts[0].username == "user_anchor"
    assert pool.get_anchor_account().username == "user_anchor"


def test_anti_ban_multiple_anchors_rejected(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    pool.register_account(username="user_anchor_1", api_key="k1", is_anchor=True)

    # Attempting to register a second anchor account must raise AntiBanViolationError
    with pytest.raises(AntiBanViolationError, match="already the designated official anchor"):
        pool.register_account(username="user_anchor_2", api_key="k2", is_anchor=True)


def test_anti_ban_worker_submission_blocked(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    pool.register_account(username="user_anchor", api_key="k1", is_anchor=True)
    worker = pool.register_account(username="user_worker_1", api_key="k2", is_anchor=False)

    sub_file = tmp_path / "submission.csv"
    sub_file.write_text("id,pred\n1,0.5\n")

    # If a worker account attempts to submit to a competition, reject immediately
    with pytest.raises(AntiBanViolationError, match="Anti-Ban Violation: Worker account"):
        pool.submit_competition_prediction(
            competition_id="titanic",
            submission_file=sub_file,
            message="attempt from worker",
            account=worker,
        )


def test_lease_worker_account_quota_priority(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    pool.register_account(username="user_anchor", api_key="ka", is_anchor=True)

    # w1 has 10h used (20h remaining)
    w1 = pool.register_account(username="w1", api_key="k1", is_anchor=False)
    with sqlite3.connect(pool.db_path) as conn:
        conn.execute("UPDATE kaggle_accounts SET gpu_hours_used = 10.0 WHERE username = 'w1'")
        conn.commit()

    # w2 has 5h used (25h remaining)
    w2 = pool.register_account(username="w2", api_key="k2", is_anchor=False)
    with sqlite3.connect(pool.db_path) as conn:
        conn.execute("UPDATE kaggle_accounts SET gpu_hours_used = 5.0 WHERE username = 'w2'")
        conn.commit()

    # Leaser should pick w2 because it has 25h remaining > 20h remaining
    leased = pool.lease_worker_account(accelerator="gpu", min_spacing_seconds=0.0)
    assert leased.username == "w2"


def test_quota_exhausted_raises(tmp_path: Path):
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    pool.register_account(username="user_anchor", api_key="ka", is_anchor=True)
    w1 = pool.register_account(username="w1", api_key="k1", is_anchor=False)

    # Exhaust w1
    with sqlite3.connect(pool.db_path) as conn:
        conn.execute("UPDATE kaggle_accounts SET gpu_hours_used = 30.0 WHERE username = 'w1'")
        conn.execute("UPDATE kaggle_accounts SET gpu_hours_used = 30.0 WHERE username = 'user_anchor'")
        conn.commit()

    with pytest.raises(QuotaExhaustedError, match="All accounts in the pool have depleted"):
        pool.lease_worker_account(accelerator="gpu", min_spacing_seconds=0.0)


@patch("subprocess.run")
def test_push_private_kernel_isolated_env(mock_run, tmp_path: Path):
    mock_run.return_value = subprocess.CompletedProcess(
        args=["kaggle", "kernels", "push"],
        returncode=0,
        stdout="Kernel pushed successfully",
        stderr="",
    )

    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    worker = pool.register_account(username="worker_alice", api_key="key_alice_secret", is_anchor=False)

    script_code = "print('Hello from private isolated kernel')\n"
    job = pool.push_private_kernel(
        account=worker,
        script_code=script_code,
        dataset_sources=["titanic"],
        accelerator="gpu",
        add_jitter=False,
    )

    assert job.status == "queued"
    assert job.is_private is True
    assert job.account_username == "worker_alice"
    assert job.kernel_slug.startswith("opt-job-")

    # Verify staging metadata is strictly private
    stage_dir = Path(job.local_staging_dir)
    assert stage_dir.exists()
    meta = json.loads((stage_dir / "kernel-metadata.json").read_text())
    assert meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true"
    assert meta["id"] == f"worker_alice/{job.kernel_slug}"

    # Verify isolated environment passed to subprocess
    call_kwargs = mock_run.call_args.kwargs
    env = call_kwargs["env"]
    assert env["KAGGLE_USERNAME"] == "worker_alice"
    assert env["KAGGLE_KEY"] == "key_alice_secret"


def test_kaggle_pool_with_operational_budgets_lifecycle(monkeypatch, tmp_path: Path):
    from mlrescurix_worker.scientific.operational_budgets import OperationalBudgets

    real_run = subprocess.run

    def selective_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if cmd and len(cmd) > 0 and cmd[0] == "kaggle":
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="Kernel pushed successfully",
                stderr="",
            )
        return real_run(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", selective_run)

    authority_bin = Path("target/debug/mlrescurix-harness").resolve()
    if not authority_bin.exists():
        pytest.skip("mlrescurix-harness binary not built")

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "budget": {"tokens": 10_000_000, "storage": 100_000_000},
        "capacity": {"threads": 4},
        "max_tasks": 10,
    }))
    harness_state_dir = tmp_path / "rust_harness_state"
    subprocess.run([
        str(authority_bin), "init",
        "--state", str(harness_state_dir),
        "--policy", str(policy_path),
    ], check=True)
    gateway_db = harness_state_dir / "state.sqlite"

    budgets = OperationalBudgets(
        db_path=gateway_db,
        project_id="kaggle_test_proj",
        authority_binary=authority_bin,
        kaggle_hours_limit=20.0,
    )

    pool = KaggleAccountPool(
        storage_dir=tmp_path / "pool",
        operational_budgets=budgets,
    )
    worker = pool.register_account(username="worker_bob", api_key="bob_secret", is_anchor=False)

    # 1. Push kernel successfully -> admitted and reserved
    job = pool.push_private_kernel(
        account=worker,
        script_code="print('running job 1')",
        job_id="job-1",
        add_jitter=False,
    )
    assert job.status == "queued"

    with sqlite3.connect(gateway_db) as conn:
        row = conn.execute(
            "SELECT status, requested_hours, actual_hours FROM operational_kaggle_jobs WHERE job_id = 'job-1'"
        ).fetchone()
        assert row is not None
        assert row[0] == "reserved"
        assert row[1] == 12.0

    # 2. Complete job -> settled as success
    pool.record_job_completion("job-1", actual_hours=1.5)
    with sqlite3.connect(gateway_db) as conn:
        row = conn.execute(
            "SELECT status, actual_hours FROM operational_kaggle_jobs WHERE job_id = 'job-1'"
        ).fetchone()
        assert row[0] == "success"
        assert row[1] == 1.5

    # Cancellation revokes the local lease but cannot stop the remote kernel.
    pool.push_private_kernel(
        account=worker,
        script_code="print('running job 2')",
        job_id="job-2",
        add_jitter=False,
    )
    pool.cancel_kernel("job-2")
    with sqlite3.connect(gateway_db) as conn:
        row = conn.execute(
            "SELECT status FROM operational_kaggle_jobs WHERE job_id = 'job-2'"
        ).fetchone()
        assert row[0] == "outcome_unknown"

    # The uncertain 12h reservation remains, so another 12h job is denied.
    with pytest.raises(QuotaExhaustedError, match="denied by operational resource authority"):
        pool.push_private_kernel(
            account=worker,
            script_code="print('running job 3')",
            job_id="job-3",
            add_jitter=False,
        )
