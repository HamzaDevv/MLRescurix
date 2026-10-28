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
