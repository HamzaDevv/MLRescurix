import hashlib
import json
import subprocess
from unittest.mock import patch

import pytest

from mlrescurix_worker.custom.admission import OOMRecoveryPlan, OOMRecoveryPolicy
from mlrescurix_worker.custom.delivery import BundleVerificationError, CustomBundlePackager
from mlrescurix_worker.custom.kaggle_backend import KaggleExecutionBackend
from mlrescurix_worker.custom.tuning import BudgetedOptunaTuner
from mlrescurix_worker.scientific.kaggle_pool import ChecksumMismatchError, KaggleAccountPool


def test_odd_microbatch_recovery_preserves_effective_batch():
    plan = OOMRecoveryPlan(microbatch_size=5, gradient_accumulation_steps=1, effective_batch_size=5)
    result = OOMRecoveryPolicy.handle_oom(plan)
    assert result.microbatch_size * result.gradient_accumulation_steps == 5


@patch("subprocess.run")
def test_remote_manifest_rejects_empty_and_escape(mock_run, tmp_path):
    mock_run.return_value = subprocess.CompletedProcess([], 0, "", "")
    pool = KaggleAccountPool(storage_dir=tmp_path / "pool")
    account = pool.register_account(username="anchor", api_key="key", is_anchor=True)
    destination = tmp_path / "output"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("secret")
    for files in ({}, {"../outside": hashlib.sha256(b"secret").hexdigest()}):
        (destination / "manifest.json").write_text(json.dumps({"files": files}))
        with pytest.raises(ChecksumMismatchError):
            pool.download_kernel_output(account, "slug", destination)


def test_remote_submission_requires_real_code(tmp_path):
    backend = KaggleExecutionBackend(storage_dir=tmp_path / "pool")
    backend.pool.register_account(username="anchor", api_key="key", is_anchor=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(ValueError, match="real kernel"):
        backend.submit(staged, {})


def test_bundle_does_not_delete_existing_destination(tmp_path):
    existing = tmp_path / "bundle"
    existing.mkdir()
    (existing / "important").write_text("preserve")
    with pytest.raises(BundleVerificationError, match="already exists"):
        CustomBundlePackager().package(None, existing, "y", "regression", "mae", 1.0, "minimize", [], [])
    assert (existing / "important").read_text() == "preserve"


def test_campaign_remaining_uses_real_ledger_field():
    class Authority:
        def get_status(self):
            return {"remaining_campaign_usd": "0.00"}

    tuner = BudgetedOptunaTuner("study", {}, campaign_authority=Authority())
    assert tuner.ask() is None


def test_worker_rejects_missing_target(tmp_path):
    from mlrescurix_worker.custom.worker import load_dataset

    data = tmp_path / "data.csv"
    data.write_text("feature,target\n1,0\n2,1\n")
    with pytest.raises(ValueError, match="Target column not found"):
        load_dataset(data, "wrong_target")
