import hashlib
import json
import subprocess
from unittest.mock import patch

import pytest

from mlrescurix_worker.custom.admission import OOMRecoveryPlan, OOMRecoveryPolicy
from mlrescurix_worker.custom.delivery import BundleVerificationError, CustomBundlePackager
from mlrescurix_worker.custom.kaggle_backend import KaggleExecutionBackend
from mlrescurix_worker.custom.tuning import BudgetedOptunaTuner, TrialState
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


def test_tuner_settlement_failure_stops_optimization_and_keeps_trial_running():
    class Authority:
        def __init__(self):
            self.settlements = 0

        def get_status(self):
            return {"remaining_campaign_usd": "1.00"}

        def reserve(self, *args):
            pass

        def settle(self, *args, **kwargs):
            self.settlements += 1
            raise RuntimeError("ledger unavailable")

    authority = Authority()
    tuner = BudgetedOptunaTuner("study", {}, campaign_authority=authority)
    with pytest.raises(RuntimeError, match="ledger unavailable"):
        tuner.optimize(lambda trial: 0.5, n_trials=1)
    assert authority.settlements == 1
    assert tuner.trials[0].state == TrialState.RUNNING
    assert tuner.total_cost_usd == 0
    assert tuner.ask() is None


def test_tuner_rejects_duplicate_terminal_report():
    tuner = BudgetedOptunaTuner("study", {})
    trial = tuner.ask()
    tuner.tell(trial, TrialState.COMPLETE, final_value=1.0, cost_usd=0.1)
    with pytest.raises(ValueError, match="running trial"):
        tuner.tell(trial, TrialState.COMPLETE, final_value=1.0, cost_usd=0.1)
    assert tuner.total_cost_usd == 0.1


def test_tuner_does_not_overwrite_existing_trial_from_another_controller(tmp_path):
    import sqlite3

    db = tmp_path / "trials.sqlite"
    first = BudgetedOptunaTuner("study", {}, storage_path=db)
    second = BudgetedOptunaTuner("study", {}, storage_path=db)
    first_trial = first.ask()
    with pytest.raises(sqlite3.IntegrityError):
        second.ask()
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*), MIN(start_time) FROM tuning_trials").fetchone() == (1, first_trial.start_time)


def test_worker_rejects_missing_target(tmp_path):
    from mlrescurix_worker.custom.worker import load_dataset

    data = tmp_path / "data.csv"
    data.write_text("feature,target\n1,0\n2,1\n")
    with pytest.raises(ValueError, match="Target column not found"):
        load_dataset(data, "wrong_target")
