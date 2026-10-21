"""Regression checks for the second S1 review; no credentials or paid calls."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from mlrescurix_worker.scientific.eda_agent import run_exploratory_python
from mlrescurix_worker.scientific.fe_ablation import _resolve_and_apply_features
from mlrescurix_worker.scientific.hypothesis_executor import HypothesisActionDispatcher
from mlrescurix_worker.scientific.workflow import WorkflowError

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("campaign_runner_followup", ROOT / "scripts/run_robustness_campaign.py")
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)


@pytest.mark.parametrize("operation", ["eda", "features"])
def test_untrusted_snippet_never_starts_host_process_or_writes_file(tmp_path, operation):
    sentinel = tmp_path / "candidate_was_executed"
    code = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')"
    frame = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with patch("subprocess.Popen", side_effect=AssertionError("Host process must not start")):
        if operation == "eda":
            receipt = run_exploratory_python(code, frame)
            assert "blocked" in receipt and len(receipt.encode()) <= 2048
        else:
            with pytest.raises(WorkflowError, match="governed sandbox unavailable"):
                _resolve_and_apply_features(code, frame, frame, "y", [])
    assert not sentinel.exists()


def test_corrupt_action_ledger_fails_closed(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text('{"actions": [')
    with pytest.raises(WorkflowError, match="ledger operation failed"):
        HypothesisActionDispatcher(ledger)
    assert ledger.read_text() == '{"actions": ['


def test_ledger_write_failure_is_not_silent(tmp_path):
    parent_file = tmp_path / "file"
    parent_file.write_text("occupied")
    dispatcher = HypothesisActionDispatcher(parent_file / "ledger.json")
    with pytest.raises(WorkflowError, match="ledger operation failed"):
        dispatcher._save_ledger()


def test_corrupt_run_summary_cannot_reset_spending(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module, "EVIDENCE_BASE", tmp_path)
    directory = tmp_path / "campaign" / "run"
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text('{"budget_spent_usd":')
    with pytest.raises(ValueError, match="Cannot safely resume"):
        runner_module.CampaignRunner("campaign", "run")


@pytest.mark.parametrize("value", ["..", "../escape", "/absolute"])
def test_run_paths_cannot_escape_campaign(tmp_path, monkeypatch, value):
    monkeypatch.setattr(runner_module, "EVIDENCE_BASE", tmp_path)
    with pytest.raises(ValueError, match="Invalid"):
        runner_module.CampaignRunner(value, "run")


def test_campaign_policy_pins_requested_endpoint_and_budget():
    policy = json.loads((ROOT / "docs/v2/BENCHMARK_SUITE.json").read_text())
    assert policy["campaign_budget"]["provider_usd_max"] == "0.30"
    assert policy["provider"]["usd_per_main_live_run_max"] == "0.05"
    assert policy["provider"]["model_id"] == "deepseek/deepseek-v4.1-flash-1017"
    assert policy["provider"]["provider_routing"] == {
        "only": ["inference-net/fp8"], "order": ["inference-net/fp8"], "allow_fallbacks": False,
    }


def test_preflight_does_not_certify_presence_and_preserves_spend(tmp_path, monkeypatch):
    import types
    monkeypatch.setattr(runner_module, "EVIDENCE_BASE", tmp_path)
    monkeypatch.setattr(runner_module, "get_git_provenance", lambda: {"commit": "test"})
    monkeypatch.setattr(runner_module.platform, "platform", lambda: "test-platform")
    monkeypatch.setitem(__import__("sys").modules, "kaggle", types.SimpleNamespace())
    monkeypatch.setattr(runner_module.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="linux 2", stderr=""))
    runner = runner_module.CampaignRunner("campaign", "run")
    runner.provider_usage_path.write_text(json.dumps({
        "total_billed_cost_usd": 0.1, "reserved_cost_usd": 0.03,
        "total_requests": 3, "remaining_budget_usd": 4.87,
    }))
    gates = runner.run_preflight()
    for gate in ("G2_RUST_HARNESS_PROVENANCE", "G3_STORAGE_SCHEMA_V11", "G4_CONTAINER_CGROUP_V2", "G5_KAGGLE_OFFICIAL_DATA", "G6_PROVIDER_GATEWAY"):
        assert gates[gate]["outcome"] != "passed"
    usage = json.loads(runner.provider_usage_path.read_text())
    assert usage["total_billed_cost_usd"] == 0.1
    assert usage["reserved_cost_usd"] == 0.03
    assert usage["total_requests"] == 3
    assert usage["remaining_budget_usd"] == 0.17


def test_campaign_budget_authority_exhaustion_before_http(tmp_path):
    policy = {
        "campaign_budget": {"provider_usd_max": "0.30", "contingency_usd": "0.05"},
        "provider": {"usd_per_main_live_run_max": "0.05"},
    }
    authority = runner_module.CampaignBudgetAuthority(tmp_path / "campaign", policy)

    # 1. Successful reservations within limits
    res1 = authority.reserve("run_01", "req_01", 0.03)
    assert float(res1) == 0.03

    # 2. Per-run cap violation (0.03 + 0.03 > 0.05 per run limit)
    with pytest.raises(runner_module.BudgetExhaustionError, match="Per-run spending ceiling"):
        authority.reserve("run_01", "req_02", 0.03)

    # 3. Settle req_01 with actual cost
    authority.settle("run_01", "req_01", 0.02)
    status = authority.get_status()
    assert status["total_billed_usd"] == "0.02"
    assert status["total_reserved_usd"] == "0.00"

    # 4. Fill campaign budget across multiple runs up to 0.30
    authority.reserve("run_02", "req_03", 0.05)
    authority.settle("run_02", "req_03", 0.05)
    authority.reserve("run_03", "req_04", 0.05)
    authority.settle("run_03", "req_04", 0.05)
    authority.reserve("run_04", "req_05", 0.05)
    authority.settle("run_04", "req_05", 0.05)
    authority.reserve("run_05", "req_06", 0.05)
    authority.settle("run_05", "req_06", 0.05)
    # Total billed so far: 0.02 + 0.05*4 = 0.22. Remaining = 0.08

    # 5. Mark uncertain retains spend against campaign ceiling
    authority.reserve("run_06", "req_07", 0.05)
    authority.mark_uncertain("run_06", "req_07", 0.05)
    status = authority.get_status()
    assert status["total_uncertain_usd"] == "0.05"
    assert status["remaining_campaign_usd"] == "0.03"

    # 6. Campaign ceiling exhaustion stops before request (0.04 > 0.03 remaining)
    with pytest.raises(runner_module.BudgetExhaustionError, match="Campaign spending ceiling"):
        authority.reserve("run_07", "req_08", 0.04)

