"""Tests for Rust Provider Gateway Adapter (Phase M1/M2 Production Governance)."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter, find_harness_binary

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("campaign_runner_mod", ROOT / "scripts/run_robustness_campaign.py")
runner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner_module)
CampaignBudgetAuthority = runner_module.CampaignBudgetAuthority


def test_find_harness_binary():
    bin_path = find_harness_binary()
    assert bin_path.is_file()
    assert os.access(bin_path, os.X_OK)


def test_rust_gateway_adapter_mock_dispatch_and_accounting(tmp_path: Path):
    state_dir = tmp_path / "harness_state"
    mock_file = tmp_path / "mock_responses.json"
    mock_responses = [
        {
            "id": "resp-mock-1",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({"action_type": "ANALYZE", "parameters": {}}),
            "tool_calls": [],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 25,
                "total_tokens": 145,
            },
            "latency_ms": 15,
        }
    ]
    mock_file.write_text(json.dumps(mock_responses), encoding="utf-8")

    pricing = {
        "verified": True,
        "input_per_million": "0.08",
        "output_per_million": "0.50",
    }

    adapter = RustGatewayAdapter(
        state_dir=state_dir,
        project_id="test_proj_gateway",
        task_id="task_gw_1",
        role_id="scientific-agent",
        mock_path=mock_file,
        pricing=pricing,
        require_policy_approval=False,
    )

    messages = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "Run analysis."},
    ]

    receipt = adapter.call_with_request_id(messages, 1000, "session-1:turn:1")
    assert receipt["gateway_route"] == "rust_mlrescurix_harness"
    assert receipt["provider_response_id"] == "resp-mock-1"
    assert receipt["usage"]["prompt_tokens"] == 120
    assert receipt["usage"]["completion_tokens"] == 25
    assert receipt["cost_is_estimate"]

    # The mock fixture is now exhausted. A retry of this stable ID must return
    # the persisted response without attempting another provider call.
    replay = adapter.call_with_request_id(messages, 1000, "session-1:turn:1")
    assert replay["provider_response_id"] == "resp-mock-1"
    assert replay["usage"] == receipt["usage"]


def test_rust_gateway_adapter_deepinfra_endpoint_mock_dispatch(tmp_path: Path):
    state_dir = tmp_path / "harness_state"
    mock_file = tmp_path / "mock_deepinfra.json"
    mock_responses = [
        {
            "id": "resp-deepinfra-1",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({"action_type": "ANALYZE", "parameters": {}}),
            "tool_calls": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "latency_ms": 25,
        }
    ]
    mock_file.write_text(json.dumps(mock_responses), encoding="utf-8")

    pricing = {
        "verified": True,
        "input_per_million": "0.16",
        "output_per_million": "0.48",
    }

    adapter = RustGatewayAdapter(
        state_dir=state_dir,
        project_id="test_proj_deepinfra",
        task_id="task_gw_di",
        role_id="scientific-agent",
        endpoint="deepinfra/fp8/flex",
        mock_path=mock_file,
        pricing=pricing,
        require_policy_approval=False,
    )

    messages = [{"role": "user", "content": "Analyze"}]
    receipt = adapter.call_with_request_id(messages, 500, "session-di:turn:1")
    assert receipt["gateway_route"] == "rust_mlrescurix_harness"
    assert receipt["provider_response_id"] == "resp-deepinfra-1"
    assert receipt["usage"]["prompt_tokens"] == 100
    assert receipt["usage"]["completion_tokens"] == 20


    # Verify SQLite provider_accounting in state_dir
    db_path = state_dir / "state.sqlite"
    assert db_path.is_file()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT prompt_tokens, completion_tokens, status FROM provider_accounting WHERE project='test_proj_deepinfra'"
        ).fetchone()
        assert row is not None
        assert row[0] == 100
        assert row[1] == 20
        assert row[2] == "ok"
        count = conn.execute(
            "SELECT count(*) FROM provider_accounting WHERE project='test_proj_deepinfra'"
        ).fetchone()[0]
        assert count == 1


def test_rust_gateway_adapter_outbound_view_staging_and_approval(tmp_path: Path):
    state_dir = tmp_path / "harness_state"
    mock_file = tmp_path / "mock_responses.json"
    mock_responses = [
        {
            "id": "resp-mock-appr",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({"action_type": "ANALYZE", "parameters": {}}),
            "tool_calls": [],
            "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
            "latency_ms": 8,
        }
    ]
    mock_file.write_text(json.dumps(mock_responses), encoding="utf-8")

    adapter = RustGatewayAdapter(
        state_dir=state_dir,
        project_id="test_proj_policy",
        mock_path=mock_file,
        pricing={"verified": True, "input_per_million": "0.08", "output_per_million": "0.50"},
        require_policy_approval=True,
    )

    messages = [{"role": "user", "content": "Check dataset metrics."}]
    request_id = "policy-session:turn:1"
    with pytest.raises(RuntimeError, match="requires a content-bound policy-approved view") as refused:
        adapter.call_with_request_id(messages, 500, request_id)
    import re
    digest = re.search(r"\(([0-9a-f]{64})\)", str(refused.value)).group(1)
    approved = subprocess.run([
        str(adapter.binary), "approve-outbound-view", "--state", str(state_dir),
        "--project", adapter.project_id, "--request-sha256", digest,
        "--actor", "test-human-reviewer", "--approved", "true", "--ttl-ms", "600000",
    ], capture_output=True, text=True)
    assert approved.returncode == 0, approved.stderr
    receipt = adapter.call_with_request_id(messages, 500, request_id)
    assert receipt["gateway_route"] == "rust_mlrescurix_harness"


def test_approval_wait_times_out_without_dispatch(tmp_path: Path):
    mock = tmp_path / "mock.json"
    mock.write_text("[]")
    adapter = RustGatewayAdapter(tmp_path / "state", "approval-wait", mock_path=mock,
        pricing={"verified": True, "input_per_million": "0.038", "output_per_million": "0.50"},
        approval_wait_seconds=1)
    with pytest.raises(TimeoutError, match="Outbound view approval timed out"):
        adapter.call_with_request_id([{"role": "user", "content": "Aggregate diagnostics only."}], 100, "wait-1")
    with sqlite3.connect(adapter.state_dir / "state.sqlite") as db:
        assert db.execute("select count(*) from gateway_dispatch_receipts").fetchone()[0] == 0


def test_autonomous_loop_with_rust_gateway_adapter(tmp_path: Path):
    # Create simple dataset
    data_path = tmp_path / "dev.csv"
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({
        "num1": rng.randn(n),
        "num2": rng.randn(n),
        "cat1": rng.choice(["A", "B"], size=n),
        "target": rng.choice([0, 1], size=n),
    })
    df.to_csv(data_path, index=False)

    folds = [
        (list(range(0, 40)), list(range(40, 60))),
        (list(range(20, 60)), list(range(0, 20))),
    ]

    campaign_dir = tmp_path / "campaign"
    policy = {
        "campaign_budget": {"provider_usd_max": "0.30", "contingency_usd": "0.05"},
        "provider": {"usd_per_main_live_run_max": "0.05"},
    }
    authority = CampaignBudgetAuthority(campaign_dir, policy)

    mock_responses = [
        {
            "id": "turn-1",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({"thought_rationale": "Analyze baseline", "action_type": "ANALYZE", "parameters": {}}),
            "tool_calls": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "latency_ms": 10,
        },
        {
            "id": "turn-2",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({"thought_rationale": "Stopping test", "action_type": "STOP_EARLY", "parameters": {"reason": "Test complete"}}),
            "tool_calls": [],
            "usage": {"prompt_tokens": 150, "completion_tokens": 20, "total_tokens": 170},
            "latency_ms": 10,
        },
    ]
    mock_file = tmp_path / "mock.json"
    mock_file.write_text(json.dumps(mock_responses), encoding="utf-8")

    state_dir = tmp_path / "harness_state"
    adapter = RustGatewayAdapter(
        state_dir=state_dir,
        project_id="test_autonomous_rust_gw",
        mock_path=mock_file,
        pricing={"verified": True, "input_per_million": "0.08", "output_per_million": "0.50"},
        require_policy_approval=False,
    )

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "workspace",
        project_id="test_autonomous_rust_gw",
        session_id="sess_rust_gw_1",
        campaign_authority=authority,
        dataset_path=data_path,
        target_column="target",
        task_type="binary_classification",
        metric_name="accuracy",
        metric_direction="maximize",
        locked_folds=folds,
        model_name="deepseek/deepseek-v4.1-flash-1017",
        endpoint="inference-net/fp8",
        max_turns=2,
        per_run_budget_usd=0.05,
        pricing={"verified": True, "input_per_million": "0.08", "output_per_million": "0.50"},
        provider_caller=adapter,
        provider_is_fixture=True,
    )

    loop_state = loop.run()
    assert loop_state["turns_completed"] == 2
    assert loop_state["stop_reason"] == "STOP_EARLY"

    # Verify durable provider receipts recorded gateway route
    for t in loop_state["turns"]:
        assert t["provider_receipt"]["gateway_route"] == "rust_mlrescurix_harness"

    # Verify campaign budget authority tracked settlements
    status = authority.get_status()
    total_billed = Decimal(str(status["total_billed_usd"]))
    assert total_billed >= Decimal("0.00")


def test_gateway_timeout_terminates_transport_descendants(tmp_path, monkeypatch):
    import sys
    import time
    import psutil
    fake = tmp_path / 'fake-harness'
    pid_path = tmp_path / 'transport.pid'
    fake.write_text('#!' + sys.executable + '\nimport subprocess,sys,time\n'
                    'from pathlib import Path\n'
                    'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\n'
                    f'Path({str(pid_path)!r}).write_text(str(child.pid))\n'
                    'time.sleep(60)\n')
    fake.chmod(0o700)
    adapter = RustGatewayAdapter.__new__(RustGatewayAdapter)
    adapter.binary = fake
    adapter.state_dir = tmp_path
    adapter.project_id, adapter.task_id, adapter.role_id = 'p', 't', 'r'
    adapter.model_name, adapter.endpoint = 'model', 'endpoint'
    adapter.pricing = None
    adapter.require_policy_approval = False
    adapter.mock_path = tmp_path / 'fixture.json'
    adapter.routing_policy = None
    original = subprocess.Popen.communicate
    def short_timeout(proc, *args, **kwargs):
        # Synchronize on the dispatch boundary rather than timing interpreter startup.
        ready_deadline = time.monotonic() + 10
        while not pid_path.exists() and proc.poll() is None and time.monotonic() < ready_deadline:
            time.sleep(.01)
        assert pid_path.exists(), "fixture never reached transport dispatch"
        return original(proc, timeout=.1)
    monkeypatch.setattr(subprocess.Popen, 'communicate', short_timeout)
    with pytest.raises(RuntimeError, match='remote outcome may be unknown'):
        adapter.call_with_request_id([], 10, 'stable')
    pid = int(pid_path.read_text())
    deadline = time.monotonic() + 2
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE: break
        time.sleep(.02)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_live_binding_cannot_open_a_fresh_campaign_allowance(tmp_path):
    adapter = RustGatewayAdapter.__new__(RustGatewayAdapter)
    adapter.mock_path = None
    with pytest.raises(ValueError, match='preserved shared ledger'):
        adapter.bind_campaign(tmp_path / 'new-campaign/campaign_budget_ledger.json', 'run')
    adapter.mock_path = tmp_path / 'offline-fixture.json'
    adapter.bind_campaign(tmp_path / 'fixture-budget.json', 'fixture-run')
    assert adapter.campaign_run == 'fixture-run'
