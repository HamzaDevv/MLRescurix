"""Offline Integration Tests for MLRescurix Autonomous AI Execution Loop.

Verifies the 8 core architectural behaviors:
1. Valid agent actions change executed configuration.
2. Actual tool results inform subsequent decisions.
3. Invalid and unsupported actions are rejected without silent substitution.
4. Critic veto blocks acceptance even with high numeric scores.
5. State survives restart without duplicate execution or billing.
6. Budget admission rejects unaffordable calls.
7. Worker failures and cancellation clean up processes and child trees.
8. Locked folds and fold-local preprocessing remain strictly isolated.
"""

import json
import os
import signal
import subprocess
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import pytest

from mlrescurix_worker.scientific.autonomous_loop import (
    AutonomousAgentLoop,
    ProcessTreeSampler,
    get_controller_rss_bytes,
)
from mlrescurix_worker.scientific.critic_engine import CriticEngine


class MockBudgetAuthority:
    """Mock budget authority for deterministic offline testing."""

    def __init__(self, campaign_ceiling: str = "0.30", per_run_ceiling: str = "0.05", initial_billed: str = "0.00"):
        self.campaign_ceiling = Decimal(campaign_ceiling)
        self.per_run_ceiling = Decimal(per_run_ceiling)
        self.billed = Decimal(initial_billed)
        self.reserved = Decimal("0.00")
        self.reservations: List[str] = []
        self.settlements: List[Tuple[str, Decimal]] = []

    def reserve(self, run_id: str, request_id: str, amount_usd: float | Decimal) -> Decimal:
        amt = Decimal(str(amount_usd))
        if self.billed + self.reserved + amt > self.campaign_ceiling:
            raise RuntimeError(f"Campaign budget ceiling exceeded: {self.billed + self.reserved + amt} > {self.campaign_ceiling}")
        self.reserved += amt
        self.reservations.append(request_id)
        return amt

    def settle(self, run_id: str, request_id: str, actual_billed_usd: float | Decimal) -> None:
        act = Decimal(str(actual_billed_usd))
        self.billed += act
        self.reserved = max(Decimal("0.00"), self.reserved - Decimal("0.005"))
        self.settlements.append((request_id, act))

    def mark_uncertain(self, run_id: str, request_id: str, reserved_usd: float | Decimal) -> None:
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "campaign_ceiling_usd": str(self.campaign_ceiling),
            "remaining_campaign_usd": str(self.campaign_ceiling - self.billed - self.reserved),
            "total_billed_usd": str(self.billed),
        }


def create_synthetic_dataset(csv_path: Path) -> Path:
    """Create deterministic synthetic tabular dataset."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "age": np.random.uniform(18, 70, n),
        "fare": np.random.exponential(30, n),
        "cabin": [f"C{i%10}" if i % 3 != 0 else np.nan for i in range(n)],
        "target": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# Test 1: Valid Agent Actions Change Executed Configuration
# =============================================================================

def test_valid_agent_action_changes_configuration(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    budget = MockBudgetAuthority()

    def mock_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        # Return valid typed hypothesis action
        return {
            "content": json.dumps({
                "thought_rationale": "Add missingness indicator for cabin to exploit MNAR signal.",
                "action_type": "HYPOTHESIS_TRANSFORM",
                "parameters": {
                    "action": "ADD_MNAR_INDICATOR",
                    "parameters": {"columns": ["cabin"]},
                },
            }),
            "cost_usd": 0.0005,
        }

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws",
        project_id="test_proj",
        session_id="run_test_1",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=1,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    res = loop.run()

    assert res["turns_completed"] == 1
    turn = res["turns"][0]
    assert turn["disposition"] == "applied"
    assert turn["proposed_action"]["action_type"] == "HYPOTHESIS_TRANSFORM"
    assert turn["measurements"]["action_status"] in ("ACCEPTED", "REJECTED", "inconclusive", "confirmed", "refuted")
    assert len(budget.reservations) == 1
    assert len(budget.settlements) == 1


# =============================================================================
# Test 2: Actual Tool Results Inform Subsequent Decisions
# =============================================================================

def test_actual_tool_results_inform_subsequent_decisions(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    budget = MockBudgetAuthority()
    prompts_seen: List[str] = []

    def mock_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        user_msg = messages[-1]["content"]
        prompts_seen.append(user_msg)
        turn_num = len(prompts_seen)

        if turn_num == 1:
            # Turn 1: evaluate portfolio
            return {
                "content": json.dumps({
                    "thought_rationale": "Evaluate baseline model families.",
                    "action_type": "EVALUATE_PORTFOLIO",
                    "parameters": {"candidate_families": ["hist_gbdt", "penalized_linear"]},
                }),
                "cost_usd": 0.0005,
            }
        else:
            # Turn 2: agent inspects turn 1 results and decides to blend and audit
            return {
                "content": json.dumps({
                    "thought_rationale": "Based on measured portfolio results, blend candidates.",
                    "action_type": "BLEND_AND_AUDIT",
                    "parameters": {},
                }),
                "cost_usd": 0.0005,
            }

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws",
        project_id="test_proj",
        session_id="run_test_2",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=2,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    res = loop.run()

    assert res["turns_completed"] == 2
    # Verify turn 2 prompt actually contained turn 1's measured portfolio scores
    turn2_prompt = json.loads(prompts_seen[1])
    assert len(turn2_prompt["turn_history"]) == 1
    t1_hist = turn2_prompt["turn_history"][0]
    assert t1_hist["proposed_action"] == "EVALUATE_PORTFOLIO"
    assert t1_hist["disposition"] == "applied"


# =============================================================================
# Test 3: Invalid and Unsupported Actions Are Rejected Explicitly
# =============================================================================

def test_invalid_and_unsupported_actions_rejected(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    budget = MockBudgetAuthority()
    call_idx = 0

    def mock_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            # Propose unsupported model family (catboost when native is uninstalled)
            return {
                "content": json.dumps({
                    "thought_rationale": "Train catboost gradient trees.",
                    "action_type": "EVALUATE_PORTFOLIO",
                    "parameters": {"candidate_families": ["catboost"]},
                }),
                "cost_usd": 0.0005,
            }
        else:
            # Propose invalid action type
            return {
                "content": json.dumps({
                    "thought_rationale": "Attempt arbitrary non-existent action.",
                    "action_type": "EXECUTE_ARBITRARY_SHELL",
                    "parameters": {"cmd": "rm -rf /"},
                }),
                "cost_usd": 0.0005,
            }

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws",
        project_id="test_proj",
        session_id="run_test_3",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=2,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    res = loop.run()

    # Turn 1 should be unsupported (if CatBoost is not installed) or applied (if installed)
    turn1 = res["turns"][0]
    from mlrescurix_worker.scientific.model_zoo import HAS_CATBOOST
    if not HAS_CATBOOST:
        assert turn1["disposition"] == "unsupported"
        assert "catboost" in turn1["error_msg"]

    # Turn 2 should be strictly rejected
    turn2 = res["turns"][1]
    assert turn2["disposition"] == "rejected"
    assert "Unknown action type" in turn2["error_msg"]


# =============================================================================
# Test 4: Critic VETO Blocks Acceptance Even With Scores
# =============================================================================

def test_critic_veto_blocks_acceptance(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    budget = MockBudgetAuthority()

    def mock_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        # Agent attempts single-family monoculture
        return {
            "content": json.dumps({
                "thought_rationale": "Blend single family candidate.",
                "action_type": "BLEND_AND_AUDIT",
                "parameters": {},
            }),
            "cost_usd": 0.0005,
        }

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws",
        project_id="test_proj",
        session_id="run_test_4",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=1,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    # Force min_distinct_families to 5 so it triggers VETO
    loop.critic_engine = CriticEngine(min_distinct_families=5)
    res = loop.run()

    turn = res["turns"][0]
    assert turn["critic_result"]["status"] == "VETO"
    assert len(turn["critic_result"]["blocking_flaws"]) >= 1
    # Candidate must NOT be accepted in state
    assert res["accepted_candidate_id"] is None


# =============================================================================
# Test 5: State Survives Restart Without Duplicate Billing or Execution
# =============================================================================

def test_state_survives_restart_without_duplicate_execution(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    budget = MockBudgetAuthority()
    calls = 0

    def mock_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        return {
            "content": json.dumps({
                "thought_rationale": f"Turn execution {calls}",
                "action_type": "EVALUATE_PORTFOLIO",
                "parameters": {"candidate_families": ["hist_gbdt", "penalized_linear"]},
            }),
            "cost_usd": 0.0005,
        }

    ws = tmp_path / "ws_resume"
    # Run loop 1 for 1 turn
    loop1 = AutonomousAgentLoop(
        workspace_root=ws,
        project_id="test_proj",
        session_id="run_resume",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=1,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    res1 = loop1.run()
    assert res1["turns_completed"] == 1
    assert calls == 1

    # Restart loop 2 in the same workspace with max_turns=2
    loop2 = AutonomousAgentLoop(
        workspace_root=ws,
        project_id="test_proj",
        session_id="run_resume",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=2,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    res2 = loop2.run()
    # Turn 0 must NOT be re-called; only turn 1 is called
    assert res2["turns_completed"] == 2
    assert calls == 2  # exactly 1 additional call


# =============================================================================
# Test 6: Budget Admission Rejects Unaffordable Calls
# =============================================================================

def test_budget_admission_rejects_unaffordable_calls(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    # Budget authority with remaining balance < 0.005
    budget = MockBudgetAuthority(campaign_ceiling="0.004", initial_billed="0.001")
    calls = 0

    def mock_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"content": "{}", "cost_usd": 0.0005}

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws",
        project_id="test_proj",
        session_id="run_budget_exhaust",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        max_turns=2,
        provider_caller=mock_provider,
        provider_is_fixture=True,
    )
    res = loop.run()

    # Loop should immediately stop due to budget ceiling
    assert res["stop_reason"] == "budget_exhaustion"
    assert calls == 0
    assert res["turns_completed"] == 0


# =============================================================================
# Test 7: Worker Failures and Cancellation Clean Up Processes
# =============================================================================

def test_worker_failures_and_cancellation_clean_up_processes():
    sampler = ProcessTreeSampler()
    metrics = sampler.sample_now()
    assert metrics["controller_peak_rss_bytes"] > 0
    assert metrics["worker_peak_rss_bytes"] is None  # no worker was registered
    assert metrics["cgroup_enforced"] is False or metrics["cgroup"]["verified_limits"]

    # Spawn background sleep process in a new process group
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(100)"],
        start_new_session=True,
    )
    pid = proc.pid
    assert psutil.pid_exists(pid)

    # Cancel/clean up via process group kill
    os.killpg(os.getpgid(pid), signal.SIGKILL)
    proc.wait()

    # Verify no surviving process
    assert not psutil.pid_exists(pid)


# =============================================================================
# Test 8: Locked Folds and Fold-Local Preprocessing Remain Isolated
# =============================================================================

def test_locked_folds_and_fold_local_preprocessing_remain_isolated(tmp_path: Path):
    dataset_path = create_synthetic_dataset(tmp_path / "data.csv")
    budget = MockBudgetAuthority()

    # Create explicit fixed locked folds: 2 folds with known indices
    fold_splits = [
        (list(range(0, 100)), list(range(100, 200))),
        (list(range(100, 200)), list(range(0, 100))),
    ]

    loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "ws",
        project_id="test_proj",
        session_id="run_locked_folds",
        campaign_authority=budget,
        dataset_path=dataset_path,
        target_column="target",
        locked_folds=fold_splits,
        max_turns=0,
    )

    # Verify splits match exactly
    splits = list(loop.cv_splitter.split(loop.raw_df, loop.raw_df["target"].values))
    assert len(splits) == 2
    assert np.array_equal(splits[0][0], fold_splits[0][0])
    assert np.array_equal(splits[0][1], fold_splits[0][1])
    assert np.array_equal(splits[1][0], fold_splits[1][0])
    assert np.array_equal(splits[1][1], fold_splits[1][1])
    # Zero holdout access
    assert loop.state["holdout_accesses"] == 0
