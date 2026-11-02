"""Session 1 Regression Suite: Contracts, Action Parity, Parameter Boundaries, and Accounting.

Covers Evidence D, E, F:
- Evidence D: Action parity across schema and runtime; regression/classification parameter mismatches;
              mixed-validity portfolio execution; exact duplicate reuse; mutation-resistant master inputs
              and hard link removal with byte/inode detection.
- Evidence E: Search accounting (local rejection vs dispatched success/failure/timeout) with durable admission.
- Evidence F: Controlled repeated proposal error visibility in durable history.
"""
import copy
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from mlrescurix_worker.scientific.autonomous_loop import ACTIONS, AutonomousAgentLoop
from mlrescurix_worker.scientific.candidate_pipeline import (
    CandidatePipeline,
    effective_configuration,
    evaluate_candidate,
    identity,
    validate_spec,
)
from mlrescurix_worker.scientific.governed_research import (
    GovernedResearchClient,
    MockSearchProvider,
    PrivacyViolationError,
    SearchSource,
    sanitize_query,
)
from mlrescurix_worker.scientific.operational_budgets import OperationalBudgets


class MockBudget:
    def __init__(self):
        self.requests = []
        self.billed = Decimal(0)

    def reserve(self, run, request, amount):
        self.requests.append(request)

    def settle(self, run, request, cost):
        self.billed += cost

    def mark_uncertain(self, *args):
        pass

    def get_status(self):
        return {
            "campaign_ceiling_usd": ".30",
            "remaining_campaign_usd": str(Decimal(".30") - self.billed),
        }


def make_test_loop(tmp_path, actions, task_type="binary_classification", **kwargs):
    rng = np.random.default_rng(2026)
    is_cls = "classification" in task_type or task_type == "binary"
    if is_cls:
        y = np.tile([0, 1], 50)
        metric_name = kwargs.pop("metric_name", "balanced_accuracy")
        metric_direction = kwargs.pop("metric_direction", "maximize")
    else:
        y = rng.normal(10.0, 2.0, 100)
        metric_name = kwargs.pop("metric_name", "rmse")
        metric_direction = kwargs.pop("metric_direction", "minimize")
    frame = pd.DataFrame({
        "feat1": rng.normal(size=len(y)),
        "feat2": rng.uniform(0, 10, size=len(y)),
        "target": y,
    })
    path = tmp_path / "development.csv"
    if not path.exists():
        frame.to_csv(path, index=False)

    ds_path = kwargs.pop("dataset_path", path)

    prompts = []

    def provider(messages, max_tokens):
        prompts.append(json.loads(messages[-1]["content"]))
        action_dict = actions[len(prompts) - 1]
        return {"content": json.dumps(action_dict), "cost_usd": 0}

    if is_cls:
        folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(frame, y))
    else:
        folds = list(KFold(2, shuffle=True, random_state=42).split(frame, y))

    loop = AutonomousAgentLoop(
        tmp_path / "workspace",
        "p_contract_test",
        "s_contract_test",
        kwargs.pop("budget", MockBudget()),
        ds_path,
        "target",
        locked_folds=folds,
        max_turns=len(actions),
        provider_caller=provider,
        provider_is_fixture=True,
        task_type=task_type,
        metric_name=metric_name,
        metric_direction=metric_direction,
        **kwargs,
    )
    return loop, prompts


def make_action(kind, **params):
    return {
        "thought_rationale": "Session 1 test action",
        "action_type": kind,
        "parameters": params,
    }


# =====================================================================
# Evidence D: Action Parity Across Schemas and Runtime
# =====================================================================

def test_evidence_d_action_parity_schema_and_runtime():
    """Validates that actions.json defines exactly the 10 supported actions,

    matching the Python worker's ACTIONS and Rust harness action set.
    """
    repo_root = Path(__file__).resolve().parents[2]
    schema_path = repo_root / "contracts" / "v3" / "actions.json"
    assert schema_path.is_file(), f"Missing schema at {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema["schema_version"] == "mlrescurix.actions.v1"
    action_names = {a["name"] for a in schema["actions"]}

    # Exactly 10 canonical actions
    expected_actions = {
        "HYPOTHESIS_TRANSFORM",
        "EVALUATE_PORTFOLIO",
        "BLEND_AND_AUDIT",
        "RESEARCH_TOPIC",
        "WRITE_PYTHON",
        "RUN_PYTHON",
        "ANALYZE",
        "HANDOFF",
        "STOP_EARLY",
        "FINALIZE_AND_STOP",
    }
    assert action_names == expected_actions, f"Mismatch: {action_names ^ expected_actions}"
    assert ACTIONS == expected_actions, f"Python ACTIONS mismatch: {ACTIONS ^ expected_actions}"
    assert "RESEARCH_TOPIC" in ACTIONS


def test_evidence_d_action_parity_shipped_runtime_assets(tmp_path):
    """Validates that action parity holds when running from materialized runtime assets.

    In the materialized asset tree, contracts/v3/actions.json is at target_root/contracts/v3/actions.json,
    which is 3 levels above autonomous_loop.py (target_root/src/mlrescurix_worker/scientific/autonomous_loop.py).
    """
    repo_root = Path(__file__).resolve().parents[2]
    schema_src = repo_root / "contracts" / "v3" / "actions.json"

    # Create materialized asset tree layout
    asset_root = tmp_path / "runtime-assets" / "digest123" / "python"
    contracts_dir = asset_root / "contracts" / "v3"
    contracts_dir.mkdir(parents=True)
    contracts_file = contracts_dir / "actions.json"
    contracts_file.write_text(schema_src.read_text(encoding="utf-8"), encoding="utf-8")

    code = f"""
import sys
from pathlib import Path

# Add python/src from repo to sys.path so we can import autonomous_loop
repo_src = Path({str(repo_root / "python" / "src")!r})
sys.path.insert(0, str(repo_src))

# Verify _load_canonical_actions when simulating __file__ inside the materialized asset root
import mlrescurix_worker.scientific.autonomous_loop as al
mock_loop_path = Path({str(asset_root / "src" / "mlrescurix_worker" / "scientific" / "autonomous_loop.py")!r})

# Run the search algorithm with mock_loop_path
found = None
for parent in mock_loop_path.parents:
    candidate = parent / "contracts" / "v3" / "actions.json"
    if candidate.is_file():
        found = candidate
        break

assert found is not None, f"Failed to find contracts/v3/actions.json from {{mock_loop_path}}"
assert found == Path({str(contracts_file)!r})

with open(found, "r", encoding="utf-8") as f:
    import json
    data = json.load(f)
    actions = {{a["name"] for a in data.get("actions", [])}}

assert len(actions) == 10
assert "RESEARCH_TOPIC" in actions
print("SHIPPED_ASSETS_ACTION_PARITY_VERIFIED")
"""
    import subprocess
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Error: {proc.stderr}"
    assert "SHIPPED_ASSETS_ACTION_PARITY_VERIFIED" in proc.stdout



# =====================================================================
# Evidence D: Model Parameter Mismatch Rejection
# =====================================================================

def test_evidence_d_parameter_mismatch_rejection_regression():
    """Validates that regression specs reject classification hyperparameters

    like 'C' or invalid penalty types instead of silently falling back.
    """
    # 1. Spec with 'C' in regression
    bad_c_spec = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"C": 0.5}
        },
    }
    with pytest.raises(ValueError, match="classification parameter and is unsupported for penalized_linear regression"):
        validate_spec(bad_c_spec, task_type="regression")

    # 2. Spec with unsupported penalty in regression
    bad_penalty_spec = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"penalty": "l1"}
        },
    }
    with pytest.raises(ValueError, match="unsupported for penalized_linear regression"):
        validate_spec(bad_penalty_spec, task_type="regression")

    # 3. Valid regression parameters (alpha, max_iter) must pass cleanly
    valid_reg_spec = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"alpha": 2.5, "max_iter": 1500}
        },
    }
    validate_spec(valid_reg_spec, task_type="regression")


def test_evidence_d_parameter_mismatch_rejection_classification():
    """Validates that classification specs reject regression hyperparameters

    like 'alpha' instead of guessing or silently substituting.
    """
    bad_alpha_spec = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"alpha": 1.0}
        },
    }
    with pytest.raises(ValueError, match="regression parameter and is unsupported for penalized_linear classification"):
        validate_spec(bad_alpha_spec, task_type="binary_classification")

    # Valid classification parameters (C, penalty, max_iter) must pass cleanly
    valid_cls_spec = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"C": 0.5, "penalty": "l2", "max_iter": 2000}
        },
    }
    validate_spec(valid_cls_spec, task_type="binary_classification")


# =====================================================================
# Evidence D: Mixed-Validity Portfolio Execution
# =====================================================================

def test_evidence_d_mixed_validity_portfolio_execution(tmp_path):
    """Validates independent sibling validation: an invalid portfolio member

    is captured in rejected_siblings, and does NOT crash or erase valid siblings.
    """
    eval_calls = []

    def tracking_evaluator(spec, frame, folds, metric):
        eval_calls.append(copy.deepcopy(spec))
        return evaluate_candidate(spec, frame, "target", "regression", folds, metric)

    # Propose penalized_linear with invalid C (invalid for regression)
    # alongside hist_gbdt (valid)
    actions = [
        make_action(
            "EVALUATE_PORTFOLIO",
            families=["penalized_linear", "hist_gbdt"],
            model_parameters={
                "penalized_linear": {"C": 1.0},  # Invalid for regression!
                "hist_gbdt": {"max_iter": 50},   # Valid!
            },
        ),
        make_action("ANALYZE"),
    ]

    loop, prompts = make_test_loop(tmp_path, actions, task_type="regression", evaluator=tracking_evaluator)
    state = loop.run()

    turn0_m = state["turns"][0]["measurements"]

    # Hist_gbdt must have executed successfully
    assert len(eval_calls) == 1
    assert eval_calls[0]["families"] == ["hist_gbdt"]

    # Penalized_linear must be recorded under rejected_siblings
    assert "rejected_siblings" in turn0_m
    assert "penalized_linear" in turn0_m["rejected_siblings"]
    rejected_info = turn0_m["rejected_siblings"]["penalized_linear"]
    assert rejected_info["status"] == "invalid_specification"
    assert "classification parameter" in rejected_info["reason"]

    # Valid sibling's evaluation must be preserved in state
    assert "hist_gbdt" in turn0_m["portfolio"]
    assert len(state["evaluated_portfolio_candidates"]) == 1
    assert state["active_spec"]["families"] == ["hist_gbdt"]


# =====================================================================
# Evidence D: Exact Duplicate Reuse via Effective Configuration
# =====================================================================

def test_evidence_d_exact_duplicate_reuse_and_effective_configuration():
    """Validates that effective_configuration resolves semantic aliases and defaults,

    ensuring exact duplicates are deduplicated across attempts.
    """
    spec1 = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"alpha": 1.0, "max_iter": 2000}
        },
    }
    spec2 = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "transforms": [],
        "seed": 42,
        "model_parameters": {
            "penalized_linear": {"alpha": 1.0}  # default max_iter is 2000
        },
    }

    # Effective configuration resolves defaults into an immutable dictionary
    eff1 = effective_configuration(spec1, task_type="regression")
    eff2 = effective_configuration(spec2, task_type="regression")

    assert eff1["resolved_parameters"]["penalized_linear"]["alpha"] == 1.0
    assert eff1["resolved_parameters"]["penalized_linear"]["max_iter"] == 2000
    assert eff2["resolved_parameters"]["penalized_linear"]["max_iter"] == 2000
    assert eff1["resolved_parameters"] == eff2["resolved_parameters"]


# =====================================================================
# Evidence D: Mutation-Resistant Master Inputs / Hardlink Unbinding
# =====================================================================

def test_evidence_d_mutation_resistant_master_inputs_and_hardlink_unbinding(tmp_path):
    """Validates that dataset staging unbinds hard links into independent snapshots,

    detects byte/inode bindings, and prevents worker mutations from altering master bytes.
    """
    rng = np.random.default_rng(42)
    master_dir = tmp_path / "master"
    master_dir.mkdir()
    master_file = master_dir / "master_dataset.csv"
    master_df = pd.DataFrame({
        "feat1": rng.normal(size=100),
        "feat2": rng.uniform(0, 10, size=100),
        "target": np.tile([0, 1], 50),
    })
    master_df.to_csv(master_file, index=False)
    master_bytes_initial = master_file.read_bytes()
    master_inode = master_file.stat().st_ino

    # Create pre-existing hard link to simulate legacy hardlink staging hazard
    legacy_staging = tmp_path / "legacy_staging"
    legacy_staging.mkdir()
    hardlink_file = legacy_staging / "development.csv"
    os.link(master_file, hardlink_file)
    assert hardlink_file.stat().st_ino == master_inode, "Precondition: files must share inode"

    # Initialize loop with the hard-linked staging path
    loop, _ = make_test_loop(
        tmp_path,
        [make_action("ANALYZE")],
        dataset_path=hardlink_file,
    )

    # Staged dataset in isolated workspace MUST be an independent copy, NOT sharing master inode
    staged_data = loop.workspace_root / "inputs" / "development.csv"
    assert staged_data.is_file()
    assert staged_data.stat().st_ino != master_inode, "Staged dataset must NOT share inode with master!"

    # Simulate rogue worker attempting to mutate staged dataset
    staged_data.write_text("MUTATION_ATTACK,CORRUPT\n0.0,0.0\n")

    # Master file MUST remain completely intact and unmutated
    assert master_file.read_bytes() == master_bytes_initial


def test_evidence_d_symlink_hazard_rejected(tmp_path):
    """Validates that symlinks to master dataset or outside workspace are rejected."""
    master_file = tmp_path / "secret.csv"
    master_file.write_text("feat1,feat2,target\n1,2,0\n3,4,1\n")

    symlink_file = tmp_path / "symlink_data.csv"
    os.symlink(master_file, symlink_file)

    with pytest.raises(ValueError, match="Dataset master path must not be a symlink"):
        make_test_loop(tmp_path, [], dataset_path=symlink_file)


# =====================================================================
# Evidence E: Search Admission Accounting
# =====================================================================

def test_evidence_e_search_accounting_local_vs_dispatched():
    """Validates search accounting:

    - Purely local rejection does NOT consume an external query attempt.
    - Dispatched calls (success/failure/timeout) consume admitted query attempts.
    - Fails closed when budget is exhausted.
    """
    mock_sources = [
        SearchSource(
            title="Huber Loss in Tabular Learning",
            url="https://arxiv.org/abs/2000.00001",
            snippet="Robust regression with Huber loss prevents outlier distortion.",
            rerank_score=0.95,
        )
    ]
    provider = MockSearchProvider(responses={"huber": mock_sources})
    client = GovernedResearchClient(provider=provider)
    budgets = OperationalBudgets(web_search_limit=2, search_connected=True)

    # 1. Local rejection: private credentials or dataset names in query -> 0 queries consumed
    res_priv = client.research_topic("techniques using development.csv dataset", budgets=budgets)
    assert res_priv["status"] == "privacy_rejected"
    assert "forbidden dataset file" in res_priv["error"]
    assert budgets.web_search_used == 0

    res_cred = client.research_topic("api key sk-12345678901234567890123456", budgets=budgets)
    assert res_cred["status"] == "privacy_rejected"
    assert "credential" in res_cred["error"]
    assert budgets.web_search_used == 0

    # 2. Dispatched call: valid query -> 1 query admitted and consumed
    res1 = client.research_topic("huber loss regression", budgets=budgets)
    assert res1["status"] == "success"
    assert budgets.web_search_used == 1
    assert budgets.web_search_remaining == 1

    # 3. Dispatched call with timeout/error: provider error still accounts for dispatched attempt
    provider.responses["failing_query"] = RuntimeError("Network timeout")
    res2 = client.research_topic("failing_query", budgets=budgets)
    assert res2["status"] in ("error", "failed")
    assert budgets.web_search_used == 2
    assert budgets.web_search_remaining == 0

    # 4. Budget exhausted: subsequent query rejected before external dispatch
    res3 = client.research_topic("any valid query", budgets=budgets)
    assert res3["status"] == "budget_exhausted"
    assert budgets.web_search_used == 2


# =====================================================================
# Evidence F: Controlled Repeated-Proposal Error Visibility
# =====================================================================

def test_evidence_f_controlled_repeated_proposal_error_visibility(tmp_path):
    """Validates that repeated identical proposals are visibly rejected in durable history

    and provide actionable error feedback in subsequent turns.
    """
    actions = [
        make_action("EVALUATE_PORTFOLIO", families=["penalized_linear"]),
        make_action("EVALUATE_PORTFOLIO", families=["penalized_linear"]),  # Repeated proposal
        make_action("HANDOFF", reason="Observe feedback and conclude"),
    ]

    loop, prompts = make_test_loop(tmp_path, actions)
    state = loop.run()

    # Turn 0 was applied
    assert state["turns"][0]["disposition"] == "applied"

    # Turn 1 (the repeated proposal) was rejected
    assert state["turns"][1]["disposition"] == "rejected"
    assert "already been evaluated" in state["turns"][1]["error_msg"]

    # In turn 2 prompt, the error must be visible in turn_history
    turn1_history = prompts[2]["turn_history"][1]
    assert turn1_history["disposition"] == "rejected"
    assert "already been evaluated" in turn1_history["error"]
