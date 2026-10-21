import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.semantic_sieve import analyze_and_lock_topology
from mlrescurix_worker.scientific.hypothesis_executor import (
    ActionExecutionError,
    HypothesisActionDispatcher,
    IdempotencyConflictError,
)


@pytest.fixture
def sample_df():
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "id": list(range(n)),
        "num1": rng.exponential(scale=2.0, size=n),
        "num2": rng.normal(loc=10.0, scale=3.0, size=n),
        "cat1": rng.choice(["A", "B", "C", "D"], size=n),
        "cat_high": [f"grp_{i % 25}" for i in range(n)],
        "ticket": [f"T_{i % 10}" for i in range(n)],
        "name": [f"Dr. John Doe {i}" for i in range(n)],
        "target": rng.randint(0, 2, size=n),
    })
    return df


@pytest.fixture
def topology(sample_df):
    return analyze_and_lock_topology(sample_df, explicit_target="target", dataset_id="test_ds")


def test_action_payload_validation():
    dispatcher = HypothesisActionDispatcher()

    # Missing fields
    with pytest.raises(ActionExecutionError, match="missing required field"):
        dispatcher.validate_action_payload({"action": "DELETE_FEATURE"})

    # Invalid action
    with pytest.raises(ActionExecutionError, match="Unknown action type"):
        dispatcher.validate_action_payload({
            "schema_version": "v2",
            "action_id": "act_01",
            "parent_candidate_id": "cand_01",
            "source_hash": "abcdef123456",
            "split_id": "split_01",
            "topology_hash": "topo123456",
            "seed": 2026,
            "action": "NON_EXISTENT_ACTION",
            "parameters": {},
            "budget_id": "b_01",
            "rationale_ref": "none",
        })


def test_idempotency_and_conflict_rejection(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()

    payload = {
        "schema_version": "v2",
        "action_id": "act_idempotent_test",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "REORDER_MODEL_PRIORITY",
        "parameters": {
            "preferred_families": ["hist_gradient_boosting", "logistic_regression"]
        },
        "budget_id": "b_01",
        "rationale_ref": "audit_decision",
    }

    # First call
    receipt1 = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"]
    )
    assert receipt1.action_id == "act_idempotent_test"
    assert receipt1.status == "confirmed"

    # Replay with identical payload: returns cached receipt
    receipt2 = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"]
    )
    assert receipt2.action_id == receipt1.action_id
    assert receipt2.payload_hash == receipt1.payload_hash

    # Replay with mutated payload: raises IdempotencyConflictError
    mutated_payload = dict(payload)
    mutated_payload["parameters"] = {"preferred_families": ["random_forest"]}
    with pytest.raises(IdempotencyConflictError):
        dispatcher.execute_action(
            mutated_payload, sample_df, topology, base_features=["num1", "num2"]
        )


def test_execute_apply_skew_transform(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()
    payload = {
        "schema_version": "v2",
        "action_id": "act_skew_num1",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "APPLY_SKEW_TRANSFORM",
        "parameters": {
            "columns": ["num1"],
            "method": "yeo_johnson",
        },
        "budget_id": "b_01",
        "rationale_ref": "reduce_skewness",
    }

    receipt = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"], fidelity="fast"
    )
    assert receipt.action_id == "act_skew_num1"
    assert receipt.status in ("confirmed", "refuted", "inconclusive")
    assert len(receipt.versioned_transform_lineage) == 1
    assert "elapsed_seconds" in receipt.resource_measurements


def test_execute_encode_high_card_cat(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()
    payload = {
        "schema_version": "v2",
        "action_id": "act_eb_high_card",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "ENCODE_HIGH_CARD_CAT",
        "parameters": {
            "columns": ["cat_high"],
            "smoothing": 10.0,
            "cv_folds": 5,
        },
        "budget_id": "b_01",
        "rationale_ref": "oof_eb_encoding",
    }

    receipt = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"], fidelity="fast"
    )
    assert receipt.action_id == "act_eb_high_card"
    assert receipt.status in ("confirmed", "refuted", "inconclusive")


def test_execute_decompose_quasi_id(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()
    payload = {
        "schema_version": "v2",
        "action_id": "act_quasi_ticket",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "DECOMPOSE_QUASI_ID",
        "parameters": {
            "id_columns": ["ticket"],
            "numeric_columns": ["num1"],
            "aggregations": ["count", "mean"],
        },
        "budget_id": "b_01",
        "rationale_ref": "ticket_aggregates",
    }

    receipt = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"], fidelity="fast"
    )
    assert receipt.action_id == "act_quasi_ticket"
    assert receipt.status in ("confirmed", "refuted", "inconclusive")


def test_execute_embed_short_text(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()
    payload = {
        "schema_version": "v2",
        "action_id": "act_text_name",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "EMBED_SHORT_TEXT",
        "parameters": {
            "text_columns": ["name"],
            "n_svd_components": 4,
            "top_k_subtokens": 8,
        },
        "budget_id": "b_01",
        "rationale_ref": "name_embeddings",
    }

    receipt = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"], fidelity="fast"
    )
    assert receipt.action_id == "act_text_name"
    assert receipt.status in ("confirmed", "refuted", "inconclusive")


def test_execute_delete_feature(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()
    payload = {
        "schema_version": "v2",
        "action_id": "act_del_num2",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "DELETE_FEATURE",
        "parameters": {
            "columns": ["num2"],
            "reason": "ablation_prune",
        },
        "budget_id": "b_01",
        "rationale_ref": "prune_noise",
    }

    receipt = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"], fidelity="fast"
    )
    assert receipt.action_id == "act_del_num2"
    assert receipt.status in ("confirmed", "refuted", "inconclusive")


def test_execute_governed_custom_sandbox_code(sample_df, topology):
    dispatcher = HypothesisActionDispatcher()
    snippet = """
import pandas as pd
import numpy as np

def build_features(df, y=None):
    X = pd.DataFrame(index=df.index)
    X['interaction'] = df['num1'] * df['num2']
    return X
"""
    payload = {
        "schema_version": "v2",
        "action_id": "act_custom_code",
        "parent_candidate_id": "cand_01",
        "source_hash": "abcdef123456",
        "split_id": "split_01",
        "topology_hash": "topo123456",
        "seed": 2026,
        "action": "CUSTOM_SANDBOX_CODE",
        "parameters": {
            "code_snippet": snippet,
            "entrypoint_function": "build_features",
        },
        "budget_id": "b_01",
        "rationale_ref": "candidate_expert_interaction",
    }

    receipt = dispatcher.execute_action(
        payload, sample_df, topology, base_features=["num1", "num2"], fidelity="fast"
    )
    assert receipt.action_id == "act_custom_code"
    assert receipt.status in ("confirmed", "refuted", "inconclusive")
