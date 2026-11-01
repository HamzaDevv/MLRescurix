"""Authoritative tests for v3 JSON schemas, multi-provider role routing,
program manifest/fit-scope validation, and exact metric mathematics.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest
import jsonschema
import numpy as np

from mlrescurix_worker.scientific.contracts.validator import (
    check_fit_scope_compatibility,
    validate_fit_scope_binding,
    validate_program_manifest,
    validate_transitive_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
V3_CONTRACTS = ROOT / "contracts" / "v3"

ALL_7_SCHEMAS = [
    "problem_spec.schema.json",
    "stage_contract.schema.json",
    "agent_routing.schema.json",
    "capability.schema.json",
    "program.schema.json",
    "fit_scope.schema.json",
    "assessment_policy.schema.json",
]


def _load_schema(name: str) -> dict:
    path = V3_CONTRACTS / name
    assert path.is_file(), f"Missing schema {name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_v3_schemas_exist_and_are_valid_draft_2020_12():
    """Verify all 7 core v3 contracts exist and parse as valid JSON draft schemas."""
    for name in ALL_7_SCHEMAS:
        schema = _load_schema(name)
        assert "$schema" in schema
        assert "$id" in schema
        assert schema.get("type") == "object"
        assert "properties" in schema
        # Validate schema itself is a valid schema
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)


def test_problem_spec_real_jsonschema_validation():
    """Validate ProblemSpec using real jsonschema.validate against valid and adversarial payloads."""
    schema = _load_schema("problem_spec.schema.json")
    valid_payload = {
        "schema_version": "mlrescurix.problem.v1",
        "campaign_id": "cmp_alpha_01",
        "task_type": "tabular_regression",
        "modalities": ["tabular"],
        "target_column": "price",
        "id_column": "sample_id",
        "metric_spec": {
            "name": "smape",
            "direction": "minimize",
            "class_order": None,
            "epsilon": None,
            "decision_threshold": None,
            "variant": "signed",
        },
        "validation_spec": {
            "strategy": "k_fold",
            "folds": 5,
            "seed": 2026,
            "embargo_horizon": None,
        },
        "requested_artifacts": ["native_weights", "inference_closure", "scorecard", "report"],
        "budget": {
            "max_wall_clock_seconds": 3600,
            "max_cost_millicents": 50000,
            "max_total_tokens": 1000000,
            "max_gpu_quota_seconds": 1800,
            "max_worker_turns": 30,
        },
        "assessment_policy": {
            "mode": "auto_if_feasible",
            "lockbox_fraction": 0.10,
            "post_assessment_refit_allowed": False,
        },
    }
    # 1. Valid instance must pass cleanly
    jsonschema.validate(instance=valid_payload, schema=schema)

    # 2. Adversarial: Missing required field
    bad_payload = dict(valid_payload)
    del bad_payload["campaign_id"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_payload, schema=schema)

    # 3. Adversarial: Invalid metric direction enum
    bad_direction = json.loads(json.dumps(valid_payload))
    bad_direction["metric_spec"]["direction"] = "maximize_or_bust"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_direction, schema=schema)

    # 4. Adversarial: Additional unexpected property
    extra_field = json.loads(json.dumps(valid_payload))
    extra_field["unknown_untracked_property"] = 123
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=extra_field, schema=schema)


def test_agent_routing_policy_real_jsonschema_and_resolution():
    """Verify AgentRoutingPolicy with separate providers and models for owner vs workers."""
    schema = _load_schema("agent_routing.schema.json")
    routing_payload = {
        "schema_version": "mlrescurix.agent_routing.v1",
        "default_worker_profile_ref": "worker_economy",
        "providers": {
            "anthropic_direct": {
                "adapter_protocol": "anthropic_messages",
                "endpoint": "https://api.anthropic.com/v1/messages",
                "secret_ref": "sec:anthropic_key",
                "permitted_data_disclosure": "redacted_only",
                "timeout_seconds": 60,
                "max_retries": 2,
            },
            "openrouter_flex": {
                "adapter_protocol": "openai_compatible",
                "endpoint": "https://openrouter.ai/api/v1",
                "secret_ref": "sec:openrouter_key",
                "permitted_data_disclosure": "redacted_only",
                "timeout_seconds": 45,
                "max_retries": 1,
            },
        },
        "models": {
            "owner_reasoning": {
                "provider_ref": "anthropic_direct",
                "model_id": "claude-3-7-sonnet-20250219",
                "context_window": 200000,
                "max_output_tokens": 8192,
                "prompt_millicents_per_1k": 300,
                "completion_millicents_per_1k": 1500,
                "supports_tools": True,
            },
            "worker_economy": {
                "provider_ref": "openrouter_flex",
                "model_id": "deepseek/deepseek-chat",
                "context_window": 64000,
                "max_output_tokens": 2048,
                "prompt_millicents_per_1k": 14,
                "completion_millicents_per_1k": 28,
                "supports_tools": True,
            },
        },
        "roles": {
            "campaign_owner": {
                "model_profile_ref": "owner_reasoning",
                "escalation_profile_ref": None,
                "max_spend_millicents": 30000,
                "temperature": 0.2,
            },
            "modeler": {
                "model_profile_ref": "worker_economy",
                "escalation_profile_ref": "owner_reasoning",
                "max_spend_millicents": 5000,
                "temperature": 0.0,
            },
            "challenger": {
                "model_profile_ref": "owner_reasoning",
                "escalation_profile_ref": None,
                "max_spend_millicents": 10000,
                "temperature": 0.1,
            },
        },
    }
    jsonschema.validate(instance=routing_payload, schema=schema)

    # Adversarial: Invalid permitted_data_disclosure enum
    bad_routing = json.loads(json.dumps(routing_payload))
    bad_routing["providers"]["anthropic_direct"]["permitted_data_disclosure"] = "leaked_to_public"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad_routing, schema=schema)


def test_program_manifest_and_fit_scope_validation():
    """Verify ProgramManifest and FitScopeBinding validation and adversarial checks."""
    valid_program = {
        "schema_version": "mlrescurix.program.v1",
        "execution_kind": "agent_authored",
        "source_tree_ref": "src_tree_hash_1234",
        "task_contract_ref": "task_contract_hash_5678",
        "environment_ref": "env_py313_verified",
        "method_id": "method_xgboost_regressor",
        "operations": {
            "train_fold": {
                "argv": ["python3", "train.py", "--fold", "0"],
                "capability": "tabular_gradient_boosting",
                "fit_scope_kind": "outer_fold_train",
            },
            "predict_fold": {
                "argv": ["python3", "predict.py", "--fold", "0"],
                "capability": "tabular_batch_inference",
                "fit_scope_kind": "consumes_matching_scope",
            },
        },
    }
    assert validate_program_manifest(valid_program) is True

    # Empty operations rejected
    bad_program = dict(valid_program, operations={})
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_program_manifest(bad_program)

    # Valid FitScopeBinding
    valid_binding = {
        "schema_version": "mlrescurix.fit_scope.v1",
        "binding_id": "scope_fold_0_train",
        "split_id": "fold_0",
        "protocol_sha256": "a" * 64,
        "fit_scope_kind": "outer_fold_train",
        "permitted_feature_row_ids_hash": "1" * 64,
        "permitted_label_row_ids_hash": "2" * 64,
        "exclusion_row_ids_hash": None,
        "transitive_fit_dependencies": [],
    }
    assert validate_fit_scope_binding(valid_binding) is True

    # Bad hash format rejected
    bad_binding = dict(valid_binding, permitted_feature_row_ids_hash="not_a_hex_hash")
    with pytest.raises(jsonschema.ValidationError):
        validate_fit_scope_binding(bad_binding)


def test_fit_scope_cross_fold_and_leakage_rejection():
    """Verify fit scope compatibility checker rejects cross-fold state sharing and label leakage."""
    producer_fold_0 = {
        "schema_version": "mlrescurix.fit_scope.v1",
        "binding_id": "scope_fold_0",
        "split_id": "fold_0",
        "protocol_sha256": "a" * 64,
        "fit_scope_kind": "outer_fold_train",
        "permitted_feature_row_ids_hash": "1" * 64,
        "permitted_label_row_ids_hash": "2" * 64,
        "exclusion_row_ids_hash": None,
        "transitive_fit_dependencies": [],
    }
    consumer_fold_1 = {
        "schema_version": "mlrescurix.fit_scope.v1",
        "binding_id": "scope_fold_1",
        "split_id": "fold_1",
        "protocol_sha256": "a" * 64,
        "fit_scope_kind": "outer_fold_train",
        "permitted_feature_row_ids_hash": "3" * 64,
        "permitted_label_row_ids_hash": "4" * 64,
        "exclusion_row_ids_hash": None,
        "transitive_fit_dependencies": [],
    }

    # Cross-fold consumption MUST fail
    ok, reason = check_fit_scope_compatibility(producer_fold_0, consumer_fold_1)
    assert not ok
    assert "cross-fold leakage" in reason

    # Same fold consumption succeeds
    consumer_fold_0 = dict(producer_fold_0, binding_id="consumer_fold_0", fit_scope_kind="consumes_matching_scope")
    ok, reason = check_fit_scope_compatibility(producer_fold_0, consumer_fold_0)
    assert ok

    # Full development train consumed by fold model MUST fail
    full_train = dict(producer_fold_0, binding_id="full_train", split_id="full", fit_scope_kind="full_development_train")
    ok, reason = check_fit_scope_compatibility(full_train, consumer_fold_0)
    assert not ok
    assert "scope violation" in reason or "cross-fold leakage" in reason

    # Evaluated label leakage check
    eval_label_hash = "2" * 64
    ok, reason = check_fit_scope_compatibility(producer_fold_0, consumer_fold_0, evaluated_label_hash=eval_label_hash)
    assert not ok
    assert "label leakage" in reason

    # Protocol mismatch rejection
    diff_proto = dict(consumer_fold_0, protocol_sha256="b" * 64)
    ok, reason = check_fit_scope_compatibility(producer_fold_0, diff_proto)
    assert not ok
    assert "protocol mismatch" in reason

    # Row-set level partial label leakage detection (A ∩ B != ∅)
    ok, reason = check_fit_scope_compatibility(
        producer_fold_0,
        consumer_fold_0,
        evaluated_label_rows={10, 11, 12},
        producer_label_rows={5, 6, 11},  # Row 11 overlaps!
    )
    assert not ok
    assert "label leakage" in reason and "11" in reason

    # Row-set level partial feature exclusion violation
    ok, reason = check_fit_scope_compatibility(
        producer_fold_0,
        consumer_fold_0,
        consumer_exclusion_rows={20, 21},
        producer_feature_rows={19, 20},  # Row 20 overlaps excluded features!
    )
    assert not ok
    assert "exclusion violation" in reason and "20" in reason

    # Stateless/pretrained producer cannot have fitted label rows
    stateless_prod = dict(producer_fold_0, fit_scope_kind="stateless", split_id="")
    ok, reason = check_fit_scope_compatibility(stateless_prod, consumer_fold_0, producer_label_rows={1, 2})
    assert not ok
    assert "cannot have fitted label rows" in reason

    # Stateless producer with empty label rows succeeds
    ok, reason = check_fit_scope_compatibility(stateless_prod, consumer_fold_0, producer_label_rows=set())
    assert ok


def test_transitive_provenance_cycle_and_integrity_checks():
    """Verify operations graph acyclicity and transitive dependency compatibility."""
    # 1. Valid acyclic graph
    scope_base = {
        "schema_version": "mlrescurix.fit_scope.v1",
        "binding_id": "scope_f0",
        "split_id": "fold_0",
        "protocol_sha256": "a" * 64,
        "fit_scope_kind": "outer_fold_train",
        "permitted_feature_row_ids_hash": "1" * 64,
        "permitted_label_row_ids_hash": "2" * 64,
        "exclusion_row_ids_hash": None,
        "transitive_fit_dependencies": [],
    }
    graph = {
        "fe": {"fit_scope": scope_base, "dependencies": []},
        "model": {"fit_scope": scope_base, "dependencies": ["fe"]},
        "eval": {"fit_scope": scope_base, "dependencies": ["model"]},
    }
    ok, reason = validate_transitive_provenance(graph)
    assert ok, reason

    # 2. Cycle detection: fe -> model -> eval -> fe
    cyclic_graph = {
        "fe": {"fit_scope": scope_base, "dependencies": ["eval"]},
        "model": {"fit_scope": scope_base, "dependencies": ["fe"]},
        "eval": {"fit_scope": scope_base, "dependencies": ["model"]},
    }
    ok, reason = validate_transitive_provenance(cyclic_graph)
    assert not ok
    assert "cyclical dependency" in reason

    # 3. Transitive row-set level label leakage rejection
    leak_graph = {
        "fe": {
            "fit_scope": scope_base,
            "fitted_label_rows": [1, 2, 3, 15],
            "dependencies": [],
        },
        "model": {
            "fit_scope": scope_base,
            "dependencies": ["fe"],
        },
        "eval": {
            "fit_scope": scope_base,
            "evaluated_label_rows": [15, 16, 17],
            "dependencies": ["model"],
        },
    }
    ok, reason = validate_transitive_provenance(leak_graph)
    assert not ok
    assert "transitive fit provenance violation" in reason
    assert "label leakage" in reason and "15" in reason

    # 4. Transitive protocol mismatch rejection
    diff_scope = dict(scope_base, protocol_sha256="b" * 64, binding_id="scope_diff_proto")
    proto_graph = {
        "fe": {"fit_scope": scope_base, "dependencies": []},
        "model": {"fit_scope": diff_scope, "dependencies": ["fe"]},
    }
    ok, reason = validate_transitive_provenance(proto_graph)
    assert not ok
    assert "protocol mismatch" in reason


def test_python_smape_formula_exact_zero_zero_and_signed():
    """Verify Section 7.2 exact SMAPE calculation:
    SMAPE(y, p) = 100/N * sum( 2|p_i - y_i| / (|y_i| + |p_i|) )
    with zero/zero contributing zero, supporting signed domain unless non_negative_only.
    """
    def compute_smape(y_true, y_pred, non_negative_only=False):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        assert len(y_true) == len(y_pred)
        if non_negative_only and ((y_true < 0).any() or (y_pred < 0).any()):
            raise ValueError("SMAPE requires non-negative values for non_negative_only variant")
        denom = np.abs(y_true) + np.abs(y_pred)
        diff = np.abs(y_pred - y_true)
        ratios = np.zeros_like(denom)
        mask = denom != 0.0
        ratios[mask] = 2.0 * diff[mask] / denom[mask]
        return float(100.0 * np.mean(ratios))

    # Identical predictions -> 0.0
    assert compute_smape([0.0, 10.0, 20.0], [0.0, 10.0, 20.0]) == 0.0

    # Hand-computed case matching Rust test:
    score = compute_smape([10.0, 20.0, 0.0], [15.0, 20.0, 0.0])
    assert abs(score - (100.0 * 0.4 / 3.0)) < 1e-12

    # Signed SMAPE handles negative values
    # row 1: y=-10, p=-15 -> 2*5 / 25 = 0.4
    # row 2: y=20, p=20 -> 0.0
    # mean = 0.4 / 2 = 0.2 -> 20.0%
    signed_score = compute_smape([-10.0, 20.0], [-15.0, 20.0], non_negative_only=False)
    assert abs(signed_score - 20.0) < 1e-12

    # non_negative_only variant rejects negative values
    with pytest.raises(ValueError, match="non-negative"):
        compute_smape([-10.0, 20.0], [-15.0, 20.0], non_negative_only=True)
