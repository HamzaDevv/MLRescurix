"""Unit tests for Phase 1 Schemas and Contracts (Draft 2020-12)."""
import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator, ValidationError


@pytest.fixture
def load_schema():
    def _load(filename: str) -> dict:
        p = Path("contracts/v2") / filename
        assert p.exists(), f"Missing schema at {p}"
        schema = json.loads(p.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        return schema
    return _load


def test_dataset_topology_schema_valid_and_enforced(load_schema):
    schema = load_schema("dataset_topology.schema.json")
    validator = Draft202012Validator(schema)

    valid_topology = {
        "schema_version": "v2",
        "dataset_id": "ds_titanic",
        "target_column": "Survived",
        "task_type": "binary_classification",
        "id_columns": ["PassengerId"],
        "entity_group_col": "Ticket",
        "temporal_col": None,
        "cv_scheme": "stratified_group_kfold",
        "cv_splits": 5,
        "column_classifications": {
            "PassengerId": "DROP_SURROGATE_ID",
            "Survived": "TARGET",
            "Pclass": "SAFE_CATEGORICAL",
            "Name": "SHORT_TEXT_EMBED",
            "Sex": "SAFE_CATEGORICAL",
            "Age": "NUMERIC_CONTINUOUS",
            "SibSp": "NUMERIC_DISCRETE",
            "Parch": "NUMERIC_DISCRETE",
            "Ticket": "QUASI_ID_DECOMPOSE",
            "Fare": "NUMERIC_CONTINUOUS",
            "Cabin": "QUASI_ID_DECOMPOSE",
            "Embarked": "SAFE_CATEGORICAL"
        },
        "column_metadata": {
            "PassengerId": {"uniqueness_ratio": 1.0, "missing_count": 0, "missing_rate": 0.0, "cardinality": 891, "coercion": None, "detected_type": "int"},
            "Fare": {"uniqueness_ratio": 0.28, "missing_count": 0, "missing_rate": 0.0, "cardinality": 248, "coercion": "currency_clean", "detected_type": "float"}
        },
        "coercions": {
            "Fare": "currency_clean"
        },
        "clean_feature_columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
        "row_count": 891,
        "column_count": 12,
        "class_balance": {"0": 549, "1": 342},
        "summary_digest": "# EDA Dossier Summary\nTarget: Survived (38.4% positive)\nRows: 891",
        "locked_at_ms": 1776500000000
    }
    validator.validate(valid_topology)

    # Negative test: missing target_column
    invalid = dict(valid_topology)
    del invalid["target_column"]
    with pytest.raises(ValidationError):
        validator.validate(invalid)

    # Negative test: invalid cv_scheme
    invalid_cv = dict(valid_topology, cv_scheme="nonexistent_split")
    with pytest.raises(ValidationError):
        validator.validate(invalid_cv)


def test_research_ledger_schema_valid_and_enforced(load_schema):
    schema = load_schema("research_ledger.schema.json")
    validator = Draft202012Validator(schema)

    valid_ledger = {
        "schema_version": "v2",
        "session_id": "sess_titanic_01",
        "project_id": "proj_titanic",
        "current_epoch": 1,
        "apex_candidate_id": "cand_epoch1_blend",
        "best_cv_score": 0.824,
        "metric_name": "balanced_accuracy",
        "metric_direction": "maximize",
        "status": "active",
        "epochs": [
            {
                "epoch": 0,
                "hypotheses": ["baseline_anchor"],
                "accepted_features": ["Pclass", "Sex", "Age"],
                "rejected_features": [],
                "ablation_results": [],
                "model_results": [
                    {"family": "lgbm", "cv_score": 0.785, "candidate_id": "cand_baseline_lgbm"}
                ],
                "ensemble_result": None,
                "critique": None,
                "m1_reflection": "Baseline anchor establishes 0.785 balanced accuracy.",
                "timestamp_ms": 1776500001000
            },
            {
                "epoch": 1,
                "hypotheses": ["decompose_ticket_and_extract_honorifics"],
                "accepted_features": ["Pclass", "Sex", "Age", "Name__honorific", "Ticket__group_size"],
                "rejected_features": ["Fare_is_nan"],
                "ablation_results": [
                    {"feature": "Name__honorific", "delta_cv": 0.024, "status": "ACCEPTED"}
                ],
                "model_results": [
                    {"family": "catboost", "cv_score": 0.819, "candidate_id": "cand_cb_01"},
                    {"family": "lgbm", "cv_score": 0.815, "candidate_id": "cand_lgb_01"}
                ],
                "ensemble_result": {
                    "cv_score": 0.824,
                    "weights": {"cand_cb_01": 0.6, "cand_lgb_01": 0.4}
                },
                "critique": {
                    "status": "APPROVED",
                    "blocking_flaws": []
                },
                "m1_reflection": "Honorifics and Ticket group size improve generalization significantly.",
                "timestamp_ms": 1776500005000
            }
        ],
        "updated_ms": 1776500005000
    }
    validator.validate(valid_ledger)

    # Negative test: missing session_id
    invalid = dict(valid_ledger)
    del invalid["session_id"]
    with pytest.raises(ValidationError):
        validator.validate(invalid)


def test_critique_receipt_schema_valid_and_enforced(load_schema):
    schema = load_schema("critique_receipt.schema.json")
    validator = Draft202012Validator(schema)

    # 1. Valid VETO receipt
    veto_receipt = {
        "schema_version": "v2",
        "critique_id": "crit_001",
        "candidate_id": "cand_overfit_tree",
        "epoch": 1,
        "status": "VETO",
        "blocking_flaws": [
            "Train-val gap exceeded: train 0.995 vs val 0.742 (gap 0.253 > 0.12)",
            "Quasi-ID column 'Ticket' used directly without decomposition"
        ],
        "soft_warnings": ["Single feature accounted for 55% of gain"],
        "audit_checks": {
            "target_memorization": {
                "passed": False,
                "train_metric": 0.995,
                "val_metric": 0.742,
                "gap": 0.253,
                "max_allowed_gap": 0.12,
                "dominant_feature": "Ticket",
                "dominant_gain_ratio": 0.55
            },
            "unexploited_signals": {
                "passed": False,
                "dropped_signals": ["Name"],
                "details": "Column 'Name' was dropped without decomposition or text embedding."
            },
            "model_diversity": {
                "passed": False,
                "distinct_families_evaluated": 1,
                "families": ["decision_tree"],
                "min_required": 3
            },
            "topology_and_imbalance": {
                "passed": True,
                "cv_scheme_used": "stratified_group_kfold",
                "required_cv_scheme": "stratified_group_kfold",
                "score_floor_met": True,
                "score": 0.742
            }
        },
        "error_slice_hypotheses": [
            "Cohort Pclass == 3 AND Sex == 'male' exhibits 42% error rate; investigate male 3rd class child survival."
        ],
        "reviewer_notes": "Candidate exhibits target memorization and fails signal utilization.",
        "timestamp_ms": 1776500010000
    }
    validator.validate(veto_receipt)

    # 2. Valid APPROVED receipt
    approved_receipt = {
        "schema_version": "v2",
        "critique_id": "crit_002",
        "candidate_id": "cand_apex_ensemble",
        "epoch": 2,
        "status": "APPROVED",
        "blocking_flaws": [],
        "soft_warnings": [],
        "audit_checks": {
            "target_memorization": {
                "passed": True,
                "train_metric": 0.852,
                "val_metric": 0.826,
                "gap": 0.026,
                "max_allowed_gap": 0.12,
                "dominant_feature": None,
                "dominant_gain_ratio": 0.22
            },
            "unexploited_signals": {
                "passed": True,
                "dropped_signals": []
            },
            "model_diversity": {
                "passed": True,
                "distinct_families_evaluated": 3,
                "families": ["catboost", "lgbm", "penalized_linear"],
                "min_required": 3
            },
            "topology_and_imbalance": {
                "passed": True,
                "cv_scheme_used": "stratified_group_kfold",
                "required_cv_scheme": "stratified_group_kfold",
                "score_floor_met": True,
                "score": 0.826
            }
        },
        "error_slice_hypotheses": [],
        "reviewer_notes": "Candidate satisfies all 4 hard assertion gates.",
        "timestamp_ms": 1776500020000
    }
    validator.validate(approved_receipt)


def test_error_forensics_schema_valid_and_enforced(load_schema):
    schema = load_schema("error_forensics.schema.json")
    validator = Draft202012Validator(schema)

    valid_forensics = {
        "schema_version": "v2",
        "forensics_id": "ef_001",
        "candidate_id": "cand_lgbm_01",
        "model_family": "lgbm",
        "cv_scores": [0.812, 0.825, 0.805, 0.831, 0.819],
        "mean_cv_score": 0.8184,
        "train_val_gap": 0.032,
        "worst_slices": [
            {
                "slice_rule": "Pclass == 3 AND Sex == 'male' AND Age_is_nan == 1",
                "sample_count": 94,
                "slice_error_rate": 0.382,
                "global_error_rate": 0.181,
                "lift": 2.11
            },
            {
                "slice_rule": "Embarked == 'S' AND Fare <= 7.89",
                "sample_count": 120,
                "slice_error_rate": 0.315,
                "global_error_rate": 0.181,
                "lift": 1.74
            }
        ],
        "feature_importance_gini": 0.42,
        "top_features_by_importance": [
            {"feature": "Sex_female", "importance": 0.31, "origin": "raw_categorical"},
            {"feature": "Name__honorific_Mr", "importance": 0.18, "origin": "text_subtoken"},
            {"feature": "Ticket__group_size", "importance": 0.14, "origin": "quasi_id_decompose"}
        ],
        "confusion_or_residual_summary": {
            "tn": 490, "fp": 59, "fn": 95, "tp": 247
        },
        "timestamp_ms": 1776500008000
    }
    validator.validate(valid_forensics)
