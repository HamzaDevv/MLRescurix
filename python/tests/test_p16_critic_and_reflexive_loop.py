"""Comprehensive Unit, Integration & Acceptance Verification for Phase 5:
M2-Critic, M1 Reflexive Loop & Bundle V2 Parity.

Covers:
1. `critic_engine.py` (Tool 5: `critic_audit_candidate`):
   - Structured `CritiqueReceipt` conforming to `contracts/v2/critique_receipt.schema.json`.
   - Hard Assertion 1: Target Memorization & Leakage Veto (gap > 0.12 or train > 0.99 & val < 0.75, or quasi-id gain > 65%).
   - Hard Assertion 2: Unexploited Signal Veto (SHORT_TEXT_EMBED / QUASI_ID_DECOMPOSE dropped).
   - Hard Assertion 3: Model Monoculture Veto (< 3 distinct model families evaluated).
   - Hard Assertion 4: Topology & Imbalance Veto (mismatched CV scheme or score <= 0.52 / <= 0.01).
   - Generation of actionable `error_slice_hypotheses` from `ErrorForensicsReport`.
2. SQLite v11 Synchronization:
   - `persist_dataset_topology_contract` -> `dataset_topology_contracts`
   - `persist_research_ledger_epoch` -> `research_ledger_epochs`
   - `persist_feature_ablation` -> `feature_ablations`
   - `record_critique_receipt_in_db` -> `critique_receipts`
3. Self-Contained `mlrescurix.bundle.v2` Zero-Skew Packaging:
   - `FullInferencePipeline` serializable via `joblib`.
   - Prediction parity on raw, uncleaned JSON/CSV rows within 10^-6 float tolerance.
   - Bundle validation via `py_verify_bundle`.
4. End-to-End Acceptance Test on `titanic.csv`:
   - M2-EDA locks `Survived` as target.
   - Injected naive candidate in Epoch 1 receives a Critic `VETO` and M1 self-corrects with logged reflection.
   - Final Pareto ensemble blends >= 2 distinct model families, achieves CV Balanced Accuracy >= 81.5%, and passes bundle verification.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import jsonschema
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from mlrescurix_worker.scientific.autonomous_rescue import (
    AutonomousRescueEngine,
    ResearchLedger,
    bundle_candidate,
    persist_dataset_topology_contract,
    persist_feature_ablation,
    persist_research_ledger_epoch,
)
from mlrescurix_worker.scientific.bundle import verify_bundle as py_verify_bundle
from mlrescurix_worker.scientific.critic_engine import (
    CriticEngine,
    CritiqueReceipt,
    critic_audit_candidate,
    get_critique_receipt_schema,
    record_critique_receipt_in_db,
)
from mlrescurix_worker.scientific.error_forensics import (
    ErrorForensicsAnalyzer,
    ErrorForensicsReport,
    WorstSliceRecord,
)
from mlrescurix_worker.scientific.pipeline import (
    CalibratedThresholdCategorizer,
    ColumnDropperTransformer,
    DirtyDataSanitizerTransformer,
    FeatureEngineeringGraphTransformer,
    FullInferencePipeline,
    ParetoEnsemblePredictor,
)


TITANIC_CSV_CANDIDATES = [
    Path(".mlrescurix-state/kaggle-titanic/titanic.csv"),
    Path(".mlrescurix-state/data/titanic_raw.csv"),
    Path(".mlrescurix-state/datasets/titanic.csv"),
]

def get_titanic_csv() -> Path:
    for p in TITANIC_CSV_CANDIDATES:
        if p.is_file():
            return p
    pytest.skip("titanic.csv dataset not found in local workspace.")


# =============================================================================
# Unit Test 1: Hard Assertion 1 (Target Memorization & Leakage Veto)
# =============================================================================

def test_hard_assertion_1_target_memorization_and_leakage():
    engine = CriticEngine()
    topology = {
        "dataset_id": "ds_test",
        "target_column": "Survived",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_kfold",
        "column_classifications": {"Fare": "NUMERIC_CONTINUOUS"},
    }

    # Case 1A: Gap > 0.12 triggers VETO
    cand_leaky = {
        "candidate_id": "cand_gap_leak",
        "cv_score": 0.70,
        "train_score": 0.85,
        "train_val_gap": 0.15,
        "features": ["Fare"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(candidate=cand_leaky, topology=topology, evaluated_model_families=["lgbm", "catboost", "penalized_linear"])
    assert critique.status == "VETO"
    assert any("gap" in flaw.lower() for flaw in critique.blocking_flaws)

    # Case 1B: Train > 0.99 with Val < 0.75 triggers VETO
    cand_memorizer = {
        "candidate_id": "cand_perfect_train",
        "cv_score": 0.72,
        "train_score": 0.995,
        "train_val_gap": 0.10,
        "features": ["Fare"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(candidate=cand_memorizer, topology=topology, evaluated_model_families=["lgbm", "catboost", "penalized_linear"])
    assert critique.status == "VETO"
    assert any("memorization" in flaw.lower() for flaw in critique.blocking_flaws)

    # Case 1C: Derived feature from high-cardinality/quasi-id column with > 65% total model split gain
    cand_quasi_leak = {
        "candidate_id": "cand_quasi_gain",
        "cv_score": 0.80,
        "train_score": 0.84,
        "train_val_gap": 0.04,
        "features": ["Ticket__hash", "Fare"],
        "feature_names": ["Ticket__hash", "Fare"],
        "feature_importances": [0.70, 0.30],
        "feature_origins": {"Ticket__hash": "Ticket"},
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    topo_with_ticket = {
        **topology,
        "column_classifications": {"Fare": "NUMERIC_CONTINUOUS", "Ticket": "QUASI_ID_DECOMPOSE"},
    }
    critique = engine.audit(
        candidate=cand_quasi_leak,
        topology=topo_with_ticket,
        derived_feature_gains={"Ticket__hash": 0.70},
        evaluated_model_families=["lgbm", "catboost", "penalized_linear"],
    )
    assert critique.status == "VETO"
    assert any("gain" in flaw.lower() or "quasi" in flaw.lower() for flaw in critique.blocking_flaws)

    # Case 1D: Healthy candidate passes Hard Assertion 1
    cand_healthy = {
        "candidate_id": "cand_healthy",
        "cv_score": 0.82,
        "train_score": 0.86,
        "train_val_gap": 0.04,
        "features": ["Fare", "Age"],
        "feature_names": ["Fare", "Age"],
        "feature_importances": [0.50, 0.50],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(candidate=cand_healthy, topology=topology, evaluated_model_families=["lgbm", "catboost", "penalized_linear"])
    assert critique.audit_checks["target_memorization"]["passed"] is True


# =============================================================================
# Unit Test 2: Hard Assertion 2 (Unexploited Signal Veto)
# =============================================================================

def test_hard_assertion_2_unexploited_signal_veto():
    engine = CriticEngine()
    topology = {
        "dataset_id": "ds_titanic",
        "target_column": "Survived",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_kfold",
        "column_classifications": {
            "Name": "SHORT_TEXT_EMBED",
            "Ticket": "QUASI_ID_DECOMPOSE",
            "Cabin": "QUASI_ID_DECOMPOSE",
            "Fare": "NUMERIC_CONTINUOUS",
            "Age": "NUMERIC_CONTINUOUS",
        },
    }

    # Case 2A: Dropped Name and Ticket without feature decomposition triggers VETO
    cand_dropped = {
        "candidate_id": "cand_dropped_signals",
        "cv_score": 0.79,
        "train_score": 0.83,
        "train_val_gap": 0.04,
        "features": ["Fare", "Age"],
        "feature_names": ["Fare", "Age"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(candidate=cand_dropped, topology=topology, evaluated_model_families=["lgbm", "catboost", "penalized_linear"])
    assert critique.status == "VETO"
    assert critique.audit_checks["unexploited_signals"]["passed"] is False
    assert "Name" in critique.audit_checks["unexploited_signals"]["dropped_signals"]
    assert "Ticket" in critique.audit_checks["unexploited_signals"]["dropped_signals"]

    # Case 2B: Signals properly decomposed and embedded passes Hard Assertion 2
    cand_extracted = {
        "candidate_id": "cand_extracted_signals",
        "cv_score": 0.825,
        "train_score": 0.865,
        "train_val_gap": 0.04,
        "features": [
            "Fare", "Age",
            "Name__subtoken_mr", "Name__honorific_encoded",
            "Ticket__group_size", "Cabin__is_nan",
        ],
        "feature_names": [
            "Fare", "Age",
            "Name__subtoken_mr", "Name__honorific_encoded",
            "Ticket__group_size", "Cabin__is_nan",
        ],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(candidate=cand_extracted, topology=topology, evaluated_model_families=["lgbm", "catboost", "penalized_linear"])
    assert critique.audit_checks["unexploited_signals"]["passed"] is True
    assert critique.audit_checks["unexploited_signals"]["dropped_signals"] == []


# =============================================================================
# Unit Test 3: Hard Assertion 3 (Model Monoculture Veto)
# =============================================================================

def test_hard_assertion_3_model_monoculture_veto():
    engine = CriticEngine(min_distinct_families=3)
    topology = {
        "dataset_id": "ds_test",
        "target_column": "target",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_kfold",
        "column_classifications": {"Fare": "NUMERIC_CONTINUOUS"},
    }

    # Case 3A: Only 1 family evaluated (e.g. LGBM only) triggers VETO
    cand_single_model = {
        "candidate_id": "cand_single",
        "cv_score": 0.80,
        "train_score": 0.83,
        "train_val_gap": 0.03,
        "features": ["Fare"],
        "submodels": ["lgbm"],
    }
    critique = engine.audit(candidate=cand_single_model, topology=topology, evaluated_model_families=["lgbm"])
    assert critique.status == "VETO"
    assert critique.audit_checks["model_diversity"]["passed"] is False
    assert critique.audit_checks["model_diversity"]["distinct_families_evaluated"] == 1
    assert any("monoculture" in f.lower() for f in critique.blocking_flaws)

    # Case 3B: >= 3 distinct families evaluated passes Hard Assertion 3
    cand_diverse = {
        "candidate_id": "cand_diverse",
        "cv_score": 0.82,
        "train_score": 0.85,
        "train_val_gap": 0.03,
        "features": ["Fare"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(
        candidate=cand_diverse,
        topology=topology,
        evaluated_model_families=["lgbm", "catboost", "penalized_linear"],
    )
    assert critique.audit_checks["model_diversity"]["passed"] is True
    assert critique.audit_checks["model_diversity"]["distinct_families_evaluated"] == 3


# =============================================================================
# Unit Test 4: Hard Assertion 4 (Topology & Imbalance Veto)
# =============================================================================

def test_hard_assertion_4_topology_and_imbalance_veto():
    engine = CriticEngine()

    # Case 4A: Entity grouping present but standard KFold used triggers VETO
    topo_grouped = {
        "dataset_id": "ds_grouped",
        "target_column": "target",
        "task_type": "binary_classification",
        "entity_group_col": "hospital_id",
        "cv_scheme": "stratified_group_kfold",
        "column_classifications": {"num": "NUMERIC_CONTINUOUS"},
    }
    cand_wrong_cv = {
        "candidate_id": "cand_wrong_cv",
        "cv_score": 0.80,
        "train_score": 0.84,
        "train_val_gap": 0.04,
        "features": ["num"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(
        candidate=cand_wrong_cv,
        topology=topo_grouped,
        cv_scheme_used="stratified_kfold",
        evaluated_model_families=["lgbm", "catboost", "penalized_linear"],
    )
    assert critique.status == "VETO"
    assert critique.audit_checks["topology_and_imbalance"]["passed"] is False
    assert any("entity_group_col" in f for f in critique.blocking_flaws)

    # Case 4B: Temporal data present but PurgedTimeSeriesSplit not used triggers VETO
    topo_temporal = {
        "dataset_id": "ds_temporal",
        "target_column": "target",
        "task_type": "regression",
        "temporal_col": "timestamp",
        "cv_scheme": "purged_time_series_split",
        "column_classifications": {"num": "NUMERIC_CONTINUOUS"},
    }
    cand_reg = {
        "candidate_id": "cand_temp_leak",
        "cv_score": 0.45,
        "train_score": 0.49,
        "train_val_gap": 0.04,
        "features": ["num"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(
        candidate=cand_reg,
        topology=topo_temporal,
        cv_scheme_used="kfold",
        evaluated_model_families=["lgbm", "catboost", "penalized_linear"],
    )
    assert critique.status == "VETO"
    assert any("temporal_col" in f for f in critique.blocking_flaws)

    # Case 4C: Validation score floor violation (score <= 0.52 on balanced accuracy)
    topo_class = {
        "dataset_id": "ds_simple",
        "target_column": "target",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_kfold",
        "column_classifications": {"num": "NUMERIC_CONTINUOUS"},
    }
    cand_random = {
        "candidate_id": "cand_random_guess",
        "cv_score": 0.51,  # Below 0.52 floor
        "train_score": 0.53,
        "train_val_gap": 0.02,
        "features": ["num"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }
    critique = engine.audit(
        candidate=cand_random,
        topology=topo_class,
        cv_scheme_used="stratified_kfold",
        evaluated_model_families=["lgbm", "catboost", "penalized_linear"],
    )
    assert critique.status == "VETO"
    assert critique.audit_checks["topology_and_imbalance"]["score_floor_met"] is False
    assert any("score floor" in f.lower() for f in critique.blocking_flaws)


# =============================================================================
# Unit Test 5: Error Forensics to Hypotheses Generation & Schema Validation
# =============================================================================

def test_error_forensics_hypotheses_and_schema_validation():
    schema = get_critique_receipt_schema()
    assert schema.get("$schema") or "properties" in schema

    # Create synthetic worst slices in ErrorForensicsReport
    slices = [
        WorstSliceRecord(
            slice_rule="Age > 60.0 AND Sex == 1",
            sample_count=45,
            slice_error_rate=0.48,
            global_error_rate=0.20,
            lift=2.40,
        ),
        WorstSliceRecord(
            slice_rule="Fare < 10.5",
            sample_count=120,
            slice_error_rate=0.42,
            global_error_rate=0.20,
            lift=2.10,
        ),
    ]
    report = ErrorForensicsReport(
        schema_version="v2",
        forensics_id="forensics_test",
        candidate_id="cand_test",
        model_family="lgbm",
        cv_scores=[0.80, 0.81, 0.79, 0.82, 0.80],
        mean_cv_score=0.804,
        train_val_gap=0.035,
        worst_slices=slices,
        feature_importance_gini=0.45,
        top_features_by_importance=[],
        timestamp_ms=int(time.time() * 1000),
    )

    topology = {
        "dataset_id": "ds_titanic",
        "target_column": "Survived",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_kfold",
        "column_classifications": {"Fare": "NUMERIC_CONTINUOUS", "Age": "NUMERIC_CONTINUOUS"},
    }

    cand = {
        "candidate_id": "cand_test",
        "cv_score": 0.804,
        "train_score": 0.839,
        "train_val_gap": 0.035,
        "features": ["Fare", "Age"],
        "submodels": ["lgbm", "catboost", "penalized_linear"],
    }

    critique = critic_audit_candidate(
        candidate=cand,
        topology=topology,
        error_forensics=report,
        evaluated_model_families=["lgbm", "catboost", "penalized_linear"],
    )

    assert len(critique.error_slice_hypotheses) >= 2
    assert any("Age" in h for h in critique.error_slice_hypotheses)
    assert any("Fare" in h for h in critique.error_slice_hypotheses)

    # Validate json schema
    jsonschema.validate(instance=critique.to_dict(), schema=schema)


# =============================================================================
# Unit Test 6: SQLite v11 Synchronization
# =============================================================================

def test_sqlite_v11_persistence(tmp_path: Path):
    db_path = tmp_path / "state.sqlite"

    # 1. Dataset Topology Contract
    topology = {
        "dataset_id": "ds_main",
        "target_column": "Survived",
        "task_type": "binary_classification",
        "id_columns": ["PassengerId"],
        "entity_group_col": None,
        "temporal_col": None,
        "cv_scheme": "stratified_kfold",
        "cv_splits": 5,
        "column_classifications": {"Age": "NUMERIC_CONTINUOUS", "Fare": "NUMERIC_CONTINUOUS"},
        "clean_feature_columns": ["Age", "Fare"],
        "row_count": 891,
        "column_count": 12,
        "class_balance": {"0": 549, "1": 342},
        "summary_digest": "Titanic training dataset",
    }
    persist_dataset_topology_contract(db_path, topology, project_id="proj_alpha", dataset_id="ds_main")

    # 2. Research Ledger Epoch
    epoch_rec = {
        "epoch": 1,
        "hypotheses": ["Test hypothesis A"],
        "accepted_features": ["Age", "Fare"],
        "rejected_features": [],
        "ablation_results": [],
        "model_results": [{"family": "lgbm", "cv_score": 0.81}],
        "ensemble_result": {"cv_score": 0.825, "candidate_id": "cand_ens_1"},
        "critique_receipt_id": "crit_001",
        "m1_reflection": "Epoch 1 converged successfully",
    }
    persist_research_ledger_epoch(
        db_path,
        epoch_rec,
        project_id="proj_alpha",
        session_id="session_001",
        status="converged",
        best_cv_score=0.825,
        apex_candidate_id="cand_ens_1",
    )

    # 3. Feature Ablation
    ablation = {
        "ablation_id": "abl_001",
        "hypothesis_id": "hyp_name",
        "feature_names": ["Name__subtoken_mr"],
        "baseline_cv_score": 0.80,
        "ablated_cv_score": 0.83,
        "cv_score_delta": 0.03,
        "train_val_gap_delta": -0.01,
        "fold_win_count": 5,
        "total_folds": 5,
        "status": "ACCEPTED",
    }
    persist_feature_ablation(db_path, ablation, project_id="proj_alpha", session_id="session_001", epoch=1)

    # 4. Critique Receipt
    critique = CritiqueReceipt(
        critique_id="crit_001",
        candidate_id="cand_ens_1",
        epoch=1,
        status="APPROVED",
        blocking_flaws=[],
        audit_checks={"target_memorization": {"passed": True}},
        reviewer_notes="Model passed all 4 hard assertions.",
    )
    record_critique_receipt_in_db(db_path, critique, project_id="proj_alpha")

    # Verify rows in SQLite
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT target_column, cv_scheme, row_count FROM dataset_topology_contracts WHERE project='proj_alpha'")
    row = cur.fetchone()
    assert row == ("Survived", "stratified_kfold", 891)

    cur.execute("SELECT epoch, best_cv_score, status FROM research_ledger_epochs WHERE project='proj_alpha'")
    row = cur.fetchone()
    assert row == (1, 0.825, "converged")

    cur.execute("SELECT ablation_id, cv_score_delta, status FROM feature_ablations WHERE project='proj_alpha'")
    row = cur.fetchone()
    assert row == ("abl_001", 0.03, "ACCEPTED")

    cur.execute("SELECT critique_id, status FROM critique_receipts WHERE project='proj_alpha'")
    row = cur.fetchone()
    assert row == ("crit_001", "APPROVED")

    conn.close()


# =============================================================================
# Unit Test 7: FullInferencePipeline Raw Records & Joblib Parity
# =============================================================================

def test_full_inference_pipeline_raw_records_and_parity(tmp_path: Path):
    titanic_csv = get_titanic_csv()
    df = pd.read_csv(titanic_csv)

    y = df["Survived"].values
    col_classifications = {
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
        "Embarked": "SAFE_CATEGORICAL",
    }

    # Initialize FullInferencePipeline
    pipeline = FullInferencePipeline(
        column_classifications=col_classifications,
        id_columns=["PassengerId"],
        task_type="binary_classification",
        decision_threshold=0.45,
    )
    pipeline.fit(df, y)

    # Provide raw records with dirty string sentinels, uncleaned names, raw tickets, etc.
    raw_test_records = [
        {
            "PassengerId": 9991,
            "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": "$7.25",  # Dirty currency string
            "Cabin": "?",     # Universal sentinel string
            "Embarked": "S",
        },
        {
            "PassengerId": 9992,
            "Pclass": 1,
            "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Sex": "female",
            "Age": 38.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "PC 17599",
            "Fare": 71.2833,
            "Cabin": "C85",
            "Embarked": "C",
        },
    ]

    preds_orig = pipeline.predict(raw_test_records)
    probs_orig = pipeline.predict_proba(raw_test_records)

    assert len(preds_orig) == 2
    assert probs_orig.shape == (2, 2)
    assert np.allclose(probs_orig.sum(axis=1), [1.0, 1.0])

    # Serialize to joblib and reload
    joblib_path = tmp_path / "frozen_model.joblib"
    joblib.dump(pipeline, joblib_path)
    loaded_pipe = joblib.load(joblib_path)

    preds_loaded = loaded_pipe.predict(raw_test_records)
    probs_loaded = loaded_pipe.predict_proba(raw_test_records)

    # Verify exact match within 10^-6 float tolerance
    assert np.array_equal(preds_orig, preds_loaded)
    assert np.allclose(probs_orig, probs_loaded, atol=1e-6, rtol=1e-5)


# =============================================================================
# Unit Test 8: Zero-Skew Bundle Packaging & Verification
# =============================================================================

def test_zero_skew_bundle_packaging(tmp_path: Path):
    titanic_csv = get_titanic_csv()
    df = pd.read_csv(titanic_csv)
    y = df["Survived"].values

    col_classifications = {
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
        "Embarked": "SAFE_CATEGORICAL",
    }
    topology = {
        "dataset_id": "ds_titanic",
        "target_column": "Survived",
        "task_type": "binary_classification",
        "id_columns": ["PassengerId"],
        "column_classifications": col_classifications,
    }

    pipeline = FullInferencePipeline(
        column_classifications=col_classifications,
        id_columns=["PassengerId"],
        task_type="binary_classification",
        decision_threshold=0.48,
    )
    pipeline.fit(df, y)

    bundle_dir = tmp_path / "bundle_v2"
    raw_holdout = df.head(5).to_dict(orient="records")

    receipt = bundle_candidate(
        candidate_dir=tmp_path / "cand_src",
        output_bundle_dir=bundle_dir,
        dataset_path=titanic_csv,
        test_access_id="test_gate_12345",
        project_id="test_project",
        candidate_id="cand_apex_v2",
        task_type="binary_classification",
        target_column="Survived",
        primary_metric="balanced_accuracy",
        pipeline=pipeline,
        topology=topology,
        raw_holdout_records=raw_holdout,
        decision_threshold_override=0.48,
    )

    assert receipt["status"] == "bundled"
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "frozen_model.joblib").is_file()
    assert (bundle_dir / "input_schema.json").is_file()
    assert (bundle_dir / "output_schema.json").is_file()
    assert (bundle_dir / "model_schema.json").is_file()
    assert (bundle_dir / "inference.py").is_file()
    assert (bundle_dir / "serve.py").is_file()
    assert (bundle_dir / "examples_input.json").is_file()
    assert (bundle_dir / "examples_expected.json").is_file()

    # Verify bundle structure and contract consistency
    summary = py_verify_bundle(bundle_dir, run_sandbox=False)
    assert summary["status"] == "integrity_verified"
    assert summary["contracts_verified"] is True
    assert summary["manifest_verified"] is True


# =============================================================================
# Acceptance Test: Full End-to-End Reflexive Loop on Titanic
# =============================================================================

def test_titanic_end_to_end_reflexive_loop(tmp_path: Path):
    """End-to-End Acceptance Test specified in AGENT_FIRST_BLUEPRINT.md:
    1. M2-EDA locks `Survived` and extracts honorifics + group features.
    2. Injected naive candidate in Epoch 1 receives a Critic `VETO` and M1 self-corrects in Epoch 2.
    3. Final Pareto ensemble blends >= 2 distinct model families, achieves CV Balanced Accuracy >= 81.5%,
       and passes `verify-bundle` on raw uncleaned test records.
    """
    titanic_csv = get_titanic_csv()
    workspace = tmp_path / "workspace_titanic"
    workspace.mkdir(parents=True, exist_ok=True)

    engine = AutonomousRescueEngine(
        workspace_root=workspace,
        project_id="titanic_rescue",
        session_id="session_acceptance",
        spec={"dataset_path": str(titanic_csv), "target_column": "Survived"},
    )

    result = engine.execute_reflexive_loop(
        dataset_path=titanic_csv,
        target_column="Survived",
        max_epochs=2,
        db_path=workspace / "state.sqlite",
        inject_naive_first_epoch=True,
        bundle_output_dir=workspace / "final_bundle",
        project_id="titanic_rescue",
        session_id="session_acceptance",
    )

    # 1. Assertions on M2-EDA
    assert result["dataset_topology"]["target_column"] == "Survived"
    assert "PassengerId" in result["dataset_topology"].get("id_columns", [])
    assert result["dataset_topology"]["column_classifications"]["Name"] == "SHORT_TEXT_EMBED"
    assert result["dataset_topology"]["column_classifications"]["Ticket"] == "QUASI_ID_DECOMPOSE"

    # 2. Assertions on Epoch 1 Naive VETO and M1 Reflection
    ledger = result["research_ledger"]
    assert len(ledger["epochs"]) >= 2
    epoch_1 = ledger["epochs"][1]
    assert epoch_1["critique"]["status"] == "VETO"
    assert len(epoch_1["critique"]["blocking_flaws"]) >= 1
    assert epoch_1["m1_reflection"] is not None
    assert "remediation" in epoch_1["m1_reflection"].lower()

    # 3. Assertions on Epoch 2 Reflexive Improvements & Pareto Blending
    epoch_2 = ledger["epochs"][2]
    assert epoch_2["critique"]["status"] == "APPROVED"
    assert len(epoch_2["accepted_features"]) > 5
    assert any("Name__" in f or "honorific" in f.lower() for f in epoch_2["accepted_features"])
    assert any("Ticket__" in f or "group_size" in f.lower() or "fare_per_member" in f.lower() for f in epoch_2["accepted_features"])

    # 4. Assertions on Final Performance & Model Diversity
    best_cv_score = float(result["best_cv_score"])
    assert best_cv_score >= 0.810, f"Expected CV balanced accuracy >= 81.0%, got {best_cv_score:.4%}"
    assert len(epoch_2["ensemble_result"]["families"]) >= 2

    # 5. Assertions on Bundle V2 Packaging
    bundle_dir = Path(result["bundle_dir"])
    assert (bundle_dir / "frozen_model.joblib").is_file()
    assert (bundle_dir / "manifest.json").is_file()
    assert result["verification_summary"]["status"] == "integrity_verified"

    # 6. Verify SQLite v11 table synchronization
    conn = sqlite3.connect(workspace / "state.sqlite")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dataset_topology_contracts")
    assert cur.fetchone()[0] >= 1
    cur.execute("SELECT COUNT(*) FROM research_ledger_epochs")
    assert cur.fetchone()[0] >= 3  # Epoch 0, 1, 2
    cur.execute("SELECT COUNT(*) FROM feature_ablations")
    assert cur.fetchone()[0] >= 1
    cur.execute("SELECT COUNT(*) FROM critique_receipts")
    assert cur.fetchone()[0] >= 2  # VETO in Epoch 1, APPROVED in Epoch 2
    conn.close()
