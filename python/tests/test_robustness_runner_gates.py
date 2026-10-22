"""Offline regression coverage for campaign runner acceptance and artifact gates."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_robustness_campaign as runner  # noqa: E402


def accepted_state(score=0.82):
    state = {
        "accepted_candidate_id": "cand_ok",
        "best_cv_score": score,
        "candidate_identity": None,
        "accepted_candidate": {"families": ["hist_gradient_boosting"], "weights": [1.0], "transforms": [], "seed": 7, "model_parameters": {}},
        "accepted_audit": {"status": "APPROVED", "candidate_id": "cand_ok"},
    }
    state["candidate_identity"] = runner.candidate_spec_identity(state["accepted_candidate"])
    state["accepted_evaluation"] = {
        "candidate_identity": state["candidate_identity"], "cv_score": score,
    }
    return state


def test_runner_rejects_missing_nonfinite_and_unaccepted_scores():
    for value in (None, float("nan"), float("inf"), "not-a-score", True):
        assert runner.finite_score(value) is None
        state = accepted_state(value)
        valid, score, _ = runner.accepted_candidate_evidence(state)
        assert not valid
        assert score is None

    no_acceptance = accepted_state()
    no_acceptance["accepted_candidate_id"] = None
    assert not runner.accepted_candidate_evidence(no_acceptance)[0]


def test_runner_requires_approved_audit_and_matching_identity():
    state = accepted_state()
    state["accepted_audit"] = {"status": "REJECTED"}
    assert not runner.accepted_candidate_evidence(state)[0]

    state = accepted_state()
    state["accepted_candidate"]["family"] = "other"
    assert not runner.accepted_candidate_evidence(state)[0]

    state = accepted_state()
    state["accepted_audit"]["candidate_id"] = "cand_other"
    assert not runner.accepted_candidate_evidence(state)[0]

    state = accepted_state()
    state["accepted_evaluation"]["cv_score"] = 0.2
    assert not runner.accepted_candidate_evidence(state)[0]
    state = accepted_state()
    state["accepted_evaluation"]["candidate_identity"] = "wrong"
    assert not runner.accepted_candidate_evidence(state)[0]


def test_runner_acceptance_is_anded_with_threshold_and_bundle_identity():
    valid, score, _ = runner.accepted_candidate_evidence(accepted_state())
    assert valid and score == pytest.approx(0.82)
    assert runner.score_meets_threshold(score, 0.75, "maximize")
    assert not runner.score_meets_threshold(score, 0.85, "maximize")
    assert runner.score_meets_threshold(0.40, 0.55, "minimize")
    assert not runner.score_meets_threshold(None, 0.55, "minimize")
    assert runner.candidate_gate_failure_reason(valid, "", score, 0.85, "maximize", True, True, "")
    assert runner.candidate_gate_failure_reason(valid, "", score, 0.75, "maximize", True, False, "identity mismatch")


def test_bundle_identity_gate_checks_manifest_and_serialized_pipeline(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "candidate_id": "cand_ok",
        "candidate_identity": "sha256:abc123",
        "model": {"weights_path": "frozen_model.joblib"},
    }), encoding="utf-8")
    joblib.dump(SimpleNamespace(candidate_identity="sha256:abc123"), tmp_path / "frozen_model.joblib")
    assert runner.bundle_matches_candidate(tmp_path, "cand_ok", "sha256:abc123")
    assert not runner.bundle_matches_candidate(tmp_path, "cand_other", "sha256:abc123")
    assert not runner.bundle_matches_candidate(tmp_path, "cand_ok", "sha256:wrong")


def test_packager_does_not_refit_or_create_fallback_candidate_without_acceptance(tmp_path):
    class MustNotFit:
        def fit_accepted_candidate(self):
            raise AssertionError("refit must not run without accepted candidate evidence")

    result = runner.package_accepted_candidate(
        MustNotFit(), {}, pd.DataFrame({"x": [1]}),
        candidate_dir=tmp_path / "candidate", bundle_dir=tmp_path / "bundle",
        dataset_path=tmp_path / "dev.csv", test_access_id="test", project_id="project",
        task_type="regression", target_column="y", primary_metric="rmse",
    )
    assert result[0:2] == (False, False)
    assert "no accepted candidate" in result[2]
    assert not (tmp_path / "bundle").exists()


def test_pinned_index_folds_map_to_selected_rows_and_fail_closed():
    folds = [{"train_indices": [20, 21], "val_indices": [22]}]
    assert runner.map_locked_index_folds(folds, [20, 21, 22]) == [([0, 1], [2])]
    with pytest.raises(ValueError, match="not fully contained"):
        runner.map_locked_index_folds(folds, [21, 22, 23])
    with pytest.raises(ValueError, match="missing"):
        runner.map_locked_index_folds([], [1, 2])
    with pytest.raises(ValueError, match="overlap"):
        runner.map_locked_index_folds([{"train_indices": [1, 2], "val_indices": [2]}], [1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        runner.map_locked_index_folds([{"train_indices": [1, 1], "val_indices": [2]}], [1, 2])


def test_k4_index_space_is_never_guessed():
    folds = [{"train_indices": [1, 2], "val_indices": [3]}]
    with pytest.raises(ValueError, match="unsupported.*index_space"):
        runner.map_k4_locked_folds(folds, 100, 103, None)
    assert runner.map_k4_locked_folds(folds, 100, 103, "bounded_dev_subset") == [([1, 2], [3])]
    source_folds = [{"train_indices": [101, 102], "val_indices": [103]}]
    assert runner.map_k4_locked_folds(source_folds, 100, 103, "source_development_rows") == [([1, 2], [3])]


def test_k4_current_manifest_does_not_fall_back_to_generated_folds():
    split_manifest = {"development_origins": [{
        "train_start_idx": 0, "train_end_idx": 3,
        "val_start_idx": 4, "val_end_idx": 5,
        "val_start_dt": 500, "val_end_dt": 600,
    }]}
    # This representative manifest has row boundaries and validation times,
    # but no declared index space, train-time boundary, or locked gap.
    with pytest.raises(ValueError, match="index_space"):
        runner.map_k4_manifest_folds(split_manifest, pd.DataFrame({"TransactionDT": [1, 2, 3]}), 422432, 472431)


def test_all_gate_thresholds_are_sourced_from_locked_policy():
    policy = json.loads((ROOT / "docs" / "v2" / "BENCHMARK_SUITE.json").read_text(encoding="utf-8"))
    assert runner.locked_quality_threshold(policy, "K1") is None
    assert runner.locked_quality_threshold(policy, "K2") is None
    assert runner.locked_quality_threshold(policy, "K3") == pytest.approx(0.75)
    assert runner.locked_quality_threshold(policy, "K4") == pytest.approx(0.85)
    assert runner.locked_quality_threshold(policy, "K5") == pytest.approx(0.55)


def test_campaign_task_contract_preserves_temporal_and_panel_topology():
    campaign = object.__new__(runner.CampaignRunner)
    campaign.policy = json.loads((ROOT / "docs" / "v2" / "BENCHMARK_SUITE.json").read_text(encoding="utf-8"))
    k4, k4_statement = campaign._task_contract("K4")
    k5, k5_statement = campaign._task_contract("K5")
    assert k4["temporal_col"] == "TransactionDT"
    assert k4["entity_join_key"] == "TransactionID"
    assert k4["cv_scheme"] == "purged_time_series_split"
    assert k5["temporal_col"] == "date"
    assert k5["entity_columns"] == ["store_nbr", "family"]
    assert k5["forecast_horizon_dates"] == 15
    assert "pinned split" in k4_statement and "pinned split" in k5_statement


def test_pinned_temporal_folds_keep_chronology_and_reject_missing_boundaries():
    frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]})
    folds = [{"train_end_date": "2026-01-02", "validation_start_date": "2026-01-03", "validation_end_date": "2026-01-04"}]
    assert runner.map_locked_temporal_folds(folds, frame, "date") == [([0, 1], [2, 3])]
    with pytest.raises(ValueError, match="lacks"):
        runner.map_locked_temporal_folds([{"train_end_date": "2026-01-02"}], frame, "date")


def test_live_provider_pricing_requires_explicit_current_revalidation():
    stale_policy = {"provider": {"pricing_snapshot": {
        "checked_local_date": "2026-10-21", "input_usd_per_million": "0.08",
        "output_usd_per_million": "0.50", "must_revalidate_before_live": True,
    }}}
    assert runner.verified_provider_pricing(stale_policy, checked_date="2026-10-22") is None
    fresh_policy = {"provider": {"pricing_snapshot": {
        "checked_local_date": "2026-10-22", "verified_for_live": True, "must_revalidate_before_live": False,
        "input_usd_per_million": "0.08", "output_usd_per_million": "0.50",
    }}}
    assert runner.verified_provider_pricing(fresh_policy, checked_date="2026-10-22") == {
        "verified": True, "input_per_million": "0.08", "output_per_million": "0.50",
    }
    contradicting_policy = {"provider": {"pricing_snapshot": {
        "checked_local_date": "2026-10-22", "verified_for_live": True, "must_revalidate_before_live": True,
        "input_usd_per_million": "0.08", "output_usd_per_million": "0.50",
    }}}
    assert runner.verified_provider_pricing(contradicting_policy, checked_date="2026-10-22") is None
