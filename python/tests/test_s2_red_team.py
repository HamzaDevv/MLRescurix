"""Independent boundary regressions; no live providers or historical state."""
from test_autonomy_regressions import make_loop, action
from mlrescurix_worker.scientific.autonomous_loop import RecoveryRequired


def test_search_admission_survives_actual_loop_reconstruction(tmp_path):
    first, _ = make_loop(tmp_path, [])
    budget = first.operational_budgets
    admitted, attempt = budget.admit_search("tabular models", "tabular models", "mock")
    assert admitted and attempt
    second, _ = make_loop(tmp_path, [])
    restored = second.operational_budgets
    assert restored.project_id == budget.project_id == first.project_id
    assert restored.db_path == budget.db_path
    assert restored.sync_search_usage() == 1


def test_uncertain_execution_cannot_be_treated_as_candidate_failure(tmp_path):
    calls = []

    def uncertain(spec, frame, folds, metric):
        calls.append(spec["families"][0])
        raise RecoveryRequired("remote dispatch outcome unknown")

    loop, _ = make_loop(tmp_path, [action("EVALUATE_PORTFOLIO", families=["hist_gbdt", "penalized_linear"])], evaluator=uncertain)
    # The outer loop may record rejection/handoff; it must not run a sibling
    # after an explicit request to reconcile an uncertain side effect.
    try:
        loop.run()
    except RecoveryRequired:
        pass
    assert calls == ["hist_gbdt"]
