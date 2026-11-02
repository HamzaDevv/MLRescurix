"""Regression tests for avoiding repeated individual experiments in portfolios."""
import copy

from test_autonomy_regressions import action, make_loop
from mlrescurix_worker.scientific.candidate_pipeline import evaluate_candidate


def test_repeated_portfolio_candidates_are_skipped_without_losing_new_families(tmp_path):
    seen = []

    def evaluator(spec, frame, folds, metric):
        seen.append(copy.deepcopy(spec))
        return evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)

    loop, prompts = make_loop(tmp_path, [
        action("EVALUATE_PORTFOLIO", families=["penalized_linear"]),
        action("EVALUATE_PORTFOLIO", families=["penalized_linear", "hist_gbdt"]),
        action("EVALUATE_PORTFOLIO", families=["penalized_linear", "hist_gbdt"]),
        action("ANALYZE"),
    ], evaluator=evaluator)
    state = loop.run()
    assert [item["families"] for item in seen] == [["penalized_linear"], ["hist_gbdt"]]
    assert "penalized_linear" in state["turns"][1]["measurements"]["previously_evaluated"]
    assert state["turns"][2]["disposition"] == "rejected"
    assert "already been evaluated" in prompts[3]["turn_history"][2]["error"]
    assert len(state["evaluated_portfolio_candidates"]) == 2


def test_portfolio_selects_approved_second_best_when_leader_is_vetoed(tmp_path):
    def evaluator(spec, frame, folds, metric):
        result = evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)
        result["cv_score"] = .9 if spec["families"] == ["penalized_linear"] else .8
        return result

    class Critique:
        def __init__(self, status): self.status = status
        def to_dict(self): return {"status": self.status}

    class Critic:
        def audit(self, candidate, **kwargs):
            return Critique("VETO" if candidate["submodels"] == ["penalized_linear"] else "APPROVED")

    loop, _ = make_loop(tmp_path, [action("EVALUATE_PORTFOLIO", families=["penalized_linear", "hist_gbdt"])], evaluator=evaluator)
    loop.critic_engine = Critic()
    state = loop.run()
    assert state["accepted_candidate"]["families"] == ["hist_gbdt"]
    assert state["selected_cv_score"] == .8
    assert state["accepted_evaluation"]["cv_score"] == .8
    audits = state["turns"][0]["measurements"]["portfolio_audits"]
    assert audits["penalized_linear"]["status"] == "VETO"
    assert audits["hist_gbdt"]["status"] == "APPROVED"


def test_blend_duplicate_identity_includes_model_parameters(tmp_path):
    seen = []
    def evaluator(spec, frame, folds, metric):
        seen.append(copy.deepcopy(spec))
        return evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)

    families = ["penalized_linear", "hist_gbdt"]
    loop, _ = make_loop(tmp_path, [
        action("BLEND_AND_AUDIT", families=families, weights=[.5, .5]),
        action("BLEND_AND_AUDIT", families=families, weights=[1, 1]),
        action("BLEND_AND_AUDIT", families=families, weights=[.5, .5], model_parameters={"hist_gbdt": {"max_iter": 50}}),
    ], evaluator=evaluator)
    state = loop.run()
    assert len(seen) == 2
    assert "Duplicate blend" in state["turns"][1]["error_msg"]
    assert state["turns"][2]["disposition"] == "applied"


def test_portfolio_allows_intentional_repeat_with_different_seed(tmp_path):
    seen = []
    def evaluator(spec, frame, folds, metric):
        seen.append(copy.deepcopy(spec))
        return evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)

    loop, _ = make_loop(tmp_path, [
        action("EVALUATE_PORTFOLIO", families=["penalized_linear"], seed=42),
        action("EVALUATE_PORTFOLIO", families=["penalized_linear"], seed=42),
        action("EVALUATE_PORTFOLIO", families=["penalized_linear"], seed=100),
    ], evaluator=evaluator)
    state = loop.run()
    assert len(seen) == 2
    assert seen[0].get("seed") == 42
    assert seen[1].get("seed") == 100
    assert state["turns"][1]["disposition"] == "rejected"
    assert "already been evaluated" in state["turns"][1]["error_msg"]
    assert state["turns"][2]["disposition"] == "applied"

