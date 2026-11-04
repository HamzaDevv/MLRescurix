"""Accounting must survive checkpoint reloads and disclose unconnected budgets."""
import json

import pytest

from mlrescurix_worker.scientific.operational_budgets import OperationalBudgets
from mlrescurix_worker.scientific.autonomous_loop import RecoveryRequired
from test_autonomy_regressions import action, make_loop


def test_checkpoint_restores_all_usage_without_double_counting(tmp_path):
    actions = [action("ANALYZE"), action("ANALYZE")]
    loop, _ = make_loop(tmp_path, actions)
    calls = iter([(10, 3), (20, 7)])

    def provider(messages, max_tokens):
        prompt, completion = next(calls)
        return {"content": json.dumps(action("ANALYZE")), "cost_usd": 0,
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}

    loop.provider_caller = provider
    loop.run()
    assert loop.operational_budgets.model_tokens_used == 40
    resumed, _ = make_loop(tmp_path, actions)
    assert resumed.operational_budgets.model_tokens_used == 40
    resumed.export_handoff()
    assert resumed.operational_budgets.model_tokens_used == 40
    status = json.loads((resumed.workspace_root / "handoff.json").read_text())["operational_budgets"]
    assert status["total_model_tokens"]["used"] == 40
    assert status["total_model_tokens"]["accounting_complete"] is True


def test_unreported_usage_and_unconnected_budgets_are_disclosed():
    budget = OperationalBudgets()
    budget.restore_token_usage([
        {"provider_receipt": {"usage": {"prompt_tokens": 11, "completion_tokens": 2}}},
        {"provider_receipt": {"cost_usd": 0}},
    ])
    status = budget.get_status()
    assert status["total_model_tokens"]["used"] == 13
    assert status["total_model_tokens"]["accounting_complete"] is False
    assert status["total_model_tokens"]["missing_receipts"] == 1
    assert status["total_model_tokens"]["remaining"] is None
    assert status["kaggle_compute"]["enforced"] is False
    assert status["web_search_queries"]["enforced"] is False


def test_connectivity_does_not_claim_global_budget_enforcement():
    budget = OperationalBudgets(search_connected=True, kaggle_connected=True)
    budget.record_search_query()
    budget.record_kaggle_compute(3600)
    status = budget.get_status()
    for key in ("web_search_queries", "kaggle_compute"):
        assert status[key]["connected"] is True
        assert status[key]["enforced"] is False
        assert status[key]["remaining" if key == "web_search_queries" else "remaining_hours"] is None


def test_invalid_config_does_not_silently_raise_budget_to_default(tmp_path):
    path = tmp_path / "budgets.json"
    path.write_text('{"budgets":{"total_model_tokens":{"limit":-1}}}')
    with pytest.raises(ValueError, match="Invalid operational budget"):
        OperationalBudgets.load(path)


def test_live_dispatch_waits_for_missing_token_receipt_reconciliation(tmp_path):
    loop, calls = make_loop(tmp_path, [action("ANALYZE")])
    loop.run()  # Fixture response deliberately lacks token usage.
    assert len(calls) == 1
    loop.provider_is_fixture = False
    loop.max_turns = 2
    with pytest.raises(RecoveryRequired, match="Token usage is missing"):
        loop.execute_turn(1, {})
    assert loop.run()["stop_reason"] == "token_usage_reconciliation_required"
    assert len(calls) == 1
    saved = json.loads(loop.state_file.read_text())
    saved["turns"][0]["provider_receipt"]["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    loop.state_file.write_text(json.dumps(saved))
    loop.max_turns = 1
    assert loop.run()["stop_reason"] == "max_turns_reached"
    assert loop.operational_budgets.model_tokens_used == 15


def test_restore_search_and_kaggle_usage_without_cumulative_multiplication():
    budget = OperationalBudgets()
    turns = [
        {"operational_budgets": {"web_search_queries": {"recorded_session_queries": 1}, "kaggle_compute": {"recorded_session_hours": 0.5}}},
        {"operational_budgets": {"web_search_queries": {"recorded_session_queries": 1}, "kaggle_compute": {"recorded_session_hours": 0.5}}},
        {"operational_budgets": {"web_search_queries": {"recorded_session_queries": 2}, "kaggle_compute": {"recorded_session_hours": 1.2}}},
    ]
    budget.restore_operational_budgets(turns)
    assert budget.web_search_used == 2
    assert budget.kaggle_hours_used == 1.2


def test_authoritative_resource_status_query(tmp_path):
    db_file = tmp_path / "state.sqlite"
    proj = "proj_auth_test"
    budget = OperationalBudgets(project_id=proj, db_path=db_file, web_search_limit=50)

    # 1. Before admission: DB doesn't exist yet, so enforced is False
    status_pre = budget.get_status()
    assert status_pre["web_search_queries"]["enforced"] is False

    # 2. Admit a search attempt via Rust harness
    admitted, attempt_id = budget.admit_search("what is boosting", "what is boosting", provider="fixture")
    assert admitted is True
    assert attempt_id is not None
    budget.settle_search(attempt_id, status="success")

    # A local attempt record does not prove campaign-wide or remote enforcement.
    status_post = budget.get_status()
    assert status_post["web_search_queries"]["enforced"] is False
    assert status_post["web_search_queries"]["recorded_session_queries"] == 1
    assert status_post["web_search_queries"]["used"] is None
    assert status_post["web_search_queries"]["remaining"] is None
    assert status_post["kaggle_compute"]["enforced"] is False
    assert status_post["kaggle_compute"]["remaining_hours"] is None

