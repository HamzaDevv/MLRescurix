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


def test_unified_four_resource_authority_lifecycle(tmp_path):
    db_file = tmp_path / "state.sqlite"
    proj = "proj_4res_test"
    budget = OperationalBudgets(project_id=proj, db_path=db_file, model_tokens_limit=50000, kaggle_hours_limit=20.0, web_search_limit=10)

    # 1. Admit and settle Kaggle job
    k_admitted, k_job = budget.admit_kaggle_job("kj_100", "kaggle_user", "nb_experiment", requested_hours=2.5)
    assert k_admitted is True
    assert k_job == "kj_100"
    budget.settle_kaggle_job("kj_100", status="success", actual_hours=2.1)

    # 2. Admit and settle Token reservation
    t_admitted, t_res = budget.admit_token_reservation("res_100", "deepseek/deepseek-v4.1-flash-1017", prompt_tokens=1200, completion_tokens=300, cost_millicents=25)
    assert t_admitted is True
    assert t_res == "res_100"
    budget.settle_token_reservation("res_100", status="ok", prompt_tokens=1150, completion_tokens=280, cost_millicents=23)

    # 3. Query authoritative 4-resource status from Rust harness
    auth_status = budget.query_authoritative_resource_status()
    assert auth_status is not None
    assert auth_status["project"] == proj
    assert auth_status["prompt_tokens"] == 1150
    assert auth_status["completion_tokens"] == 280
    assert auth_status["total_tokens"] == 1430
    assert auth_status["total_spend_millicents"] == 23
    assert auth_status["kaggle_jobs_count"] == 1
    assert auth_status["kaggle_hours_used"] == 2.1

    # In-memory budget counters must match authoritative Rust SQLite state
    assert budget.kaggle_hours_used == 2.1
    assert budget.model_tokens_used == 1430
    assert budget.prompt_tokens_used == 1150
    assert budget.completion_tokens_used == 280


def test_token_reservation_conflicting_settlement_rejected(tmp_path):
    db_file = tmp_path / "state.sqlite"
    proj = "proj_conflict_test"
    budget = OperationalBudgets(project_id=proj, db_path=db_file, model_tokens_limit=50000)

    admitted, res_id = budget.admit_token_reservation("res_conf_1", "deepseek/deepseek-v4.1-flash-1017", prompt_tokens=1000, completion_tokens=500, cost_millicents=29)
    assert admitted is True

    # First settlement succeeds
    budget.settle_token_reservation("res_conf_1", status="ok", prompt_tokens=900, completion_tokens=400, cost_millicents=25)

    # Idempotent replay with identical parameters succeeds without warning/error
    budget.settle_token_reservation("res_conf_1", status="ok", prompt_tokens=900, completion_tokens=400, cost_millicents=25)

    # Conflicting settlement with different tokens must be rejected
    from mlrescurix_worker.scientific.operational_budgets import _find_harness_binary
    harness_bin = _find_harness_binary()
    assert harness_bin is not None
    import subprocess
    cmd = [
        str(harness_bin),
        "token-settle",
        "--state", str(db_file),
        "--project", proj,
        "--reservation-id", "res_conf_1",
        "--status", "ok",
        "--prompt-tokens", "1200",  # changed prompt tokens
        "--completion-tokens", "400",
        "--cost-millicents", "25",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "rejected conflicting settlement" in proc.stderr


def test_failed_kaggle_jobs_consume_quota(tmp_path):
    db_file = tmp_path / "state.sqlite"
    proj = "proj_kaggle_quota"
    # Budget has limit of 5.0 hours
    budget = OperationalBudgets(project_id=proj, db_path=db_file, kaggle_hours_limit=5.0)

    # Job 1 requests 4.0 hours, runs for 3.5 hours, and then fails
    admitted, j1 = budget.admit_kaggle_job("job_fail_1", "user", "nb_1", requested_hours=4.0)
    assert admitted is True
    budget.settle_kaggle_job("job_fail_1", status="failed", actual_hours=3.5, error="OOM")

    # Authoritative status must record 3.5 hours used
    auth_status = budget.query_authoritative_resource_status()
    assert auth_status["kaggle_hours_used"] == 3.5

    # Job 2 requests 2.0 hours. Since 3.5 + 2.0 = 5.5 > 5.0, it must be rejected!
    admitted2, j2 = budget.admit_kaggle_job("job_fail_2", "user", "nb_2", requested_hours=2.0)
    assert admitted2 is False
    assert j2 is None

    # Job 3 requests 1.0 hour. Since 3.5 + 1.0 = 4.5 <= 5.0, it must be admitted!
    admitted3, j3 = budget.admit_kaggle_job("job_fail_3", "user", "nb_3", requested_hours=1.0)
    assert admitted3 is True
    assert j3 == "job_fail_3"


def test_settle_search_persists_evidence(tmp_path):
    import sqlite3
    db_file = tmp_path / "state.sqlite"
    proj = "proj_search_ev"
    budget = OperationalBudgets(project_id=proj, db_path=db_file, web_search_limit=5)

    admitted, attempt_id = budget.admit_search(
        query="multivariate regression feature engineering",
        sanitized_query="multivariate regression feature engineering",
        provider="serp_fixture",
    )
    assert admitted is True
    assert attempt_id is not None

    evidence = json.dumps({
        "provider": "serp_fixture",
        "query": "multivariate regression feature engineering",
        "sources": [{"title": "OLS Guide", "url": "https://example.com/ols", "snippet": "Ordinary Least Squares"}],
    })
    budget.settle_search(attempt_id, status="success", evidence=evidence, cost_millicents=50)

    # Verify evidence is recorded in SQLite
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("SELECT status, cost_millicents, response_evidence FROM operational_search_attempts WHERE attempt_id = ?", (attempt_id,))
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "success"
    assert row[1] == 50
    assert "OLS Guide" in row[2]
    conn.close()


def test_outcome_unknown_conflicting_settlement_rejected(tmp_path):
    from mlrescurix_worker.scientific.operational_budgets import _find_harness_binary
    harness_bin = _find_harness_binary()
    assert harness_bin is not None
    import subprocess

    db_file = tmp_path / "state.sqlite"
    proj = "proj_outcome_unknown"
    budget = OperationalBudgets(project_id=proj, db_path=db_file, model_tokens_limit=50000)

    admitted, res_id = budget.admit_token_reservation("res_unk_1", "deepseek", prompt_tokens=1000, completion_tokens=500, cost_millicents=25)
    assert admitted is True

    # Settle as outcome_unknown
    budget.settle_token_reservation("res_unk_1", status="outcome_unknown", prompt_tokens=1000, completion_tokens=500, cost_millicents=25)

    # Idempotent settlement succeeds
    budget.settle_token_reservation("res_unk_1", status="outcome_unknown", prompt_tokens=1000, completion_tokens=500, cost_millicents=25)

    # Conflicting settlement of outcome_unknown reservation with status 'ok' must be rejected!
    cmd = [
        str(harness_bin),
        "token-settle",
        "--state", str(db_file),
        "--project", proj,
        "--reservation-id", "res_unk_1",
        "--status", "ok",
        "--prompt-tokens", "1000",
        "--completion-tokens", "500",
        "--cost-millicents", "25",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "rejected conflicting settlement" in proc.stderr
