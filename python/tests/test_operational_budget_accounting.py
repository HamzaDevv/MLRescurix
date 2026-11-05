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


def test_autonomous_loop_migrates_legacy_operational_ledger_and_fences_old_writers(tmp_path):
    import sqlite3
    import subprocess
    from mlrescurix_worker.scientific.operational_budgets import _find_harness_binary
    from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
    from test_autonomy_regressions import Budget

    harness_bin = _find_harness_binary()
    assert harness_bin is not None

    ws = tmp_path / "ws"
    ws.mkdir()
    legacy_file = ws / "operational_search.sqlite"
    harness_state_dir = ws / "rust_harness_state"
    target_state = harness_state_dir / "state.sqlite"

    policy_file = ws / "policy.json"
    policy_file.write_text(json.dumps({
        "budget": {"tokens": 10_000_000, "storage": 100_000_000},
        "capacity": {"threads": 4},
        "max_tasks": 10,
    }))

    # 1. Initialize Rust harness state in target_state
    init_cmd = [
        str(harness_bin),
        "init",
        "--state", str(harness_state_dir),
        "--policy", str(policy_file),
    ]
    subprocess.run(init_cmd, check=True)

    # 2. Populate legacy operational ledger
    with sqlite3.connect(legacy_file) as conn:
        conn.execute("""
            CREATE TABLE operational_search_attempts (
                project TEXT, attempt_id TEXT PRIMARY KEY, query_text TEXT, sanitized_query TEXT,
                provider TEXT, status TEXT, cost_millicents INTEGER, created_ms INTEGER,
                settled_ms INTEGER, error_message TEXT, response_evidence TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE operational_token_reservations (
                project TEXT, reservation_id TEXT PRIMARY KEY, role_id TEXT, model_id TEXT,
                prompt_tokens INTEGER, completion_tokens INTEGER, cost_millicents INTEGER,
                status TEXT, created_ms INTEGER, settled_ms INTEGER, error_message TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE operational_kaggle_jobs (
                project TEXT, job_id TEXT PRIMARY KEY, account_username TEXT, slug TEXT,
                accelerator TEXT, requested_hours REAL, actual_hours REAL, status TEXT,
                created_ms INTEGER, settled_ms INTEGER, error_message TEXT
            )
        """)
        conn.execute(
            "INSERT INTO operational_search_attempts VALUES ('proj_m', 'att_1', 'ols', 'ols', 'ddg', 'success', 0, 100, 150, NULL, 'ev')"
        )
        conn.execute(
            "INSERT INTO operational_token_reservations VALUES ('proj_m', 'tok_1', 'worker', 'model', 100, 50, 1, 'ok', 200, 250, NULL)"
        )
        conn.commit()

    dev_csv = ws / "dev.csv"
    dev_csv.write_text("x,target\n1,0\n2,1\n3,0\n4,1\n5,0\n6,1\n7,0\n8,1\n9,0\n10,1\n")

    # 3. Instantiate AutonomousAgentLoop - should trigger automatic migration of legacy_file into target_state
    loop = AutonomousAgentLoop(
        ws, "proj_m", "sess_m", Budget(), dev_csv, "target",
        provider_is_fixture=True, max_turns=1
    )

    # 4. Verify operational budgets now uses the target state
    assert loop.operational_budgets.db_path == str(target_state)

    # 5. Verify target state has the migrated records
    with sqlite3.connect(target_state) as conn:
        count_s = conn.execute("SELECT COUNT(*) FROM operational_search_attempts WHERE project='proj_m'").fetchone()[0]
        count_t = conn.execute("SELECT COUNT(*) FROM operational_token_reservations WHERE project='proj_m'").fetchone()[0]
        assert count_s == 1
        assert count_t == 1

    # 6. Verify legacy database has migration triggers that abort old writers
    with sqlite3.connect(legacy_file) as conn:
        with pytest.raises((sqlite3.OperationalError, sqlite3.IntegrityError), match="operational ledger migrated; use controller authority"):
            conn.execute(
                "INSERT INTO operational_search_attempts VALUES ('proj_m', 'att_new', 'q', 'q', 'p', 'admitted', 0, 300, NULL, NULL, NULL)"
            )


def test_autonomous_loop_migration_fails_closed_on_tampered_legacy_ledger(tmp_path):
    import sqlite3
    import subprocess
    from mlrescurix_worker.scientific.operational_budgets import _find_harness_binary
    from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
    from test_autonomy_regressions import Budget

    harness_bin = _find_harness_binary()
    assert harness_bin is not None

    ws = tmp_path / "ws_tampered"
    ws.mkdir()
    legacy_file = ws / "operational_search.sqlite"
    harness_state_dir = ws / "rust_harness_state"
    target_state = harness_state_dir / "state.sqlite"

    policy_file = ws / "policy.json"
    policy_file.write_text(json.dumps({
        "budget": {"tokens": 10_000_000, "storage": 100_000_000},
        "capacity": {"threads": 4},
        "max_tasks": 10,
    }))

    subprocess.run([
        str(harness_bin), "init",
        "--state", str(harness_state_dir),
        "--policy", str(policy_file),
    ], check=True)

    # Corrupt/uncoordinated table in legacy ledger
    with sqlite3.connect(legacy_file) as conn:
        conn.executescript("CREATE TABLE provider_accounting (id INT); INSERT INTO provider_accounting VALUES (1);")

    dev_csv = ws / "dev.csv"
    dev_csv.write_text("x,target\n1,0\n2,1\n")

    # AutonomousAgentLoop must fail closed with RuntimeError rather than swallowing
    with pytest.raises(RuntimeError, match="Legacy operational ledger migration failed"):
        AutonomousAgentLoop(
            ws, "proj_fail", "sess_fail", Budget(), dev_csv, "target",
            provider_is_fixture=True, max_turns=1
        )


def test_governance_live_provider_dispatch_admits_and_settles_tokens(tmp_path):
    from decimal import Decimal
    from mlrescurix_worker.scientific.operational_budgets import OperationalBudgets, _find_harness_binary
    from mlrescurix_worker.scientific.governance import GovernanceService
    from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter
    import subprocess

    harness_bin = _find_harness_binary()
    assert harness_bin is not None

    state_dir = tmp_path / "gov_state"
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({
        "budget": {"tokens": 10_000_000, "storage": 100_000_000},
        "capacity": {"threads": 4},
        "max_tasks": 10,
    }))
    subprocess.run([
        str(harness_bin), "init",
        "--state", str(state_dir),
        "--policy", str(policy_file),
    ], check=True)
    db_file = state_dir / "state.sqlite"

    budgets = OperationalBudgets(
        db_path=db_file,
        project_id="gov_test_proj",
        authority_binary=harness_bin,
        model_tokens_limit=100_000,
    )

    class MockRustGateway(RustGatewayAdapter):
        def __init__(self):
            self.state_dir = state_dir
            self.binary = harness_bin
            self.project_id = "gov_test_proj"
            self.pricing = {"verified": True, "input_per_million": 140, "output_per_million": 280}
        def reservation(self, messages, max_tokens):
            return Decimal("0.0005")
        def call_with_request_id(self, messages, max_tokens, request_id):
            import sqlite3
            with sqlite3.connect(db_file) as conn:
                reserved = conn.execute(
                    "SELECT prompt_tokens, completion_tokens FROM operational_token_reservations WHERE project=? AND reservation_id=?",
                    (self.project_id, request_id),
                ).fetchone()
            assert reserved == (len(json.dumps(messages).encode()) + 4096, max_tokens)
            return {
                "content": "hello from model",
                "cost_usd": "0.0002",
                "cost_is_estimate": False,
                "usage": {"prompt_tokens": 120, "completion_tokens": 40},
            }

    class DummyCampaignAuthority:
        def __init__(self):
            self.reserved = []
            self.settled = []
        def get_status(self):
            return {"campaign_ceiling_usd": "1.00"}
        def reserve(self, session_id, req_id, amt):
            self.reserved.append((session_id, req_id, amt))
        def settle(self, session_id, req_id, amt, cost_kind=None):
            self.settled.append((session_id, req_id, amt, cost_kind))

    class DummySession:
        def __init__(self):
            self.session_id = "sess_gov_1"
            self.model_name = "deepseek/deepseek-chat"
            self.campaign_authority = DummyCampaignAuthority()
            self.operational_budgets = budgets
            self.provider_is_fixture = False
            self.provider_caller = MockRustGateway()
            self.per_run_budget_usd = Decimal("0.05")
            self.pricing = {"verified": True, "input_per_million": 140, "output_per_million": 280}
            self.state = {"turns": [], "pending_turn": {"request_id": "req_gov_1"}}
            self.events = []
        def _reservation(self, messages, max_tokens):
            return Decimal("0.0005")
        def _event(self, name, **kwargs):
            self.events.append((name, kwargs))
        def _save_state(self, state):
            self.state = state

    session = DummySession()
    service = GovernanceService()

    res = service.dispatch(
        session,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )

    assert res["content"] == "hello from model"

    # Verify token reservation was recorded and settled in the authoritative SQLite database
    import sqlite3
    with sqlite3.connect(db_file) as conn:
        row = conn.execute(
            "SELECT role_id, model_id, prompt_tokens, completion_tokens, status FROM operational_token_reservations WHERE project='gov_test_proj'"
        ).fetchone()
        assert row is not None
        assert row[0] == "governance_dispatch"
        assert row[1] == "deepseek/deepseek-chat"
        assert row[2] == 120
        assert row[3] == 40
        assert row[4] == "ok"
