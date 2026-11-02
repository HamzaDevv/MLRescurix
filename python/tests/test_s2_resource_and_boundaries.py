"""S2 Verification Gates C, E, F Integration Tests.

Validates:
Gate C:
- Shared atomic search quota admission across two independent controllers/budget instances sharing a project_id.
- Pre-turn durable search attempt admission and idempotent settlement in SQLite.
- Timeout accounting: TimeoutError raised by search provider is recorded as 'timeout' and remains counted.
- Zero external dispatch when search admission is denied.
- Local validation/privacy rejection causes zero usage consumption.

Gate E:
- Sibling runtime failure isolation: a runtime failure during fit/evaluate in one sibling does not abort valid siblings.

Gate F:
- Strict action schema validation against contracts/v3/actions.json.
- Rejection of unknown actions and schema parameter violations without silent fallback.
"""
import json
import sqlite3
import uuid
import pytest

from mlrescurix_worker.scientific.governed_research import (
    GovernedResearchClient,
    MockSearchProvider,
    PrivacyViolationError,
    SearchSource,
    sanitize_query,
)
from mlrescurix_worker.scientific.operational_budgets import (
    OperationalBudgets,
)
from mlrescurix_worker.scientific.candidate_pipeline import (
    evaluate_candidate,
)
from test_autonomy_regressions import make_loop, action


def test_gate_c_shared_atomic_search_quota_and_zero_dispatch():
    project_id = f"test_gate_c_{uuid.uuid4().hex[:8]}"

    # Controller 1
    budget_ctrl1 = OperationalBudgets(project_id=project_id, web_search_limit=2, search_connected=True)
    # Controller 2 shares same project_id
    budget_ctrl2 = OperationalBudgets(project_id=project_id, web_search_limit=2, search_connected=True)

    sources = [
        SearchSource(
            title="Tabular Ensembles",
            url="https://arxiv.org/abs/test1",
            snippet="Ensemble diversity improves tabular metrics.",
        )
    ]
    dispatched_calls = []

    class CountingMockProvider:
        name = "counting_mock"

        def is_configured(self) -> bool:
            return True

        def search(self, query, max_results=5):
            dispatched_calls.append(query)
            return sources

    provider = CountingMockProvider()
    client1 = GovernedResearchClient(provider=provider)
    client2 = GovernedResearchClient(provider=provider)

    # Controller 1 performs 1st query -> admitted and dispatched
    res1 = client1.research_topic("gradient boosting loss formulations", budgets=budget_ctrl1)
    assert res1["status"] == "success"
    assert len(dispatched_calls) == 1
    assert budget_ctrl1.web_search_used == 1

    # Controller 2 queries durable status -> sees 1 query used
    assert budget_ctrl2.sync_search_usage() == 1
    assert budget_ctrl2.web_search_remaining == 1

    # Controller 2 performs 2nd query -> admitted and dispatched
    res2 = client2.research_topic("neural tabular embeddings", budgets=budget_ctrl2)
    assert res2["status"] == "success"
    assert len(dispatched_calls) == 2
    assert budget_ctrl2.web_search_used == 2
    assert budget_ctrl1.sync_search_usage() == 2

    # Controller 1 attempts 3rd query -> denied admission by SQLite authority!
    res3 = client1.research_topic("third query over limit", budgets=budget_ctrl1)
    assert res3["status"] == "budget_exhausted"
    # Zero external dispatch occurred!
    assert len(dispatched_calls) == 2
    assert budget_ctrl1.web_search_used == 2


def test_gate_c_timeout_accounting_and_persistence():
    project_id = f"test_timeout_{uuid.uuid4().hex[:8]}"
    budget = OperationalBudgets(project_id=project_id, web_search_limit=3, search_connected=True)

    # Provider configured to raise TimeoutError
    provider = MockSearchProvider(responses={"timeout_topic": TimeoutError})
    client = GovernedResearchClient(provider=provider)

    res = client.research_topic("timeout_topic research query", budgets=budget)
    assert res["status"] == "failed"
    assert res["error_type"] == "timeout"
    assert "TimeoutError" in res["error"] or "timeout" in res["error"].lower()
    # Timed-out query remains counted as an admitted attempt
    assert budget.web_search_used == 1

    # Verify attempt recorded in durable SQLite with 'timeout' status
    db_path = budget.db_path
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT status, error_message FROM operational_search_attempts WHERE project = ? ORDER BY created_ms DESC LIMIT 1",
        (project_id,),
    )
    row = cur.fetchone()
    conn.close()
    assert row is not None
    assert row[0] == "timeout"


def test_gate_c_local_privacy_rejection_causes_zero_dispatch_and_zero_usage():
    project_id = f"test_privacy_{uuid.uuid4().hex[:8]}"
    budget = OperationalBudgets(project_id=project_id, web_search_limit=5, search_connected=True)
    client = GovernedResearchClient(provider=MockSearchProvider(responses={}))

    # Direct sanitizer check
    with pytest.raises(PrivacyViolationError):
        sanitize_query("query with secret sk-1234567890123456789012345")

    # Client call rejects before admission/dispatch
    res = client.research_topic("query with secret sk-1234567890123456789012345", budgets=budget)
    assert res["status"] == "privacy_rejected"
    assert "secret or credential" in res["error"]

    # Zero usage consumed
    assert budget.web_search_used == 0


def test_gate_e_portfolio_runtime_sibling_failure_isolation(tmp_path):
    def sabotaged_evaluator(spec, frame, folds, metric):
        if spec.get("families") == ["penalized_linear"]:
            raise RuntimeError("Simulated runtime failure in linear sibling fit")
        return evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)

    loop, _ = make_loop(
        tmp_path,
        [
            action("EVALUATE_PORTFOLIO", families=["penalized_linear", "hist_gbdt"]),
            action("ANALYZE"),
        ],
        evaluator=sabotaged_evaluator,
    )

    state = loop.run()

    # Sibling isolation: loop did not crash
    assert len(state["turns"]) == 2
    turn0 = state["turns"][0]
    assert turn0["disposition"] == "applied"

    # Linear failure is captured in sibling_failures
    sibling_failures = turn0["measurements"].get("sibling_failures", {})
    assert "penalized_linear" in sibling_failures
    assert "Simulated runtime failure" in sibling_failures["penalized_linear"]["error"]

    # Valid sibling (hist_gbdt) successfully evaluated and accepted
    assert state.get("accepted_candidate") is not None
    assert state["accepted_candidate"]["families"] == ["hist_gbdt"]


def test_gate_f_action_schema_validation(tmp_path):
    loop, _ = make_loop(
        tmp_path,
        [
            # Turn 0: Canonical action with valid parameters
            action("EVALUATE_PORTFOLIO", families=["tree", "linear"], seed=2026),
            # Turn 1: Unknown action fails closed
            {"thought_rationale": "Invalid", "action_type": "MAGIC_WAND_OPTIMIZE", "parameters": {}},
            # Turn 2: Missing required parameters fails closed
            {"thought_rationale": "Missing req", "action_type": "HYPOTHESIS_TRANSFORM", "parameters": {}},
            action("ANALYZE"),
        ],
    )

    state = loop.run()
    turns = state["turns"]

    # Turn 0 applied
    assert turns[0]["disposition"] == "applied"

    # Turn 1 rejected for unknown action type
    assert turns[1]["disposition"] == "rejected"
    assert "Unknown action type 'MAGIC_WAND_OPTIMIZE'" in turns[1]["error_msg"]

    # Turn 2 rejected for missing required parameters
    assert turns[2]["disposition"] == "rejected"
    assert "missing required parameter" in turns[2]["error_msg"]


def test_gate_e_exact_duplicate_candidate_one_fit(tmp_path):
    fit_count = 0
    def counting_evaluator(spec, frame, folds, metric):
        nonlocal fit_count
        fit_count += 1
        return evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)

    loop, _ = make_loop(
        tmp_path,
        [
            # Turn 0: Explicit max_iter=2000 for penalized_linear
            action("EVALUATE_PORTFOLIO", families=["penalized_linear"], model_parameters={"penalized_linear": {"max_iter": 2000}}),
            # Turn 1: Semantic duplicate with omitted max_iter (default resolves to 2000)
            action("EVALUATE_PORTFOLIO", families=["penalized_linear"], model_parameters={"penalized_linear": {}}),
            action("ANALYZE"),
        ],
        evaluator=counting_evaluator,
    )

    state = loop.run()
    assert fit_count == 1, f"Expected exactly 1 fit for duplicate candidate, got {fit_count}"
    turn1 = state["turns"][1]
    assert turn1["disposition"] == "rejected"
    assert "Every proposed portfolio candidate has already been evaluated" in turn1["error_msg"]
    assert "penalized_linear" in turn1["error_msg"]

