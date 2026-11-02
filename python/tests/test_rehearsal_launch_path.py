"""Complete launch-path rehearsal test.

Exercises the full launch path:
1. Staged development dataset and pinned folds
2. Multi-turn AutonomousAgentLoop with RustGatewayAdapter
3. Policy staging (stage-outbound-view) and operator approval (approve-outbound-view)
4. Independent evaluation, critic audit, and candidate acceptance
5. Model fitting, artifact generation, parity verification, and handoff
6. Unapproved view / failure case fails closed
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
from mlrescurix_worker.scientific.candidate_pipeline import identity
from mlrescurix_worker.scientific.development_stage import stage_development, sha256
from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter, find_harness_binary
from mlrescurix_worker.scientific.export import export_model, verify_export_parity


def test_complete_launch_path_rehearsal(tmp_path: Path):
    harness_bin = find_harness_binary()
    assert harness_bin.is_file()

    # 1. Prepare staged dataset and pinned folds from Amazon catalog data
    source_csv = tmp_path / "raw_data.csv"
    # Create realistic synthetic catalog rows
    np.random.seed(42)
    n_rows = 200
    rows = []
    for i in range(n_rows):
        words = ["Pack", "Bottle", "Ounce", "Organic", "Sauce", "Coffee", "Tea", "Snack", "Spicy", "Sweet"]
        title = " ".join(np.random.choice(words, size=4))
        desc = f"Item {i} {title} Value: {np.random.randint(1, 100)} Unit: Count"
        # Create meaningful relationship with text so model achieves positive R^2
        price = float(5.0 + len(title) * 1.5 + float(np.random.uniform(0.1, 1.0)))
        rows.append({"catalog_content": f"{title}\n{desc}", "price": price})
    df_raw = pd.DataFrame(rows)
    df_raw.to_csv(source_csv, index=False)

    split_manifest = tmp_path / "split_manifest.json"
    split_manifest.write_text(json.dumps({
        "development_index_start": 0,
        "development_index_end": n_rows - 1,
        "task_id": "rehearsal-test",
    }))

    staged_csv = tmp_path / "staged" / "development.csv"
    stage_development(
        source_csv,
        split_manifest,
        staged_csv,
        sha256(source_csv),
        sha256(split_manifest),
    )
    assert staged_csv.is_file()
    assert staged_csv.with_suffix(".stage.json").is_file()

    # Create 2 pinned folds
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    locked_folds = [(tr.tolist(), va.tolist()) for tr, va in kf.split(range(n_rows))]

    # 2. Setup mock responses for 4 turns
    mock_responses = [
        # Turn 0: ANALYZE
        {
            "id": "rehearsal-turn-0",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({
                "thought_rationale": "Analyze the dataset diagnostics to plan features.",
                "action_type": "ANALYZE",
                "parameters": {},
            }),
            "tool_calls": [],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30, "total_tokens": 230},
            "latency_ms": 20,
        },
        # Turn 1: HYPOTHESIS_TRANSFORM (TEXT_TFIDF_SVD)
        {
            "id": "rehearsal-turn-1",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({
                "thought_rationale": "Apply TF-IDF and SVD on catalog content text.",
                "action_type": "HYPOTHESIS_TRANSFORM",
                "parameters": {
                    "action": "TEXT_TFIDF_SVD",
                    "parameters": {"columns": ["catalog_content"]},
                },
            }),
            "tool_calls": [],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340},
            "latency_ms": 30,
        },
        # Turn 2: EVALUATE_PORTFOLIO
        {
            "id": "rehearsal-turn-2",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({
                "thought_rationale": "Evaluate candidate regression models with log1p target transform.",
                "action_type": "EVALUATE_PORTFOLIO",
                "parameters": {
                    "families": ["penalized_linear", "hist_gbdt"],
                    "target_transform": "log1p",
                },
            }),
            "tool_calls": [],
            "usage": {"prompt_tokens": 450, "completion_tokens": 50, "total_tokens": 500},
            "latency_ms": 40,
        },
        # Turn 3: FINALIZE_AND_STOP
        {
            "id": "rehearsal-turn-3",
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "content": json.dumps({
                "thought_rationale": "Lock accepted candidate and finish autonomous exploration.",
                "action_type": "FINALIZE_AND_STOP",
                "parameters": {},
            }),
            "tool_calls": [],
            "usage": {"prompt_tokens": 600, "completion_tokens": 30, "total_tokens": 630},
            "latency_ms": 20,
        },
    ]

    mock_file = tmp_path / "mock_provider.json"
    mock_file.write_text(json.dumps(mock_responses), encoding="utf-8")

    # 3. Setup Mock Campaign Authority
    class RehearsalAuthority:
        def __init__(self):
            self.reservations = []
            self.settlements = []
            self.ledger_path = tmp_path / "rehearsal_ledger.json"
            self.campaign_ceiling = Decimal("0.30")

        def get_status(self):
            return {
                "total_billed_usd": "0.00100",
                "uncertain_reserved_usd": "0.00000",
                "remaining_campaign_usd": "0.19000",
                "campaign_ceiling_usd": "0.30000",
            }

        def reserve(self, run_id: str, req_id: str, amount):
            self.reservations.append((run_id, req_id, amount))
            return amount

        def settle(self, run_id: str, req_id: str, actual):
            self.settlements.append((run_id, req_id, actual))

        def mark_uncertain(self, run_id: str, req_id: str, amount):
            pass

    authority = RehearsalAuthority()

    # 4. Setup RustGatewayAdapter with policy approval enabled
    harness_state = tmp_path / "harness_state"
    adapter = RustGatewayAdapter(
        state_dir=harness_state,
        project_id="rehearsal_proj",
        mock_path=mock_file,
        pricing={"verified": True, "input_per_million": "0.038", "output_per_million": "0.50"},
        require_policy_approval=True,
        approval_wait_seconds=10,
    )

    # 5. Background Operator thread to inspect and approve outbound views
    stop_operator = threading.Event()
    approved_shas = []

    def operator_daemon():
        db_path = harness_state / "state.sqlite"
        while not stop_operator.is_set():
            if db_path.is_file():
                try:
                    with sqlite3.connect(db_path) as db:
                        rows = db.execute(
                            "SELECT request_sha256, approved FROM outbound_views WHERE project='rehearsal_proj' AND approved=0"
                        ).fetchall()
                        for req_sha, app in rows:
                            # Inspect view using harness inspect-outbound-view
                            cmd_inspect = [
                                str(harness_bin),
                                "inspect-outbound-view",
                                "--state", str(harness_state),
                                "--project", "rehearsal_proj",
                                "--request-sha256", req_sha,
                            ]
                            res_insp = subprocess.run(cmd_inspect, capture_output=True, text=True, timeout=5)
                            assert res_insp.returncode == 0, f"inspect failed: {res_insp.stderr}"

                            # Approve view using harness approve-outbound-view
                            cmd_app = [
                                str(harness_bin),
                                "approve-outbound-view",
                                "--state", str(harness_state),
                                "--project", "rehearsal_proj",
                                "--request-sha256", req_sha,
                                "--actor", "rehearsal-operator",
                                "--approved", "true",
                            ]
                            res_app = subprocess.run(cmd_app, capture_output=True, text=True, timeout=5)
                            assert res_app.returncode == 0, f"approve failed: {res_app.stderr}"
                            approved_shas.append(req_sha)
                except Exception:
                    pass
            time.sleep(0.1)

    op_thread = threading.Thread(target=operator_daemon, daemon=True)
    op_thread.start()

    # 6. Run AutonomousAgentLoop
    workspace = tmp_path / "workspace"
    loop = AutonomousAgentLoop(
        workspace_root=workspace,
        project_id="rehearsal_proj",
        session_id="rehearsal_session_1",
        campaign_authority=authority,
        dataset_path=staged_csv,
        target_column="price",
        task_type="regression",
        metric_name="smape",
        metric_direction="minimize",
        locked_folds=locked_folds,
        model_name="deepseek/deepseek-v4.1-flash-1017",
        endpoint="wafer/us",
        max_turns=4,
        per_run_budget_usd=0.05,
        pricing={"verified": True, "input_per_million": "0.038", "output_per_million": "0.50"},
        provider_caller=adapter,
        provider_is_fixture=True,
    )

    try:
        loop_state = loop.run()
    finally:
        stop_operator.set()
        op_thread.join(timeout=2)

    # 7. Verify Turn Execution & Stop Reason
    for t in loop_state["turns"]:
        print("TURN:", t["turn_index"], "action:", t["proposed_action"].get("action_type"), "disp:", t.get("disposition"), "error:", t.get("error_message") or t.get("error"), "stop:", loop_state.get("stop_reason"))
    assert loop_state["turns_completed"] == 4
    assert loop_state["stop_reason"] == "FINALIZE_AND_STOP"
    assert len(approved_shas) == 4, f"Expected 4 approved outbound views, got {len(approved_shas)}"

    # 8. Verify Candidate Acceptance & Critic Audit
    accepted_id = loop_state.get("accepted_candidate_id")
    assert accepted_id is not None
    assert accepted_id.startswith("cand_")
    audit = loop_state.get("accepted_audit", {})
    assert audit.get("status") == "APPROVED"
    assert loop_state.get("best_cv_score") is not None

    # 9. Fit the Accepted Candidate & Export
    fitted_pipeline = loop.fit_accepted_candidate()
    assert fitted_pipeline is not None
    assert hasattr(fitted_pipeline, "predict")

    # Save fitted model
    model_artifact = workspace / "accepted_model.joblib"
    import joblib
    joblib.dump(fitted_pipeline, model_artifact)
    assert model_artifact.is_file()

    # 10. Verify Prediction Parity & Holdout Independence
    features_only = df_raw[["catalog_content"]]
    predictions = fitted_pipeline.predict(features_only)
    assert len(predictions) == n_rows
    assert np.all(np.isfinite(predictions))
    # Prices must be positive
    assert np.all(predictions > 0)

    # 11. Export bundle and verify parity via verify_export_parity
    export_dir = workspace / "exported_bundle"
    export_manifest = export_model(
        model=fitted_pipeline,
        export_dir=export_dir,
        task="regression",
        features=["catalog_content"],
        formats=["joblib"],
    )
    assert (export_dir / "frozen_model.joblib").is_file()
    assert (export_dir / "export_manifest.json").is_file()

    test_inputs = features_only.iloc[:10].to_dict(orient="records")
    parity_result = verify_export_parity(
        model=fitted_pipeline,
        export_dir=export_dir,
        test_inputs=test_inputs,
        task="regression",
    )
    assert parity_result["verified"] is True
    assert "joblib" in parity_result["formats"]

    # 12. Verify Durable Handoff and Journey Documentation
    handoff_json = workspace / "handoff.json"
    handoff_md = workspace / "HANDOFF.md"
    trace_jsonl = workspace / "autonomous_trace.jsonl"
    assert handoff_json.is_file()
    assert handoff_md.is_file()
    assert trace_jsonl.is_file()

    handoff_data = json.loads(handoff_json.read_text())
    assert handoff_data["accepted_candidate_id"] == accepted_id
    assert handoff_data["stop_reason"] == "FINALIZE_AND_STOP"
    assert len(handoff_data["apex_trajectory"]) >= 1

    md_content = handoff_md.read_text()
    assert "Model Creation Journey (Apex Iterations)" in md_content
    assert "Accepted candidate" in md_content

    # 13. Verify Failure Case: Unapproved View Fails Closed
    unapproved_harness_state = tmp_path / "unapproved_harness_state"
    unapproved_adapter = RustGatewayAdapter(
        state_dir=unapproved_harness_state,
        project_id="unapproved_proj",
        mock_path=mock_file,
        pricing={"verified": True, "input_per_million": "0.038", "output_per_million": "0.50"},
        require_policy_approval=True,
        approval_wait_seconds=1,  # Short timeout with no operator approval
    )
    unapproved_loop = AutonomousAgentLoop(
        workspace_root=tmp_path / "unapproved_ws",
        project_id="unapproved_proj",
        session_id="unapproved_session_1",
        campaign_authority=authority,
        dataset_path=staged_csv,
        target_column="price",
        task_type="regression",
        metric_name="smape",
        metric_direction="minimize",
        locked_folds=locked_folds,
        max_turns=1,
        provider_caller=unapproved_adapter,
        provider_is_fixture=True,
    )
    with pytest.raises(TimeoutError, match="Outbound view approval timed out"):
        unapproved_loop.run()
