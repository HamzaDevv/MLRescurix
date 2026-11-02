"""Regression checks at real autonomy/evidence boundaries (all provider calls offline)."""
import copy
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop, RecoveryRequired
from mlrescurix_worker.scientific.candidate_pipeline import CandidatePipeline, evaluate_candidate, identity, score_metric


class Budget:
    def __init__(self): self.requests = []; self.billed = Decimal(0)
    def reserve(self, run, request, amount): self.requests.append(request)
    def settle(self, run, request, cost): self.billed += cost
    def mark_uncertain(self, *args): pass
    def get_status(self): return {"campaign_ceiling_usd": ".30", "remaining_campaign_usd": str(Decimal('.30') - self.billed)}


def make_loop(tmp_path, actions, **kwargs):
    rng = np.random.default_rng(41)
    y = np.tile([0, 1], 60)
    # Missingness itself carries signal; categorical raw value is otherwise constant.
    frame = pd.DataFrame({"age": rng.normal(size=len(y)), "cabin": np.where(y == 0, None, "A"), "y": y})
    path = tmp_path / "dev.csv"
    if not path.exists(): frame.to_csv(path, index=False)
    prompts = []
    def provider(messages, max_tokens):
        prompts.append(json.loads(messages[-1]["content"]))
        action = actions[len(prompts) - 1]
        return {"content": json.dumps(action), "cost_usd": 0}
    folds = list(StratifiedKFold(2, shuffle=True, random_state=42).split(frame, y))
    loop = AutonomousAgentLoop(tmp_path / "workspace", "p", "s", kwargs.pop("budget", Budget()), path, "y",
        locked_folds=folds, max_turns=len(actions), provider_caller=provider, provider_is_fixture=True, **kwargs)
    return loop, prompts


def action(kind, **params): return {"thought_rationale": "Follow observed evidence", "action_type": kind, "parameters": params}


def test_transform_is_fold_local_and_mutates_later_candidate(tmp_path):
    traces = []
    def evaluator(spec, frame, folds, metric):
        traces.append(copy.deepcopy(spec))
        result = evaluate_candidate(spec, frame, "y", "binary_classification", folds, metric)
        # Paired branch wiring, independent of whether this toy indicator improves an already categorical baseline.
        if spec["transforms"]: result["cv_score"] += .01
        return result
    loop, prompts = make_loop(tmp_path, [action("HYPOTHESIS_TRANSFORM", action="ADD_MNAR_INDICATOR", columns=["cabin"]),
        action("EVALUATE_PORTFOLIO", families=["penalized_linear"])], evaluator=evaluator)
    state = loop.run()
    assert state["turns"][0]["measurements"]["pipeline_updated"] is True
    assert traces[-1]["transforms"][0]["action"] == "ADD_MNAR_INDICATOR"
    assert "cabin__missing" in state["current_active_features"]
    feedback = prompts[1]["turn_history"][0]
    assert feedback["measurements"]["paired_metric_delta"] == pytest.approx(.01)
    assert feedback["parameters"]["columns"] == ["cabin"]
    assert feedback["measurements"]["after"]["fold_receipts"]
    assert "y" not in state["current_active_features"]
    assert state["iterations"][0]["critic_status"] == "NOT_AUDITED"
    with pytest.raises(ValueError, match="not eligible"):
        loop.select_apex_iteration(1)


def test_raw_candidate_preprocessor_never_fits_validation_categories():
    train = pd.DataFrame({"x": [1., np.nan, 3., 4.], "cat": ["a", "a", "b", "b"], "y": [0, 0, 1, 1]})
    validation = pd.DataFrame({"x": [9999.], "cat": ["validation_only"], "y": [0]})
    spec = {"families": ["penalized_linear"], "weights": [1], "transforms": [], "seed": 42}
    pipeline = CandidatePipeline(spec, "y", "binary_classification").fit(train)
    assert pipeline.imputer_.statistics_[0] == 3
    assert "validation_only" not in pipeline.encoder_.categories_[0]
    assert pipeline.predict(validation).shape == (1,)
    assert pipeline.imputer_.statistics_[0] == 3


def test_non_object_response_is_rejected_then_error_in_next_prompt(tmp_path):
    loop, prompts = make_loop(tmp_path, [[], action("HANDOFF", reason="repair", next_steps=["Check malformed proposal"])])
    state = loop.run()
    assert state["turns"][0]["disposition"] == "rejected"
    assert "JSON object" in prompts[1]["turn_history"][0]["error"]
    handoff = json.loads((loop.workspace_root / "handoff.json").read_text())
    assert handoff["stop_reason"] == "HANDOFF"
    assert handoff["next_steps"] == ["Check malformed proposal"]


def test_dynamic_script_write_error_repair_run_feedback(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific import python_tool
    executed = []
    def tool(path, root, **kwargs):
        executed.append(path.read_text())
        return {"ok": len(executed) > 1, "status": "script_error" if len(executed) == 1 else "success",
            "stderr": "NameError: missing" if len(executed) == 1 else "", "stdout": "42" if len(executed) > 1 else "", "artifacts": []}
    monkeypatch.setattr(python_tool, "run_python_script", tool)
    actions = [action("WRITE_PYTHON", filename="analysis.py", code="print(missing)"), action("RUN_PYTHON", filename="analysis.py"),
        action("WRITE_PYTHON", filename="analysis.py", code="print(42)"), action("RUN_PYTHON", filename="analysis.py"), action("HANDOFF", reason="done")]
    loop, prompts = make_loop(tmp_path, actions, outbound_script_output_approved=True)
    loop.run()
    assert "NameError" in prompts[2]["turn_history"][-1]["measurements"]["python_result"]["stderr"]
    assert executed == ["print(missing)", "print(42)"]
    assert prompts[4]["turn_history"][-1]["measurements"]["python_result"]["stdout"] == "42"
    assert loop.state["accepted_candidate_id"] is None


def test_script_execution_uses_checked_snapshot_when_original_changes(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific import python_tool
    observed = []
    def tool(path, root, **kwargs):
        original = root / "analysis.py"
        original.write_text("print('changed after staging')")
        observed.append(path.read_text())
        return {"status": "success", "stdout": "", "stderr": "", "artifacts": []}
    monkeypatch.setattr(python_tool, "run_python_script", tool)
    loop, _ = make_loop(tmp_path, [action("WRITE_PYTHON", filename="analysis.py", code="print('checked')"),
                                  action("RUN_PYTHON", filename="analysis.py")])
    loop.run()
    assert observed == ["print('checked')"]
    assert not list((loop.workspace_root / "scripts").glob(".run-*"))


def test_script_snapshot_rejects_fifo_without_blocking(tmp_path):
    import os
    loop, _ = make_loop(tmp_path, [action("WRITE_PYTHON", filename="analysis.py", code="print(1)"),
                                  action("RUN_PYTHON", filename="analysis.py")])
    loop.execute_turn(0, loop.build_diagnostic_dossier())
    path = loop.workspace_root / "scripts" / "analysis.py"
    path.unlink()
    os.mkfifo(path)
    receipt = loop.execute_turn(1, loop.build_diagnostic_dossier())
    assert receipt["disposition"] == "rejected"
    assert "bounded regular file" in receipt["error_msg"]


def test_script_staged_data_hash_is_checked_before_launch(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific import execution_service, python_tool
    original_copy = execution_service.shutil.copyfile
    def corrupt_copy(src, dst):
        original_copy(src, dst)
        with open(dst, "a") as stream:
            stream.write("corrupted\n")
    monkeypatch.setattr(execution_service.shutil, "copyfile", corrupt_copy)
    monkeypatch.setattr(python_tool, "run_python_script", lambda *a, **k: pytest.fail("must reject before Docker"))
    loop, _ = make_loop(tmp_path, [action("WRITE_PYTHON", filename="analysis.py", code="print(1)"),
                                  action("RUN_PYTHON", filename="analysis.py")])
    state = loop.run()
    assert state["turns"][1]["disposition"] == "rejected"
    assert "Development data changed" in state["turns"][1]["error_msg"]


def test_staged_script_and_development_data_run_in_docker(tmp_path):
    import shutil
    import subprocess
    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "version"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=5).returncode:
        pytest.skip("Docker daemon is unavailable")
    code = ("import os, pathlib\n"
            "data=pathlib.Path(os.environ['MLRESCURIX_DATA'])/'development.csv'\n"
            "out=pathlib.Path(os.environ['MLRESCURIX_OUTPUTS'])/'count.txt'\n"
            "out.write_text(str(len(data.read_text().splitlines()) - 1))\n")
    loop, _ = make_loop(tmp_path, [action("WRITE_PYTHON", filename="analysis.py", code=code),
                                  action("RUN_PYTHON", filename="analysis.py")])
    state = loop.run()
    result = state["turns"][1]["measurements"]["python_result"]
    assert result["status"] == "success", result
    assert (Path(result["artifact_dir"]) / "count.txt").read_text() == "120"


def test_provider_crash_is_not_redispatched_on_restart(tmp_path):
    budget = Budget()
    loop, _ = make_loop(tmp_path, [], budget=budget)
    def crash(*args): raise RuntimeError("lost response")
    loop.provider_caller = crash
    with pytest.raises(RuntimeError, match="lost response"): loop.execute_turn(0, {})
    assert len(budget.requests) == 1
    with pytest.raises(RecoveryRequired, match="Interrupted dispatch"): make_loop(tmp_path, [], budget=budget)
    assert len(budget.requests) == 1


def test_changed_dataset_and_corrupt_checkpoint_fail_closed(tmp_path):
    loop, _ = make_loop(tmp_path, [])
    frame = pd.read_csv(loop.dataset_path); frame.loc[0, "age"] = 99; frame.to_csv(loop.dataset_path, index=False)
    with pytest.raises(RecoveryRequired, match="binding mismatch"): make_loop(tmp_path, [])
    loop.state_file.write_text("not-json")
    with pytest.raises(RecoveryRequired, match="Corrupt checkpoint"): make_loop(tmp_path, [])


def test_finalize_without_acceptance_rejected(tmp_path):
    loop, _ = make_loop(tmp_path, [action("FINALIZE_AND_STOP")])
    state = loop.run()
    assert state["turns"][0]["disposition"] == "rejected"
    assert state["accepted_candidate_id"] is None
    with pytest.raises(ValueError, match="No accepted"): loop.fit_accepted_candidate()


def test_metric_semantics_and_roc_auc():
    assert score_metric("rmse", [1, 4], [1, 2]) == pytest.approx(np.sqrt(2))
    assert score_metric("rmse_log_price", [1, 4], [1, 2]) == pytest.approx(np.log(2) / np.sqrt(2))
    assert score_metric("roc_auc", [0, 1], [0, 1], np.array([[.8,.2],[.1,.9]]), np.array([0,1])) == 1
    with pytest.raises(ValueError): score_metric("unknown", [0], [0])


def test_blend_uses_requested_weights_and_refit_matches_audited_spec(tmp_path, monkeypatch):
    calls = []
    def evaluator(spec, frame, folds, metric):
        calls.append(copy.deepcopy(spec))
        return {"cv_score": .8, "cv_scores": [.79,.81], "fold_receipts": [{"critic_validation_score": .79}, {"critic_validation_score": .81}],
            "critic_train_score": .83, "critic_validation_score": .8, "feature_names": ["age", "cabin"], "candidate_identity": identity(spec)}
    class Receipt:
        def __init__(self, candidate): self.candidate = candidate
        def to_dict(self): return {"status": "APPROVED", "candidate_id": self.candidate["candidate_id"], "blocking_flaws": []}
    loop, _ = make_loop(tmp_path, [action("BLEND_AND_AUDIT", families=["penalized_linear", "hist_gbdt"], weights=[.9,.1])], evaluator=evaluator)
    monkeypatch.setattr(loop.critic_engine, "audit", lambda **kw: Receipt(kw["candidate"]))
    state = loop.run()
    pipeline = loop.fit_accepted_candidate()
    assert pipeline.candidate_identity == state["candidate_identity"] == identity(calls[0])
    assert pipeline.spec["weights"] == [.9,.1]
    assert state["accepted_audit"]["candidate_id"] == state["accepted_candidate_id"]
    pipeline.spec["weights"] = [1,1]  # modifying a returned object cannot alter persisted state
    assert state["accepted_candidate"]["weights"] == [.9,.1]


def test_required_locked_folds_cannot_fall_back(tmp_path):
    pd.DataFrame({"x": range(10), "y": [0,1]*5}).to_csv(tmp_path / "dev.csv", index=False)
    with pytest.raises(ValueError, match="Pinned development folds required"):
        AutonomousAgentLoop(tmp_path / "ws", "p", "s", Budget(), tmp_path / "dev.csv", "y", require_locked_folds=True, provider_is_fixture=True)


def test_stale_controller_refreshes_checkpoint_without_repeating_calls(tmp_path):
    actions = [action("ANALYZE")]
    first, prompts = make_loop(tmp_path, actions)
    second, second_prompts = make_loop(tmp_path, actions)
    first.run()
    second.run()
    assert len(prompts) == 1
    assert second_prompts == []


def test_freeform_script_output_stays_local_without_outbound_approval(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific import python_tool
    monkeypatch.setattr(python_tool, "run_python_script", lambda *a, **kw: {"status":"script_error", "stdout":"PRIVATE_ROW", "stderr":"ValueError: PRIVATE_ROW"})
    loop, prompts = make_loop(tmp_path, [action("WRITE_PYTHON", filename="a.py", code="print(1)"), action("RUN_PYTHON", filename="a.py"), action("HANDOFF")])
    loop.run()
    assert "PRIVATE_ROW" not in json.dumps(prompts[-1])
    assert "ValueError" in json.dumps(prompts[-1])
    assert loop.state["turns"][1]["measurements"]["python_result"]["stdout"] == "PRIVATE_ROW"


def test_unapproved_script_result_does_not_upload_other_worker_controlled_fields(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific import python_tool
    secret = "PrivateCustomerSecret"
    monkeypatch.setattr(python_tool, "run_python_script", lambda *a, **kw: {
        "status": secret, "stdout": secret, "stderr": f"{secret}Error: {secret}",
        "error": secret, "artifacts": [{"path": secret}], "artifact_dir": secret,
        "resources": {"detail": secret}, "extra": secret})
    loop, prompts = make_loop(tmp_path, [action("WRITE_PYTHON", filename="a.py", code="print(1)"),
        action("RUN_PYTHON", filename="a.py"), action("HANDOFF")])
    loop.run()
    result = prompts[-1]["turn_history"][1]["measurements"]["python_result"]
    assert secret not in json.dumps(result)
    assert result["status"] == "unknown"
    assert loop.state["turns"][1]["measurements"]["python_result"]["artifact_dir"] == secret


def test_live_callback_cannot_bypass_verified_pricing(tmp_path):
    loop, _ = make_loop(tmp_path, [])
    loop.provider_is_fixture = False
    with pytest.raises(ValueError, match="verified"): loop._reservation([], 100)


def test_valid_json_checkpoint_with_invalid_shape_is_rejected(tmp_path):
    loop, _ = make_loop(tmp_path, [])
    loop.state_file.write_text(json.dumps({"binding": loop.binding, "turns": []}))
    with pytest.raises(RecoveryRequired, match="schema"): make_loop(tmp_path, [])


@pytest.mark.parametrize("limit", ["memory", "wall"])
def test_real_worker_is_terminated_on_resource_limit(tmp_path, monkeypatch, limit):
    import subprocess
    import sys
    from mlrescurix_worker.scientific import autonomous_loop
    loop, _ = make_loop(tmp_path, [])
    actual_popen = subprocess.Popen
    children = []
    def sleeping_worker(*args, **kwargs):
        process = actual_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(autonomous_loop.subprocess, "Popen", sleeping_worker)
    monkeypatch.setattr(loop.sampler, "sample_now", lambda: {"workers":{"rss_bytes": 9999999999 if limit == "memory" else 1}})
    loop.evaluation_timeout_seconds = .2
    with pytest.raises(RuntimeError, match="RSS budget|wall-time"):
        loop._evaluate(loop.state["active_spec"])
    assert children and children[0].poll() is not None


def test_multiple_failed_worker_executions_keep_each_resource_receipt(tmp_path, monkeypatch):
    import subprocess
    import sys
    from mlrescurix_worker.scientific import autonomous_loop
    loop, _ = make_loop(tmp_path, [])
    actual_popen = subprocess.Popen
    monkeypatch.setattr(autonomous_loop.subprocess, "Popen",
                        lambda *a, **kw: actual_popen([sys.executable, "-c", "raise SystemExit(1)"], **kw))
    loop._turn_worker_resource_receipts = []
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Candidate worker failed"):
            loop._evaluate(loop.state["active_spec"])
    assert len(loop._turn_worker_resource_receipts) == 2


def test_log_target_pipeline_inverts_predictions_and_accepts_raw_records():
    frame = pd.DataFrame({"x": np.arange(1, 21), "cat": ["a", "b"]*10, "y": np.exp(np.arange(1, 21)/10)})
    spec = {"families":["penalized_linear"],"weights":[1],"transforms":[],"seed":42,"target_transform":"log"}
    pipeline = CandidatePipeline(spec, "y", "regression").fit(frame)
    assert not hasattr(pipeline, "predict_proba")
    preds = pipeline.predict(frame.drop(columns="y").to_dict("records"))
    assert np.all(preds > 0)
    assert score_metric("rmse_log_price", frame.y, preds) < .2


def test_abrupt_worker_exit_is_feedback_and_agent_can_continue(tmp_path, monkeypatch):
    import subprocess
    import sys
    from mlrescurix_worker.scientific import autonomous_loop
    loop, prompts = make_loop(tmp_path, [action("EVALUATE_PORTFOLIO", families=["hist_gbdt"]), action("ANALYZE")])
    actual_popen = subprocess.Popen
    def killed_worker(*args, **kwargs):
        return actual_popen([sys.executable, "-c", "import os,signal; os.kill(os.getpid(),signal.SIGKILL)"], **kwargs)
    monkeypatch.setattr(autonomous_loop.subprocess, "Popen", killed_worker)
    state = loop.run()
    assert state["turns"][0]["disposition"] == "rejected"
    assert "Candidate worker failed" in prompts[1]["turn_history"][0]["error"]
    assert state["turns"][1]["disposition"] == "applied"
    assert state["accepted_candidate_id"] is None


def test_temporal_contract_rejects_shuffled_pinned_folds(tmp_path):
    frame = pd.DataFrame({"time":range(10), "x":range(10), "y":[0,1]*5})
    frame.to_csv(tmp_path / "dev.csv", index=False)
    with pytest.raises(ValueError, match="temporal order"):
        AutonomousAgentLoop(tmp_path / "ws", "p", "s", Budget(), tmp_path / "dev.csv", "y", provider_is_fixture=True,
            locked_folds=[([0,2,4,6,8],[1,3,5,7,9])], require_locked_folds=True,
            topology={"temporal_col":"time", "cv_scheme":"purged_time_series_split"})


def test_script_repair_locations_are_bounded_without_exporting_exception_data(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific import python_tool
    monkeypatch.setattr(python_tool, 'run_python_script', lambda *a, **k: {
        'status': 'script_error', 'stdout': 'PRIVATE_ROW',
        'stderr': 'File "/scripts/analysis.py", line 1\nNameError: PRIVATE_ROW', 'artifacts': []})
    loop, prompts = make_loop(tmp_path, [action('WRITE_PYTHON', filename='analysis.py', code='print(missing)'),
        action('RUN_PYTHON', filename='analysis.py'), action('HANDOFF', reason='repair required')])
    loop.run()
    result = prompts[2]['turn_history'][-1]['measurements']['python_result']
    assert result['repair_locations'] == [{'script': 'analysis.py', 'line': 1}]
    assert result['stderr'] == 'NameError'
    assert 'PRIVATE_ROW' not in json.dumps(result)
