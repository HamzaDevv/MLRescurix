"""Offline observational probes; outputs are findings, not acceptance assertions.

Run from the repository: python3 docs/evidence/refactor/S1/red_team/reproduce_remaining_gaps.py
Uses only synthetic reference evidence, mock providers and disposable state copies.
The resulting observations may change after the S2 repairs.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "python/src"), str(ROOT / "python/tests")]

from test_s1_contracts_and_parity import make_action, make_test_loop
from mlrescurix_worker.scientific.candidate_pipeline import identity, resolve_effective_config
from mlrescurix_worker.scientific.governed_research import GovernedResearchClient, MockSearchProvider
from mlrescurix_worker.scientific.model_zoo import create_estimator
from mlrescurix_worker.scientific.operational_budgets import OperationalBudgets


def python_probes():
    out = {}
    for family in ["xgboost", "penalized_linear"]:
        spec = {"families": [family], "weights": [1], "transforms": [], "seed": 42}
        resolved = resolve_effective_config(spec, "regression")["resolved_parameters"][family]
        actual = create_estimator(family, "regression", 42).get_params()
        out[family] = {"declared": resolved, "actual_n_estimators": actual.get("n_estimators"),
                       "actual_max_iter": actual.get("model__max_iter")}
    a = {"families": ["penalized_linear"], "weights": [1], "transforms": [], "seed": 42,
         "model_parameters": {"penalized_linear": {"alpha": 1}}}
    b = json.loads(json.dumps(a))
    b["model_parameters"]["penalized_linear"]["max_iter"] = 2000
    out["equal_resolved_configs"] = (resolve_effective_config(a, "regression")["resolved_parameters"] ==
                                     resolve_effective_config(b, "regression")["resolved_parameters"])
    for spec in [a, b]:
        spec["effective_configuration"] = resolve_effective_config(spec, "regression")
    out["equal_execution_keys"] = identity(a) == identity(b)
    budget = OperationalBudgets(web_search_limit=1, search_connected=True)
    GovernedResearchClient(provider=MockSearchProvider()).research_topic("robust tabular regression", budgets=budget)
    restored = OperationalBudgets(web_search_limit=1, search_connected=True)
    restored.restore_operational_budgets([])
    out["search_after_dispatch"] = budget.web_search_used
    out["search_after_restart_before_turn_commit"] = restored.web_search_used
    with tempfile.TemporaryDirectory() as scratch:
        calls = []

        def failed_evaluator(spec, frame, folds, metric):
            calls.append(spec["families"][0])
            raise RuntimeError("synthetic infrastructure failure")

        loop, _ = make_test_loop(Path(scratch), [make_action("EVALUATE_PORTFOLIO", families=["hist_gbdt", "penalized_linear"])],
                                 task_type="regression", evaluator=failed_evaluator)
        state = loop.run()
        out["siblings_attempted_after_first_failure"] = calls
        out["turn_disposition"] = state["turns"][0]["disposition"]
    return out


def resume_probes():
    evidence = HERE.parent
    original = json.loads((evidence / "cli_outcome.json").read_text())
    out = {}
    for case in ["changed_problem", "changed_weights"]:
        with tempfile.TemporaryDirectory() as scratch:
            state = Path(scratch) / "state"
            shutil.copytree(evidence / "cli_state", state)
            spec = json.loads((evidence / "problem_spec.json").read_text())
            spec["dataset_path"] = str(ROOT / spec["dataset_path"])
            if case == "changed_problem":
                spec["metric"], spec["seed"] = "rmse", 7
            else:
                path = state / "worker_workspace/weights.json"
                weights = json.loads(path.read_text())
                weights["weights"][0] += 100
                path.write_text(json.dumps(weights))
                for manifest in state.glob("delivery_bundle_*/manifest.json"):
                    manifest.unlink()
            spec_path = Path(scratch) / "spec.json"
            spec_path.write_text(json.dumps(spec))
            proc = subprocess.run([str(ROOT / "target/debug/mlrescurix-harness"), "autonomous-create", "--offline-fixture",
                                   "--state", str(state), "--spec", str(spec_path)], capture_output=True, text=True, cwd=ROOT)
            result = {"exit_code": proc.returncode, "stderr": proc.stderr.strip()}
            if proc.returncode == 0:
                got = json.loads(proc.stdout)
                result.update(requested_metric=spec["metric"], requested_seed=spec["seed"],
                              reported_protocol_sha256=got["protocol_sha256"], original_protocol_sha256=original["protocol_sha256"],
                              same_validation_score=got["validation_score"] == original["validation_score"],
                              same_lockbox_score=got["lockbox_score"] == original["lockbox_score"],
                              same_weights_sha256=got["weights_sha256"] == original["weights_sha256"])
            out[case] = result
    return out


if __name__ == "__main__":
    for name, result in [("python_boundary_probes.json", python_probes()), ("resume_boundary_probes.json", resume_probes())]:
        (HERE / name).write_text(json.dumps(result, indent=2) + "\n")
        print(name, json.dumps(result, indent=2))
