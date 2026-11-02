"""Offline observational probes, not acceptance assertions. Uses disposable synthetic state.

Run from the repository after cargo build -p mlrescurix-harness.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "python/src"), str(ROOT / "python/tests")]
from mlrescurix_worker.scientific.governed_research import GovernedResearchClient, MockSearchProvider
from mlrescurix_worker.scientific.operational_budgets import OperationalBudgets


def search_probes(scratch):
    budget = OperationalBudgets(project_id="probe", db_path=str(scratch / "search.sqlite"))
    _, attempt = budget.admit_search("tabular models", "tabular models", "mock")
    budget.settle_search(attempt, "success")
    budget.settle_search(attempt, "timeout", "late contradictory replay")
    import sqlite3
    with sqlite3.connect(budget.db_path) as conn:
        status = conn.execute("SELECT status FROM operational_search_attempts WHERE attempt_id=?", (attempt,)).fetchone()[0]
    provider = MockSearchProvider()
    ungoverned = GovernedResearchClient(provider=provider).research_topic("tabular models", budgets=None)
    return {"terminal_success_after_contradictory_settlement": status,
            "without_admission_authority_status": ungoverned["status"],
            "without_admission_authority_provider_calls": len(provider.recorded_queries)}


def source_rebinding_probe(scratch):
    data = scratch / "data.csv"
    data.write_text("x,target\n" + "".join(f"{i / 30},{i / 15}\n" for i in range(30)))
    spec = scratch / "spec.json"
    spec.write_text(json.dumps({"project_id": "source_probe", "dataset_path": str(data), "target_column": "target"}))
    state = scratch / "state"
    command = [str(ROOT / "target/debug/mlrescurix-harness"), "autonomous-create", "--offline-fixture", "--state", str(state), "--spec", str(spec)]
    first = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if first.returncode:
        return {"initial_exit": first.returncode, "stderr": first.stderr}
    before = json.loads(first.stdout)
    branch_source = next((state / "controller/workflows/source_probe/branches").glob("branch_*/solution.py"))
    original = branch_source.read_text()
    altered = original.replace("val = sum(w[j] * phi[j] for j in range(dim))", "val = 1000000.0")
    assert altered != original
    branch_source.write_text(altered)
    bundle = Path(before["bundle_dir"])
    (bundle / "manifest.json").unlink()
    second = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    result = {"repack_exit": second.returncode, "stderr": second.stderr.strip()}
    if second.returncode == 0:
        after = json.loads(second.stdout)
        worker = state / "worker_workspace"
        preds = scratch / "changed_source_predictions.json"
        replay = subprocess.run([sys.executable, str(worker / "solution.py"), "infer", "--features-in", str(worker / "val_features.csv"),
                                 "--weights-in", str(worker / "weights.json"), "--predictions-out", str(preds)], capture_output=True, text=True)
        assert replay.returncode == 0, replay.stderr
        result.update(same_validation_score=before["validation_score"] == after["validation_score"],
                      same_holdout_score=before["lockbox_score"] == after["lockbox_score"],
                      same_weights=before["weights_sha256"] == after["weights_sha256"],
                      exported_source_changed=(bundle / "solution.py").read_text() != original,
                      new_source_predictions_differ_from_scored_predictions=json.loads(preds.read_text()) != json.loads((worker / "val_predictions.json").read_text()))
    return result


def wheel_probe(scratch):
    package = scratch / "python"
    shutil.copytree(ROOT / "python", package, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "build", ".pytest_cache"))
    wheels = scratch / "wheels"
    build = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps", "--no-build-isolation",
                            str(package), "--wheel-dir", str(wheels)], text=True, capture_output=True)
    if build.returncode:
        return {"build_exit": build.returncode, "stderr": build.stderr}
    wheel = next(wheels.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        schemas = [name for name in archive.namelist() if name.endswith("actions.json")]
    code = "import sys; sys.path.insert(0, sys.argv[1]); import mlrescurix_worker.scientific.autonomous_loop"
    imported = subprocess.run([sys.executable, "-I", "-c", code, str(wheel)], cwd=scratch, text=True, capture_output=True)
    return {"build_exit": 0, "action_schema_files": schemas, "isolated_wheel_import_exit": imported.returncode,
            "error_tail": imported.stderr.splitlines()[-2:]}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        observations = {"search": search_probes(scratch), "source_rebinding": source_rebinding_probe(scratch), "built_wheel": wheel_probe(scratch)}
    (HERE / "remaining_gap_observations.json").write_text(json.dumps(observations, indent=2) + "\n")
    print(json.dumps(observations, indent=2))
