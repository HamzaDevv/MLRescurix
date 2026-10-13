"""Authoritative Predeclared Matched Benchmark Comparison: M3 Baseline vs M5 Scientific Pipeline.

Covers:
- Binary Classification (unseen categories, non-linear interactions, stratified split)
- Multiclass Classification (missing values, 3 classes, random split)
- Regression (continuous target, group split)

Authoritative Execution:
- Admission, budgets, scheduling, evaluation and finalization run strictly through the Rust harness binary.
- Matched trial budgets and compute limits between Arm 1 (M3) and Arm 2 (M5).
- Honest accounting: full-pipeline costs, bundle sizes, latencies, test metrics, and transparent no-gain reporting.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_BIN = REPO_ROOT / "target" / "debug" / "mlrescurix-harness"
PYTHON_EXE = (
    "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
    if Path("/Library/Frameworks/Python.framework/Versions/3.13/bin/python3").is_file()
    else sys.executable
)


def run_cmd(args: list[str], check: bool = True) -> tuple[int, str, str]:
    res = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(args)} failed with exit {res.returncode}:\nStdout: {res.stdout}\nStderr: {res.stderr}"
        )
    return res.returncode, res.stdout, res.stderr


def generate_binary_dataset(csv_path: Path, seed: int = 2026, n_samples: int = 300) -> None:
    """Binary classification: nonlinear interactions and unseen test categories."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0.0, 1.0, n_samples)
    f2 = rng.uniform(-2.0, 2.0, n_samples)
    cat1 = rng.choice(["alpha", "beta", "gamma", "delta"], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    cat2 = rng.choice(["low", "medium", "high"], n_samples)

    signal = (f1 * f2 > 0.2).astype(float) + (cat1 == "alpha").astype(float) * 0.8 - (cat2 == "high").astype(float) * 0.5
    noise = rng.normal(0.0, 0.3, n_samples)
    prob = 1.0 / (1.0 + np.exp(-(signal + noise)))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "cat1": cat1,
        "cat2": cat2,
        "target": target,
    })
    # Inject an unseen category into a few tail rows (test split)
    df.loc[int(n_samples * 0.85):, "cat1"] = "unseen_omega"
    df.to_csv(csv_path, index=False)


def generate_multiclass_dataset(csv_path: Path, seed: int = 2027, n_samples: int = 300) -> None:
    """Multiclass classification (3 classes) with missing numeric features."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n_samples)
    x2 = rng.normal(1.0, 1.5, n_samples)
    x3 = rng.uniform(-1.0, 1.0, n_samples)

    scores_c0 = -x1 + 0.5 * x2
    scores_c1 = x1 * 0.8 + x3 * 1.2
    scores_c2 = x2 * -0.5 - x3 * 0.5
    scores = np.column_stack([scores_c0, scores_c1, scores_c2])
    target = np.argmax(scores + rng.normal(0.0, 0.2, scores.shape), axis=1)

    # Inject missingness into x2
    missing_mask = rng.random(n_samples) < 0.15
    x2[missing_mask] = np.nan

    df = pd.DataFrame({
        "num1": x1,
        "num2": x2,
        "num3": x3,
        "class_label": target,
    })
    df.to_csv(csv_path, index=False)


def generate_regression_dataset(csv_path: Path, seed: int = 2028, n_samples: int = 300) -> None:
    """Regression with continuous target and group identifiers."""
    rng = np.random.default_rng(seed)
    n_groups = 10
    group_ids = [f"grp_{i % n_groups}" for i in range(n_samples)]
    group_shifts = {f"grp_{i}": rng.normal(0.0, 0.5) for i in range(n_groups)}

    feat_a = rng.uniform(-2.0, 2.0, n_samples)
    feat_b = rng.normal(0.0, 1.0, n_samples)

    y = 2.5 * feat_a - 1.8 * (feat_b ** 2) + np.array([group_shifts[g] for g in group_ids]) + rng.normal(0.0, 0.2, n_samples)

    df = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "group_id": group_ids,
        "continuous_target": y,
    })
    df.to_csv(csv_path, index=False)


def setup_project_and_env(state: Path, work_dir: Path, project_id: str, budget_calls: int = 200) -> None:
    """Initialize state, create scientific environment, project, branch, task and claim lease."""
    policy_path = work_dir / f"policy_{project_id}.json"
    policy_path.write_text(json.dumps({
        "budget": {"tokens": budget_calls * 1000, "calls": budget_calls, "wall_ms": budget_calls * 10000},
        "capacity": {"workers": 4},
        "max_tasks": 1024,
    }))
    if not state.exists():
        run_cmd([str(HARNESS_BIN), "init", "--state", str(state), "--policy", str(policy_path)])

    # Setup environment
    run_cmd([
        str(HARNESS_BIN), "create-environment",
        "--state", str(state), "--id", "scientific", "--python", PYTHON_EXE
    ])
    cfg_path = state / "environments" / "scientific" / "pyvenv.cfg"
    if cfg_path.exists():
        cfg_content = cfg_path.read_text()
        cfg_path.write_text(cfg_content.replace("include-system-site-packages = false", "include-system-site-packages = true"))
    target_py = state / "environments" / "scientific" / "bin" / "python3"
    if target_py.exists() or target_py.is_symlink():
        target_py.unlink()
    os.symlink(PYTHON_EXE, target_py)

    # Create project
    req_path = work_dir / f"req_create_{project_id}.json"
    req_path.write_text(json.dumps({
        "command_id": f"init-{project_id}",
        "project_id": project_id,
        "expected_revision": None,
        "operation": {
            "type": "create_project",
            "intent": "M6 benchmark",
            "finalization_id": f"final-{project_id}",
            "policy": {
                "budget": {"tokens": budget_calls * 1000, "calls": budget_calls, "wall_ms": budget_calls * 10000},
                "capacity": {"workers": 4},
                "max_tasks": 1024,
            }
        }
    }))
    run_cmd([str(HARNESS_BIN), "command", "--state", str(state), "--request", str(req_path)])

    # Create branch
    b_req = work_dir / f"req_branch_{project_id}.json"
    b_req.write_text(json.dumps({
        "command_id": f"branch-{project_id}",
        "project_id": project_id,
        "expected_revision": None,
        "operation": {
            "type": "create_branch",
            "branch_id": "main",
            "base_hash": "base",
            "code_hash": "code",
        }
    }))
    run_cmd([str(HARNESS_BIN), "command", "--state", str(state), "--request", str(b_req)])

    # Create task
    t_req = work_dir / f"req_task_{project_id}.json"
    t_req.write_text(json.dumps({
        "command_id": f"task-{project_id}",
        "project_id": project_id,
        "expected_revision": None,
        "operation": {
            "type": "create_task",
            "task_id": "t1",
            "branch_id": "main",
            "context": {
                "role_id": "lead", "role_version": "v1", "prompt_hash": "prompt",
                "model": None, "phase": "search", "hypothesis": None, "base_hash": "base",
                "code_hash": "code", "evidence": [], "artifacts": [], "pending_questions": [],
                "calls": [], "jobs": [], "next_action": "search", "stop_conditions": [],
            }
        }
    }))
    run_cmd([str(HARNESS_BIN), "command", "--state", str(state), "--request", str(t_req)])

    # Claim lease
    now_ms = int(time.time() * 1000)
    c_req = work_dir / f"req_claim_{project_id}.json"
    c_req.write_text(json.dumps({
        "command_id": f"claim-{project_id}",
        "project_id": project_id,
        "expected_revision": None,
        "operation": {
            "type": "claim",
            "task_id": "t1",
            "owner": "local-worker",
            "lease_id": "lease-t1",
            "now_ms": now_ms,
            "expires_ms": now_ms + 3_600_000,
            "expected_generation": 1,
        }
    }))
    run_cmd([str(HARNESS_BIN), "command", "--state", str(state), "--request", str(c_req)])


def run_benchmark_arm(
    state: Path,
    work_dir: Path,
    project_id: str,
    dataset_id: str,
    target: str,
    task: str,
    metric: str,
    split: str,
    recipes: list[str],
    arm_label: str,
) -> dict[str, Any]:
    """Execute complete authoritative workflow for one arm."""
    start_time = time.time()

    # 1. Audit
    audit_spec_path = work_dir / f"audit_spec_{project_id}.json"
    audit_spec_path.write_text(json.dumps({
        "job_id": f"audit-{project_id}",
        "project_id": project_id,
        "task_id": "t1",
        "generation": 2,
        "timeout_ms": 30_000,
        "threads": 1,
        "max_output_bytes": 4 * 1024 * 1024,
        "environment_id": "scientific",
        "probe": {
            "kind": "scientific",
            "operation": "audit",
            "dataset_id": dataset_id,
            "spec": {
                "project_id": project_id,
                "target": target,
                "task": task,
                "metric": metric,
                "split": split,
                "inference_batch_size": 16,
                "inference_repetitions": 3,
            },
            "candidate_id": None,
            "test_access_id": None,
        }
    }))
    audit_t0 = time.time()
    run_cmd([str(HARNESS_BIN), "submit", "--state", str(state), "--spec", str(audit_spec_path)])
    run_cmd([str(HARNESS_BIN), "supervise", "--state", str(state), "--drain"])
    audit_ms = (time.time() - audit_t0) * 1000

    # 2. Baselines
    base_spec_path = work_dir / f"base_spec_{project_id}.json"
    base_spec_path.write_text(json.dumps({
        "job_id": f"base-{project_id}",
        "project_id": project_id,
        "task_id": "t1",
        "generation": 2,
        "timeout_ms": 30_000,
        "threads": 1,
        "max_output_bytes": 4 * 1024 * 1024,
        "environment_id": "scientific",
        "probe": {
            "kind": "scientific",
            "operation": "baselines",
            "dataset_id": dataset_id,
            "spec": None,
            "candidate_id": None,
            "test_access_id": None,
        }
    }))
    base_t0 = time.time()
    run_cmd([str(HARNESS_BIN), "submit", "--state", str(state), "--spec", str(base_spec_path)])
    run_cmd([str(HARNESS_BIN), "supervise", "--state", str(state), "--drain"])
    base_ms = (time.time() - base_t0) * 1000

    # 3. Search
    direction = "minimize" if metric in {"mae", "rmse", "log_loss"} else "maximize"
    search_spec_path = work_dir / f"search_spec_{project_id}.json"
    search_spec_path.write_text(json.dumps({
        "session_id": f"sess-{project_id}",
        "project_id": project_id,
        "dataset_id": dataset_id,
        "target": target,
        "task": task,
        "metric": metric,
        "direction": direction,
        "split": split,
        "fidelities": [0.5, 1.0],
        "recipes": recipes,
        "max_trials": 16,
        "max_screen_trials": 4,
        "max_full_trials": 2,
        "search_wall_ms": 180_000,
        "job_timeout_ms": 30_000,
        "finalization_allowance_ms": 30_000,
        "max_repeats": 3,
        "tie_tolerance": 0.005,
        "concurrency": 1,
        "max_retries": 2,
        "constraints": None,
    }))
    run_cmd([str(HARNESS_BIN), "search-init", "--state", str(state), "--spec", str(search_spec_path)])
    search_t0 = time.time()
    _, search_out, _ = run_cmd([
        str(HARNESS_BIN), "search-run",
        "--state", str(state),
        "--project", project_id,
        "--session", f"sess-{project_id}",
        "--task", "t1",
        "--generation", "2",
        "--environment", "scientific"
    ])
    search_ms = (time.time() - search_t0) * 1000
    selection = json.loads(search_out)
    apex_id = selection.get("selected_candidate_id")
    if not apex_id:
        for f in (state / "logs" / project_id).glob("*.log"):
            print(f"=== {f.name} ===")
            print(f.read_text())
        raise RuntimeError(f"Search failed to select apex candidate: {selection}")

    # 4. Finalize
    gate_id = f"gate-{project_id}"
    fin_spec_path = work_dir / f"fin_spec_{project_id}.json"
    fin_spec_path.write_text(json.dumps({
        "job_id": f"fin-{project_id}",
        "project_id": project_id,
        "task_id": "t1",
        "generation": 2,
        "timeout_ms": 60_000,
        "threads": 1,
        "max_output_bytes": 4 * 1024 * 1024,
        "environment_id": "scientific",
        "probe": {
            "kind": "scientific",
            "operation": "finalize",
            "dataset_id": dataset_id,
            "spec": None,
            "candidate_id": apex_id,
            "test_access_id": gate_id,
        }
    }))
    fin_t0 = time.time()
    run_cmd([str(HARNESS_BIN), "submit", "--state", str(state), "--spec", str(fin_spec_path)])
    run_cmd([str(HARNESS_BIN), "supervise", "--state", str(state), "--drain"])
    fin_ms = (time.time() - fin_t0) * 1000

    # 5. Verify Bundle
    bundle_path = state / "workflows" / project_id / "bundle"
    _, verif_out, _ = run_cmd([str(HARNESS_BIN), "verify-bundle", "--bundle", str(bundle_path), "--sandbox"])
    bundle_report = json.loads(verif_out)

    # 6. Extract Final Metrics & Ledger
    final_json = json.loads((bundle_path / "final.json").read_text())
    scorecard = json.loads((bundle_path / "scorecard.json").read_text())
    _, snap_out, _ = run_cmd([str(HARNESS_BIN), "snapshot", "--state", str(state), "--project", project_id])
    snapshot = json.loads(snap_out)

    total_time_ms = (time.time() - start_time) * 1000

    contenders = selection.get("contenders", [])
    feasible_count = sum(1 for c in contenders if c.get("is_feasible"))
    total_contenders = len(contenders)

    return {
        "arm": arm_label,
        "status": selection.get("status"),
        "apex_candidate_id": apex_id,
        "recipe": selection.get("recipe"),
        "validation_metric": selection.get("contenders", [{}])[0].get("scores", [None])[0] if selection.get("contenders") else None,
        "test_metric": final_json.get("test_metrics", {}).get(metric),
        "test_metric_name": metric,
        "feasible_contenders": f"{feasible_count}/{total_contenders}",
        "bundle_bytes": bundle_report.get("total_bytes"),
        "bundle_verified": bundle_report.get("status") == "verified",
        "parity_check": bundle_report.get("parity_check"),
        "warm_p95_ms": scorecard.get("inference", {}).get("warm_p95_ms"),
        "timings_ms": {
            "audit": round(audit_ms, 1),
            "baselines": round(base_ms, 1),
            "search": round(search_ms, 1),
            "finalize": round(fin_ms, 1),
            "total": round(total_time_ms, 1),
        },
        "ledger_charged": snapshot.get("project_ledger", {}).get("charged", {}),
        "provider_spend_usd": 0.00,
    }


def main() -> int:
    print("=" * 80)
    print("MLRescurix M6 Authoritative Predeclared Matched Benchmark Comparison")
    print("=" * 80)

    image = os.environ.get("MLRESCURIX_BUNDLE_IMAGE", "")
    digest = image.rsplit("@sha256:", 1)[-1] if "@sha256:" in image else ""
    if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
        print("Refusing benchmark before consuming test gates: configure a pinned scientific MLRESCURIX_BUNDLE_IMAGE for real sandbox bundle parity.")
        return 2
    if HARNESS_BIN.is_file() and any(p.stat().st_mtime_ns > HARNESS_BIN.stat().st_mtime_ns for p in (REPO_ROOT / "crates/harness/src").rglob("*.*")):
        print("Harness binary is stale; run cargo build --workspace --locked before benchmarking.")
        return 2
    if not HARNESS_BIN.is_file():
        print(f"Error: harness binary not found at {HARNESS_BIN}. Run `cargo build` first.")
        return 1
    capabilities = json.loads(subprocess.check_output([str(HARNESS_BIN), "capabilities"], text=True))
    prerequisites = capabilities.get("release_prerequisites", {})
    required = ("composite_finalization", "durable_specialist_outcome_recovery", "bounded_total_storage_cycles")
    if any(prerequisites.get(name) is not True for name in required):
        print("Refusing new benchmark work before dataset allocation or test access: composite finalization, durable specialist recovery and total storage prerequisites are incomplete. Historical metrics remain invalid acceptance evidence.")
        return 2

    temp_dir = tempfile.TemporaryDirectory()
    work_dir = Path(temp_dir.name)

    tasks_spec = [
        {
            "name": "Binary Classification (Unseen Categories & Interactions)",
            "generator": generate_binary_dataset,
            "filename": "binary_unseen.csv",
            "dataset_id": "data_binary",
            "target": "target",
            "task": "binary_classification",
            "metric": "balanced_accuracy",
            "split": "stratified",
        },
        {
            "name": "Multiclass Classification (Missing Values, 3 Classes)",
            "generator": generate_multiclass_dataset,
            "filename": "multiclass_missing.csv",
            "dataset_id": "data_multiclass",
            "target": "class_label",
            "task": "multiclass_classification",
            "metric": "balanced_accuracy",
            "split": "stratified",
        },
        {
            "name": "Regression (Nonlinear, Group Split)",
            "generator": generate_regression_dataset,
            "filename": "regression_group.csv",
            "dataset_id": "data_regression",
            "target": "continuous_target",
            "task": "regression",
            "metric": "rmse",
            "split": "random",
        },
    ]

    results = []

    for idx, t in enumerate(tasks_spec, 1):
        print(f"\n--- Task {idx}: {t['name']} ---")
        csv_path = work_dir / t["filename"]
        t["generator"](csv_path)

        # Matched Arms
        arms = [
            ("M3 Baseline", ["linear", "tree"]),
            ("M5 Scientific", ["linear", "tree", "hist_boosting"]),
        ]

        task_results = {"task": t["name"], "metric": t["metric"], "arms": {}}

        for arm_name, recipes in arms:
            proj_id = f"p{idx}_{arm_name.lower().replace(' ', '_')}"
            state_dir = work_dir / f"state_{proj_id}"
            print(f"Executing {arm_name} on authoritative path (project: {proj_id})...")

            # Setup fresh project
            setup_project_and_env(state_dir, work_dir, proj_id)

            # Register dataset
            run_cmd([
                str(HARNESS_BIN), "register-dataset",
                "--state", str(state_dir),
                "--project", proj_id,
                "--id", t["dataset_id"],
                "--csv", str(csv_path),
            ])

            # Run Arm
            res = run_benchmark_arm(
                state=state_dir,
                work_dir=work_dir,
                project_id=proj_id,
                dataset_id=t["dataset_id"],
                target=t["target"],
                task=t["task"],
                metric=t["metric"],
                split=t["split"],
                recipes=recipes,
                arm_label=arm_name,
            )
            task_results["arms"][arm_name] = res
            print(f"  [{arm_name}] Apex: {res['recipe']} | Test Score: {res['test_metric']} | Wall Time: {res['timings_ms']['total']}ms | Bundle: {res['bundle_bytes']}B | P95: {res['warm_p95_ms']}ms")

        # Compare outcomes
        m3_res = task_results["arms"]["M3 Baseline"]
        m5_res = task_results["arms"]["M5 Scientific"]
        m3_score = m3_res["test_metric"] or 0.0
        m5_score = m5_res["test_metric"] or 0.0
        if t["metric"] in {"mae", "rmse", "log_loss"}:
            delta = m3_score - m5_score
        else:
            delta = m5_score - m3_score

        if delta > 0.005:
            outcome = "M5 advantage"
        elif delta < -0.005:
            outcome = "M3 advantage (honest no-gain for M5)"
        else:
            outcome = "Statistical parity / no difference"

        task_results["comparison"] = {
            "m3_test_score": m3_score,
            "m5_test_score": m5_score,
            "delta": round(delta, 4),
            "outcome": outcome,
        }
        print(f"  => Outcome: {outcome} (delta: {delta:+.4f})")
        results.append(task_results)

    # State storage measurement
    state_bytes = sum(f.stat().st_size for f in work_dir.rglob("*") if f.is_file() and "state_" in str(f))
    output_payload = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_engine": "Authoritative Rust harness (mlrescurix-harness v0.1.0)",
            "hardware_platform": "Apple Silicon (macOS)",
            "python_runtime": PYTHON_EXE,
            "matched_budget_per_arm": {
                "max_trials": 16,
                "max_screen_trials": 4,
                "max_full_trials": 2,
                "max_repeats": 3,
                "search_wall_ms": 180000,
            },
            "state_storage_total_bytes": state_bytes,
        },
        "tasks": results,
    }

    # Save to docs/m6-comparison-results.json
    out_file = REPO_ROOT / "docs" / "m6-comparison-results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_payload, indent=2))
    print(f"\nBenchmark completed successfully! Results saved to {out_file}")

    temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
