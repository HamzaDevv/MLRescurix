#!/usr/bin/env python3
"""MLRescurix Robustness Campaign Preflight & Resumable Runner.

Adheres to ROBUSTNESS_MASTER_PLAN.md:
- Preflight: environment, binary provenance, schema version (up to v11), Docker daemon,
  Kaggle credentials, provider key presence and budget limits.
- Resumable: run_id directory layout under docs/evidence/v2/robustness/<campaign>/<run_id>/.
- Explicit outcomes: passed, failed, skipped, blocked, inconclusive.
- Worker resource traces: controller RSS, worker process tree RSS, cgroup on Linux.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python/src"))
MANIFEST_PATH = REPO_ROOT / "docs" / "v2" / "BENCHMARK_SUITE.json"
EVIDENCE_BASE = REPO_ROOT / "docs" / "evidence" / "v2" / "robustness"


from mlrescurix_worker.scientific.benchmark_contracts import (
    finite_score, score_meets_threshold, locked_quality_threshold, verified_provider_pricing, candidate_gate_failure_reason, accepted_candidate_evidence, candidate_spec_identity, bundle_matches_candidate, map_locked_index_folds, map_k4_locked_folds, map_k4_manifest_folds, map_locked_temporal_folds, package_accepted_candidate
)


def compute_file_sha256(path: Path | str) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_sha256(directory: Path | str) -> str:
    d = Path(directory)
    if not d.exists():
        return "not_found"
    files = []
    for root, _, filenames in os.walk(d):
        rel_dir = Path(root).relative_to(d)
        if any(part.startswith(".") or part == "__pycache__" for part in rel_dir.parts):
            continue
        for fn in filenames:
            if fn.startswith(".") or fn.endswith(".pyc"):
                continue
            fpath = Path(root) / fn
            rel_file = (rel_dir / fn).as_posix()
            try:
                content = fpath.read_bytes()
                files.append((rel_file, hashlib.sha256(content).hexdigest()))
            except Exception:
                pass
    files.sort(key=lambda x: x[0])
    h = hashlib.sha256()
    for rel_path, file_sha in files:
        h.update(f"{rel_path}:{file_sha}\n".encode("utf-8"))
    return h.hexdigest()


def get_git_provenance() -> Dict[str, Any]:
    try:
        p_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        commit = p_head.stdout.strip() if p_head.returncode == 0 else "unknown"
        p_diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        diff_text = p_diff.stdout if p_diff.returncode == 0 else ""
        diff_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest() if diff_text else "clean"
        p_status = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        is_dirty = bool(p_status.stdout.strip())
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "dirty_diff_sha256": diff_sha,
        }
    except Exception as e:
        return {"commit": "unknown", "error": str(e), "is_dirty": True, "dirty_diff_sha256": "error"}


from mlrescurix_worker.scientific.governance import BudgetExhaustionError, CampaignBudgetAuthority


class CampaignRunner:
    def __init__(self, campaign: str = "20261021-robustness", run_id: Optional[str] = None):
        for label, value in (("campaign", campaign), ("run_id", run_id)):
            if value is not None and (not value or Path(value).name != value or value in (".", "..")):
                raise ValueError(f"Invalid {label}: expected a single directory name")
        self.policy = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.campaign = campaign
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = EVIDENCE_BASE / self.campaign / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        self.campaign_authority = CampaignBudgetAuthority.open_preserved(EVIDENCE_BASE / "20261021-robustness", self.policy)

        self.summary_path = self.run_dir / "summary.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.traces_path = self.run_dir / "traces.jsonl"
        self.resources_path = self.run_dir / "resources.jsonl"
        self.provider_usage_path = self.run_dir / "provider_usage.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.faults_path = self.run_dir / "faults.json"
        self.artifacts_path = self.run_dir / "artifacts.json"

        # Load or initialize summary with strict resume checks
        if self.summary_path.exists():
            try:
                loaded = json.loads(self.summary_path.read_text(encoding="utf-8"))
                if loaded.get("run_id") and loaded["run_id"] != self.run_id:
                    raise ValueError(f"Run ID mismatch: expected {self.run_id}, found {loaded['run_id']}")
                if loaded.get("campaign") and loaded["campaign"] != self.campaign:
                    raise ValueError(f"Campaign mismatch: expected {self.campaign}, found {loaded['campaign']}")
                # Verify manifest immutability on resume
                if self.manifest_path.exists() and MANIFEST_PATH.exists():
                    saved_sha = compute_file_sha256(self.manifest_path)
                    cur_sha = compute_file_sha256(MANIFEST_PATH)
                    if saved_sha and cur_sha and saved_sha != cur_sha:
                        raise ValueError("Cannot safely resume: benchmark manifest policy has been modified")
                self.summary = loaded
            except Exception as e:
                raise ValueError("Cannot safely resume invalid summary") from e
        else:
            self.summary = self._init_summary()

    def _init_summary(self) -> Dict[str, Any]:
        return {
            "campaign": self.campaign,
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "git": get_git_provenance(),
            "gates": {},
            "budget_spent_usd": 0.0,
            "holdout_accesses": 0,
        }

    def _atomic_write_json(self, path: Path, data: Any) -> None:
        if isinstance(data, dict) and ("total_billed_cost_usd" in data or "budget_spent_usd" in data):
            # Preserve compatibility fields without presenting bookkeeping as invoices.
            data = dict(data, cost_basis="ledger_recorded_estimates_or_unverified_accounting",
                        verified_provider_charge_usd=None,
                        legacy_billed_fields_are_verified_charges=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def record_trace(self, event_name: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **payload,
        }
        with open(self.traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_resource(self, stage: str, metrics: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **metrics,
        }
        with open(self.resources_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary(self) -> None:
        # Keep run-level accounting aligned with the campaign's durable ledger.
        # Historical summaries written before this reconciliation remain
        # unchanged and are corrected by the accompanying audit note.
        ledger = self.campaign_authority.get_status()
        run_ledger = ledger.get("runs", {}).get(self.run_id, {})
        self.summary["budget_spent_usd"] = float(Decimal(str(run_ledger.get("billed_usd", "0"))))
        self.summary["budget_reserved_usd"] = float(Decimal(str(run_ledger.get("reserved_usd", "0"))))
        self.summary["budget_uncertain_usd"] = float(Decimal(str(run_ledger.get("uncertain_usd", "0"))))
        self.summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write_json(self.summary_path, self.summary)

    def _task_contract(self, task_id):
        from mlrescurix_worker.scientific.task_adapters import benchmark_adapter
        return benchmark_adapter(self.policy, task_id, REPO_ROOT).contract()

    def _execution_contract_block(self, task_id):
        problem = next(p for p in self.policy["problems"] if p["id"] == task_id)
        contract = problem.get("execution_contract", {})
        if contract.get("status", "").startswith("blocked"):
            outcome = {"status": "blocked", "reason": contract}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome
        return None

    def _live_gateway_block(self) -> Optional[Dict[str, Any]]:
        """Verify Rust harness binary exists for production gateway execution."""
        from mlrescurix_worker.scientific.rust_gateway import find_harness_binary
        try:
            find_harness_binary()
            return None
        except FileNotFoundError as exc:
            outcome = {
                "status": "blocked",
                "reason": f"Rust production provider gateway/orchestrator binary is missing: {exc}",
            }
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

    def run_preflight(self) -> Dict[str, Any]:
        """Perform comprehensive preflight checks across all system layers."""
        self.record_trace("preflight_start", {"action": "evaluating_prerequisites"})
        t0 = time.time()

        # 1. Host and environment
        host_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "arch": platform.machine(),
            "cores": os.cpu_count(),
        }

        # 2. Scientific packages
        packages = {}
        for pkg in ["numpy", "pandas", "scipy", "sklearn", "pytest", "jsonschema", "psutil"]:
            try:
                import importlib.metadata
                packages[pkg] = importlib.metadata.version(pkg)
            except Exception:
                try:
                    mod = __import__(pkg)
                    packages[pkg] = getattr(mod, "__version__", "present")
                except ImportError:
                    packages[pkg] = None

        # 3. Rust harness binary provenance
        bin_debug = REPO_ROOT / "target" / "debug" / "mlrescurix-harness"
        bin_dist = REPO_ROOT / "dist" / "mlrescurix-harness"
        rust_provenance = {
            "debug_binary": {
                "exists": bin_debug.is_file(),
                "sha256": compute_file_sha256(bin_debug),
            },
            "dist_binary": {
                "exists": bin_dist.is_file(),
                "sha256": compute_file_sha256(bin_dist),
            },
            "rust_source_sha256": compute_directory_sha256(REPO_ROOT / "crates" / "harness" / "src"),
        }

        # 4. Storage schema compatibility check (accepts schema versions up to 11)
        storage_check = {"max_supported_version": 11, "never_downgrade_to_v10": True,
                         "status": "unverified", "reason": "Runtime database migration not exercised"}

        # 5. Docker daemon & Linux cgroup v2 connectivity
        docker_reachable = False
        docker_info_str = ""
        docker_bin = shutil.which("docker")
        if docker_bin:
            try:
                res = subprocess.run([docker_bin, "info", "--format", "{{.OSType}} {{.CgroupVersion}}"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    docker_reachable = True
                    docker_info_str = res.stdout.strip()
                else:
                    docker_info_str = res.stderr.strip()
            except Exception as e:
                docker_info_str = str(e)

        docker_status = {
            "docker_bin": docker_bin,
            "daemon_reachable": docker_reachable,
            "info": docker_info_str,
            "cgroup_v2_certified": ("2" in docker_info_str and "linux" in docker_info_str.lower()),
        }

        # 6. Kaggle prerequisites (check without failing)
        kaggle_installed = False
        try:
            import kaggle  # type: ignore
            kaggle_installed = True
        except Exception:
            pass

        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        kaggle_access_token = Path.home() / ".kaggle" / "access_token"
        local_kaggle_token = REPO_ROOT / ".config" / "kaggle.token"
        kaggle_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or bool(os.environ.get("KAGGLE_API_TOKEN"))
        has_kaggle_creds = kaggle_json.is_file() or kaggle_access_token.is_file() or local_kaggle_token.is_file() or kaggle_env
        pinned_tasks = {}
        for t_id, dir_name, main_file in [
            ("K1", "k1_titanic", "train.csv"),
            ("K2", "k2_house_prices", "train.csv"),
            ("K3", "k3_forest_cover", "train.csv"),
            ("K4", "k4_ieee_fraud", "train_transaction.csv"),
            ("K5", "k5_store_sales", "train.csv"),
        ]:
            d = REPO_ROOT / "data" / "official" / dir_name
            manifest = d / "split_manifest.json"
            data_f = d / main_file
            pinned_tasks[t_id] = manifest.is_file() and data_f.is_file()

        pinned_count = sum(1 for v in pinned_tasks.values() if v)
        kaggle_status = {
            "client_installed": kaggle_installed,
            "credentials_present": has_kaggle_creds,
            "official_data_status": "ready" if pinned_count == 5 else ("partial_pinned" if pinned_count > 0 else "not_pinned"),
            "pinned_tasks": pinned_tasks,
        }

        # 7. Provider credentials (check existence WITHOUT printing secrets)
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key_file = openrouter_key_file.is_file()
        has_env_key = bool(os.environ.get("OPENROUTER_API_KEY"))
        rust_binary_present = False
        try:
            from mlrescurix_worker.scientific.rust_gateway import find_harness_binary
            find_harness_binary()
            rust_binary_present = True
        except Exception:
            rust_binary_present = False

        provider_status = {
            "key_present": (has_key_file or has_env_key),
            "key_source": "environment" if has_env_key else ("local_file" if has_key_file else "none"),
            "rust_gateway_binary_present": rust_binary_present,
            "validity_verified": False,  # unverified until authorized live solve
            "spending_cap_usd": float(self.policy["campaign_budget"]["provider_usd_max"]),
            "per_run_cap_usd": float(self.policy["provider"]["usd_per_main_live_run_max"]),
            "model_id": self.policy["provider"]["model_id"],
            "provider_routing": self.policy["provider"]["provider_routing"],
        }

        if not rust_binary_present:
            g6_outcome = "blocked"
            g6_reason = "Rust harness binary not found; build with 'cargo build --release'"
        elif not provider_status["key_present"]:
            g6_outcome = "blocked"
            g6_reason = "Provider credentials missing; expected .config/openrouter.key or OPENROUTER_API_KEY"
        else:
            g6_outcome = "inconclusive"
            g6_reason = "Rust production gateway wired with outbound-view policy and budget accounting; live request unverified"

        # Compile gate assessments
        gates = {
            "G0_HOST_ENVIRONMENT": {
                "outcome": "passed",
                "details": host_info,
            },
            "G1_PYTHON_SCIENTIFIC_STACK": {
                "outcome": "passed" if all(packages[p] is not None for p in ["numpy", "pandas", "sklearn", "pytest"]) else "failed",
                "details": packages,
            },
            "G2_RUST_HARNESS_PROVENANCE": {
                "outcome": "inconclusive",
                "reason": "Binary hashes collected; source-to-build binding unverified",
                "details": rust_provenance,
            },
            "G3_STORAGE_SCHEMA_V11": {
                "outcome": "inconclusive",
                "details": storage_check,
            },
            "G4_CONTAINER_CGROUP_V2": {
                "outcome": "inconclusive" if docker_status["cgroup_v2_certified"] else "blocked",
                "reason": "Docker daemon inaccessible or not Linux cgroup v2" if not docker_status["cgroup_v2_certified"] else "Daemon reports cgroup v2; per-job enforcement unverified",
                "details": docker_status,
            },
            "G5_KAGGLE_OFFICIAL_DATA": {
                "outcome": "inconclusive" if kaggle_status["official_data_status"] == "ready" else "blocked",
                "reason": "Official dataset files pinned; campaign verification required" if kaggle_status["official_data_status"] == "ready" else f"Official datasets pinned: {pinned_count}/5",
                "details": kaggle_status,
            },
            "G6_PROVIDER_GATEWAY": {
                "outcome": g6_outcome,
                "reason": g6_reason,
                "details": provider_status,
            },
        }

        self.summary["gates"].update(gates)
        preflight_elapsed = time.time() - t0

        # Snapshot immutable manifest to evidence dir (preserve on resume)
        if MANIFEST_PATH.is_file() and not self.manifest_path.exists():
            shutil.copy2(MANIFEST_PATH, self.manifest_path)

        # Write artifacts record atomically
        artifacts_record = {
            "source_provenance": self.summary["git"],
            "rust_binaries": rust_provenance,
            "python_worker_sha256": compute_directory_sha256(REPO_ROOT / "python" / "src" / "mlrescurix_worker"),
            "benchmark_manifest_sha256": compute_file_sha256(self.manifest_path) if self.manifest_path.exists() else None,
        }
        self._atomic_write_json(self.artifacts_path, artifacts_record)

        # Record provider usage baseline (preserve on resume)
        if not self.provider_usage_path.exists():
            provider_record = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_billed_cost_usd": 0.0,
                "reserved_cost_usd": 0.0,
                "remaining_budget_usd": float(self.policy["campaign_budget"]["provider_usd_max"]),
                "budget_scope": "run_observation_only_not_authoritative_campaign_ledger",
            }
            self._atomic_write_json(self.provider_usage_path, provider_record)

        # Recompute this run's observation under the current lower cap without
        # erasing spend/reservations. This is not cross-run admission authority.
        usage = json.loads(self.provider_usage_path.read_text(encoding="utf-8"))
        cap = Decimal(self.policy["campaign_budget"]["provider_usd_max"])
        spent = Decimal(str(usage["total_billed_cost_usd"]))
        reserved = Decimal(str(usage["reserved_cost_usd"]))
        if not spent.is_finite() or not reserved.is_finite() or spent < 0 or reserved < 0:
            raise ValueError("Invalid provider accounting; cannot resume")
        usage["remaining_budget_usd"] = float(max(Decimal("0"), cap - spent - reserved))
        usage["spending_cap_usd"] = float(cap)
        usage["budget_scope"] = "run_observation_only_not_authoritative_campaign_ledger"
        self._atomic_write_json(self.provider_usage_path, usage)

        self.record_trace("preflight_completed", {"duration_s": preflight_elapsed, "blocked_gates": [k for k, v in gates.items() if v["outcome"] == "blocked"]})
        self.save_summary()
        return gates

    def run_fault_reproducers(self) -> Dict[str, Any]:
        """Run RB-001 - RB-004 reproducers and fault regressions."""
        self.record_trace("faults_start", {"action": "running_reproducers"})
        t0 = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "python" / "tests" / "test_s1_robustness_reproducers.py"),
            str(REPO_ROOT / "python" / "tests" / "test_hypothesis_executor.py"),
            "-q",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python" / "src")

        p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
        elapsed = time.time() - t0

        fault_log_path = self.run_dir / "logs" / "fault_reproducers.log"
        fault_log_path.write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")

        passed = (p.returncode == 0)
        faults_result = {
            "test_file": "test_s1_robustness_reproducers.py, test_hypothesis_executor.py",
            "exit_code": p.returncode,
            "duration_s": elapsed,
            "passed": passed,
            "stdout_summary": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "",
        }

        self.faults_path.write_text(json.dumps(faults_result, indent=2), encoding="utf-8")

        gate_outcome = "passed" if passed else "failed"
        self.summary["gates"]["G_REPRODUCERS_RB001_RB004"] = {
            "outcome": gate_outcome,
            "exit_code": p.returncode,
            "duration_s": elapsed,
        }
        self.summary["gates"]["G_TYPED_ACTION_DISPATCHER"] = {
            "outcome": gate_outcome,
            "exit_code": p.returncode,
            "duration_s": elapsed,
        }

        self.record_trace("faults_completed", {"outcome": gate_outcome, "duration_s": elapsed})
        self.save_summary()
        return faults_result

    def run_deterministic_dev(self) -> Dict[str, Any]:
        """Run deterministic dev benchmark verification on synthetic fixtures."""
        self.record_trace("deterministic_dev_start", {"action": "evaluating_benchmarks"})
        t0 = time.time()
        import resource
        rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "python" / "tests" / "test_kaggle_benchmarks.py"),
            str(REPO_ROOT / "python" / "tests" / "test_gate4_benchmarks.py"),
            "-q",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python" / "src")

        p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
        elapsed = time.time() - t0
        rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self.record_resource("deterministic_dev", {
            "elapsed_s": elapsed,
            "controller_lifetime_peak_rss_bytes": rss_end if sys.platform == "darwin" else rss_end * 1024,
            "controller_lifetime_peak_delta_bytes": (rss_end - rss_start) if sys.platform == "darwin" else (rss_end - rss_start) * 1024,
            "worker_measurement_status": "unavailable",
        })

        dev_log_path = self.run_dir / "logs" / "deterministic_dev.log"
        dev_log_path.write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")

        passed = (p.returncode == 0)
        dev_result = {
            "test_files": "test_kaggle_benchmarks.py, test_gate4_benchmarks.py",
            "exit_code": p.returncode,
            "duration_s": elapsed,
            "passed": passed,
            "stdout_summary": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "",
        }

        self._atomic_write_json(self.run_dir / "deterministic_dev.json", dev_result)
        gate_outcome = "passed" if passed else "failed"
        self.summary["gates"]["G_DETERMINISTIC_DEV_BENCHMARKS"] = {
            "outcome": gate_outcome,
            "exit_code": p.returncode,
            "duration_s": elapsed,
        }
        self.summary["status"] = "inconclusive" if passed else "failed"
        self.record_trace("deterministic_dev_completed", {"outcome": gate_outcome, "duration_s": elapsed})
        self.save_summary()
        return dev_result

    def _call_openrouter_specialist(self, role: str, messages: List[Dict[str, str]], max_tokens: int = 1200) -> Dict[str, Any]:
        raise RuntimeError("Legacy direct provider transport disabled; use campaign-bound RustGatewayAdapter with a stable request identity")

    def run_benchmark(self, task_id):
        from mlrescurix_worker.scientific.benchmark_runner import run_benchmark
        return run_benchmark(self, task_id)

    # Public compatibility entry points share the same lifecycle.
    def run_live_dev(self): return self.run_benchmark("K1")
    def run_live_dev_k2(self): return self.run_benchmark("K2")
    def run_live_dev_k3(self): return self.run_benchmark("K3")
    def run_live_dev_k4(self): return self.run_benchmark("K4")
    def run_live_dev_k5(self): return self.run_benchmark("K5")

    def run_certify(self) -> Dict[str, Any]:
        """Execute certification gates. Never succeeds without Linux cgroup v2 verification."""
        self.record_trace("certify_start", {"action": "evaluating_certification_gates"})
        gates = self.run_preflight()
        blocked = [k for k, v in gates.items() if v["outcome"] == "blocked"]
        failed = [k for k, v in gates.items() if v["outcome"] == "failed"]
        outcome = {
            "status": "blocked" if blocked else ("failed" if failed else "inconclusive"),
            "reason": f"Certification rejected: blocked gates {blocked}" if blocked else "Certification requires Linux cgroup v2 and complete campaign passes",
            "blocked_gates": blocked,
            "failed_gates": failed,
        }
        self.summary["gates"]["G_CERTIFICATION"] = {
            "outcome": outcome["status"],
            "reason": outcome["reason"],
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("certify_rejected", outcome)
        self.save_summary()
        return outcome


def main() -> Optional[int]:
    parser = argparse.ArgumentParser(description="MLRescurix Robustness Campaign Runner (S1-S5)")
    parser.add_argument("--mode", choices=["preflight", "faults", "deterministic_dev", "live_dev", "certify", "all"], default="preflight", help="Execution mode")
    parser.add_argument("--task", choices=["K1", "K2", "K3", "K4", "K5"], default="K3", help="Task for live_dev mode (default: K3)")
    parser.add_argument("--campaign", default="20261021-robustness", help="Campaign name")
    parser.add_argument("--run-id", default=None, help="Resumable run ID")
    args = parser.parse_args()

    print(f"=== MLRescurix Robustness Campaign: mode={args.mode}, task={args.task}, campaign={args.campaign} ===")
    runner = CampaignRunner(campaign=args.campaign, run_id=args.run_id)
    print(f"Evidence Run Directory: {runner.run_dir}")

    if args.mode in ("preflight", "all"):
        print("\n--- Running Layered Preflight ---")
        gates = runner.run_preflight()
        for g_id, g_info in gates.items():
            status = g_info["outcome"].upper()
            reason = f" ({g_info.get('reason')})" if g_info.get("reason") else ""
            print(f"  [{status:<7}] {g_id}{reason}")

    if args.mode in ("faults", "all"):
        print("\n--- Running S1 Fault Reproducers & Typed Hypothesis Dispatcher ---")
        res = runner.run_fault_reproducers()
        print(f"  Status: {'PASSED' if res['passed'] else 'FAILED'} in {res['duration_s']:.2f}s")
        print(f"  Summary: {res['stdout_summary']}")
        if not res["passed"]:
            return 1

    if args.mode in ("deterministic_dev", "all"):
        print("\n--- Running Deterministic Dev Benchmarks ---")
        runner.run_preflight()
        res = runner.run_deterministic_dev()
        print(f"  Status: {'PASSED' if res['passed'] else 'FAILED'} in {res['duration_s']:.2f}s")
        print(f"  Summary: {res['stdout_summary']}")
        if not res["passed"]:
            return 1

    if args.mode == "live_dev":
        res = runner.run_benchmark(args.task)
        print(json.dumps(res, indent=2, default=str))
        return 0 if res["status"] == "passed" else 1

    if args.mode == "certify":
        print("\n--- Evaluating Certification Gates ---")
        res = runner.run_certify()
        print(f"  Status: {res['status'].upper()}: {res['reason']}")
        print(f"\n=== Robustness Campaign Certification Blocked ===")
        print(f"Summary written to: {runner.summary_path}")
        return 1

    outcomes = [g["outcome"] for g in runner.summary["gates"].values()]
    runner.summary["status"] = ("failed" if "failed" in outcomes else
                                 "blocked" if "blocked" in outcomes else
                                 "inconclusive" if "inconclusive" in outcomes else "completed")
    runner.save_summary()

    print("\n=== Robustness Campaign Run Complete ===")
    print(f"Summary written to: {runner.summary_path}")
    return 0 if runner.summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
