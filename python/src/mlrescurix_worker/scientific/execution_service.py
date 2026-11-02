"""Execution lifecycle for trusted installed workers and isolated generated scripts.

Trusted host processes are development-only, not a hostile-code sandbox. Script
execution always uses the bounded Docker backend; its metrics remain advisory.
"""
import copy
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from .candidate_pipeline import identity
from .evidence import atomic_json, read_worker_result, seal_result, source_identity, streaming_file_sha256
REPO_ROOT = Path(__file__).resolve().parents[4]


class ExecutionService:
    def __init__(self, backends: dict | None = None, admission_controller: object | None = None):
        self.backends = backends or {}
        self.admission_controller = admission_controller

    def register_backend(self, backend: object) -> None:
        self.backends[getattr(backend, "backend_id", str(backend))] = backend

    def get_backend(self, backend_id: str) -> object | None:
        return self.backends.get(backend_id)

    def evaluate(self, session, spec, operation="evaluate"):
        if source_identity(REPO_ROOT) != session.binding['source_sha256']:
            raise ValueError('Worker source changed since session binding')
        if streaming_file_sha256(session.dataset_path) != session.binding["data_sha256"]:
            raise ValueError("Development data changed since checkpoint binding")
        if operation == "evaluate" and session.state.get("evaluation_count", 0) >= 12:
            raise ValueError("Candidate evaluation limit (12) reached; handoff required")
        if operation == "evaluate":
            session.state["evaluation_count"] = session.state.get("evaluation_count", 0) + 1
            if "evaluated_families" not in session.state:
                session.state["evaluated_families"] = []
            for fam in spec.get("families", []):
                if fam not in session.state["evaluated_families"]:
                    session.state["evaluated_families"].append(fam)
            session._save_state(session.state)
        if session.evaluator is not None and operation == "evaluate":
            if not session.provider_is_fixture:
                raise ValueError("Injected evaluators are restricted to explicit offline fixtures")
            result = session.evaluator(copy.deepcopy(spec), session.raw_df.copy(), session.locked_folds, session.metric_name)
            return seal_result(result, session.binding, "offline_fixture_evaluator")
        # Only trusted installed evaluator code runs on host; generated Python uses Docker tool.
        token = identity({"spec": spec, "binding": session.binding, "operation": operation})
        jobs = session.workspace_root / "evaluations"; jobs.mkdir(exist_ok=True)
        request, output = jobs / f"{token}.request.json", jobs / f"{token}.result.json"
        request_data = {"spec": spec, "dataset": str(session.dataset_path), "target": session.target_column,
            "task_type": session.task_type, "metric": session.metric_name, "folds": session.locked_folds,
            "topology": session.topology, "binding": session.binding,
            "cpu_seconds": int(session.evaluation_timeout_seconds), "memory_bytes": session.worker_memory_bytes,
            "operation": operation, "model_path": str(jobs / f"{token}.joblib")}
        request_identity = identity(request_data)
        if output.exists():
            return read_worker_result(output, request_identity)
        atomic_json(request, request_data)
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR")}
        env.update(PYTHONPATH=str(REPO_ROOT / "python" / "src"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (jobs / f"{token}.worker.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "mlrescurix_worker.scientific.candidate_worker", str(request), str(output)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            session.sampler.register_worker(proc.pid)
            worker_rss_peak = None
            worker_sample_count = 0
            worker_sample_status = "unavailable"
            worker_cgroup = None
            try:
                deadline = time.monotonic() + session.evaluation_timeout_seconds
                while proc.poll() is None:
                    if time.monotonic() >= deadline: raise subprocess.TimeoutExpired(proc.args, session.evaluation_timeout_seconds)
                    metrics = session.sampler.sample_now()
                    worker_metrics = metrics.get("workers", {})
                    observed_memory = worker_metrics.get("rss_bytes")
                    worker_sample_count += 1
                    worker_sample_status = worker_metrics.get("status", "unavailable")
                    if observed_memory is not None:
                        worker_rss_peak = max(worker_rss_peak or 0, observed_memory)
                    if metrics.get("cgroup", {}).get("scope") == "workers":
                        worker_cgroup = metrics["cgroup"]
                    if observed_memory is not None and observed_memory > session.worker_memory_bytes:
                        raise RuntimeError("Candidate worker exceeded observed RSS budget; terminated (not cgroup containment)")
                    try: proc.wait(timeout=.1)
                    except subprocess.TimeoutExpired: pass
                if proc.returncode != 0 or not output.exists(): raise RuntimeError(f"Candidate worker failed ({proc.returncode}); see {token}.worker.log")
            except subprocess.TimeoutExpired as exc: raise RuntimeError("Candidate worker wall-time limit exceeded") from exc
            finally:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
                final_metrics = session.sampler.sample_now()
                final_worker_metrics = final_metrics.get("workers", {})
                final_rss = final_worker_metrics.get("rss_bytes")
                if final_rss is not None:
                    worker_rss_peak = max(worker_rss_peak or 0, final_rss)
                if final_metrics.get("cgroup", {}).get("scope") == "workers":
                    worker_cgroup = final_metrics["cgroup"]
                if not hasattr(session, "_turn_worker_resource_receipts"):
                    session._turn_worker_resource_receipts = []
                session._turn_worker_resource_receipts.append({
                    "sampling_status": worker_sample_status,
                    "sample_count": worker_sample_count,
                    "observed_rss_peak_bytes": worker_rss_peak,
                    "configured_memory_budget_bytes": session.worker_memory_bytes,
                    "memory_enforcement": "RLIMIT_AS plus sampled RSS terminate on Linux" if sys.platform.startswith("linux") else "sampled RSS terminate only",
                    "cpu_limit_seconds": int(session.evaluation_timeout_seconds),
                    "cpu_enforcement": "RLIMIT_CPU in candidate worker",
                    "wall_limit_seconds": session.evaluation_timeout_seconds,
                    "wall_enforcement": "controller deadline with process-group termination",
                    "cgroup": worker_cgroup or {"scope": "workers", "verified_limits": False, "available": False},
                    "cgroup_enforced": bool(worker_cgroup and worker_cgroup.get("scope") == "workers" and worker_cgroup.get("verified_limits")),
                })
                session.sampler.unregister_worker(proc.pid)
        return read_worker_result(output, request_identity)

    def run_script(self, session, params):
        if source_identity(REPO_ROOT) != session.binding['source_sha256']:
            raise ValueError('Worker source changed since session binding')
        from .python_tool import run_python_script
        filename = session._script_name(params.get("filename"))
        registered = session.state["script_registry"].get(filename)
        scripts = session.workspace_root / "scripts"
        if scripts.is_symlink() or scripts.resolve().parent != session.workspace_root:
            raise ValueError("Script directory must be confined to workspace")
        path = scripts / filename
        if not registered or path.is_symlink():
            raise ValueError("Script missing or changed since governed write")
        # Snapshot the checked bytes and mount only that snapshot. Checking a
        # source file and subsequently mounting the original leaves a race.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 100000:
                raise ValueError("Script must be a bounded regular file")
            source = stream.read(100001)
        if len(source) > 100000 or hashlib.sha256(source).hexdigest() != registered["sha256"]:
            raise ValueError("Script missing or changed since governed write")
        with tempfile.TemporaryDirectory(prefix=".run-", dir=scripts) as stage:
            stage = Path(stage)
            staged_script = stage / filename
            staged_script.write_bytes(source)
            staged_script.chmod(0o444)
            data = stage / "data"
            data.mkdir(mode=0o755)
            staged_data = data / "development.csv"
            shutil.copyfile(session.dataset_path, staged_data)
            if streaming_file_sha256(staged_data) != session.binding["data_sha256"]:
                raise ValueError("Development data changed since checkpoint binding")
            staged_data.chmod(0o444)
            result = run_python_script(staged_script, scripts, data_root=data, sandbox_image=session.sandbox_image)
        if hasattr(result, "to_dict"): result = result.to_dict()
        return {"python_result": result, "advisory_only": True}
