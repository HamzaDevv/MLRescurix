"""Continuous process and cgroup resource sampling.

The sampler reports controller and worker processes separately. A process is
only called a worker when it was explicitly registered as one. Cgroup
containment is reported only when every sampled PID's cgroup membership can be
resolved and both finite memory and CPU limits are present.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import psutil


class ProcessTreeSampler:
    """Sample a controller process tree and explicitly registered workers.

    ``sample_now`` remains a synchronous, dictionary-returning API. ``start``
    runs sampling at ``interval_seconds`` in a daemon thread; callers should
    use ``stop`` or the context manager to clean it up.
    """

    def __init__(self, pid: Optional[int] = None, interval_seconds: float = 0.05):
        self.target_pid = int(pid if pid is not None else os.getpid())
        self.interval = max(0.01, float(interval_seconds))
        self.worker_pids: set[int] = set()
        self.worker_peak_rss_bytes: Optional[int] = None
        self.controller_peak_rss_bytes: Optional[int] = None
        self.worker_cpu_samples: deque[float] = deque(maxlen=4096)
        self.controller_cpu_samples: deque[float] = deque(maxlen=4096)
        self.cgroup_enforced = False
        self.cgroup_memory_peak_bytes: Optional[int] = None
        self.cgroup_oom_events: Optional[int] = None
        self.containment_status = "unverified"
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processes: dict[int, psutil.Process] = {}
        self._cpu_primed: set[int] = set()
        self._last_metrics: Dict[str, Any] = self._empty_metrics()

    def register_worker(self, pid: int) -> None:
        """Register a governed worker PID for subsequent independent sampling."""
        with self._lock:
            self.worker_pids.add(int(pid))

    def unregister_worker(self, pid: int) -> None:
        with self._lock:
            pid = int(pid)
            self.worker_pids.discard(pid)
            self._processes.pop(pid, None)
            self._cpu_primed.discard(pid)

    def start(self, worker_pids: Optional[Iterable[int]] = None) -> "ProcessTreeSampler":
        if worker_pids is not None:
            with self._lock:
                self.worker_pids.update(int(pid) for pid in worker_pids)
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self
            self._stop_event.clear()
            # Prime persistent psutil objects. The first CPU reading is
            # intentionally discarded because psutil defines it as meaningless.
            self.sample_now()
            self._thread = threading.Thread(target=self._sampling_loop, name="resource-sampler", daemon=True)
            self._thread.start()
        return self

    def stop(self, timeout: Optional[float] = 2.0) -> Dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            self._thread = None
            self._processes.clear()
            self._cpu_primed.clear()
            return dict(self._last_metrics)

    def __enter__(self) -> "ProcessTreeSampler":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()

    def sample_now(self, pid: Optional[int] = None) -> Dict[str, Any]:
        """Sample the selected process tree immediately.

        If ``pid`` is provided it acts as the controller root for this sample;
        separately registered worker PIDs remain independently labeled.
        Inaccessible or vanished processes are reported as unavailable rather
        than being silently converted to zero usage.
        """
        controller_pid = int(pid if pid is not None else self.target_pid)
        with self._lock:
            controller_pids, controller_error = self._tree_pids(controller_pid)
            worker_roots = set(self.worker_pids)
            worker_pids: set[int] = set()
            worker_errors: list[str] = []
            for worker_pid in worker_roots:
                found, error = self._tree_pids(worker_pid)
                worker_pids.update(found)
                if error:
                    worker_errors.append(error)
            # Explicit worker registration takes precedence if a worker is a
            # descendant of the controller process.
            controller_pids -= worker_pids
            controller = self._sample_group(controller_pids)
            workers = self._sample_group(worker_pids)
            if controller["rss_bytes"] is not None:
                self.controller_peak_rss_bytes = max(self.controller_peak_rss_bytes or 0, controller["rss_bytes"])
            if workers["rss_bytes"] is not None:
                self.worker_peak_rss_bytes = max(self.worker_peak_rss_bytes or 0, workers["rss_bytes"])
            if controller["cpu_percent"] is not None:
                self.controller_cpu_samples.append(controller["cpu_percent"])
            if workers["cpu_percent"] is not None:
                self.worker_cpu_samples.append(workers["cpu_percent"])
            all_pids = controller_pids | worker_pids
            # Resource containment claims describe governed workers. If there
            # are no registered workers, inspect the controller for diagnostics
            # while preserving the explicit scope in the returned record.
            cgroup_scope = "workers" if worker_pids else "controller"
            cgroup = self._inspect_cgroups(worker_pids or controller_pids)
            cgroup["scope"] = cgroup_scope
            self.cgroup_enforced = bool(cgroup.get("verified_limits"))
            self.containment_status = "cgroup_v2_limited" if self.cgroup_enforced else "unverified_or_uncontained"
            self.cgroup_memory_peak_bytes = cgroup.get("memory_peak_bytes")
            self.cgroup_oom_events = cgroup.get("oom_events")
            metrics = {
                "controller": controller,
                "workers": workers,
                "controller_pid": controller_pid,
                "worker_pids": sorted(worker_roots),
                "current_tree_rss_bytes": self._sum_if_available(controller["rss_bytes"], workers["rss_bytes"]),
                "worker_peak_rss_bytes": self.worker_peak_rss_bytes,
                "controller_peak_rss_bytes": self.controller_peak_rss_bytes,
                "avg_cpu_percent": self._mean(self.controller_cpu_samples + self.worker_cpu_samples),
                "cgroup": cgroup,
                "cgroup_enforced": self.cgroup_enforced,
                "containment_status": self.containment_status,
                "sampling_status": "available" if controller["status"] == "available" or workers["status"] == "available" else "unavailable",
                "errors": [e for e in [controller_error, *worker_errors] if e],
            }
            self._last_metrics = metrics
            return dict(metrics)

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self.interval):
            try:
                self.sample_now()
            except Exception:
                # A periodic monitor must not crash the governed workload.
                continue

    def _tree_pids(self, pid: int) -> tuple[set[int], Optional[str]]:
        try:
            process = self._processes.get(pid)
            if process is None:
                process = psutil.Process(pid)
                self._processes[pid] = process
        except psutil.AccessDenied:
            return set(), f"permission_denied:{pid}"
        except psutil.NoSuchProcess:
            return set(), f"process_unavailable:{pid}"
        except PermissionError:
            return set(), f"permission_denied:{pid}"
        except OSError as exc:
            return set(), f"process_error:{pid}:{type(exc).__name__}"
        try:
            children = process.children(recursive=True)
            return {pid, *(child.pid for child in children)}, None
        except (psutil.AccessDenied, PermissionError):
            # The root itself may still be readable. Retain it and surface that
            # descendant enumeration was partial in the diagnostics.
            return {pid}, f"tree_enumeration_permission_denied:{pid}"
        except psutil.NoSuchProcess:
            return set(), f"process_unavailable:{pid}"
        except OSError as exc:
            return {pid}, f"tree_enumeration_error:{pid}:{type(exc).__name__}"

    def _sample_group(self, pids: set[int]) -> Dict[str, Any]:
        rss = 0
        rss_ok = False
        cpu_values: list[float] = []
        errors: list[str] = []
        for pid in sorted(pids):
            try:
                process = self._processes.get(pid)
                if process is None:
                    process = psutil.Process(pid)
                    self._processes[pid] = process
                rss += int(process.memory_info().rss)
                rss_ok = True
                cpu = float(process.cpu_percent(interval=None))
                if pid not in self._cpu_primed:
                    self._cpu_primed.add(pid)
                else:
                    cpu_values.append(cpu)
            except (psutil.AccessDenied, PermissionError):
                errors.append(f"permission_denied:{pid}")
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                errors.append(f"process_unavailable:{pid}")
            except OSError as exc:
                errors.append(f"process_error:{pid}:{type(exc).__name__}")
        return {
            "rss_bytes": rss if rss_ok else None,
            "cpu_percent": self._mean(cpu_values),
            "process_count": len(pids),
            "status": "available" if rss_ok else "unavailable",
            "errors": errors,
        }

    def _inspect_cgroups(self, pids: set[int]) -> Dict[str, Any]:
        unavailable = {"available": False, "verified_limits": False, "membership_verified": False,
                       "memory_limit_bytes": None, "cpu_quota_cores": None, "memory_current_bytes": None,
                       "memory_peak_bytes": None, "oom_events": None, "errors": []}
        if not pids or not Path("/proc/self/cgroup").exists():
            return unavailable
        roots: set[Path] = set()
        errors: list[str] = []
        try:
            mount = Path("/sys/fs/cgroup")
            if not mount.is_dir():
                return unavailable
            for pid in pids:
                try:
                    content = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
                    unified = next((line.split("::", 1)[1] for line in content.splitlines() if "::" in line), None)
                    if unified is None:
                        errors.append(f"no_unified_cgroup:{pid}")
                        continue
                    candidate = (mount / unified.lstrip("/")).resolve()
                    mount_resolved = mount.resolve()
                    if candidate != mount_resolved and mount_resolved not in candidate.parents:
                        errors.append(f"cgroup_path_outside_mount:{pid}")
                        continue
                    roots.add(candidate)
                except (OSError, PermissionError) as exc:
                    errors.append(f"membership_unavailable:{pid}:{type(exc).__name__}")
            if len(roots) != 1 or errors:
                unavailable["errors"] = errors or ["processes_have_different_cgroups"]
                return unavailable
            root = next(iter(roots))
            memory_limit = self._read_limit(root / "memory.max")
            cpu_quota = self._read_cpu_limit(root / "cpu.max")
            current = self._read_int(root / "memory.current")
            peak = self._read_int(root / "memory.peak")
            events = self._read_key_value(root / "memory.events")
            oom = events.get("oom") if events else None
            verified = memory_limit is not None and cpu_quota is not None
            return {"available": True, "verified_limits": verified, "membership_verified": True,
                    "path": str(root), "memory_limit_bytes": memory_limit, "cpu_quota_cores": cpu_quota,
                    "memory_current_bytes": current, "memory_peak_bytes": peak, "oom_events": oom,
                    "errors": [] if verified else ["finite_memory_and_cpu_limits_not_both_verified"]}
        except (OSError, PermissionError):
            unavailable["errors"] = ["cgroup_inspection_permission_denied"]
            return unavailable

    @staticmethod
    def _read_limit(path: Path) -> Optional[int]:
        try:
            value = path.read_text(encoding="utf-8").strip()
            return int(value) if value and value != "max" and int(value) > 0 else None
        except (OSError, ValueError, PermissionError):
            return None

    @staticmethod
    def _read_cpu_limit(path: Path) -> Optional[float]:
        try:
            quota, period = path.read_text(encoding="utf-8").split()[:2]
            if quota == "max":
                return None
            q, p = int(quota), int(period)
            return q / p if q > 0 and p > 0 else None
        except (OSError, ValueError, PermissionError):
            return None

    @staticmethod
    def _read_int(path: Path) -> Optional[int]:
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError, PermissionError):
            return None

    @staticmethod
    def _read_key_value(path: Path) -> Dict[str, int]:
        try:
            result: Dict[str, int] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                key, value = line.split()[:2]
                result[key] = int(value)
            return result
        except (OSError, ValueError, PermissionError):
            return {}

    @staticmethod
    def _mean(values: list[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _sum_if_available(first: Optional[int], second: Optional[int]) -> Optional[int]:
        return first + second if first is not None and second is not None else None

    def _empty_metrics(self) -> Dict[str, Any]:
        return {"sampling_status": "unavailable", "controller": {"status": "unavailable"},
                "workers": {"status": "unavailable"}, "current_tree_rss_bytes": None,
                "worker_peak_rss_bytes": None, "controller_peak_rss_bytes": None,
                "avg_cpu_percent": None, "cgroup_enforced": False,
                "containment_status": "unverified_or_uncontained"}
