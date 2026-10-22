"""Offline tests for truthful process and cgroup resource monitoring."""

import time

import psutil

from mlrescurix_worker.scientific.resource_monitor import ProcessTreeSampler


def test_samples_controller_and_registered_worker_separately_and_stops(monkeypatch):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.cpu_calls = 0

        def children(self, recursive=True):
            return []

        def memory_info(self):
            return type("Memory", (), {"rss": self.pid * 10})()

        def cpu_percent(self, interval=None):
            self.cpu_calls += 1
            return 0.0 if self.cpu_calls == 1 else float(self.pid % 100)

    monkeypatch.setattr(psutil, "Process", FakeProcess)
    sampler = ProcessTreeSampler(pid=101, interval_seconds=0.02)
    sampler.register_worker(202)
    sampler.start()
    time.sleep(0.10)
    metrics = sampler.sample_now()
    stopped = sampler.stop()
    assert metrics["sampling_status"] == "available"
    assert metrics["controller"]["status"] == "available"
    assert metrics["workers"]["status"] == "available"
    assert metrics["workers"]["rss_bytes"] == 2020
    assert metrics["worker_peak_rss_bytes"] >= metrics["workers"]["rss_bytes"]
    assert metrics["workers"]["cpu_percent"] == 2.0
    assert metrics["cgroup"]["scope"] == "workers"
    assert metrics["containment_status"] != "cgroup_v2_limited" or metrics["cgroup"]["verified_limits"]
    assert sampler._thread is None
    assert not sampler._processes


def test_permission_denial_is_unavailable_not_zero(monkeypatch):
    sampler = ProcessTreeSampler(pid=88888888)

    def denied(pid):
        raise psutil.AccessDenied(pid=pid)

    monkeypatch.setattr(psutil, "Process", denied)
    metrics = sampler.sample_now()
    assert metrics["sampling_status"] == "unavailable"
    assert metrics["controller"]["rss_bytes"] is None
    assert metrics["controller"]["cpu_percent"] is None
    assert metrics["current_tree_rss_bytes"] is None
    assert metrics["cgroup_enforced"] is False
    assert any("permission_denied" in error for error in metrics["errors"])


def test_child_enumeration_denial_retains_readable_root(monkeypatch):
    class PartiallyReadableProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=True):
            raise psutil.AccessDenied(pid=self.pid)

        def memory_info(self):
            return type("Memory", (), {"rss": 4096})()

        def cpu_percent(self, interval=None):
            return 0.0

    monkeypatch.setattr(psutil, "Process", PartiallyReadableProcess)
    sampler = ProcessTreeSampler(pid=456789)
    metrics = sampler.sample_now()
    assert metrics["controller"]["rss_bytes"] == 4096
    assert metrics["controller"]["status"] == "partial"
    assert metrics["sampling_status"] == "partial"
    assert any("tree_enumeration_permission_denied" in error for error in metrics["errors"])
    assert metrics["cgroup"]["scope"] == "controller"


def test_context_manager_runs_and_cleans_sampler_thread():
    with ProcessTreeSampler(interval_seconds=0.02) as sampler:
        assert sampler._thread is not None and sampler._thread.is_alive()
        time.sleep(0.06)
    assert sampler._thread is None
    assert not sampler._processes
