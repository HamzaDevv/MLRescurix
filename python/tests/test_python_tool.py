import io
import base64
import json
import subprocess
import shutil
import sys
import threading
from pathlib import Path

from mlrescurix_worker.scientific import python_tool


class FakeChild:
    def __init__(self, args, rc=0, user_stdout=b"ok\n", user_stderr=b"", hang=False):
        self.args = args
        self.pid = 12345
        status = {"exit_code": rc, "stdout_b64": base64.b64encode(user_stdout).decode(),
                  "stderr_b64": base64.b64encode(user_stderr).decode(),
                  "output_limit": False, "artifact_error": None,
                  "resources": {"available": True, "source": "test cgroup receipt",
                                "cgroup_v2": {"memory_peak_bytes": 1024}}}
        self.stdout = io.BytesIO(b"" if hang else b"MLR_PYTHON_RESULT:" + json.dumps(status).encode() + b"\n")
        self.stderr = io.BytesIO(b"")
        # The supervisor stays live after reporting status until the host copies outputs.
        self.returncode = None
        self.hang = hang

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0 if not self.hang else -9
        return self.returncode

    def kill(self):
        self.returncode = -9


def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    script = root / "analysis.py"
    script.write_text("print('sample')\n", encoding="utf-8")
    return root, script


def install_fake_docker(monkeypatch, factory):
    import tarfile
    monkeypatch.setattr(python_tool.shutil, "which", lambda _: "/fake/docker")

    def fake_popen(args, **kwargs):
        if args[1:2] == ["exec"]:
            payload = io.BytesIO()
            content = b'{"score":0.8}'
            with tarfile.open(fileobj=payload, mode="w") as archive:
                entry = tarfile.TarInfo("./metrics.json")
                entry.size = len(content)
                archive.addfile(entry, io.BytesIO(content))
            child = FakeChild(args)
            child.stdout = io.BytesIO(payload.getvalue())
            child.returncode = 0
            return child
        return factory(args, **kwargs)

    monkeypatch.setattr(python_tool.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(python_tool.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b""))


def test_success_returns_bounded_result_and_artifact_hash(tmp_path, monkeypatch):
    root, script = workspace(tmp_path)

    install_fake_docker(monkeypatch, lambda *a, **k: FakeChild(a[0]))
    result = python_tool.run_python_script(script, root, sandbox_image="test-image")
    assert result["status"] == "success"
    assert result["stdout"] == "ok\n"
    assert result["artifacts"][0]["path"] == "metrics.json"
    assert len(result["artifacts"][0]["sha256"]) == 64
    assert Path(result["artifact_dir"], "metrics.json").exists()
    assert result["resources"]["cgroup_v2"]["memory_peak_bytes"] == 1024


def test_nonzero_exit_is_structured_error(tmp_path, monkeypatch):
    root, script = workspace(tmp_path)
    install_fake_docker(monkeypatch, lambda *a, **k: FakeChild(a[0], rc=7, user_stderr=b"bad"))
    result = python_tool.run_python_script(script, root, sandbox_image="test-image")
    assert result["status"] == "script_error"
    assert result["exit_code"] == 7
    assert "status 7" in result["error"]


def test_timeout_kills_process_group_and_container(tmp_path, monkeypatch):
    root, script = workspace(tmp_path)
    child = FakeChild([], hang=True)
    install_fake_docker(monkeypatch, lambda *a, **k: child)
    killed = []
    monkeypatch.setattr(python_tool, "_kill_container", lambda name: killed.append(name))
    monkeypatch.setattr(python_tool.os, "killpg", lambda *args: None)
    result = python_tool.run_python_script(script, root, sandbox_image="test-image",
                                           limits=python_tool.PythonToolLimits(timeout_seconds=0.02))
    assert result["status"] == "timeout"
    assert killed
    assert child.returncode == -9


def test_path_escape_fails_before_spawning(tmp_path, monkeypatch):
    root, _ = workspace(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    monkeypatch.setattr(python_tool.shutil, "which", lambda _: "/fake/docker")
    def should_not_spawn(*args, **kwargs):
        raise AssertionError("must validate path before launch")
    monkeypatch.setattr(python_tool.subprocess, "Popen", should_not_spawn)
    result = python_tool.run_python_script(outside, root, sandbox_image="test-image")
    assert result["status"] == "execution_error"
    assert "escapes" in result["error"]


def test_refuses_host_fallback_without_docker(tmp_path, monkeypatch):
    root, script = workspace(tmp_path)
    monkeypatch.setattr(python_tool.shutil, "which", lambda _: None)
    result = python_tool.run_python_script(script, root)
    assert result["status"] == "sandbox_unavailable"
    assert "refusing host Python" in result["error"]


def test_container_supervisor_source_compiles():
    compile(python_tool._CONTAINER_SUPERVISOR, "<python-tool-supervisor>", "exec")


def test_nonfinite_execution_limits_are_rejected_before_launch():
    import pytest
    for limits in (python_tool.PythonToolLimits(timeout_seconds=float("nan")),
                   python_tool.PythonToolLimits(timeout_seconds=float("inf")),
                   python_tool.PythonToolLimits(cpus=float("nan")),
                   python_tool.PythonToolLimits(cpus=float("inf"))):
        with pytest.raises(ValueError, match="limits must be positive"):
            python_tool._validate_limits(limits)


def test_drain_observes_short_line_while_pipe_writer_remains_open():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys,time;sys.stdout.write('status\\n');sys.stdout.flush();time.sleep(1)"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    captured, exceeded, ready = bytearray(), threading.Event(), threading.Event()
    reader = threading.Thread(target=python_tool._drain,
                              args=(proc.stdout, 1024, captured, exceeded, ready), daemon=True)
    reader.start()
    try:
        assert ready.wait(0.5)
        assert proc.poll() is None
        assert captured == b"status\n"
        assert not exceeded.is_set()
    finally:
        proc.kill()
        proc.wait()
        reader.join(timeout=2)


def test_docker_integration_when_daemon_available(tmp_path):
    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "version"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=5).returncode:
        import pytest
        pytest.skip("Docker daemon is unavailable")
    root, script = workspace(tmp_path)
    data = tmp_path / "development-data"
    data.mkdir()
    (data / "values.txt").write_text("dev-only", encoding="utf-8")
    script.write_text(
        "import os, pathlib\n"
        "value = pathlib.Path(os.environ['MLRESCURIX_DATA'], 'values.txt').read_text()\n"
        "pathlib.Path(os.environ['MLRESCURIX_OUTPUTS'], 'result.txt').write_text(value)\n"
        "print(value)\n", encoding="utf-8")
    result = python_tool.run_python_script(script, root, data_root=data,
                                           limits=python_tool.PythonToolLimits(timeout_seconds=20))
    assert result["status"] == "success"
    assert result["stdout"].strip() == "dev-only"
    assert Path(result["artifact_dir"], "result.txt").read_text(encoding="utf-8") == "dev-only"


def test_docker_candidate_predictions_are_scored_against_controller_labels(tmp_path):
    """Exercise isolated generated code and independently score its output.

    This is a synthetic component test, not an integrated promotion path.
    """
    import csv
    import hashlib
    import pytest

    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "version"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=5).returncode:
        pytest.skip("Docker daemon is unavailable")
    root, script = workspace(tmp_path)
    data = tmp_path / "development-inputs"
    data.mkdir()
    (data / "train.csv").write_text("x,y\n0,1\n1,3\n2,5\n", encoding="utf-8")
    (data / "validation.csv").write_text("id,x\n10,3\n11,4\n", encoding="utf-8")
    # Validation labels remain in the controller process and are never mounted.
    private_labels = {"10": 7.0, "11": 9.0}
    script.write_text(
        "import csv, os, pathlib\n"
        "data=pathlib.Path(os.environ['MLRESCURIX_DATA'])\n"
        "out=pathlib.Path(os.environ['MLRESCURIX_OUTPUTS'])\n"
        "with (data/'train.csv').open() as f: train=list(csv.DictReader(f))\n"
        "slope=(float(train[-1]['y'])-float(train[0]['y']))/(float(train[-1]['x'])-float(train[0]['x']))\n"
        "intercept=float(train[0]['y'])-slope*float(train[0]['x'])\n"
        "with (data/'validation.csv').open() as f: rows=list(csv.DictReader(f))\n"
        "with (out/'predictions.csv').open('w', newline='') as f:\n"
        " w=csv.DictWriter(f, fieldnames=['id','prediction']); w.writeheader()\n"
        " for row in rows: w.writerow({'id':row['id'], 'prediction':slope*float(row['x'])+intercept})\n",
        encoding="utf-8")
    source_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    result = python_tool.run_python_script(script, root, data_root=data,
                                           limits=python_tool.PythonToolLimits(timeout_seconds=20))
    assert result["status"] == "success"
    prediction_path = Path(result["artifact_dir"], "predictions.csv")
    with prediction_path.open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == len(private_labels)
    assert {row["id"] for row in predictions} == set(private_labels)
    mean_absolute_error = sum(abs(float(row["prediction"]) - private_labels[row["id"]])
                              for row in predictions) / len(private_labels)
    assert mean_absolute_error == 0
    assert result["artifacts"][0]["sha256"] == hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    assert hashlib.sha256(script.read_bytes()).hexdigest() == source_hash


def test_docker_worker_cannot_impersonate_supervisor_or_leave_detached_child(tmp_path):
    import pytest
    import time

    docker = shutil.which("docker")
    if docker is None or subprocess.run([docker, "version"], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=5).returncode:
        pytest.skip("Docker daemon is unavailable")
    root, script = workspace(tmp_path)
    script.write_text(
        "import os, pathlib, signal, time\n"
        "out=pathlib.Path(os.environ['MLRESCURIX_OUTPUTS'])\n"
        "caps=[line for line in pathlib.Path('/proc/self/status').read_text().splitlines() if line.startswith('CapEff:')][0]\n"
        "(out/'capabilities').write_text(caps.split()[1])\n"
        "try:\n"
        " os.kill(1, signal.SIGKILL)\n"
        "except PermissionError:\n"
        " (out/'supervisor-protected').write_text('yes')\n"
        "else:\n"
        " (out/'supervisor-protected').write_text('no')\n"
        "r,w=os.pipe()\n"
        "if os.fork() == 0:\n"
        " os.setsid()\n"
        " os.write(w,b'1')\n"
        " time.sleep(0.2)\n"
        " (out/'late-write').write_text('escaped')\n"
        " os._exit(0)\n"
        "os.read(r,1)\n"
        "print('finished')\n", encoding="utf-8")
    result = python_tool.run_python_script(script, root,
                                           limits=python_tool.PythonToolLimits(timeout_seconds=20))
    assert result["status"] == "success", result
    artifacts = Path(result["artifact_dir"])
    assert (artifacts / "supervisor-protected").read_text() == "yes"
    assert int((artifacts / "capabilities").read_text(), 16) == 0
    time.sleep(0.4)
    assert not (artifacts / "late-write").exists()


def test_streamed_artifact_archive_rejects_escape_links_and_oversize(tmp_path):
    import tarfile
    import pytest

    for name, kind, size in [("../escape.txt", tarfile.REGTYPE, 1),
                             ("link", tarfile.SYMTYPE, 0),
                             ("large.txt", tarfile.REGTYPE, 5)]:
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            entry.size = size
            if kind == tarfile.SYMTYPE:
                entry.linkname = "../escape.txt"
            archive.addfile(entry, io.BytesIO(b"x" * size) if size else None)
        payload.seek(0)
        root = tmp_path / name.replace("/", "_")
        root.mkdir()
        with pytest.raises(ValueError):
            python_tool._extract_bounded_artifacts(payload, root, max_bytes=4, max_files=2)
    assert not (tmp_path / "escape.txt").exists()


def test_collection_checks_archive_before_host_extraction(tmp_path, monkeypatch):
    import tarfile
    import pytest
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("../escape.txt")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    commands = []
    def fake_popen(args, **kwargs):
        commands.append(args)
        child = FakeChild(args)
        child.stdout = io.BytesIO(payload.getvalue())
        child.returncode = 0
        return child
    monkeypatch.setattr(python_tool.subprocess, "Popen", fake_popen)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        python_tool._copy_container_artifacts("docker", "container", out)
    assert commands[0][1] == "exec"
    assert not (tmp_path / "escape.txt").exists()


def test_streamed_artifact_archive_copies_bounded_regular_file(tmp_path):
    import tarfile
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        entry = tarfile.TarInfo("./results/value.txt")
        entry.size = 2
        archive.addfile(entry, io.BytesIO(b"42"))
    payload.seek(0)
    python_tool._extract_bounded_artifacts(payload, tmp_path, max_bytes=2, max_files=2)
    assert (tmp_path / "results/value.txt").read_bytes() == b"42"
