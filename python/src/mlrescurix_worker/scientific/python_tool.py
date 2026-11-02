"""Bounded execution for generated scientific Python scripts.

Generated code is untrusted. This module only executes it in an OCI container and
never falls back to the host interpreter. Container isolation is a useful boundary,
not a proof against every kernel/runtime escape.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import os
import shutil
import signal
import subprocess
import tarfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PythonToolLimits:
    timeout_seconds: float = 60.0
    memory_mb: int = 1024
    cpus: float = 1.0
    pids: int = 64
    max_output_bytes: int = 1_000_000
    max_artifact_bytes: int = 100_000_000
    max_artifact_files: int = 1_000


@dataclass
class PythonToolResult:
    ok: bool
    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    elapsed_ms: int
    artifacts: list[dict[str, Any]]
    error: str | None = None
    artifact_dir: str | None = None
    resources: dict[str, Any] = field(default_factory=lambda: {
        "available": False, "reason": "resource receipt unavailable"
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contained(path: Path, root: Path) -> Path:
    root_real = root.resolve(strict=True)
    path_real = path.resolve(strict=True)
    if path_real == root_real or root_real not in path_real.parents:
        raise ValueError(f"path escapes its permitted root: {path}")
    return path_real


def _reject_symlink_components(path: Path) -> None:
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlink path components are forbidden: {cursor}")


def _validate_limits(limits: PythonToolLimits) -> None:
    integer_limits = (limits.memory_mb, limits.pids, limits.max_output_bytes,
                      limits.max_artifact_bytes, limits.max_artifact_files)
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in integer_limits)
            or not math.isfinite(limits.timeout_seconds) or not math.isfinite(limits.cpus)
            or limits.timeout_seconds <= 0 or limits.memory_mb <= 0 or limits.cpus <= 0
            or limits.pids <= 0 or limits.max_output_bytes <= 0
            or limits.max_artifact_bytes <= 0 or limits.max_artifact_files <= 0):
        raise ValueError("all Python tool limits must be positive")


def _drain(pipe: Any, cap: int, captured: bytearray, exceeded: threading.Event,
           ready: threading.Event | None = None) -> None:
    total = 0
    while True:
        # BufferedReader.read() may wait for the full size or EOF. Docker's
        # supervisor emits its status line and then stays alive for docker cp.
        if hasattr(pipe, "read1"):
            chunk = pipe.read1(8192)
        else:
            chunk = os.read(pipe.fileno(), 8192)
        if not chunk:
            return
        total += len(chunk)
        if len(captured) < cap:
            captured.extend(chunk[: cap - len(captured)])
        if total > cap:
            exceeded.set()
        if ready is not None and b"\n" in chunk:
            ready.set()


def _kill_container(name: str) -> None:
    docker = shutil.which("docker")
    if docker:
        try:
            subprocess.run([docker, "kill", name], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=3, check=False, start_new_session=True)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            subprocess.run([docker, "rm", "-f", name], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=3, check=False, start_new_session=True)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _copy_container_artifacts(docker: str, name: str, output_dir: Path,
                              max_bytes: int = 100_000_000, max_files: int = 1_000) -> None:
    # Never let docker cp extract untrusted container paths onto the host.
    # Stream a tar archive and validate every entry before creating it.
    exec_proc = subprocess.Popen([docker, "exec", name, "tar", "-C", "/outputs", "-cf", "-", "."],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
    timer = threading.Timer(10, exec_proc.kill)
    timer.start()
    try:
        _extract_bounded_artifacts(exec_proc.stdout, output_dir, max_bytes, max_files)
        if exec_proc.wait(timeout=2) != 0:
            raise RuntimeError("sandbox artifact stream failed or timed out")
    finally:
        timer.cancel()
        if exec_proc.poll() is None:
            exec_proc.kill()
        exec_proc.wait(timeout=2)
        exec_proc.stdout.close()


def _extract_bounded_artifacts(stream, output_dir: Path, max_bytes: int, max_files: int) -> None:
    """Copy regular tar members only; never trust archive paths, links or sizes."""
    total = count = 0
    root = output_dir.resolve()
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if member.name in (".", "./") and member.isdir():
                continue
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("artifact archive path escapes output directory")
            destination = root / relative
            if destination.resolve() == root or root not in destination.resolve().parents:
                raise ValueError("artifact archive path escapes output directory")
            count += 1
            if count > max_files:
                raise ValueError("artifact archive exceeded entry limit")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isreg() or member.issparse():
                raise ValueError("artifact archive contains a link, sparse, or non-regular file")
            total += member.size
            if member.size < 0 or total > max_bytes:
                raise ValueError("artifact archive exceeded byte limit")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("artifact archive file contents unavailable")
            with source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=65536)


def _artifact_manifest(output_dir: Path, cap: int, max_files: int) -> list[dict[str, Any]]:
    files = sorted(output_dir.rglob("*"))
    manifest: list[dict[str, Any]] = []
    total = 0
    for path in files:
        if path.is_symlink():
            raise ValueError(f"artifact symlinks are forbidden: {path.relative_to(output_dir)}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if len(manifest) >= max_files:
            raise ValueError("artifact output exceeded configured file count")
        total += size
        if total > cap:
            raise ValueError("artifact output exceeded configured byte limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append({"path": path.relative_to(output_dir).as_posix(), "size_bytes": size,
                         "sha256": digest.hexdigest()})
    return manifest


_CONTAINER_SUPERVISOR = r'''import base64, json, os, resource, signal, subprocess, sys, threading
limit_out, limit_err = int(os.environ["MLR_MAX_STDOUT"]), int(os.environ["MLR_MAX_STDERR"])
limit_bytes, limit_files = int(os.environ["MLR_MAX_ARTIFACT_BYTES"]), int(os.environ["MLR_MAX_ARTIFACT_FILES"])
captures, overflow = [bytearray(), bytearray()], [False, False]
def drain(pipe, i, cap):
    while True:
        data = pipe.read1(8192) if hasattr(pipe, "read1") else os.read(pipe.fileno(), 8192)
        if not data: break
        room = max(0, cap - len(captures[i]))
        captures[i].extend(data[:room])
        if len(data) > room: overflow[i] = True
def drop_privileges():
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
proc = subprocess.Popen([sys.executable, sys.argv[1]], stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                        preexec_fn=drop_privileges)
threads = [threading.Thread(target=drain, args=(proc.stdout, 0, limit_out)),
           threading.Thread(target=drain, args=(proc.stderr, 1, limit_err))]
for t in threads: t.start()
code = proc.wait()
try: os.killpg(proc.pid, signal.SIGKILL)
except ProcessLookupError: pass
# A worker can create a new session and escape its original process group.
# PID 1 is the privileged supervisor; signal every remaining process in this
# container's PID namespace before inspecting or collecting its outputs.
try: os.kill(-1, signal.SIGKILL)
except ProcessLookupError: pass
for t in threads: t.join(2)
count, total = 0, 0
try:
    for root, dirs, files in os.walk("/outputs", followlinks=False):
        for name in list(dirs):
            if os.path.islink(os.path.join(root, name)): raise ValueError("artifact symlink")
        for name in files:
            p = os.path.join(root, name)
            if os.path.islink(p) or not os.path.isfile(p): raise ValueError("non-regular artifact")
            count += 1
            if count > limit_files: raise ValueError("artifact file count limit exceeded")
            total += os.path.getsize(p)
            if total > limit_bytes: raise ValueError("artifact byte limit exceeded")
    artifact_error = None
except Exception as e: artifact_error = str(e)
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = {"available": False, "scope": "container cgroup plus direct child fallback",
           "host_process_tree_sampled": False, "source": None,
           "direct_child_peak_rss_bytes": int(usage.ru_maxrss * 1024),
           "user_cpu_seconds": float(usage.ru_utime), "system_cpu_seconds": float(usage.ru_stime),
           "cgroup_v2": {"memory_peak_bytes": None, "memory_max_bytes": None,
                         "cpu_usage_usec": None, "cpu_user_usec": None,
                         "cpu_system_usec": None, "cpu_quota_usec": None, "cpu_period_usec": None}}
if receipt["direct_child_peak_rss_bytes"] > 0 or receipt["user_cpu_seconds"] > 0 or receipt["system_cpu_seconds"] > 0:
    receipt["available"], receipt["source"] = True, "resource.getrusage(RUSAGE_CHILDREN)"
try:
    cg = "/sys/fs/cgroup"
    def read_int(name):
        try:
            value = open(os.path.join(cg, name), encoding="ascii").read().strip()
            return None if value == "max" else int(value)
        except (OSError, ValueError): return None
    mem_peak, mem_max = read_int("memory.peak"), read_int("memory.max")
    try:
        cpu = {}
        for line in open(os.path.join(cg, "cpu.stat"), encoding="ascii"):
            key, value = line.split()
            cpu[key] = int(value)
    except (OSError, ValueError): cpu = {}
    try:
        quota_text, period_text = open(os.path.join(cg, "cpu.max"), encoding="ascii").read().split()
        quota, period = (None if quota_text == "max" else int(quota_text)), int(period_text)
    except (OSError, ValueError): quota, period = None, None
    cg_receipt = receipt["cgroup_v2"]
    cg_receipt.update(memory_peak_bytes=mem_peak, memory_max_bytes=mem_max,
                      cpu_usage_usec=cpu.get("usage_usec"), cpu_user_usec=cpu.get("user_usec"),
                      cpu_system_usec=cpu.get("system_usec"), cpu_quota_usec=quota,
                      cpu_period_usec=period)
    if any(v is not None for v in (mem_peak, mem_max, cpu.get("usage_usec"), quota)):
        receipt["available"], receipt["source"] = True, "container cgroup v2 and resource.getrusage"
except Exception: pass
status = {"exit_code": code, "stdout_b64": base64.b64encode(captures[0]).decode("ascii"),
          "stderr_b64": base64.b64encode(captures[1]).decode("ascii"),
          "output_limit": any(overflow), "artifact_error": artifact_error,
          "resources": receipt}
print("MLR_PYTHON_RESULT:" + json.dumps(status, separators=(",", ":")), flush=True)
signal.pause()
'''


def run_python_script(
    script_path: str | Path,
    workspace_root: str | Path,
    *,
    data_root: str | Path | None = None,
    limits: PythonToolLimits | None = None,
    sandbox_image: str | None = None,
) -> dict[str, Any]:
    """Run a script in Docker with explicit read-only inputs and isolated outputs.

    `script_path` must be inside `workspace_root`. If provided, `data_root` is mounted
    separately at `/data` read-only. Only the requested script file is mounted; the
    rest of the host workspace is hidden from generated code. Script arguments are deliberately not accepted;
    generated scripts cannot dispatch arbitrary host commands through this API.
    """
    limits = limits or PythonToolLimits()
    _validate_limits(limits)
    started = time.monotonic()
    container = f"mlr-py-{os.getpid()}-{uuid.uuid4().hex[:16]}"
    output_dir: Path | None = None
    child: subprocess.Popen[bytes] | None = None
    try:
        workspace_input = Path(workspace_root)
        if workspace_input.is_symlink():
            raise ValueError("workspace must not be a symlink")
        workspace = workspace_input.resolve(strict=True)
        _reject_symlink_components(workspace)
        if not workspace.is_dir():
            raise ValueError("workspace must be a real directory")
        requested_script = Path(script_path)
        if not requested_script.is_absolute():
            requested_script = workspace / requested_script
        _reject_symlink_components(requested_script)
        script = _contained(requested_script, workspace)
        if not script.is_file() or script.suffix != ".py":
            raise ValueError("script must be a regular .py file inside the workspace")
        data_input = Path(data_root) if data_root is not None else workspace
        if data_input.is_symlink():
            raise ValueError("data_root must not be a symlink")
        _reject_symlink_components(data_input)
        data = data_input.resolve(strict=True)
        if not data.is_dir():
            raise ValueError("data_root must be a real directory")
        docker = shutil.which("docker")
        if not docker:
            return PythonToolResult(False, "sandbox_unavailable", None, "", "", 0, [],
                                    "Docker is required; refusing host Python execution").to_dict()
        image = sandbox_image or os.environ.get("MLRESCURIX_SANDBOX_IMAGE")
        if not image:
            image_ref = Path(__file__).resolve().parents[4] / "docker" / "scientific" / "image.ref"
            try:
                image = image_ref.read_text(encoding="utf-8").strip()
            except OSError:
                image = ""
        if not image:
            raise ValueError("scientific sandbox image is not configured")
        artifact_parent = workspace / ".python_tool_outputs"
        if artifact_parent.is_symlink():
            raise ValueError("artifact output directory cannot be a symlink")
        artifact_parent.mkdir(exist_ok=True)
        artifact_parent = _contained(artifact_parent, workspace)
        output_dir = artifact_parent / uuid.uuid4().hex
        output_dir.mkdir(mode=0o700)
        output_tmpfs_mb = max(1, (limits.max_artifact_bytes + 1024 * 1024 - 1) // (1024 * 1024))
        command = [docker, "run", "--pull=never", "--name", container,
                   "--network=none", "--read-only", "--user", "0:0",
                   "--cap-drop=ALL", "--cap-add=SETUID", "--cap-add=SETGID", "--cap-add=KILL",
                   "--security-opt=no-new-privileges:true",
                   "--memory", f"{limits.memory_mb}m", "--cpus", str(limits.cpus),
                   "--pids-limit", str(limits.pids),
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,uid=1000,gid=1000",
                   "--tmpfs", f"/outputs:rw,noexec,nosuid,size={output_tmpfs_mb}m,nr_inodes={limits.max_artifact_files},uid=1000,gid=1000",
                   "-v", f"{script}:/script.py:ro",
                   "-w", "/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "PYTHONUNBUFFERED=1", "--env", "TMPDIR=/tmp",
                   "--env", "MLRESCURIX_OUTPUTS=/outputs",
                   "--env", f"MLR_MAX_STDOUT={limits.max_output_bytes}",
                   "--env", f"MLR_MAX_STDERR={limits.max_output_bytes}",
                   "--env", f"MLR_MAX_ARTIFACT_BYTES={limits.max_artifact_bytes}",
                   "--env", f"MLR_MAX_ARTIFACT_FILES={limits.max_artifact_files}"]
        if data_root is not None:
            command.extend(["-v", f"{data}:/data:ro", "--env", "MLRESCURIX_DATA=/data"])
        else:
            command.extend(["--env", "MLRESCURIX_DATA="])
        threads = str(max(1, int(limits.cpus + 0.999)))
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
            command.extend(["--env", f"{var}={threads}"])
        command.extend(["--entrypoint", "python3", image, "-c", _CONTAINER_SUPERVISOR, "/script.py"])
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, start_new_session=True)
        status_buf, cli_err_buf = bytearray(), bytearray()
        overflow = threading.Event()
        status_ready = threading.Event()
        status_cap = limits.max_output_bytes * 3 + 64 * 1024
        out_thread = threading.Thread(target=_drain, args=(child.stdout, status_cap, status_buf, overflow, status_ready), daemon=True)
        err_thread = threading.Thread(target=_drain, args=(child.stderr, 16 * 1024, cli_err_buf, overflow), daemon=True)
        out_thread.start(); err_thread.start()
        deadline = started + limits.timeout_seconds
        timed_out = False
        while child.poll() is None:
            if overflow.is_set():
                break
            if status_ready.is_set():
                break
            if time.monotonic() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
        if timed_out or overflow.is_set():
            _kill_container(container)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
        status = None
        if not timed_out and not overflow.is_set() and status_ready.is_set():
            marker = b"MLR_PYTHON_RESULT:"
            line = bytes(status_buf).splitlines()[0]
            if not line.startswith(marker):
                raise RuntimeError("sandbox returned malformed execution status")
            status = json.loads(line[len(marker):])
            if status.get("artifact_error") is None:
                _copy_container_artifacts(docker, container, output_dir, limits.max_artifact_bytes, limits.max_artifact_files)
            _kill_container(container)
            child.wait(timeout=3)
        out_thread.join(timeout=3); err_thread.join(timeout=3)
        if not timed_out and not status_ready.is_set():
            detail = cli_err_buf.decode("utf-8", "replace")
            return PythonToolResult(False, "sandbox_error", child.returncode, "", "",
                                    int((time.monotonic()-started)*1000), [],
                                    detail or "container exited without execution status").to_dict()
        if timed_out:
            return PythonToolResult(False, "timeout", None, "", "",
                                    int((time.monotonic()-started)*1000), [],
                                    f"execution exceeded {limits.timeout_seconds:g}s").to_dict()
        if overflow.is_set():
            return PythonToolResult(False, "output_limit", child.returncode, "", "",
                                    int((time.monotonic()-started)*1000), [],
                                        f"stdout or stderr exceeded the {limits.max_output_bytes}-byte per-stream limit").to_dict()
        if status.get("artifact_error"):
            return PythonToolResult(False, "artifact_error", status["exit_code"],
                                    base64.b64decode(status["stdout_b64"]).decode("utf-8", "replace"),
                                    base64.b64decode(status["stderr_b64"]).decode("utf-8", "replace"),
                                    int((time.monotonic()-started)*1000), [], status["artifact_error"],
                                    resources=status.get("resources", {})).to_dict()
        manifest = _artifact_manifest(output_dir, limits.max_artifact_bytes, limits.max_artifact_files)
        code = status["exit_code"]
        stdout = base64.b64decode(status["stdout_b64"]).decode("utf-8", "replace")
        stderr = base64.b64decode(status["stderr_b64"]).decode("utf-8", "replace")
        if status["output_limit"]:
            return PythonToolResult(False, "output_limit", code, stdout, stderr,
                                    int((time.monotonic()-started)*1000), manifest,
                                    f"stdout or stderr exceeded the {limits.max_output_bytes}-byte per-stream limit",
                                    str(output_dir), status.get("resources", {})).to_dict()
        return PythonToolResult(code == 0, "success" if code == 0 else "script_error", code,
                                stdout, stderr,
                                int((time.monotonic()-started)*1000), manifest,
                                None if code == 0 else f"script exited with status {code}",
                                str(output_dir), status.get("resources", {})).to_dict()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as exc:
        return PythonToolResult(False, "execution_error", None, "", "",
                                int((time.monotonic()-started)*1000), [], str(exc)).to_dict()
    finally:
        if child is not None and child.poll() is None:
            _kill_container(container)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if child is not None:
            _kill_container(container)
        # Successful artifacts stay in the workspace for the next agent turn.
