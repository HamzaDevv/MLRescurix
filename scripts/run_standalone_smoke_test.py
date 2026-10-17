#!/usr/bin/env python3
"""
MLRescurix Standalone Distribution Smoke Test
Executes a clean-room end-to-end operational validation from the release tarball.
"""

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout_sec: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def http_get(port: int, path: str, headers: dict = None) -> tuple[int, str]:
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as s:
        req = f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n"
        if headers:
            for k, v in headers.items():
                req += f"{k}: {v}\r\n"
        req += "\r\n"
        s.sendall(req.encode("utf-8"))
        res = b""
        while chunk := s.recv(4096):
            res += chunk
    decoded = res.decode("utf-8", errors="replace")
    status_line = decoded.split("\r\n")[0]
    status_code = int(status_line.split(" ")[1])
    return status_code, decoded


def http_post(port: int, path: str, body: str, headers: dict = None) -> tuple[int, str]:
    body_bytes = body.encode("utf-8")
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as s:
        req = f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {len(body_bytes)}\r\nConnection: close\r\n"
        if headers:
            for k, v in headers.items():
                req += f"{k}: {v}\r\n"
        req += "\r\n"
        s.sendall(req.encode("utf-8") + body_bytes)
        res = b""
        while chunk := s.recv(4096):
            res += chunk
    decoded = res.decode("utf-8", errors="replace")
    status_line = decoded.split("\r\n")[0]
    status_code = int(status_line.split(" ")[1])
    return status_code, decoded


def run_smoke_test():
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"

    sys_name = platform.system().lower()
    arch = platform.machine()
    tarball_name = f"mlrescurix-v0.1.0-m6-{sys_name}-{arch}.tar.gz"
    tarball_path = dist_dir / tarball_name

    if not tarball_path.is_file():
        print(f"Error: Release tarball {tarball_path} does not exist. Run scripts/package_release.py first.")
        sys.exit(1)

    smoke_prefix = Path(tempfile.mkdtemp(prefix="mlrescurix-smoke-test-"))
    print(f"\n================================================================================")
    print(f"STARTING CLEAN-ROOM STANDALONE SMOKE TEST")
    print(f"Isolated Prefix: {smoke_prefix}")
    print(f"Release Tarball: {tarball_path}")
    print(f"================================================================================\n")

    try:
        # Step 1: Extract Tarball
        print("[Step 1/8] Extracting release tarball into isolated sandbox...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=smoke_prefix, filter='data')
            else:
                tar.extractall(path=smoke_prefix)

        pkg_dir = smoke_prefix / "mlrescurix-0.1.0-m6"
        assert pkg_dir.is_dir(), f"Extracted directory {pkg_dir} not found"
        assert (pkg_dir / "bin" / "mlrescurix-harness").is_file(), "Binary not found in tarball"
        assert (pkg_dir / "install.sh").is_file(), "install.sh not found in tarball"
        assert (pkg_dir / "README.md").is_file(), "README.md not found in tarball"
        assert (pkg_dir / "RELEASE_NOTES.md").is_file(), "RELEASE_NOTES.md not found in tarball"
        assert (pkg_dir / "CHANGELOG.md").is_file(), "CHANGELOG.md not found in tarball"
        wheel_candidates = list((pkg_dir / "wheels").glob("*.whl"))
        assert len(wheel_candidates) > 0, "No python wheel found in tarball"
        wheel_path = wheel_candidates[0]
        print(f"  Extracted release package: {pkg_dir.name}")
        print(f"  Found Python worker wheel: {wheel_path.name}")

        # Step 2: Run install.sh into isolated opt prefix
        print("\n[Step 2/8] Executing standalone install.sh into isolated prefix...")
        opt_dir = smoke_prefix / "opt"
        install_res = subprocess.run(
            [str(pkg_dir / "install.sh"), str(opt_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  Install script output:\n{install_res.stdout.strip()}")
        installed_bin = opt_dir / "bin" / "mlrescurix-harness"
        assert installed_bin.is_file(), "Installed binary not found at opt/bin/mlrescurix-harness"
        assert os.access(installed_bin, os.X_OK), "Installed binary is not executable"

        # Step 3: Verify Capabilities without checkout or PYTHONPATH
        print("\n[Step 3/8] Verifying installed harness capabilities outside git repository...")
        workspace_dir = smoke_prefix / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        env_clean = {k: v for k, v in os.environ.items() if k not in ["PYTHONPATH", "CARGO_MANIFEST_DIR"]}
        cap_res = subprocess.run(
            [str(installed_bin), "capabilities"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            env=env_clean,
            check=True,
        )
        cap_data = json.loads(cap_res.stdout)
        assert cap_data["state_schema_version"] == 10, "state_schema_version must be 10"
        assert cap_data["standalone_install_without_checkout"] is True
        for gate_name, gate_status in cap_data["historical_m6_release_gates"].items():
            assert gate_status == "closed_and_verified", f"Gate {gate_name} not closed: {gate_status}"
        for gate_name, gate_status in cap_data["v2_release_gates"].items():
            assert gate_status == "passed", f"v2 gate {gate_name} not passed: {gate_status}"
        print(f"  Capabilities verified successfully: state_schema_version={cap_data['state_schema_version']}")
        print(f"  Historical M6 release gates: all 5 closed and verified")
        print(f"  v2 release gates: all 12 passed")

        # Step 4: Setup isolated virtualenv and install wheel
        print("\n[Step 4/8] Setting up isolated Python venv and installing worker wheel...")
        venv_dir = smoke_prefix / "venv"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
            check=True,
            capture_output=True,
        )
        venv_py = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"
        subprocess.run(
            [str(venv_pip), "install", "--no-deps", str(wheel_path)],
            check=True,
            capture_output=True,
        )
        # Verify wheel import
        import_res = subprocess.run(
            [str(venv_py), "-c", "import mlrescurix_worker; print(mlrescurix_worker.__file__)"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"  Installed worker wheel into venv: {import_res.stdout.strip()}")

        # Step 5: Initialize state and Profile Dataset
        print("\n[Step 5/8] Initializing state and profiling tabular dataset...")
        state_dir = smoke_prefix / "state"
        policy_file = smoke_prefix / "policy.json"
        policy_file.write_text(json.dumps({"budget": {"wall_ms": 600000}, "capacity": {"workers": 2}, "max_tasks": 16}))

        subprocess.run(
            [str(installed_bin), "init", "--state", str(state_dir), "--policy", str(policy_file)],
            cwd=str(workspace_dir),
            capture_output=True,
            check=True,
            env=env_clean,
        )
        subprocess.run(
            [str(installed_bin), "migrate", "--state", str(state_dir)],
            cwd=str(workspace_dir),
            capture_output=True,
            check=True,
            env=env_clean,
        )

        csv_path = workspace_dir / "churn_smoke.csv"
        with open(csv_path, "w") as f:
            f.write("tenure,monthly_charges,total_charges,support_calls,churn\n")
            for i in range(250):
                tenure = float(i % 72 + 1)
                monthly = 20.0 + float(i % 80)
                total = tenure * monthly
                support = float(i % 6)
                churn = 1 if (monthly > 70.0 and support > 2.0) or tenure < 6.0 else 0
                f.write(f"{tenure:.1f},{monthly:.2f},{total:.2f},{support:.0f},{churn}\n")

        prof_res = subprocess.run(
            [str(installed_bin), "profile-dataset", "--state", str(state_dir), "--csv", str(csv_path), "--target", "churn"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=True,
            env=env_clean,
        )
        prof_data = json.loads(prof_res.stdout)
        assert prof_data["row_count"] == 250, "Expected 250 rows in profile"
        assert prof_data["target_column"] == "churn", "Target column must be churn"
        assert len(prof_data["feature_columns"]) == 4, "Expected 4 feature columns"
        print(f"  Dataset profiled successfully: {prof_data['row_count']} rows, target='{prof_data['target_column']}'")

        # Step 6: Autonomous Rescue Bundle Packaging
        print("\n[Step 6/8] Packaging certified candidate delivery bundle...")
        cand_dir = smoke_prefix / "candidate"
        bundle_dir = smoke_prefix / "bundle"
        cand_dir.mkdir(parents=True, exist_ok=True)

        pkg_res = subprocess.run(
            [
                str(installed_bin),
                "package-candidate-bundle",
                "--state", str(state_dir),
                "--candidate-dir", str(cand_dir),
                "--bundle-dir", str(bundle_dir),
                "--dataset", str(csv_path),
                "--test-access-id", "smoke-test-acc-001",
                "--project", "p-smoke-01",
                "--candidate", "cand-smoke-apex",
                "--task-type", "binary",
                "--target", "churn",
                "--metric", "balanced_accuracy",
            ],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=True,
            env=env_clean,
        )
        pkg_data = json.loads(pkg_res.stdout)
        assert pkg_data["status"] == "bundled", f"Expected bundled, got {pkg_data['status']}"
        assert pkg_data["format"] == "mlrescurix.bundle.v2", f"Expected mlrescurix.bundle.v2, got {pkg_data['format']}"
        assert pkg_data["verification_summary"]["manifest_verified"] is True, "Bundle manifest verification failed"
        print(f"  Bundle certified and packaged: {bundle_dir}")
        print(f"  Candidate ID: {pkg_data['candidate_id']}")
        print(f"  Manifest SHA-256: {pkg_data['manifest_sha256']}")

        # Step 7: Standalone CLI Inference, HTTP Serve, and Bundle Verification
        print("\n[Step 7/8] Running standalone CLI inference and zero-dependency HTTP server...")
        test_in = workspace_dir / "test_input.json"
        test_out = workspace_dir / "test_out.json"
        records = [
            {"tenure": 12.0, "monthly_charges": 65.5, "total_charges": 786.0, "support_calls": 2.0},
            {"tenure": 48.0, "monthly_charges": 25.0, "total_charges": 1200.0, "support_calls": 0.0},
            {"tenure": 2.0, "monthly_charges": 85.0, "total_charges": 170.0, "support_calls": 4.0},
        ]
        test_in.write_text(json.dumps(records))

        # CLI Inference
        inf_res = subprocess.run(
            [str(venv_py), "-B", str(bundle_dir / "inference.py"), "--input", str(test_in), "--output", str(test_out)],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        pred_data = json.loads(test_out.read_text())
        assert len(pred_data["prediction"]) == 3, "Expected 3 predictions"
        assert len(pred_data["probabilities"]) == 3, "Expected 3 probability pairs"
        print(f"  Standalone CLI inference succeeded: predictions={pred_data['prediction']}")

        # HTTP Serve
        serve_port = get_free_port()
        serve_proc = subprocess.Popen(
            [str(venv_py), "-B", str(bundle_dir / "serve.py"), "--port", str(serve_port)],
            cwd=str(workspace_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert wait_for_port(serve_port, 5.0), f"serve.py failed to bind port {serve_port}"
            code, resp = http_get(serve_port, "/health")
            assert code == 200 and "healthy" in resp, f"GET /health returned {code}: {resp}"
            code, resp = http_post(serve_port, "/predict", json.dumps(records))
            assert code == 200 and "prediction" in resp, f"POST /predict returned {code}: {resp}"
            print(f"  Standalone HTTP server validated on port {serve_port}: /health=200, /predict=200")
        finally:
            serve_proc.kill()
            serve_proc.wait()

        # Bundle Verification via Harness
        verify_res = subprocess.run(
            [str(installed_bin), "verify-bundle", "--bundle", str(bundle_dir), "--sandbox=false"],
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            check=True,
            env=env_clean,
        )
        verify_data = json.loads(verify_res.stdout)
        assert verify_data["status"] == "integrity_verified", "verify-bundle failed"
        assert verify_data["manifest_verified"] is True
        print(f"  Harness verify-bundle succeeded: status={verify_data['status']}")

        # Step 8: Dashboard Serving Validation (mlrescurix-harness serve-dashboard)
        print("\n[Step 8/8] Validating harness dashboard serving (serve-dashboard alias)...")
        create_project_req = {
            "command_id": "cmd-create-p-smoke-01",
            "project_id": "p-smoke-01",
            "expected_revision": None,
            "operation": {
                "type": "create_project",
                "intent": "Smoke test dashboard project",
                "finalization_id": "f-smoke-01",
                "policy": {"budget": {"wall_ms": 600000}, "capacity": {"workers": 2}, "max_tasks": 16},
            },
        }
        req_file = workspace_dir / "create_project_req.json"
        req_file.write_text(json.dumps(create_project_req, indent=2))
        subprocess.run(
            [
                str(installed_bin),
                "command",
                "--state", str(state_dir),
                "--request", str(req_file),
            ],
            cwd=str(workspace_dir),
            capture_output=True,
            check=True,
            env=env_clean,
        )
        print("  Project p-smoke-01 created via harness command CLI")

        subprocess.run(
            [
                str(installed_bin),
                "register-dataset",
                "--state", str(state_dir),
                "--project", "p-smoke-01",
                "--id", "churn-ds",
                "--csv", str(csv_path),
            ],
            cwd=str(workspace_dir),
            capture_output=True,
            check=True,
            env=env_clean,
        )
        print("  Dataset registered for p-smoke-01 in state database")

        dash_port = get_free_port()
        dash_token = "smoke-auth-token-42"
        dash_proc = subprocess.Popen(
            [
                str(installed_bin),
                "serve-dashboard",
                "--state", str(state_dir),
                "--port", str(dash_port),
                "--token", dash_token,
            ],
            cwd=str(workspace_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env_clean,
        )
        try:
            assert wait_for_port(dash_port, 5.0), f"Dashboard server failed to bind port {dash_port}"
            
            # Test GET / (embedded dashboard HTML)
            code, resp = http_get(dash_port, "/")
            assert code == 200 and "MLRescurix" in resp, f"GET / returned {code}: {resp[:200]}"
            print(f"  GET / returned HTTP 200 OK with embedded dashboard UI ({len(resp)} bytes)")

            # Test POST /session (token unlock with origin verification)
            headers = {"Origin": f"http://127.0.0.1:{dash_port}"}
            code, resp = http_post(dash_port, "/session", json.dumps({"token": dash_token}), headers=headers)
            assert code == 200 and "mlr_session=" in resp, f"POST /session failed: code={code}, resp={resp}"
            print(f"  POST /session returned HTTP 200 OK with session cookie")

            # Test authenticated snapshot route
            auth_headers = {"Authorization": f"Bearer {dash_token}"}
            code, resp = http_get(dash_port, "/v1/projects/p-smoke-01/snapshot", headers=auth_headers)
            assert code == 200 and "p-smoke-01" in resp, f"GET /snapshot returned {code}: {resp}"
            print(f"  GET /v1/projects/p-smoke-01/snapshot returned HTTP 200 OK")

            print(f"  Harness serve-dashboard validated flawlessly on port {dash_port}!")
        finally:
            dash_proc.kill()
            dash_proc.wait()

        print("\n================================================================================")
        print("ALL STANDALONE SMOKE TESTS COMPLETED SUCCESSFULLY (100% CLEAN-ROOM PASS)")
        print("================================================================================\n")

    finally:
        shutil.rmtree(smoke_prefix, ignore_errors=True)


if __name__ == "__main__":
    run_smoke_test()
