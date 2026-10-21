#!/usr/bin/env python3
"""
kaggle_runner.py - Dispatch & Execute Local Scripts on Kaggle Cloud GPU/TPU

Features:
- Bundles any local Python script (and optional extra files/folders).
- Configures Kaggle kernel metadata with GPU/TPU & Internet enabled.
- Pushes execution to Kaggle Cloud via `kaggle kernels push`.
- Live-polls remote execution status (QUEUED -> RUNNING -> COMPLETE / ERROR).
- Automatically downloads output logs and generated artifacts to a local directory.
- Displays stdout/stderr execution logs directly in the terminal.

Usage:
    python3 scripts/kaggle_runner.py scripts/test_gpu_workload.py --gpu
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_kaggle_username() -> str:
    """Resolve the Kaggle username from env, kaggle.json, or Kaggle API."""
    # 1. Environment variable
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"].strip()

    # 2. ~/.kaggle/kaggle.json
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "username" in data and data["username"]:
                    return data["username"].strip()
        except Exception:
            pass

    # 3. Python KaggleApi
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        user = api.get_config_value("username")
        if user:
            return user.strip()
    except Exception:
        pass

    # 4. Fallback default
    return "ahkhan786"


def sanitize_slug(name: str) -> str:
    """Sanitize title or name into a valid Kaggle kernel slug (5-50 chars, [a-z0-9-])."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if not s:
        s = "job"
    # Ensure min 5 chars
    if len(s) < 5:
        s = f"{s}-run"
    # Truncate to max 45 to leave room for timestamp if needed
    return s[:45].rstrip("-")


def run_cmd(cmd: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Helper to run shell/CLI commands."""
    return subprocess.run(
        cmd,
        text=True,
        capture_output=capture,
        check=check,
    )


class KaggleJobRunner:
    def __init__(
        self,
        script_path: Path | str,
        title: Optional[str] = None,
        slug: Optional[str] = None,
        gpu: bool = True,
        tpu: bool = False,
        internet: bool = True,
        output_dir: Optional[Path | str] = None,
        extra_files: Optional[List[str]] = None,
        datasets: Optional[List[str]] = None,
        competitions: Optional[List[str]] = None,
        staging_dir: Optional[Path | str] = None,
        poll_interval: int = 8,
        timeout_minutes: int = 60,
    ):
        self.script_path = Path(script_path).resolve()
        if not self.script_path.exists():
            raise FileNotFoundError(f"Target script not found: {self.script_path}")

        self.username = get_kaggle_username()
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M%S")

        # Generate slug and title
        base_slug = slug or sanitize_slug(self.script_path.stem)
        # Ensure unique slug with timestamp
        self.slug = f"{base_slug}-{timestamp}"[:50].rstrip("-")
        self.title = title or f"{self.script_path.stem} {timestamp}"
        self.kernel_id = f"{self.username}/{self.slug}"

        self.gpu = gpu
        self.tpu = tpu
        self.internet = internet
        self.extra_files = extra_files or []
        self.datasets = datasets or []
        self.competitions = competitions or []
        self.poll_interval = poll_interval
        self.timeout_minutes = timeout_minutes

        self.output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else (Path.cwd() / "kaggle_outputs" / self.slug).resolve()
        )
        self.staging_dir = (
            Path(staging_dir).resolve()
            if staging_dir
            else (Path.cwd() / ".kaggle_staging" / self.slug).resolve()
        )

    def prepare_staging(self) -> Path:
        """Create the staging folder with code and kernel-metadata.json."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

        # Copy target script
        target_code_file = "main.py"
        shutil.copy2(self.script_path, self.staging_dir / target_code_file)

        # Copy any extra files or folders
        for extra in self.extra_files:
            src = Path(extra).resolve()
            if not src.exists():
                print(f"[Warning] Extra file/directory not found: {src}")
                continue
            dst = self.staging_dir / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        # Build kernel-metadata.json
        metadata: Dict[str, Any] = {
            "id": self.kernel_id,
            "title": self.title,
            "code_file": target_code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if self.gpu else "false",
            "enable_tpu": "true" if self.tpu else "false",
            "enable_internet": "true" if self.internet else "false",
            "machine_shape": "",
            "dataset_sources": self.datasets,
            "competition_sources": self.competitions,
            "kernel_sources": [],
            "model_sources": [],
        }

        with open(self.staging_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return self.staging_dir

    def push(self) -> str:
        """Push kernel to Kaggle."""
        print(f"📦 Staging kernel in: {self.staging_dir}")
        self.prepare_staging()

        print(f"🚀 Pushing kernel to Kaggle Cloud...")
        print(f"   • Kernel ID   : {self.kernel_id}")
        print(f"   • GPU Enabled : {self.gpu}")
        print(f"   • Internet    : {self.internet}")
        print(f"   • Script      : {self.script_path.name}")

        result = run_cmd(["kaggle", "kernels", "push", "-p", str(self.staging_dir)])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to push kernel:\n{result.stderr}\n{result.stdout}")

        output_msg = result.stdout.strip() or result.stderr.strip()
        print(f"   • Status      : {output_msg}")
        return output_msg

    def get_status(self) -> str:
        """Query kernel worker status from Kaggle."""
        res = run_cmd(["kaggle", "kernels", "status", self.kernel_id])
        output = res.stdout.strip()
        # Parse status: e.g. "ahkhan786/slug has status "KernelWorkerStatus.RUNNING""
        # or JSON / plain text
        status_match = re.search(r'status\s+"?KernelWorkerStatus\.?([A-Z_]+)"?', output, re.IGNORECASE)
        if status_match:
            return status_match.group(1).upper()
        
        # Check raw keywords
        for candidate in ["COMPLETE", "RUNNING", "QUEUED", "ERROR", "CANCELLED", "CANCELING"]:
            if candidate in output.upper():
                return candidate
        return output

    def wait_for_completion(self) -> str:
        """Poll Kaggle until kernel finishes running."""
        start_time = time.time()
        max_duration = self.timeout_minutes * 60
        last_status = None

        print(f"\n⏳ Monitoring remote execution (poll interval: {self.poll_interval}s)...")
        while True:
            elapsed = int(time.time() - start_time)
            if elapsed > max_duration:
                print(f"\n❌ Timed out after {self.timeout_minutes} minutes.")
                return "TIMEOUT"

            status = self.get_status()
            if status != last_status:
                timestamp_str = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"   [{timestamp_str}] Elapsed: {elapsed:3d}s | Remote Status: {status}")
                last_status = status

            if status in ["COMPLETE", "ERROR", "CANCELLED"]:
                return status

            time.sleep(self.poll_interval)

    def fetch_output(self) -> Path:
        """Download remote output and print logs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📥 Downloading execution outputs to: {self.output_dir}")

        res = run_cmd(["kaggle", "kernels", "output", self.kernel_id, "-p", str(self.output_dir)])
        if res.returncode != 0:
            print(f"Warning: Failed to fetch output:\n{res.stderr}\n{res.stdout}")

        # Search for log file
        log_files = list(self.output_dir.glob("*.log"))
        if log_files:
            log_file = log_files[0]
            print(f"\n📄 Remote Execution Console Output ({log_file.name}):")
            print("=" * 70)
            try:
                content = log_file.read_text(encoding="utf-8", errors="replace")
                print(content.strip() if content.strip() else "[Log is empty]")
            except Exception as e:
                print(f"Failed to read log: {e}")
            print("=" * 70)

        # List all downloaded artifacts
        artifacts = list(self.output_dir.iterdir())
        if artifacts:
            print(f"\n📦 Downloaded Artifacts ({len(artifacts)} files):")
            for item in artifacts:
                size_kb = item.stat().st_size / 1024
                print(f"   • {item.name:<35} ({size_kb:.1f} KB)")
        else:
            print("   (No output artifacts produced)")

        return self.output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run any local Python script on Kaggle Cloud GPU/TPU resources."
    )
    parser.add_argument("script", help="Path to Python script to run on Kaggle.")
    parser.add_argument("--title", default=None, help="Kaggle notebook title.")
    parser.add_argument("--slug", default=None, help="Custom kernel slug.")
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=True, help="Enable GPU accelerator (default: True).")
    parser.add_argument("--tpu", action=argparse.BooleanOptionalAction, default=False, help="Enable TPU accelerator (default: False).")
    parser.add_argument("--internet", action=argparse.BooleanOptionalAction, default=True, help="Enable internet access (default: True).")
    parser.add_argument("--output-dir", default=None, help="Local directory to download artifacts.")
    parser.add_argument("--extra-files", nargs="*", default=[], help="Additional files or folders to include.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Kaggle dataset sources (e.g. user/dataset).")
    parser.add_argument("--competitions", nargs="*", default=[], help="Kaggle competition sources (e.g. titanic).")
    parser.add_argument("--poll-interval", type=int, default=8, help="Polling interval in seconds (default: 8).")
    parser.add_argument("--timeout", type=int, default=60, help="Max wait timeout in minutes (default: 60).")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for completion; return immediately after push.")
    parser.add_argument("--clean-staging", action="store_true", help="Remove staging folder after pushing.")

    args = parser.parse_args()

    runner = KaggleJobRunner(
        script_path=args.script,
        title=args.title,
        slug=args.slug,
        gpu=args.gpu,
        tpu=args.tpu,
        internet=args.internet,
        output_dir=args.output_dir,
        extra_files=args.extra_files,
        datasets=args.datasets,
        competitions=args.competitions,
        poll_interval=args.poll_interval,
        timeout_minutes=args.timeout,
    )

    runner.push()

    if args.clean_staging:
        shutil.rmtree(runner.staging_dir, ignore_errors=True)

    if args.no_wait:
        print(f"\nJob pushed successfully. Check status anytime with:")
        print(f"  kaggle kernels status {runner.kernel_id}")
        print(f"  kaggle kernels output {runner.kernel_id} -p {runner.output_dir}")
        return

    final_status = runner.wait_for_completion()
    runner.fetch_output()

    if final_status == "COMPLETE":
        print(f"\n🎉 Kaggle GPU Job Finished Successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Kaggle Job Ended with status: {final_status}")
        sys.exit(1)


if __name__ == "__main__":
    main()
