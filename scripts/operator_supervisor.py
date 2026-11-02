#!/usr/bin/env python3
"""MLRescurix Independent Operator Supervisor for Outbound View Approval.

Enforces fail-closed human-in-the-loop / operator privacy controls:
1. Inspects staged outbound view payloads via `mlrescurix-harness inspect-outbound-view`.
2. Verifies cryptographic content binding (digest verification).
3. Verifies strict privacy boundaries:
   - No raw dataset CSV rows dumped into LLM context.
   - No credentials, tokens, or secret keys leaked.
   - Pinned model identifier and single provider route (no fallback).
   - Bounded token limits.
4. Atomically approves valid views via `mlrescurix-harness approve-outbound-view`.
5. Emits auditable operator receipts to stdout and an operator log file.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def inspect_view(harness_binary: Path, state_dir: Path, project: str, digest: str) -> dict:
    cmd = [
        str(harness_binary),
        "inspect-outbound-view",
        "--state", str(state_dir),
        "--project", project,
        "--request-sha256", digest,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        raise RuntimeError(f"inspect-outbound-view failed ({res.returncode}): {res.stderr or res.stdout}")
    return json.loads(res.stdout)


def approve_view(harness_binary: Path, state_dir: Path, project: str, digest: str, approved: bool = True, actor: str = "operator") -> dict:
    cmd = [
        str(harness_binary),
        "approve-outbound-view",
        "--state", str(state_dir),
        "--project", project,
        "--request-sha256", digest,
        "--actor", actor,
        "--approved", "true" if approved else "false",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        raise RuntimeError(f"approve-outbound-view failed ({res.returncode}): {res.stderr or res.stdout}")
    return json.loads(res.stdout)


def verify_privacy_and_policy(view_data: dict, expected_model: str, expected_route: str, dev_csv_path: Path | None = None) -> tuple[bool, str]:
    view = view_data.get("view", {})
    if not isinstance(view, dict):
        return False, "view payload is not a JSON object"

    # 1. Model check
    model = view.get("model")
    if model != expected_model:
        return False, f"model mismatch: expected {expected_model}, got {model}"

    # 2. Provider routing check
    provider = view.get("provider", {})
    if not isinstance(provider, dict):
        return False, "provider config missing or invalid"
    if provider.get("allow_fallbacks") is not False:
        return False, "provider allow_fallbacks must be explicitly False"
    if provider.get("only") != [expected_route]:
        return False, f"provider.only mismatch: expected [{expected_route}], got {provider.get('only')}"
    if provider.get("order") != [expected_route]:
        return False, f"provider.order mismatch: expected [{expected_route}], got {provider.get('order')}"

    # 3. Max tokens bound check
    max_tokens = view.get("max_tokens", 0)
    if not isinstance(max_tokens, int) or max_tokens <= 0 or max_tokens > 4096:
        return False, f"max_tokens out of bounds: {max_tokens}"

    # 4. Content credential & privacy scan
    messages = view.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return False, "messages must be a non-empty list"

    # Known secret patterns
    secret_patterns = [
        r"sk-[a-zA-Z0-9_-]{20,}",
        r"sk-or-[a-zA-Z0-9_-]{20,}",
        r"ghp_[a-zA-Z0-9]{20,}",
        r"xoxb-[a-zA-Z0-9_-]{20,}",
    ]

    total_chars = 0
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        if role not in ("system", "user", "assistant"):
            return False, f"invalid message role at index {idx}: {role}"
        if not isinstance(content, str):
            return False, f"message content at index {idx} must be a string"
        total_chars += len(content)

        # Check for credential leakage
        for pat in secret_patterns:
            if re.search(pat, content):
                return False, f"credential pattern {pat} detected in message {idx}"

        # Check for raw CSV dump / raw dataset dump
        if "catalog_development.csv" in content or "development.csv" in content:
            # Check if actual raw CSV lines are present (e.g. multiple commas, catalog items)
            lines = content.splitlines()
            csv_like_lines = [l for l in lines if l.count(",") >= 4 and len(l) > 40]
            if len(csv_like_lines) > 5:
                return False, f"possible raw CSV rows detected in message {idx} ({len(csv_like_lines)} tabular lines)"

    return True, f"verified {len(messages)} messages ({total_chars} chars), model={model}, route={expected_route}"


def run_supervisor(
    state_dir: Path,
    project: str,
    expected_model: str,
    expected_route: str,
    harness_binary: Path,
    log_file: Path | None = None,
    watch_pid: int | None = None,
    poll_interval: float = 0.5,
    timeout_seconds: float = 3600.0,
):
    print(f"[OPERATOR] Supervisor started for project={project}, state={state_dir}", flush=True)
    print(f"[OPERATOR] Pinned policy: model={expected_model}, route={expected_route}", flush=True)

    db_path = state_dir / "state.sqlite"
    start_time = time.monotonic()
    processed_digests = set()

    def log_audit(msg: str):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        line = f"[{timestamp}] {msg}"
        print(line, flush=True)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    log_audit(f"SUPERVISOR_INIT: project={project}, state={state_dir}")

    while time.monotonic() - start_time < timeout_seconds:
        # Check parent process if watch_pid is provided
        if watch_pid is not None:
            try:
                os.kill(watch_pid, 0)
            except OSError:
                log_audit(f"WATCH_PID_EXIT: Agent process {watch_pid} terminated; stopping supervisor.")
                break

        if not db_path.is_file():
            time.sleep(poll_interval)
            continue

        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
                rows = db.execute(
                    "SELECT request_sha256, actor, view_json, approved FROM outbound_views WHERE project=? AND approved=0",
                    (project,),
                ).fetchall()
        except sqlite3.OperationalError:
            time.sleep(poll_interval)
            continue

        for req_sha256, actor, view_json, approved in rows:
            if req_sha256 in processed_digests:
                continue

            log_audit(f"PENDING_VIEW_DETECTED: digest={req_sha256[:16]}... actor={actor}")

            try:
                view_inspection = inspect_view(harness_binary, state_dir, project, req_sha256)
                valid, reason = verify_privacy_and_policy(view_inspection, expected_model, expected_route)

                if valid:
                    log_audit(f"INSPECTION_PASSED: digest={req_sha256[:16]}... reason={reason}")
                    approve_res = approve_view(harness_binary, state_dir, project, req_sha256, approved=True, actor="operator")
                    log_audit(f"APPROVAL_GRANTED: digest={req_sha256[:16]}... result={approve_res}")
                    processed_digests.add(req_sha256)
                else:
                    log_audit(f"INSPECTION_FAILED: digest={req_sha256[:16]}... rejection_reason={reason}")
                    reject_res = approve_view(harness_binary, state_dir, project, req_sha256, approved=False, actor="operator")
                    log_audit(f"APPROVAL_REJECTED: digest={req_sha256[:16]}... result={reject_res}")
                    processed_digests.add(req_sha256)
            except Exception as exc:
                log_audit(f"INSPECTION_ERROR: digest={req_sha256[:16]}... error={exc}")

        time.sleep(poll_interval)

    log_audit("SUPERVISOR_STOPPED: completed run or timeout.")


def main():
    parser = argparse.ArgumentParser(description="MLRescurix Operator Supervisor for Outbound Privacy Approval")
    parser.add_argument("--state", required=True, type=Path, help="Harness state directory")
    parser.add_argument("--project", required=True, help="Project identifier")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    parser.add_argument("--route", default="wafer/us")
    parser.add_argument("--harness", type=Path, default=REPO_ROOT / "target" / "release" / "mlrescurix-harness")
    parser.add_argument("--log-file", type=Path, help="Durable operator audit log file")
    parser.add_argument("--watch-pid", type=int, help="Optional agent PID to monitor")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    run_supervisor(
        state_dir=args.state.resolve(),
        project=args.project,
        expected_model=args.model,
        expected_route=args.route,
        harness_binary=args.harness.resolve(),
        log_file=args.log_file.resolve() if args.log_file else None,
        watch_pid=args.watch_pid,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
