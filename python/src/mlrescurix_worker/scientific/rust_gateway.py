"""Rust Provider Gateway Adapter (Phase M1/M2 Production Governance).

Integrates AutonomousAgentLoop with the compiled mlrescurix-harness:
1. OutboundFilter pattern redaction and content-bound outbound_views policy approval.
2. Verified model pricing lookup and transaction-safe SQLite provider_accounting recording.
3. Stable request identification, durable local response receipts, and ambiguous-outcome replay fencing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]


def find_harness_binary(custom_path: Optional[Path | str] = None) -> Path:
    """Locate the compiled mlrescurix-harness executable."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()

    candidates = [
        REPO_ROOT / "target" / "release" / "mlrescurix-harness",
        REPO_ROOT / "target" / "debug" / "mlrescurix-harness",
    ]
    existing = [c.resolve() for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if existing:
        existing.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return existing[0]

    which_path = shutil.which("mlrescurix-harness")
    if which_path:
        return Path(which_path).resolve()

    raise FileNotFoundError(
        "mlrescurix-harness binary not found. Build with 'cargo build' before running production gateway."
    )


class RustGatewayAdapter:
    """Production provider transport executing through the compiled Rust harness."""

    def __init__(
        self,
        state_dir: Path | str,
        project_id: str,
        task_id: str = "default-task",
        role_id: str = "scientific-agent",
        model_name: str = "deepseek/deepseek-v4.1-flash-1017",
        endpoint: str = "inference-net/fp8",
        harness_binary: Optional[Path | str] = None,
        require_policy_approval: bool = True,
        mock_path: Optional[Path | str] = None,
        pricing: Optional[Dict[str, Any]] = None,
    ):
        self.state_dir = Path(state_dir).resolve()
        self.project_id = project_id
        self.task_id = task_id
        self.role_id = role_id
        self.model_name = model_name
        self.endpoint = endpoint
        self.binary = find_harness_binary(harness_binary)
        self.require_policy_approval = require_policy_approval
        self.mock_path = Path(mock_path).resolve() if mock_path is not None else None
        if pricing is not None:
            if not pricing.get("verified"):
                raise ValueError("Rust gateway pricing must be marked verified before provider dispatch")
            for key in ("input_per_million", "output_per_million"):
                value = Decimal(str(pricing.get(key, "nan")))
                if not value.is_finite() or value < 0:
                    raise ValueError(f"verified pricing requires a finite nonnegative {key}")
        self.pricing = pricing

        # Initialize/migrate harness state
        self._init_harness_state()

    def _init_harness_state(self) -> None:
        """Ensure state root is initialized and model pricing is registered."""
        # 1. Initialize harness state if not yet owned
        owned_marker = self.state_dir / "OWNED"
        if not owned_marker.is_file():
            # If directory exists but is empty (or unowned), remove it so init create_dir succeeds
            if self.state_dir.is_dir() and not any(self.state_dir.iterdir()):
                self.state_dir.rmdir()
            elif not self.state_dir.exists():
                self.state_dir.parent.mkdir(parents=True, exist_ok=True)

            policy_payload = {
                "budget": {
                    "calls": 1000,
                    "tokens": 5000000,
                    "wall_ms": 3600000,
                },
                "capacity": {
                    "workers": 4,
                    "cpu_threads": 4,
                },
                "max_tasks": 1024,
            }
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(policy_payload, f)
                policy_path = f.name
            try:
                cmd_init = [str(self.binary), "init", "--state", str(self.state_dir), "--policy", policy_path]
                res_init = subprocess.run(cmd_init, capture_output=True, text=True, timeout=15)
                if res_init.returncode != 0:
                    raise RuntimeError(f"Harness init failed ({res_init.returncode}): {res_init.stderr or res_init.stdout}")
            finally:
                try:
                    os.unlink(policy_path)
                except OSError:
                    pass

        # 2. Run migrate to ensure schema v11 is present
        cmd_migrate = [str(self.binary), "migrate", "--state", str(self.state_dir)]
        res = subprocess.run(cmd_migrate, capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            raise RuntimeError(f"Harness migrate failed ({res.returncode}): {res.stderr or res.stdout}")

        # 2. Register model pricing if provided
        if self.pricing is not None:
            # Convert dollars per million to millicents per 1k:
            # $0.08 per 1M tokens = $0.00008 per 1k = 8 millicents per 1k ($1 = 100,000 millicents)
            in_rate = Decimal(str(self.pricing.get("input_per_million", "0.08")))
            out_rate = Decimal(str(self.pricing.get("output_per_million", "0.50")))
            prompt_mc = int((in_rate * Decimal(100)).to_integral_value(rounding=ROUND_CEILING))
            comp_mc = int((out_rate * Decimal(100)).to_integral_value(rounding=ROUND_CEILING))

            cmd_pricing = [
                str(self.binary),
                "register-model-pricing",
                "--state", str(self.state_dir),
                "--model", self.model_name,
                "--prompt-millicents-per-1k", str(prompt_mc),
                "--completion-millicents-per-1k", str(comp_mc),
            ]
            res_p = subprocess.run(cmd_pricing, capture_output=True, text=True, timeout=15)
            if res_p.returncode != 0:
                raise RuntimeError(f"Register model pricing failed ({res_p.returncode}): {res_p.stderr or res_p.stdout}")

    def __call__(self, messages: List[Dict[str, Any]], max_tokens: int = 3500) -> Dict[str, Any]:
        raise ValueError("RustGatewayAdapter requires a stable campaign request_id; use call_with_request_id")

    def call_with_request_id(
        self, messages: List[Dict[str, Any]], max_tokens: int, request_id: str
    ) -> Dict[str, Any]:
        """Dispatch with a caller-owned stable session/turn ID for crash-safe replay fencing."""
        if not request_id or len(request_id) > 256 or any(ord(c) < 32 for c in request_id):
            raise ValueError("request_id must be a nonempty stable campaign identifier")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        req_payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": [],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "reasoning": {
                "max_tokens": 200,
            },
            "provider": {
                "only": [self.endpoint],
                "order": [self.endpoint],
                "allow_fallbacks": False,
            },
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            req_file = tmp_path / "chat_request.json"
            req_file.write_text(json.dumps(req_payload), encoding="utf-8")

            # If policy approval is required, stage and approve outbound view
            if self.require_policy_approval:
                stage_cmd = [
                    str(self.binary),
                    "stage-outbound-view",
                    "--state", str(self.state_dir),
                    "--project", self.project_id,
                    "--request", str(req_file),
                    "--actor", "scientific-controller",
                ]
                res_stage = subprocess.run(stage_cmd, capture_output=True, text=True, timeout=15)
                if res_stage.returncode != 0:
                    raise RuntimeError(f"stage-outbound-view failed ({res_stage.returncode}): {res_stage.stderr or res_stage.stdout}")

                try:
                    stage_info = json.loads(res_stage.stdout)
                    req_sha256 = stage_info["request_sha256"]
                except Exception as exc:
                    raise RuntimeError(f"Cannot parse stage-outbound-view output: {res_stage.stdout}") from exc

                # Approval is deliberately an external human action. Include the
                # digest in a later refusal so an operator can inspect/approve it.

            # Dispatch through gateway-dispatch
            disp_cmd = [
                str(self.binary),
                "gateway-dispatch",
                "--state", str(self.state_dir),
                "--project", self.project_id,
                "--task", self.task_id,
                "--role", self.role_id,
                "--request-id", request_id,
                "--request", str(req_file),
            ]
            if self.mock_path is not None:
                disp_cmd.extend(["--mock", str(self.mock_path)])
            if self.require_policy_approval:
                disp_cmd.extend(["--require-policy-approval", "true"])

            try:
                res_disp = subprocess.run(disp_cmd, capture_output=True, text=True, timeout=75)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"gateway timed out for request_id={request_id}; remote outcome may be unknown; "
                    "retry only with the same request_id"
                ) from exc
            if res_disp.returncode != 0:
                raise RuntimeError(
                    f"gateway-dispatch failed ({res_disp.returncode}): {res_disp.stderr or res_disp.stdout}"
                )

            try:
                data = json.loads(res_disp.stdout)
            except Exception as exc:
                raise RuntimeError(f"Failed to parse gateway-dispatch JSON output: {res_disp.stdout}") from exc

            usage = data.get("usage", {})
            cost_usd = str(data.get("cost_usd", "0.00"))

            return {
                "content": data.get("content", ""),
                "usage": usage,
                "cost_usd": cost_usd,
                "cost_is_estimate": bool(data.get("cost_is_estimate", True)),
                "provider_response_id": data.get("id"),
                "latency_ms": data.get("latency_ms", 0),
                "request_sha256": data.get("request_sha256"),
                "request_id": data.get("request_id", request_id),
                "gateway_route": "rust_mlrescurix_harness",
            }
