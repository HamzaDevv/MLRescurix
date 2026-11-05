"""Operational budgets management for MLRescurix autonomous execution.

Tracks configured budgets for the autonomous session:
1. Total model token usage (counting prompt/input and completion/output tokens across all agents and calls).
2. Total Kaggle compute usage (measured in accelerator hours from the Kaggle environment).
3. Total web search queries.

Preserves existing dollar-denominated campaign budget caps without deleting or resetting history.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("mlrescurix.scientific.operational_budgets")

DEFAULT_TOKEN_LIMIT = 5_000_000
DEFAULT_KAGGLE_HOURS_LIMIT = 30.0
DEFAULT_SEARCH_QUERIES_LIMIT = 500


def _find_harness_binary() -> Optional[Path]:
    custom = os.environ.get("MLRESCURIX_HARNESS_BIN")
    if custom:
        p = Path(custom)
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "target" / "release" / "mlrescurix-harness",
        repo_root / "target" / "debug" / "mlrescurix-harness",
    ]
    existing = [c.resolve() for c in candidates if c.is_file() and os.access(c, os.X_OK)]
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)
    which_path = shutil.which("mlrescurix-harness")
    if which_path:
        return Path(which_path).resolve()
    return None


@dataclass
class OperationalBudgets:
    """Tracks consumption and enforces limits for the 3 operational budgets."""
    model_tokens_limit: int = DEFAULT_TOKEN_LIMIT
    model_tokens_used: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    missing_token_receipts: int = 0

    kaggle_hours_limit: float = DEFAULT_KAGGLE_HOURS_LIMIT
    kaggle_hours_used: float = 0.0

    web_search_limit: int = DEFAULT_SEARCH_QUERIES_LIMIT
    web_search_used: int = 0

    search_connected: bool = False
    kaggle_connected: bool = False

    config_source: Optional[str] = None
    ledger_path: Optional[str] = None
    db_path: Optional[str] = None
    authority_binary: Optional[str] = None
    project_id: str = field(default_factory=lambda: f"campaign_{uuid.uuid4().hex[:8]}")

    def __post_init__(self):
        if not self.db_path:
            env_db = os.environ.get("MLRESCURIX_STATE_DB")
            if env_db:
                self.db_path = env_db
            elif self.ledger_path and Path(self.ledger_path).parent.joinpath("state.sqlite").is_file():
                self.db_path = str(Path(self.ledger_path).parent / "state.sqlite")
            else:
                clean_proj = re.sub(r"[^\w.-]", "_", self.project_id)
                self.db_path = str(Path(tempfile.gettempdir()) / f"mlrescurix_search_{clean_proj}.sqlite")
        self.web_search_used = max(self.web_search_used, self._query_durable_search_count())

    def _authority_binary(self) -> Optional[Path]:
        if self.authority_binary:
            path = Path(self.authority_binary)
            return path if path.is_file() and os.access(path, os.X_OK) else None
        return _find_harness_binary()

    def _query_durable_search_count(self) -> int:
        if not self.db_path or not Path(self.db_path).is_file():
            return 0
        harness_bin = self._authority_binary()
        if not harness_bin:
            return 0
        try:
            cmd = [
                str(harness_bin),
                "search-count",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            res = json.loads(proc.stdout)
            return int(res.get("count", 0))
        except Exception as exc:
            logger.warning(f"Rust harness search-count failed: {exc}")
            return 0

    def admit_search(self, query: str, sanitized_query: str, provider: str = "unknown") -> tuple[bool, Optional[str]]:
        """Atomically admit a search query before dispatch. Returns (admitted: bool, attempt_id: Optional[str]).

        Delegates strictly to mlrescurix-harness authority. Fails closed if the Rust authority is unavailable.
        """
        if not self.db_path:
            logger.error("Durable state database missing; search admission failed closed")
            return False, None

        harness_bin = self._authority_binary()
        if not harness_bin:
            logger.error("Single Rust resource authority binary not found; search admission failed closed")
            return False, None

        try:
            cmd = [
                str(harness_bin),
                "search-admit",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
                "--query", str(query),
                "--sanitized", str(sanitized_query),
                "--provider", str(provider),
                "--limit", str(self.web_search_limit),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            res = json.loads(proc.stdout)
            admitted = bool(res.get("admitted", False))
            attempt_id = res.get("attempt_id")
            if admitted:
                self.web_search_used = self._query_durable_search_count()
            return admitted, attempt_id
        except Exception as exc:
            logger.error(f"Rust harness search-admit failed closed: {exc}")
            return False, None

    def settle_search(
        self,
        attempt_id: str,
        status: str = "success",
        error: Optional[str] = None,
        evidence: Optional[str] = None,
        cost_millicents: int = 0,
    ) -> None:
        """Idempotently settle an admitted search attempt via the single Rust resource authority."""
        if not self.db_path:
            return
        harness_bin = self._authority_binary()
        if not harness_bin:
            logger.error(f"Cannot settle search attempt {attempt_id}: Rust harness authority binary missing")
            raise RuntimeError(f"Cannot settle search attempt {attempt_id}: Rust harness authority binary missing")
        try:
            cmd = [
                str(harness_bin),
                "search-settle",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
                "--attempt-id", str(attempt_id),
                "--status", str(status),
                "--cost-millicents", str(cost_millicents),
            ]
            if error:
                cmd.extend(["--error", str(error)])
            if evidence:
                cmd.extend(["--evidence", str(evidence)])
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.sync_authoritative_status()
        except Exception as exc:
            logger.error(f"Rust harness search-settle failed: {exc}")
            raise RuntimeError(f"Rust harness search-settle failed for attempt {attempt_id}: {exc}") from exc

    def admit_kaggle_job(
        self,
        job_id: str,
        account_username: str,
        notebook_title: str,
        gpu_type: str = "gpu",
        requested_hours: float = 12.0,
    ) -> tuple[bool, Optional[str]]:
        """Atomically admit a Kaggle compute job via Rust resource authority."""
        if not self.db_path:
            return False, None
        harness_bin = self._authority_binary()
        if not harness_bin:
            logger.error("Single Rust resource authority binary not found; kaggle admission failed closed")
            return False, None
        try:
            cmd = [
                str(harness_bin),
                "kaggle-admit",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
                "--job-id", str(job_id),
                "--account-username", str(account_username),
                "--notebook-title", str(notebook_title),
                "--gpu-type", str(gpu_type),
                "--max-hours", str(requested_hours),
                "--quota-hours-limit", str(self.kaggle_hours_limit),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            res = json.loads(proc.stdout)
            admitted = bool(res.get("admitted", False))
            return admitted, job_id if admitted else None
        except Exception as exc:
            logger.error(f"Rust harness kaggle-admit failed closed: {exc}")
            return False, None

    def settle_kaggle_job(
        self,
        job_id: str,
        status: str = "success",
        actual_hours: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Idempotently settle a Kaggle job via Rust resource authority."""
        if not self.db_path:
            return
        harness_bin = self._authority_binary()
        if not harness_bin:
            logger.error(f"Cannot settle kaggle job {job_id}: Rust harness authority binary missing")
            raise RuntimeError(f"Cannot settle kaggle job {job_id}: Rust harness authority binary missing")
        try:
            cmd = [
                str(harness_bin),
                "kaggle-settle",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
                "--job-id", str(job_id),
                "--status", str(status),
                "--actual-hours", str(actual_hours),
            ]
            if error:
                cmd.extend(["--error", str(error)])
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.sync_authoritative_status()
        except Exception as exc:
            logger.error(f"Rust harness kaggle-settle failed: {exc}")
            raise RuntimeError(f"Rust harness kaggle-settle failed for job {job_id}: {exc}") from exc

    def admit_token_reservation(
        self,
        reservation_id: Optional[str] = None,
        model: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_millicents: int = 0,
        role_id: str = "worker",
        spend_limit_millicents: int = 10_000_000,
        model_id: Optional[str] = None,
        token_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> tuple[bool, Optional[str]]:
        """Atomically admit a model token reservation via Rust resource authority."""
        if not self.db_path:
            return False, None
        harness_bin = self._authority_binary()
        if not harness_bin:
            logger.error("Single Rust resource authority binary not found; token reservation failed closed")
            return False, None
        actual_model = model or model_id or "unknown"
        actual_res_id = reservation_id or f"res_{uuid.uuid4().hex[:12]}"
        actual_token_limit = token_limit if token_limit is not None else self.model_tokens_limit
        try:
            cmd = [
                str(harness_bin),
                "token-admit",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
                "--reservation-id", str(actual_res_id),
                "--role-id", str(role_id),
                "--model", str(actual_model),
                "--estimated-prompt-tokens", str(prompt_tokens),
                "--estimated-completion-tokens", str(completion_tokens),
                "--cost-millicents", str(cost_millicents),
                "--token-limit", str(actual_token_limit),
                "--spend-limit-millicents", str(spend_limit_millicents),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            res = json.loads(proc.stdout)
            admitted = bool(res.get("admitted", False))
            return admitted, actual_res_id if admitted else None
        except Exception as exc:
            logger.error(f"Rust harness token-admit failed closed: {exc}")
            return False, None

    def settle_token_reservation(
        self,
        reservation_id: str,
        status: str = "ok",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_millicents: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Idempotently settle a model token reservation via Rust resource authority."""
        if not self.db_path:
            return
        harness_bin = self._authority_binary()
        if not harness_bin:
            logger.error(f"Cannot settle token reservation {reservation_id}: Rust harness authority binary missing")
            raise RuntimeError(f"Cannot settle token reservation {reservation_id}: Rust harness authority binary missing")
        try:
            cmd = [
                str(harness_bin),
                "token-settle",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
                "--reservation-id", str(reservation_id),
                "--status", str(status),
                "--prompt-tokens", str(prompt_tokens),
                "--completion-tokens", str(completion_tokens),
                "--cost-millicents", str(cost_millicents),
            ]
            if error:
                cmd.extend(["--error", str(error)])
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.sync_authoritative_status()
        except Exception as exc:
            logger.error(f"Rust harness token-settle failed: {exc}")
            raise RuntimeError(f"Rust harness token-settle failed for reservation {reservation_id}: {exc}") from exc

    def query_authoritative_resource_status(self) -> Optional[Dict[str, Any]]:
        """Query single authoritative resource status from Rust harness."""
        if not self.db_path:
            return None
        harness_bin = self._authority_binary()
        if not harness_bin:
            return None
        try:
            cmd = [
                str(harness_bin),
                "resource-status",
                "--state", str(self.db_path),
                "--project", str(self.project_id),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(proc.stdout)
        except Exception as exc:
            logger.warning(f"Rust harness resource-status failed: {exc}")
            return None

    def sync_authoritative_status(self) -> Optional[Dict[str, Any]]:
        """Sync in-memory usage counters with single authoritative Rust SQLite state."""
        status = self.query_authoritative_resource_status()
        if not status:
            return None
        # Locally recorded usage remains a useful lower bound even when provider
        # accounting is incomplete. The verification flag is reported separately.
        if status.get("prompt_tokens") is not None:
            self.prompt_tokens_used = max(self.prompt_tokens_used, int(status["prompt_tokens"]))
        if status.get("completion_tokens") is not None:
            self.completion_tokens_used = max(self.completion_tokens_used, int(status["completion_tokens"]))
        if status.get("total_tokens") is not None:
            self.model_tokens_used = max(self.model_tokens_used, int(status["total_tokens"]))
        if status.get("search_attempts") is not None:
            self.web_search_used = max(self.web_search_used, int(status["search_attempts"]))
        if status.get("kaggle_hours_used") is not None:
            self.kaggle_hours_used = max(self.kaggle_hours_used, float(status["kaggle_hours_used"]))
        return status

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None, workspace_root: Optional[Union[str, Path]] = None,
             *, project_id: Optional[str] = None, db_path: Optional[Union[str, Path]] = None,
             authority_binary: Optional[Union[str, Path]] = None) -> "OperationalBudgets":
        """Load budgets from config file or return generous defaults."""
        binding = {}
        if project_id is not None:
            binding["project_id"] = project_id
        if db_path is not None:
            binding["db_path"] = str(db_path)
        if authority_binary is not None:
            binding["authority_binary"] = str(authority_binary)
        candidates = []
        if config_path:
            candidates.append(Path(config_path))
        if workspace_root:
            candidates.append(Path(workspace_root) / "operational_budgets.json")
            candidates.append(Path(workspace_root) / ".config" / "operational_budgets.json")
            candidates.append(Path(workspace_root) / "config" / "operational_budgets.json")

        repo_root = Path(__file__).resolve().parents[4]
        candidates.extend([
            repo_root / "config" / "operational_budgets.json",
            repo_root / ".config" / "operational_budgets.json",
            Path.cwd() / "config" / "operational_budgets.json",
            Path.cwd() / ".config" / "operational_budgets.json",
        ])

        found_path: Optional[Path] = None
        for c in candidates:
            if c.is_file():
                found_path = c.resolve()
                break

        if not found_path:
            logger.info("No operational_budgets.json found; initializing generous defaults")
            return cls(**binding)

        try:
            data = json.loads(found_path.read_text(encoding="utf-8"))
            budgets = data.get("budgets", {})
            token_cfg = budgets.get("total_model_tokens", {})
            kaggle_cfg = budgets.get("kaggle_compute", {})
            search_cfg = budgets.get("web_search_queries", {})

            token_limit = int(token_cfg.get("limit", DEFAULT_TOKEN_LIMIT))
            kaggle_limit = float(kaggle_cfg.get("limit_hours", DEFAULT_KAGGLE_HOURS_LIMIT))
            search_limit = int(search_cfg.get("limit", DEFAULT_SEARCH_QUERIES_LIMIT))
            if token_limit < 0 or search_limit < 0 or not math.isfinite(kaggle_limit) or kaggle_limit < 0:
                raise ValueError("Budget limits must be nonnegative and finite")

            preservation = data.get("spending_cap_preservation", {})
            ledger_path = preservation.get("campaign_ledger_path")

            search_conn = False
            kaggle_conn = False
            try:
                from .governed_research import GovernedResearchClient
                search_conn = GovernedResearchClient().is_provider_available()
            except Exception:
                pass
            try:
                from .kaggle_pool import KaggleAccountPool
                pool = KaggleAccountPool()
                kaggle_conn = len(pool.list_accounts()) > 0 and pool.get_anchor_account() is not None
            except Exception:
                pass

            return cls(
                model_tokens_limit=token_limit,
                kaggle_hours_limit=kaggle_limit,
                web_search_limit=search_limit,
                search_connected=search_conn,
                kaggle_connected=kaggle_conn,
                config_source=str(found_path),
                ledger_path=ledger_path,
                **binding,
            )
        except Exception as exc:
            raise ValueError(f"Invalid operational budget configuration at {found_path}: {exc}") from exc

    @property
    def model_tokens_remaining(self) -> int:
        return max(0, self.model_tokens_limit - self.model_tokens_used)

    @property
    def kaggle_hours_remaining(self) -> float:
        return max(0.0, round(self.kaggle_hours_limit - self.kaggle_hours_used, 4))

    @property
    def web_search_remaining(self) -> int:
        return max(0, self.web_search_limit - self.web_search_used)

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> int:
        """Records token usage from a completed LLM interaction."""
        p = max(0, int(prompt_tokens))
        c = max(0, int(completion_tokens))
        total = p + c
        self.prompt_tokens_used += p
        self.completion_tokens_used += c
        self.model_tokens_used += total
        return total

    def restore_operational_budgets(self, turns: list) -> None:
        """Rebuild usage for all operational budgets from durable turn history."""
        self.restore_token_usage(turns)
        session_queries = [
            int(t.get("operational_budgets", {}).get("web_search_queries", {}).get("recorded_session_queries", 0)
                or t.get("operational_budgets", {}).get("web_search_queries", {}).get("used") or 0)
            for t in turns
        ]
        durable_searches = self._query_durable_search_count()
        self.web_search_used = max(max(session_queries, default=0), durable_searches)
        session_hours = [
            float(t.get("operational_budgets", {}).get("kaggle_compute", {}).get("recorded_session_hours", 0.0)
                  or t.get("operational_budgets", {}).get("kaggle_compute", {}).get("used_hours") or 0.0)
            for t in turns
        ]
        self.kaggle_hours_used = max(session_hours, default=0.0)

    def restore_token_usage(self, turns: list) -> None:
        """Rebuild usage from every durable turn; safe on repeated checkpoint reloads."""
        self.model_tokens_used = self.prompt_tokens_used = self.completion_tokens_used = 0
        self.missing_token_receipts = 0
        for turn in turns:
            self.record_provider_usage(turn.get("provider_receipt") or {})

    def record_provider_usage(self, receipt: dict) -> None:
        usage = receipt.get("usage")
        if not isinstance(usage, dict) or "prompt_tokens" not in usage or "completion_tokens" not in usage:
            self.missing_token_receipts += 1
            return
        try:
            self.record_tokens(usage["prompt_tokens"], usage["completion_tokens"])
        except (TypeError, ValueError, OverflowError):
            self.missing_token_receipts += 1

    def record_kaggle_compute(self, duration_seconds: float) -> float:
        """Records Kaggle accelerator compute in hours."""
        hours = max(0.0, float(duration_seconds) / 3600.0)
        self.kaggle_hours_used += hours
        return hours

    def record_search_query(self, count: int = 1) -> int:
        """Records web search query executions."""
        cnt = max(0, int(count))
        self.web_search_used += cnt
        return cnt

    def has_tokens_available(self, estimated_tokens: int = 0) -> bool:
        if self.db_path:
            self.sync_authoritative_status()
        return (self.model_tokens_used + max(0, estimated_tokens)) <= self.model_tokens_limit

    def has_kaggle_available(self, estimated_hours: float = 0.0) -> bool:
        if self.db_path:
            self.sync_authoritative_status()
        return (self.kaggle_hours_used + max(0.0, estimated_hours)) <= self.kaggle_hours_limit

    def sync_search_usage(self) -> int:
        """Query durable SQLite and update in-memory web_search_used count."""
        if self.db_path:
            self.web_search_used = max(self.web_search_used, self._query_durable_search_count())
        return self.web_search_used

    def has_search_available(self, count: int = 1) -> bool:
        if self.db_path:
            self.web_search_used = max(self.web_search_used, self._query_durable_search_count())
        return (self.web_search_used + max(0, count)) <= self.web_search_limit

    def get_status(self) -> Dict[str, Any]:
        """Structured dictionary of current usage against all three operational budgets."""
        if self.db_path:
            self.sync_authoritative_status()
        # A status response does not establish campaign-wide enforcement or
        # verified remote quotas; expose only observed session usage.
        search_status = {
            "limit": self.web_search_limit,
            "recorded_session_queries": self.web_search_used,
            "unit": "queries",
            "enforced": False,
            "connected": self.search_connected,
            "used": None,
            "remaining": None,
            "accounting_scope": "session observations only; unified authority not verified",
            "exhausted": None,
        }
        kaggle_status = {
            "limit_hours": self.kaggle_hours_limit,
            "recorded_session_hours": round(self.kaggle_hours_used, 4),
            "unit": "hours",
            "enforced": False,
            "connected": self.kaggle_connected,
            "used_hours": None,
            "remaining_hours": None,
            "accounting_scope": "session observations only; unified authority not verified",
            "exhausted": None,
        }

        return {
            "total_model_tokens": {
                "limit": self.model_tokens_limit,
                "used": self.model_tokens_used,
                "remaining": self.model_tokens_remaining if self.missing_token_receipts == 0 else None,
                "nominal_remaining": self.model_tokens_remaining,
                "prompt_tokens_used": self.prompt_tokens_used,
                "completion_tokens_used": self.completion_tokens_used,
                "accounting_complete": self.missing_token_receipts == 0,
                "missing_receipts": self.missing_token_receipts,
                "scope": "recorded calls in this autonomous session",
                "unit": "tokens",
                "exhausted": not self.has_tokens_available(),
            },
            "kaggle_compute": kaggle_status,
            "web_search_queries": search_status,
            "config_source": self.config_source,
        }

    def format_display(self) -> str:
        """Formatted human-readable string displaying actual usage and remaining budget."""
        status = self.get_status()
        tok = status["total_model_tokens"]
        kag = status["kaggle_compute"]
        sea = status["web_search_queries"]
        kag_display = (
            f"{kag['used_hours']:.2f}h / {kag['limit_hours']:.2f}h ({kag['remaining_hours']:.2f}h remaining)"
            if kag["enforced"]
            else f"usage unknown; configured {kag['limit_hours']:.2f}h (enforcement not verified)"
        )
        sea_display = (
            f"{sea['used']} / {sea['limit']} ({sea['remaining']} remaining)"
            if sea["enforced"]
            else f"usage unknown; configured {sea['limit']} queries (enforcement not verified)"
        )
        return (
            f"Operational Budgets Status:\n"
            f"  1. Model Tokens:   {tok['used']:,} recorded / {tok['limit']:,} {tok['unit']} (accounting complete: {tok['accounting_complete']})\n"
            f"  2. Kaggle Compute: {kag_display}\n"
            f"  3. Web Search:     {sea_display}"
        )
