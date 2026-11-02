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
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("mlrescurix.scientific.operational_budgets")

DEFAULT_TOKEN_LIMIT = 5_000_000
DEFAULT_KAGGLE_HOURS_LIMIT = 30.0
DEFAULT_SEARCH_QUERIES_LIMIT = 500


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
        self._ensure_db_schema()
        self.web_search_used = max(self.web_search_used, self._query_durable_search_count())

    def _ensure_db_schema(self) -> None:
        if not self.db_path:
            return
        try:
            p = Path(self.db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(p), timeout=10.0) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS operational_search_attempts (
                        project TEXT NOT NULL,
                        attempt_id TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        sanitized_query TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')),
                        cost_millicents INTEGER NOT NULL DEFAULT 0,
                        created_ms INTEGER NOT NULL,
                        settled_ms INTEGER,
                        error_message TEXT,
                        PRIMARY KEY(project, attempt_id)
                    );"""
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_search_attempts_status ON operational_search_attempts(project, status);")
        except Exception as exc:
            logger.warning(f"Failed to ensure search attempt database schema at {self.db_path}: {exc}")

    def _query_durable_search_count(self) -> int:
        if not self.db_path or not Path(self.db_path).is_file():
            return 0
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM operational_search_attempts WHERE project = ? AND status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')",
                    (self.project_id,)
                ).fetchone()
                return int(row[0]) if row else 0
        except Exception:
            return 0

    def _record_durable_search_attempt(self, attempt_id: str, query: str, sanitized: str, provider: str, status: str = "admitted") -> None:
        if not self.db_path:
            return
        try:
            now_ms = int(time.time() * 1000)
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute(
                    """INSERT INTO operational_search_attempts(project, attempt_id, query_text, sanitized_query, provider, status, cost_millicents, created_ms, settled_ms, error_message)
                       VALUES(?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL)
                       ON CONFLICT(project, attempt_id) DO NOTHING""",
                    (self.project_id, attempt_id, query, sanitized, provider, status, now_ms)
                )
        except Exception as exc:
            logger.warning(f"Failed to record search attempt {attempt_id}: {exc}")

    def _update_durable_search_attempt(self, attempt_id: str, status: str = "success", error: Optional[str] = None) -> None:
        if not self.db_path:
            return
        try:
            now_ms = int(time.time() * 1000)
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute(
                    """UPDATE operational_search_attempts
                       SET status = ?, settled_ms = ?, error_message = ?
                       WHERE project = ? AND attempt_id = ?""",
                    (status, now_ms, error, self.project_id, attempt_id)
                )
        except Exception as exc:
            logger.warning(f"Failed to update search attempt {attempt_id}: {exc}")

    def admit_search(self, query: str, sanitized_query: str, provider: str = "unknown") -> tuple[bool, Optional[str]]:
        """Atomically admit a search query before dispatch. Returns (admitted: bool, attempt_id: Optional[str])."""
        if not self.db_path:
            if not self.has_search_available(count=1):
                return False, None
            self.web_search_used += 1
            return True, f"search_attempt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        now_ms = int(time.time() * 1000)
        attempt_id = f"search_attempt_{now_ms}_{uuid.uuid4().hex[:8]}"
        try:
            with sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None) as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT COUNT(*) FROM operational_search_attempts WHERE project = ? AND status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')",
                    (self.project_id,)
                ).fetchone()
                current_count = int(row[0]) if row else 0
                if current_count >= self.web_search_limit:
                    conn.execute("ROLLBACK")
                    self.web_search_used = current_count
                    return False, None
                conn.execute(
                    """INSERT INTO operational_search_attempts(project, attempt_id, query_text, sanitized_query, provider, status, cost_millicents, created_ms, settled_ms, error_message)
                       VALUES(?, ?, ?, ?, ?, 'admitted', 0, ?, NULL, NULL)""",
                    (self.project_id, attempt_id, query, sanitized_query, provider, now_ms)
                )
                conn.execute("COMMIT")
                self.web_search_used = current_count + 1
                return True, attempt_id
        except Exception as exc:
            logger.warning(f"Failed atomic search admission: {exc}")
            return False, None

    def settle_search(self, attempt_id: str, status: str = "success", error: Optional[str] = None) -> None:
        """Idempotently settle an admitted search attempt."""
        self._update_durable_search_attempt(attempt_id, status=status, error=error)

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None, workspace_root: Optional[Union[str, Path]] = None,
             *, project_id: Optional[str] = None, db_path: Optional[Union[str, Path]] = None) -> "OperationalBudgets":
        """Load budgets from config file or return generous defaults."""
        binding = {}
        if project_id is not None:
            binding["project_id"] = project_id
        if db_path is not None:
            binding["db_path"] = str(db_path)
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
        return (self.model_tokens_used + max(0, estimated_tokens)) <= self.model_tokens_limit

    def has_kaggle_available(self, estimated_hours: float = 0.0) -> bool:
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
        search_status: Dict[str, Any] = {
            "limit": self.web_search_limit,
            "recorded_session_queries": self.web_search_used,
            "unit": "queries",
            "enforced": self.search_connected,
        }
        if self.search_connected:
            search_status["used"] = self.web_search_used
            search_status["remaining"] = self.web_search_remaining
            search_status["accounting_scope"] = "connected to governed web search provider"
            search_status["exhausted"] = not self.has_search_available()
        else:
            search_status["used"] = None
            search_status["remaining"] = None
            search_status["accounting_scope"] = "not connected to active web search provider (missing credentials)"
            search_status["exhausted"] = None

        kaggle_status: Dict[str, Any] = {
            "limit_hours": self.kaggle_hours_limit,
            "recorded_session_hours": round(self.kaggle_hours_used, 4),
            "unit": "hours",
            "enforced": self.kaggle_connected,
        }
        if self.kaggle_connected:
            kaggle_status["used_hours"] = round(self.kaggle_hours_used, 4)
            kaggle_status["remaining_hours"] = self.kaggle_hours_remaining
            kaggle_status["accounting_scope"] = "measured Kaggle accelerator hours"
            kaggle_status["exhausted"] = not self.has_kaggle_available()
        else:
            kaggle_status["used_hours"] = None
            kaggle_status["remaining_hours"] = None
            kaggle_status["accounting_scope"] = "not connected to Kaggle dispatch or account quota"
            kaggle_status["exhausted"] = None

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
            else f"usage unknown; configured {kag['limit_hours']:.2f}h (not connected or enforced)"
        )
        sea_display = (
            f"{sea['used']} / {sea['limit']} ({sea['remaining']} remaining)"
            if sea["enforced"]
            else f"usage unknown; configured {sea['limit']} queries (not connected or enforced)"
        )
        return (
            f"Operational Budgets Status:\n"
            f"  1. Model Tokens:   {tok['used']:,} recorded / {tok['limit']:,} {tok['unit']} (accounting complete: {tok['accounting_complete']})\n"
            f"  2. Kaggle Compute: {kag_display}\n"
            f"  3. Web Search:     {sea_display}"
        )
