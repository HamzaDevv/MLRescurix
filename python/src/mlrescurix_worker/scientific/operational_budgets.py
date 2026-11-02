"""Operational budgets management for MLRescurix autonomous execution.

Enforces and tracks the three configurable operational budgets:
1. Total model token usage (counting prompt/input and completion/output tokens across all agents and calls).
2. Total Kaggle compute usage (measured in accelerator hours from the Kaggle environment).
3. Total web search queries.

Preserves existing dollar-denominated campaign budget caps without deleting or resetting history.
"""
from __future__ import annotations

import copy
import json
import logging
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

    kaggle_hours_limit: float = DEFAULT_KAGGLE_HOURS_LIMIT
    kaggle_hours_used: float = 0.0

    web_search_limit: int = DEFAULT_SEARCH_QUERIES_LIMIT
    web_search_used: int = 0

    config_source: Optional[str] = None
    ledger_path: Optional[str] = None

    @classmethod
    def load(cls, config_path: Optional[Union[str, Path]] = None, workspace_root: Optional[Union[str, Path]] = None) -> "OperationalBudgets":
        """Load budgets from config file or return generous defaults."""
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
            return cls()

        try:
            data = json.loads(found_path.read_text(encoding="utf-8"))
            budgets = data.get("budgets", {})
            token_cfg = budgets.get("total_model_tokens", {})
            kaggle_cfg = budgets.get("kaggle_compute", {})
            search_cfg = budgets.get("web_search_queries", {})

            token_limit = int(token_cfg.get("limit", DEFAULT_TOKEN_LIMIT))
            kaggle_limit = float(kaggle_cfg.get("limit_hours", DEFAULT_KAGGLE_HOURS_LIMIT))
            search_limit = int(search_cfg.get("limit", DEFAULT_SEARCH_QUERIES_LIMIT))

            preservation = data.get("spending_cap_preservation", {})
            ledger_path = preservation.get("campaign_ledger_path")

            return cls(
                model_tokens_limit=token_limit,
                kaggle_hours_limit=kaggle_limit,
                web_search_limit=search_limit,
                config_source=str(found_path),
                ledger_path=ledger_path,
            )
        except Exception as exc:
            logger.warning(f"Failed parsing operational budgets from {found_path}: {exc}; using defaults")
            return cls(config_source=str(found_path))

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

    def has_search_available(self, count: int = 1) -> bool:
        return (self.web_search_used + max(0, count)) <= self.web_search_limit

    def get_status(self) -> Dict[str, Any]:
        """Structured dictionary of current usage against all three operational budgets."""
        return {
            "total_model_tokens": {
                "limit": self.model_tokens_limit,
                "used": self.model_tokens_used,
                "remaining": self.model_tokens_remaining,
                "prompt_tokens_used": self.prompt_tokens_used,
                "completion_tokens_used": self.completion_tokens_used,
                "unit": "tokens",
                "exhausted": not self.has_tokens_available(),
            },
            "kaggle_compute": {
                "limit_hours": self.kaggle_hours_limit,
                "used_hours": round(self.kaggle_hours_used, 4),
                "remaining_hours": self.kaggle_hours_remaining,
                "unit": "hours",
                "exhausted": not self.has_kaggle_available(),
            },
            "web_search_queries": {
                "limit": self.web_search_limit,
                "used": self.web_search_used,
                "remaining": self.web_search_remaining,
                "unit": "queries",
                "exhausted": not self.has_search_available(),
            },
            "config_source": self.config_source,
        }

    def format_display(self) -> str:
        """Formatted human-readable string displaying actual usage and remaining budget."""
        tok = self.get_status()["total_model_tokens"]
        kag = self.get_status()["kaggle_compute"]
        sea = self.get_status()["web_search_queries"]
        return (
            f"Operational Budgets Status:\n"
            f"  1. Model Tokens:   {tok['used']:,} / {tok['limit']:,} {tok['unit']} ({tok['remaining']:,} remaining)\n"
            f"  2. Kaggle Compute: {kag['used_hours']:.2f} / {kag['limit_hours']:.2f} {kag['unit']} ({kag['remaining_hours']:.2f} remaining)\n"
            f"  3. Web Search:     {sea['used']} / {sea['limit']} {sea['unit']} ({sea['remaining']} remaining)"
        )
