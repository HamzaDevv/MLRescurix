"""Experimental compatibility wrapper for an explicitly selected Rust gateway.

Direct provider dispatch and automatic fallback are disabled. The Rust adapter
and its caller retain responsibility for policy, budget and reconciliation.
The local pricing table is historical and is not an authoritative cost bound.
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

MODEL_TRANSLATION_MAP = {
    "deepseek/deepseek-chat": "deepseek-chat",
    "deepseek/deepseek-v4.1-flash": "deepseek-chat",
    "deepseek/deepseek-v4.1-flash-1017": "deepseek-chat",
    "deepseek/deepseek-r1": "deepseek-reasoner",
    "deepseek-chat": "deepseek-chat",
    "deepseek-reasoner": "deepseek-reasoner",
}

PRICING_TABLE = {
    "deepseek/deepseek-v4.1-flash": {"input_per_m": 0.08, "output_per_m": 0.50},
    "deepseek/deepseek-chat": {"input_per_m": 0.14, "output_per_m": 0.28},
    "deepseek-chat": {"input_per_m": 0.14, "output_per_m": 0.28},
    "deepseek-reasoner": {"input_per_m": 0.55, "output_per_m": 2.19},
}


class RoutePolicy:
    AUTO_FAILOVER = "auto_failover"
    PRIMARY_ONLY = "primary_only"
    FALLBACK_ONLY = "fallback_only"


class DualTierGateway:
    """Delegates an explicit primary-only request to a supplied Rust adapter."""

    def __init__(
        self,
        openrouter_key: Optional[str] = None,
        deepseek_key: Optional[str] = None,
        primary_model: str = "deepseek/deepseek-v4.1-flash",
        timeout_seconds: float = 35.0,
        route_policy: str = RoutePolicy.AUTO_FAILOVER,
        max_attempts_per_route: int = 2,
        rust_adapter: Optional[Any] = None,
        campaign_authority: Optional[Any] = None,
        model_map: Optional[Dict[str, str]] = None,
        pricing_table: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.openrouter_key = openrouter_key or self._resolve_key("OPENROUTER_API_KEY", ".config/openrouter.key")
        self.deepseek_key = deepseek_key or self._resolve_key("DEEPSEEK_API_KEY", ".config/deepseek.key")
        self.primary_model = primary_model
        self.timeout_seconds = timeout_seconds
        self.route_policy = route_policy
        self.max_attempts_per_route = max(1, int(max_attempts_per_route))
        self.campaign_ledger: Optional[Path] = None
        self.campaign_run: Optional[str] = None
        self.rust_adapter = rust_adapter
        self.campaign_authority = campaign_authority
        self.model_map = dict(MODEL_TRANSLATION_MAP)
        if model_map:
            self.model_map.update(model_map)
        self.pricing_table = dict(PRICING_TABLE)
        if pricing_table:
            self.pricing_table.update(pricing_table)

    @staticmethod
    def _resolve_key(env_var: str, file_path: str) -> str:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
        p = Path(file_path)
        if p.exists():
            txt = p.read_text().strip()
            if txt:
                return txt
        return ""

    def bind_campaign(
        self,
        ledger_path: Union[str, Path],
        run_id: str,
        campaign_authority: Optional[Any] = None,
    ) -> None:
        """Bind gateway to a shared campaign ledger for accounting governance."""
        self.campaign_ledger = Path(ledger_path).resolve()
        self.campaign_run = run_id
        if campaign_authority is not None:
            self.campaign_authority = campaign_authority

    def reservation(self, messages: List[Dict[str, Any]], max_tokens: int) -> Decimal:
        """Return a historical estimate; this is not suitable for budget admission."""
        primary_pricing = self.pricing_table.get(self.primary_model, {"input_per_m": 0.20, "output_per_m": 0.60})
        fallback_model = self.model_map.get(self.primary_model, self.primary_model)
        fallback_pricing = self.pricing_table.get(fallback_model, {"input_per_m": 0.20, "output_per_m": 0.60})
        max_in = max(primary_pricing["input_per_m"], fallback_pricing["input_per_m"])
        max_out = max(primary_pricing["output_per_m"], fallback_pricing["output_per_m"])
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        # Conservative token upper bound: at most 1 token per byte + 512 framing (bytes // 3 is not an upper bound)
        est_prompt_tokens = prompt_bytes + 512
        in_cost = (est_prompt_tokens / 1_000_000) * max_in
        out_cost = (max_tokens / 1_000_000) * max_out
        return Decimal(str(round(in_cost + out_cost, 6)))

    def call_with_request_id(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2500,
        request_id: str = "",
        policy_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delegate one explicitly selected route; direct calls and failover are disabled."""
        # Direct HTTP retries do not have durable per-attempt reservations or
        # ambiguity fencing. Keep them unreachable until that integration exists.
        if self.rust_adapter is None:
            raise RuntimeError("Direct dual-gateway dispatch is disabled; use the governed Rust gateway")
        effective_policy = policy_override or self.route_policy
        if effective_policy != RoutePolicy.PRIMARY_ONLY:
            raise RuntimeError("Dual-gateway fallback is unavailable until governed routing is implemented")
        if not request_id:
            raise ValueError("A stable request_id is required for governed dispatch")
        if self.campaign_authority is not None:
            raise RuntimeError("Dual-gateway campaign settlement is disabled; use the authoritative dispatch path")
        return self.rust_adapter.call_with_request_id(messages, max_tokens, request_id)
