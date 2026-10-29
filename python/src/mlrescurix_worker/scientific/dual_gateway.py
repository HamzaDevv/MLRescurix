"""Dual-Tier LLM Gateway: OpenRouter Primary with DeepSeek Direct Failover.

Provides robust, production-grade LLM dispatch:
1. Tier 1 (Primary): OpenRouter API (https://openrouter.ai/api/v1/chat/completions).
2. Tier 2 (Fallback): Direct DeepSeek API (https://api.deepseek.com/chat/completions).
3. Dynamic Model Translation: Maps OpenRouter slug format (e.g. `deepseek/deepseek-chat`)
   to direct DeepSeek format (`deepseek-chat`) during fallback.
4. Error Interception: Triggers failover on HTTP 429 (rate limit), 500/502/503/504 (provider outage),
   and connection timeouts.
5. Governed Route Policies:
   - AUTO_FAILOVER: Tries primary first, fails over to direct DeepSeek.
   - PRIMARY_ONLY: Restricts dispatch strictly to OpenRouter.
   - FALLBACK_ONLY: Direct dispatch to DeepSeek without OpenRouter.
6. Campaign Budget & Replay Accounting:
   - Enforces campaign binding and bounded retry attempts per route.
"""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

logger = logging.getLogger("mlrescurix.dual_gateway")

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
    """Dispatches chat completions to OpenRouter with automatic failover to DeepSeek Direct."""

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
        """Estimate upper-bound cost for budget reservation covering primary and fallback rates."""
        primary_pricing = PRICING_TABLE.get(self.primary_model, {"input_per_m": 0.20, "output_per_m": 0.60})
        fallback_model = MODEL_TRANSLATION_MAP.get(self.primary_model, "deepseek-chat")
        fallback_pricing = PRICING_TABLE.get(fallback_model, {"input_per_m": 0.20, "output_per_m": 0.60})
        max_in = max(primary_pricing["input_per_m"], fallback_pricing["input_per_m"])
        max_out = max(primary_pricing["output_per_m"], fallback_pricing["output_per_m"])
        prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        est_prompt_tokens = prompt_bytes // 3 + 512
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
        """Dispatch completion under governed route policy with bounded retry attempts."""
        start_time = time.monotonic()
        last_error = None
        effective_policy = policy_override or self.route_policy
        req_id = request_id or f"req-{int(time.time() * 1000)}"
        run_id = self.campaign_run or "governed_llm"

        # Apply pre-dispatch campaign reservation
        res_cost = self.reservation(messages, max_tokens)
        if self.campaign_authority and hasattr(self.campaign_authority, "reserve"):
            try:
                self.campaign_authority.reserve(run_id, req_id, res_cost)
            except Exception as exc:
                logger.error("Campaign reservation rejected for request %s: %s", req_id, exc)
                raise

        # If rust_adapter is configured, route via Rust control plane
        if self.rust_adapter is not None:
            try:
                res = self.rust_adapter.call_with_request_id(messages, max_tokens, req_id)
                if self.campaign_authority and hasattr(self.campaign_authority, "settle"):
                    billed = Decimal(str(res.get("cost_usd", "0.0")))
                    self.campaign_authority.settle(run_id, req_id, billed, cost_kind="provider_verified")
                return res
            except Exception as exc:
                if self.campaign_authority and hasattr(self.campaign_authority, "mark_uncertain"):
                    self.campaign_authority.mark_uncertain(run_id, req_id, res_cost)
                raise

        # 1. Primary Attempt: OpenRouter (unless restricted to FALLBACK_ONLY)
        if effective_policy in (RoutePolicy.AUTO_FAILOVER, RoutePolicy.PRIMARY_ONLY):
            if self.openrouter_key:
                for attempt in range(1, self.max_attempts_per_route + 1):
                    try:
                        logger.info(
                            "Dispatching to Tier 1: OpenRouter (%s), attempt %d/%d",
                            self.primary_model,
                            attempt,
                            self.max_attempts_per_route,
                        )
                        res = self._post_json(
                            url=OPENROUTER_URL,
                            api_key=self.openrouter_key,
                            model=self.primary_model,
                            messages=messages,
                            max_tokens=max_tokens,
                            extra_headers={"HTTP-Referer": "https://github.com/mlrescurix", "X-Title": "MLRescurix"},
                        )
                        res["provider_source"] = "openrouter"
                        res["duration_seconds"] = time.monotonic() - start_time
                        res["request_id"] = req_id
                        res["attempts"] = attempt
                        res["route_policy"] = effective_policy
                        if self.campaign_authority and hasattr(self.campaign_authority, "settle"):
                            try:
                                billed = Decimal(str(res.get("cost_usd", "0.0")))
                                self.campaign_authority.settle(run_id, req_id, billed, cost_kind="unverified")
                            except Exception:
                                pass
                        return res
                    except Exception as exc:
                        last_error = exc
                        logger.warning(
                            "Tier 1 OpenRouter attempt %d failed: %s",
                            attempt,
                            exc,
                        )
            else:
                last_error = RuntimeError("OpenRouter API key not configured")

            if effective_policy == RoutePolicy.PRIMARY_ONLY:
                raise RuntimeError(
                    f"RoutePolicy.PRIMARY_ONLY dispatch failed after {self.max_attempts_per_route} attempts. Last error: {last_error}"
                )

        # 2. Fallback Attempt: DeepSeek Direct (for AUTO_FAILOVER or FALLBACK_ONLY)
        if effective_policy in (RoutePolicy.AUTO_FAILOVER, RoutePolicy.FALLBACK_ONLY):
            if self.deepseek_key:
                fallback_model = MODEL_TRANSLATION_MAP.get(self.primary_model, "deepseek-chat")
                for attempt in range(1, self.max_attempts_per_route + 1):
                    try:
                        logger.info(
                            "Dispatching to Tier 2: DeepSeek Direct (%s), attempt %d/%d",
                            fallback_model,
                            attempt,
                            self.max_attempts_per_route,
                        )
                        res = self._post_json(
                            url=DEEPSEEK_URL,
                            api_key=self.deepseek_key,
                            model=fallback_model,
                            messages=messages,
                            max_tokens=max_tokens,
                        )
                        res["provider_source"] = "deepseek_direct"
                        res["duration_seconds"] = time.monotonic() - start_time
                        res["request_id"] = req_id
                        res["attempts"] = attempt
                        res["route_policy"] = effective_policy
                        if self.campaign_authority and hasattr(self.campaign_authority, "settle"):
                            try:
                                billed = Decimal(str(res.get("cost_usd", "0.0")))
                                self.campaign_authority.settle(run_id, req_id, billed, cost_kind="unverified")
                            except Exception:
                                pass
                        return res
                    except Exception as exc:
                        last_error = exc
                        logger.error(
                            "Tier 2 DeepSeek Direct attempt %d failed: %s",
                            attempt,
                            exc,
                        )
            else:
                last_error = RuntimeError("DeepSeek API key not configured")

        if self.campaign_authority and hasattr(self.campaign_authority, "mark_uncertain"):
            try:
                self.campaign_authority.mark_uncertain(run_id, req_id, res_cost)
            except Exception:
                pass

        raise RuntimeError(f"All permitted LLM gateway tiers failed under policy '{effective_policy}'. Last error: {last_error}")

    def _post_json(
        self,
        url: str,
        api_key: str,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }).encode("utf-8")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MLRescurix/2.0",
        }
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"Provider returned no choices: {data}")

        content = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        pricing = PRICING_TABLE.get(model, {"input_per_m": 0.14, "output_per_m": 0.28})
        cost_usd = (prompt_tokens / 1_000_000) * pricing["input_per_m"] + (completion_tokens / 1_000_000) * pricing["output_per_m"]

        return {
            "content": content,
            "usage": usage,
            "cost_usd": float(round(cost_usd, 6)),
            "cost_is_estimate": True,
            "model": model,
            "raw_response": data,
        }
