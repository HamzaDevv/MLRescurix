"""Governed search client interface for MLRescurix research.

Integrates with GovernedResearchClient:
- Validates query against privacy invariants (no raw dataset rows, csv filenames, or credentials).
- Detects configured search providers (Tavily, SerpApi, Brave, Kagi, or Mock).
- Fails closed when no provider is configured, reporting the exact missing dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from .governed_research import (
    BaseSearchProvider,
    GovernedResearchClient,
    MockSearchProvider,
    PrivacyViolationError,
    SearchSource,
)

DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    summary: str
    rerank_score: float = 0.0


class DeepSeekSearchClient:
    """Governed research client delegating to GovernedResearchClient with strict privacy enforcement."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_DEEPSEEK_ENDPOINT,
        timeout_seconds: float = 25.0,
        provider: Optional[BaseSearchProvider] = None,
        forbidden_tokens: Optional[List[str]] = None,
        budgets: Optional[Any] = None,
    ):
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.budgets = budgets
        self.governed_client = GovernedResearchClient(
            provider=provider,
            forbidden_tokens=forbidden_tokens,
        )
        self.last_status = "idle"
        self.last_error: Optional[str] = None

    def search(
        self, query: str, count: int = 5, max_results: Optional[int] = None, need_content: bool = False
    ) -> List[SearchResult]:
        limit = max_results or count
        res = self.governed_client.research_topic(query, max_results=limit, budgets=self.budgets)
        self.last_status = res["status"]
        if res["status"] == "success":
            self.last_error = None
            return [
                SearchResult(
                    title=s["title"],
                    url=s["url"],
                    snippet=s["snippet"],
                    summary=s["snippet"],
                    rerank_score=s.get("rerank_score", 0.0),
                )
                for s in res.get("sources", [])
            ]
        self.last_error = res.get("error", "Search unavailable")
        return []

    def fetch(self, target_url: str) -> str:
        """Report that fetching is unavailable without transmitting the URL."""
        self.last_status = "unavailable"
        self.last_error = "No verified, governed web fetch provider configured"
        return "[Web fetch unavailable: no verified, governed provider configured]"

    def research_topic(self, query: str, max_results: int = 5, rationale: str = "") -> str:
        """Context Firewall: Execute search and compress into < 2 KB high-signal Markdown."""
        res = self.governed_client.research_topic(query, rationale=rationale, max_results=max_results, budgets=self.budgets)
        self.last_status = res["status"]
        self.last_error = res.get("error")
        return res.get("summary", "")
