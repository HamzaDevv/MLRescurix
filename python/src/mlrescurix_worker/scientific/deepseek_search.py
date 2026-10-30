"""Legacy research interface, disabled until a governed search provider is verified."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    summary: str
    rerank_score: float = 0.0


class DeepSeekSearchClient:
    """Compatibility interface; a DeepSeek model key does not provide web search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_DEEPSEEK_ENDPOINT,
        timeout_seconds: float = 25.0,
    ):
        # This disabled interface has no reason to read configured credentials.
        self.api_key = api_key or ""
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def search(
        self, query: str, count: int = 5, max_results: Optional[int] = None, need_content: bool = False
    ) -> List[SearchResult]:
        """Fail closed without sending the query to an unverified endpoint."""
        self.last_status = "unavailable"
        self.last_error = "No verified, governed web search provider configured"
        return []

    def fetch(self, target_url: str) -> str:
        """Report that fetching is unavailable without transmitting the URL."""
        self.last_status = "unavailable"
        self.last_error = "No verified, governed web fetch provider configured"
        return "[Web fetch unavailable: no verified, governed provider configured]"

    def research_topic(self, query: str, max_results: int = 5) -> str:
        """Context Firewall: Execute search and compress into < 2 KB high-signal Markdown."""
        results = self.search(query, max_results=max_results)
        if not results:
            return f"### Research Findings for '{query}'\nWeb search is unavailable."

        lines = [f"### Research Findings for '{query}'", ""]
        for i, res in enumerate(results, start=1):
            title = res.title.strip()
            summary = res.summary.strip() or res.snippet.strip()
            # Clean extraneous spaces/newlines
            summary = re.sub(r"\s+", " ", summary)
            if len(summary) > 300:
                summary = summary[:300] + "..."
            lines.append(f"**{i}. [{title}]({res.url})**")
            lines.append(f"> {summary}")
            lines.append("")

        compressed = "\n".join(lines)
        if len(compressed.encode("utf-8")) > 2048:
            compressed = compressed[:2000] + "\n\n... [truncated by Context Firewall]"
        return compressed
