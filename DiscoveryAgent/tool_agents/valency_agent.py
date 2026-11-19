"""Valency Bond scholarly research tool agent for DiscoveryAgent.

Integrates with Valency's Model Context Protocol (MCP) server (mcp.valency.io)
and REST/OAuth endpoints to grant autonomous discovery agents direct semantic
access to 44M+ scholarly preprints and peer-reviewed publications.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from ..config import LLMConfig
from ..utils.logger import Logger
from .base import AgentToolConfig, BaseAgent


DEFAULT_VALENCY_ENDPOINT = "https://mcp.valency.io"

DEFAULT_VALENCY_REPORT_INSTRUCTIONS = (
    "Return your literature review as Markdown with these sections:\n\n"
    "# Academic Literature Synthesis (Valency Bond)\n"
    "## Executive Summary\n"
    "- 3-5 core theoretical and empirical takeaways grounded in peer-reviewed literature\n\n"
    "## Direct Evidence & Critical Findings\n"
    "- Contrastive analysis across cited papers\n"
    "- Explicitly note methodological limits, sample sizes, and whether findings are causal or correlational\n\n"
    "## Citation Graph & Trajectory\n"
    "- Trace forward and backward citations linking seminal foundational papers to current preprints\n\n"
    "## Guardrails & Scientific Critique\n"
    "- Highlight potential outcome leakage, unproven dynamical assumptions, or flawed statistical metrics\n\n"
    "## Bibliography & Identifiers\n"
    "- Bulleted citations with paper title, authors, year, DOI / arXiv identifier, and URL\n"
)


@dataclass
class ValencyAgentConfig(AgentToolConfig):
    endpoint_url: str = DEFAULT_VALENCY_ENDPOINT
    api_token: Optional[str] = None
    log_dir: str = "logs"
    max_output_tokens: int = 5000
    timeout: int = 300


class ValencyScholarlyAgent(BaseAgent):
    """Scholarly search and citation graph agent powered by Valency Bond."""

    def __init__(self, config: ValencyAgentConfig, logger: Optional[Logger] = None):
        self.config = config
        self.endpoint_url = config.endpoint_url or os.getenv("VALENCY_MCP_URL", DEFAULT_VALENCY_ENDPOINT)
        self.api_token = config.api_token or os.getenv("VALENCY_API_KEY")
        self.logger = logger or Logger("valency_scholarly", log_dir=config.log_dir)

    def execute(self, query: str, searches_dir: Optional[str] = None) -> str:
        prompt = f"{query.strip()}\n\n{DEFAULT_VALENCY_REPORT_INSTRUCTIONS}"
        self.logger.log_prompt("user", prompt, "Valency Scholarly Query")

        # Headers for authenticated requests to Valency Bond MCP / REST endpoints
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DiscoveryAgent-ValencyBond/1.0",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        # Attempt structured search via Valency MCP / API
        result_text = ""
        try:
            # Query MCP tools or fallback search endpoint
            response = requests.post(
                f"{self.endpoint_url.rstrip('/')}/v1/search",
                json={
                    "query": query,
                    "max_results": 10,
                    "include_citations": True,
                },
                headers=headers,
                timeout=self.config.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                result_text = self._format_response_markdown(query, data)
            elif response.status_code == 401:
                # Informative response when OAuth token is needed
                result_text = (
                    f"### Valency Bond Authentication Notice\n\n"
                    f"The Valency Bond endpoint (`{self.endpoint_url}`) returned HTTP 401 Unauthorized.\n"
                    f"To activate live literature graph queries, authenticate your session at "
                    f"`https://mcp.valency.io` or set `VALENCY_API_KEY` in your environment.\n\n"
                    f"**Simulated synthesis for prompt:** {query}\n"
                )
            else:
                result_text = f"Valency Bond request failed with status {response.status_code}: {response.text}"
        except Exception as e:
            result_text = (
                f"### Valency Bond Connectivity Report\n\n"
                f"Endpoint: `{self.endpoint_url}`\n"
                f"Note: Network query raised: {str(e)}.\n"
                f"Ensure connectivity to `mcp.valency.io` or provide a valid OAuth Bearer token.\n"
            )

        if searches_dir:
            try:
                os.makedirs(searches_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"valency_{timestamp}.md"
                with open(os.path.join(searches_dir, fname), "w", encoding="utf-8") as f:
                    f.write(f"# Query: {query}\n\n{result_text}")
            except Exception as log_err:
                self.logger.log_prompt("warning", f"Could not write search log: {log_err}")

        return result_text

    def _format_response_markdown(self, query: str, data: Dict[str, Any]) -> str:
        lines = [
            f"# Valency Academic Literature Synthesis",
            f"**Query**: {query}\n",
            "## Key Publications & Evidence",
        ]
        results = data.get("results") or data.get("papers") or []
        for i, paper in enumerate(results, start=1):
            title = paper.get("title", "Untitled")
            authors = ", ".join(paper.get("authors", [])) or "Unknown authors"
            year = paper.get("year", "n.d.")
            abstract = paper.get("abstract") or paper.get("summary") or "No abstract available."
            doi = paper.get("doi") or paper.get("arxiv_id") or ""
            url = paper.get("url") or f"https://doi.org/{doi}" if doi else ""
            lines.append(f"### {i}. {title} ({year})")
            lines.append(f"- **Authors**: {authors}")
            if url:
                lines.append(f"- **Link / DOI**: [{doi or 'Link'}]({url})")
            lines.append(f"- **Abstract / Summary**: {abstract}\n")
        return "\n".join(lines)
