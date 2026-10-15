"""Anthropic web-search tool agent."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import anthropic

from ..config import LLMConfig
from ..utils.logger import Logger
from ..utils.llm_diagnostics import format_api_exception, summarize_response_blocks
from .base import AgentToolConfig, BaseAgent


DEFAULT_SYSTEM_PROMPT = """
You are tasked with completing a simple web search for the user's query.
Your internal knowledge is limited, so please gather information from the web before reporting any findings or answering the user's query.
Use the available research tools to gather relevant information about the user's query. If Semantic Scholar or PubMed tools are available, prefer them. Otherwise, use the web_search tool. Then produce a concise, factual answer with source attributions when available.

IMPORTANT: You must use the web_search tool to gather information.
"""


def _extract_text_from_anthropic_response(response: Any) -> str:
    return "\n".join(
        block.text for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    )


def _format_anthropic_exception(exc: Exception, *, model_name: str, query: str, include_traceback: bool = False) -> str:
    return format_api_exception(
        exc,
        headline="Single web search failed.",
        context={"Model": model_name, "Query preview": query[:200]},
        include_traceback=include_traceback,
    )


@dataclass
class SingleSearchConfig(AgentToolConfig):
    llm_config: LLMConfig = None
    log_dir: str = "logs"
    max_tokens: int = 1200
    max_web_searches: int = 3
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    request_timeout_seconds: float = 180.0


class SingleSearchAgent(BaseAgent):
    """Run one Anthropic Messages call with at most three web searches."""

    def __init__(self, config: SingleSearchConfig, logger=None):
        self.config = config
        self.llm_config = config.llm_config
        if self.llm_config.api_provider != "anthropic":
            raise ValueError("SingleSearchAgent requires an Anthropic model")
        if not self.llm_config.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(
            api_key=self.llm_config.api_key,
            timeout=config.request_timeout_seconds,
        )
        self.logger = logger or Logger("single_search", log_dir=config.log_dir)

    def execute(self, query: str, searches_dir: Optional[str] = None) -> str:
        self.logger.log_prompt("user", query, "Single Web Search Query")
        self.logger.log_action(
            "single_search_request",
            f"model={self.llm_config.model_name}; max_web_searches={self.config.max_web_searches}",
        )
        try:
            response = self.client.messages.create(
                model=self.llm_config.model_name,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                messages=[{"role": "user", "content": query}],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": self.config.max_web_searches,
                }],
            )
        except Exception as exc:
            self.logger.log_error(_format_anthropic_exception(
                exc, model_name=self.llm_config.model_name, query=query, include_traceback=True,
            ))
            raise RuntimeError(_format_anthropic_exception(
                exc, model_name=self.llm_config.model_name, query=query,
            )) from exc

        result = _extract_text_from_anthropic_response(response)
        self.logger.log_action(
            "single_search_response",
            f"stop_reason={getattr(response, 'stop_reason', None)}; blocks={summarize_response_blocks(response)}",
        )
        if not result.strip():
            result = "Single web search completed but returned no text."
            self.logger.log_error(result)
        self.logger.log_response(result, "single_search_result")
        if searches_dir:
            directory = Path(searches_dir)
            directory.mkdir(parents=True, exist_ok=True)
            name = f"single_search_{self.llm_config.model_name}_{datetime.now():%Y%m%d_%H%M%S}.md"
            (directory / name).write_text(
                f"# Single Search Output\n\n## Query\n{query}\n\n## Model\n{self.llm_config.model_name}\n\n{result}",
                encoding="utf-8",
            )
        return result
