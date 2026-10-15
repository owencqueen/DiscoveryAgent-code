"""Perplexity deep-research tool agent."""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Optional

import requests

from ..config import LLMConfig
from ..utils.logger import Logger
from .base import AgentToolConfig, BaseAgent


DEFAULT_REPORT_INSTRUCTIONS = (
    "Return your answer as Markdown with these sections:\n\n"
    "# Title\n"
    "## Executive Summary\n"
    "- 3-5 bullet points with the most important findings\n\n"
    "## Key Findings\n"
    "- Provide well-structured analysis with evidence\n"
    "- Note any disagreements across sources\n\n"
    "## Methodology\n"
    "- Briefly outline how you conducted the research\n\n"
    "## Limitations\n\n"
    "## Sources\n"
    "- Use a bulleted list of sources with titles and direct URLs\n"
)


def _extract_text_from_perplexity_response(json_resp: Any) -> str:
    """Extract text from Perplexity API JSON responses in a robust way."""
    try:
        # Common Perplexity response shapes may include 'choices' or nested 'message'/'content'
        if isinstance(json_resp, dict):
            # choices -> [ { message: { content: "..." } } ]
            choices = json_resp.get("choices")
            if choices and isinstance(choices, list):
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("content") or first.get("text")
                    if isinstance(msg, dict):
                        content = msg.get("content") or msg.get("text")
                        if isinstance(content, str):
                            return content
                    if isinstance(msg, str):
                        return msg

            # fallback keys
            for key in ("output", "answer", "text", "content", "response"):
                val = json_resp.get(key)
                if isinstance(val, str) and val.strip():
                    return val

        # As a last resort, stringify the whole response
        import json as _json

        return _json.dumps(json_resp)
    except Exception:
        return ""


@dataclass
class DeepResearchConfig(AgentToolConfig):
    llm_config: Optional[LLMConfig] = None
    log_dir: str = "logs"
    max_output_tokens: int = 3000
    timeout: int = 1800


class DeepResearchAgent(BaseAgent):
    """Return a source-linked report from ``sonar-deep-research``."""

    def __init__(self, config: DeepResearchConfig, logger=None):
        self.config = config
        self.llm_config = config.llm_config
        if self.llm_config is None or self.llm_config.model_name != "sonar-deep-research":
            raise ValueError("DeepResearchAgent requires sonar-deep-research")
        self.logger = logger or Logger("deep_research", log_dir=config.log_dir)

    def execute(self, query: str, searches_dir: Optional[str] = None) -> str:
        key = self.llm_config.api_key or os.getenv("PERPLEXITY_API_KEY")
        if not key:
            raise ValueError("PERPLEXITY_API_KEY is required for deep research")

        prompt = f"{query.strip()}\n\n{DEFAULT_REPORT_INSTRUCTIONS}"
        self.logger.log_prompt("user", prompt, "Deep Research Query")
        result = ""
        error: Optional[Exception] = None
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                json={
                    "model": "sonar-deep-research",
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=self.config.timeout,
            )
            if response.ok:
                result = _extract_text_from_perplexity_response(response.json())
            else:
                error = Exception(f"Perplexity API returned status {response.status_code}: {response.text}")
        except Exception as exc:
            error = exc
        if error is not None and not result:
            message = f"Error during deep research: {error}"
            self.logger.log_error(message)
            return message

        # Filter out any <think>...</think> blocks from the response
        if result:
            think_start = result.find("<think>")
            while think_start != -1:
                think_end = result.find("</think>", think_start)
                if think_end == -1:
                    break
                result = result[:think_start] + result[think_end + 8:]
                think_start = result.find("<think>")

        # Ensure some minimal Markdown structure if the model did not follow instructions
        if result and ("# " not in result):
            result = "# Research Report\n\n" + result.strip()

        if searches_dir:
            directory = Path(searches_dir)
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / f"deep_research_sonar-deep-research_{datetime.now():%Y%m%d_%H%M%S}.md"
            output.write_text(
                f"# Deep Research Output\n\n## Query\n{query}\n\n"
                f"## Model\nsonar-deep-research\n\n{result}",
                encoding="utf-8",
            )
        self.logger.log_response(result, "deep_research_result")
        return result
