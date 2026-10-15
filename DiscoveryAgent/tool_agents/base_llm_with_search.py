"""One-shot Anthropic reviewer with native web search."""

from typing import Optional

import anthropic

from ..base_llm import anthropic_api_calculate_cost
from ..config import LLMConfig
from ..utils.logger import Logger


class BaseLLMWithSearch:
    """Web-enabled reviewer call used when ``review_search`` is set."""

    def __init__(
        self,
        llm_config: LLMConfig,
        log_dir: str = "logs",
        max_tokens: int = None,
        max_web_searches: int = 3,
        system_prompt: Optional[str] = None,
        logger=None,
    ):
        if llm_config.api_provider != "anthropic":
            raise ValueError("Review web search requires an Anthropic model")
        if not llm_config.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for review web search")
        self.llm_config = llm_config
        self.model_name = llm_config.model_name
        self.max_tokens = max_tokens or llm_config.max_tokens or 1200
        self.max_web_searches = max_web_searches
        self.system_prompt = system_prompt or llm_config.system_prompt or ""
        self.logger = logger or Logger("base_llm_with_search", log_dir=log_dir)
        self.client = anthropic.Anthropic(api_key=llm_config.api_key)

    def forward(self, input: str, max_tokens: int = None, response_format=None, get_cost: bool = False):
        if response_format is not None:
            raise ValueError("Structured reviewer output is unsupported with web search")
        self.logger.log_prompt("user", input, "base_llm_with_search_query")
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": input}],
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": self.max_web_searches,
                }],
            )
        except Exception as exc:
            self.logger.log_error(f"Review web search failed: {exc}")
            raise
        result = "\n".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        )
        self.logger.log_response(result, "base_llm_with_search_result")
        if get_cost:
            return result, anthropic_api_calculate_cost(response, model=self.model_name)
        return result
