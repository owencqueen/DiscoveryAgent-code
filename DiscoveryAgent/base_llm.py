"""One-shot Anthropic calls used by review design and empirical memory."""

import anthropic

from .config import LLMConfig


# Approximate USD prices per million input/output tokens, used only when a caller requests a cost.
_ANTHROPIC_PRICING_PER_MILLION = {
    "claude-opus-4-6": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}


def anthropic_api_calculate_cost(response, model="claude-opus-4-6"):
    input_price, output_price = _ANTHROPIC_PRICING_PER_MILLION[model]
    usage = response.usage
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000


class OnePassLLM:
    """Send one prompt to an Anthropic model and return text or tool input."""

    def __init__(self, system_prompt: str = None, llm_config: LLMConfig = None):
        self.llm_config = llm_config or LLMConfig(model_name="claude-opus-4-6")
        if self.llm_config.api_provider != "anthropic":
            raise ValueError("OnePassLLM requires an Anthropic model")
        self.model_name = self.llm_config.model_name
        self.temperature = self.llm_config.temperature
        self.max_tokens = self.llm_config.max_tokens
        self.system_prompt = system_prompt or ""
        self.client = anthropic.Anthropic(api_key=self.llm_config.api_key)

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def forward(self, input: str, max_tokens: int = None, response_format=None, get_cost: bool = False):
        params = {
            "model": self.model_name,
            "max_tokens": max_tokens or self.max_tokens or 2000,
            "temperature": self.temperature,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": input}],
        }
        if response_format is not None:
            params["tools"] = [{
                "name": "analyze_experimental_evidence",
                "description": "Analyze experimental evidence and provide structured findings",
                "input_schema": response_format,
            }]
            params["tool_choice"] = {"type": "tool", "name": "analyze_experimental_evidence"}
        response = self.client.messages.create(**params)
        if not response.content:
            raise ValueError("Anthropic returned no content")
        first = response.content[0]
        if response_format is not None and getattr(first, "type", None) == "tool_use":
            output = first.input
        else:
            output = first.text
        if get_cost:
            return output, anthropic_api_calculate_cost(response, model=self.model_name)
        return output

    def __call__(self, input, max_tokens=None, get_cost=False):
        return self.forward(input, max_tokens=max_tokens, get_cost=get_cost)
