import os
from types import SimpleNamespace

import pytest

# Ignore local Claude runtime overrides while importing the modules.
from DiscoveryAgent.hyp_agents.director_agent import DirectorAgent
from DiscoveryAgent.hyp_agents.hypothesis_gen_agent import HypothesisGenAgent
from DiscoveryAgent.tool_agents.single_search import _format_anthropic_exception
from DiscoveryAgent.utils.llm_diagnostics import (
    REFUSAL_ADVICE,
    describe_request,
    describe_response,
    first_refusal,
    format_api_exception,
    log_api_exception,
    log_api_exchange,
    refusal_notice,
    summarize_response_blocks,
)


class CaptureLogger:
    def __init__(self):
        self.actions = []
        self.errors = []

    def log_action(self, action, details=None):
        self.actions.append((action, details))

    def log_error(self, message, code=None):
        self.errors.append((message, code))


def block(**values):
    return SimpleNamespace(**values)


def response(*, stop_reason="end_turn", text="answer", stop_details=None):
    return block(
        model="claude-opus-4-6",
        stop_reason=stop_reason,
        stop_details=stop_details,
        _request_id="req-123",
        content=[] if text is None else [block(type="text", text=text)],
        usage=block(
            input_tokens=120,
            output_tokens=40,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=10,
        ),
    )


def test_request_and_response_diagnostics_include_decision_metadata():
    request_summary = describe_request(
        {
            "model": "claude-opus-4-6",
            "max_tokens": 4096,
            "messages": [{}, {}],
            "system": "system prompt",
            "thinking": {"type": "enabled", "budget_tokens": 512},
            "tools": [{"name": "search"}],
            "mcp_servers": [{"name": "papers"}],
        }
    )
    response_summary = describe_response(response())

    assert "model=claude-opus-4-6" in request_summary
    assert "max_tokens=4096" in request_summary
    assert "messages=2" in request_summary
    assert "system_chars=13" in request_summary
    assert "thinking=" in request_summary
    assert "tools=1" in request_summary
    assert "mcp_servers=1" in request_summary
    assert "stop_reason=end_turn" in response_summary
    assert "request_id=req-123" in response_summary
    assert "usage(input=120, output=40, cache_read_input=20, cache_creation_input=10)" in response_summary
    assert "0:text" in response_summary
    assert "chars=6" in response_summary
    assert "preview='answer'" in response_summary


def test_response_block_summary_covers_thinking_search_and_empty_content():
    described = summarize_response_blocks(
        block(
            content=[
                block(type="thinking", thinking="reasoning trace"),
                block(type="web_search_tool_result", id="search-1"),
                block(type="tool_use", name="calculator"),
            ]
        )
    )

    assert "0:thinking" in described and "chars=15" in described
    assert "1:web_search_tool_result (id=search-1)" in described
    assert "2:tool_use (name=calculator)" in described
    assert summarize_response_blocks(block(content=[])) == "<no content blocks>"


def test_empty_response_is_an_error_while_healthy_exchange_is_an_action():
    logger = CaptureLogger()
    empty = response(stop_reason="max_tokens", text=None)

    log_api_exchange(
        logger,
        "DirectorAgent",
        {"model": "claude-opus-4-6", "max_tokens": 4096, "messages": [{}]},
        empty,
        "",
    )

    assert not logger.actions
    assert len(logger.errors) == 1
    assert "[EMPTY RESPONSE]" in logger.errors[0][0]
    assert "stop_reason=max_tokens" in logger.errors[0][0]
    assert "extracted_text_chars=0" in logger.errors[0][0]

    healthy_logger = CaptureLogger()
    log_api_exchange(healthy_logger, "DirectorAgent", {"model": "m"}, response(), "answer")
    assert healthy_logger.actions[0][0] == "api_exchange"
    assert not healthy_logger.errors


@pytest.mark.parametrize(
    "details, category",
    [({"category": "bio"}, "bio"), (block(category="policy"), "policy")],
)
def test_refusal_diagnostics_preserve_category_and_retry_advice(details, category):
    refused = response(stop_reason="refusal", text=None, stop_details=details)
    notice = refusal_notice(refused)
    logger = CaptureLogger()

    log_api_exchange(logger, "DirectorAgent", {"model": "m"}, refused, "")

    assert notice is not None
    assert f"category: {category}" in notice
    assert REFUSAL_ADVICE in notice
    assert "[REFUSAL]" in logger.errors[0][0]
    assert not logger.actions
    assert first_refusal(block(last_refusal=None), block(last_refusal=notice)) == notice


def test_partial_text_refusal_remains_an_error():
    refused = response(stop_reason="refusal", text="partial answer")
    logger = CaptureLogger()

    log_api_exchange(logger, "DirectorAgent", {"model": "m"}, refused, "partial answer")

    assert not logger.actions
    assert "[REFUSAL]" in logger.errors[0][0]
    assert "extracted_text_chars=14" in logger.errors[0][0]


def test_api_exception_diagnostics_include_sdk_fields_and_bound_bodies():
    class APIProblem(RuntimeError):
        status_code = 400
        request_id = "request-42"
        body = "b" * 1100
        response = block(status_code=502, text="r" * 1200)

    problem = APIProblem("bad request")
    formatted = format_api_exception(
        problem,
        headline="LLM API call failed.",
        context={"Source": "DirectorAgent"},
    )

    assert "Exception: APIProblem: bad request" in formatted
    assert "HTTP status: 400" in formatted
    assert "Request ID: request-42" in formatted
    assert "Response status: 502" in formatted
    assert "...(+100 chars)" in formatted
    assert "...(+200 chars)" in formatted

    single_search = _format_anthropic_exception(
        problem,
        model_name="claude-opus-4-6",
        query="find evidence",
    )
    assert "Single web search failed." in single_search
    assert "Model: claude-opus-4-6" in single_search
    assert "Query preview: find evidence" in single_search


def test_diagnostic_helpers_never_mask_the_original_failure():
    class BrokenLogger:
        def log_action(self, *_args, **_kwargs):
            raise RuntimeError("logger unavailable")

        def log_error(self, *_args, **_kwargs):
            raise RuntimeError("logger unavailable")

    class BrokenResponse:
        @property
        def content(self):
            raise RuntimeError("content unavailable")

    logger = BrokenLogger()
    log_api_exchange(logger, "Agent", None, BrokenResponse(), None)
    log_api_exception(logger, ValueError("original"), source="Agent")
    assert summarize_response_blocks(BrokenResponse()) == "<content unavailable>"


def _director_for(response_or_error):
    agent = DirectorAgent.__new__(DirectorAgent)
    agent.logger = CaptureLogger()
    agent.llm_config = block(api_provider="anthropic", model_name="claude-opus-4-6")
    agent.messages = []
    agent.mcp_servers = None
    agent.compaction_enabled = False
    agent.compaction_trigger_tokens = 150000
    agent.compaction_instructions = None

    def create(**_kwargs):
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return response_or_error

    agent.client = block(messages=block(create=create), beta=block(messages=block(create=create)))
    return agent


def _hypothesis_agent_for(response_or_error):
    agent = HypothesisGenAgent.__new__(HypothesisGenAgent)
    agent.logger = CaptureLogger()
    agent.llm_config = block(api_provider="anthropic", model_name="claude-opus-4-6")
    agent.messages = []

    def create(**_kwargs):
        if isinstance(response_or_error, BaseException):
            raise response_or_error
        return response_or_error

    agent.client = block(messages=block(create=create))
    return agent


@pytest.mark.parametrize("factory", [_director_for, _hypothesis_agent_for])
def test_anthropic_agents_log_exchange_and_expose_last_refusal(factory):
    refused = response(stop_reason="refusal", text=None, stop_details={"category": "bio"})
    agent = factory(refused)

    result = agent._call_api_with_retry({"model": "claude-opus-4-6"}, max_retries=0)

    assert result == ""
    assert "[REFUSAL]" in agent.last_refusal
    assert "category: bio" in agent.last_refusal
    assert "source=" in agent.logger.errors[0][0]


@pytest.mark.parametrize("factory", [_director_for, _hypothesis_agent_for])
def test_anthropic_agents_log_unexpected_api_exceptions_before_reraising(factory):
    agent = factory(ValueError("SDK exploded"))

    with pytest.raises(ValueError, match="SDK exploded"):
        agent._call_api_with_retry({"model": "claude-opus-4-6"}, max_retries=0)

    diagnostic = agent.logger.errors[-1][0]
    assert "LLM API call failed." in diagnostic
    assert "Exception: ValueError: SDK exploded" in diagnostic
    assert "Retryable: no" in diagnostic
    assert "Traceback:" in diagnostic


def test_text_extraction_failure_logs_the_response_diagnostic():
    class BrokenContent:
        model = "claude-opus-4-6"
        stop_reason = "end_turn"
        stop_details = None
        usage = None

        @property
        def content(self):
            raise RuntimeError("cannot decode blocks")

    agent = DirectorAgent.__new__(DirectorAgent)
    agent.logger = CaptureLogger()

    assert agent._extract_text_from_anthropic_response(BrokenContent()) == ""
    assert "Failed to extract text from response: RuntimeError: cannot decode blocks" in agent.logger.errors[0][0]
    assert "stop_reason=end_turn" in agent.logger.errors[0][0]
    assert "blocks=[<content unavailable>]" in agent.logger.errors[0][0]
