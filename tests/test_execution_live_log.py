import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from DiscoveryAgent.tool_agents.execution_agent_cc import ExecutionAgentCC
from DiscoveryAgent.tool_agents.execution_agent_cc import (
    ExecutionAgentEmptyResponseError,
    ExecutionAgentMalformedReportError,
    ExecutionAgentMissingFinalResponseError,
    ExecutionAgentRefusalError,
    ExecutionAgentSDKResponseError,
)
from DiscoveryAgent.tool_agents.execution_sdk_tracker import (
    EXECUTION_LOG_TRUNCATION_MARKER,
    ExecutionLogWriter,
    ExecutionSDKTracker,
    sanitize_execution_output,
)
from DiscoveryAgent.utils.query_artifacts import execution_log_filename, query_suffix


class _Logger:
    def log_error(self, _message):
        pass

    def log_action(self, *_args, **_kwargs):
        pass


def _query_dir(tmp_path: Path, number: int = 1) -> Path:
    path = tmp_path / f"query_{number:02d}"
    path.mkdir()
    return path


def test_execution_log_uses_minimum_two_digit_query_identity():
    assert [(query_suffix(n), execution_log_filename(n)) for n in (1, 9, 10, 100)] == [
        ("01", "execution_log_01.txt"),
        ("09", "execution_log_09.txt"),
        ("10", "execution_log_10.txt"),
        ("100", "execution_log_100.txt"),
    ]


def test_tracker_creates_private_log_and_flushes_tool_output_in_order(tmp_path):
    query_dir = _query_dir(tmp_path)
    tracker = ExecutionSDKTracker(logger=_Logger())
    tracker.set_query_log_dir(str(query_dir), query_num=1)
    log_path = query_dir / "execution_log_01.txt"
    assert log_path.is_file()
    assert os.stat(log_path).st_mode & 0o777 == 0o600

    asyncio.run(tracker.pre_tool_use_hook(
        {"tool_name": "Bash", "tool_input": {"command": "printf hello"}}, "call-1", None,
    ))
    asyncio.run(tracker.post_tool_use_hook(
        {"tool_response": {"stdout": "hello", "stderr": "warning"}}, "call-1", None,
    ))
    tracker.log_observable("Execution complete")

    text = log_path.read_text(encoding="utf-8")
    positions = [text.index(value) for value in (
        "Tool start", "printf hello", "stdout: hello", "stderr: warning", "Tool end", "Execution complete",
    )]
    assert positions == sorted(positions)
    assert "Director" not in text


def test_duplicate_setup_is_idempotent_and_reopen_adds_one_resume_boundary(tmp_path):
    query_dir = _query_dir(tmp_path)
    tracker = ExecutionSDKTracker(logger=_Logger())
    tracker.set_query_log_dir(str(query_dir), query_num=1)
    tracker.log_observable("first")
    tracker.set_query_log_dir(str(query_dir), query_num=1)
    tracker.log_observable("second")
    tracker.close()
    first_session = (query_dir / "execution_log_01.txt").read_text(encoding="utf-8")
    assert first_session.count("Execution started") == 1
    assert "Execution resumed" not in first_session

    resumed = ExecutionSDKTracker(logger=_Logger())
    resumed.set_query_log_dir(str(query_dir), query_num=1)
    resumed.log_observable("after restart")
    resumed.close()
    text = (query_dir / "execution_log_01.txt").read_text(encoding="utf-8")
    assert text.count("Execution resumed") == 1
    assert text.index("first") < text.index("second") < text.index("Execution resumed") < text.index("after restart")


def test_sanitizer_strips_terminal_controls_and_redacts_credentials(monkeypatch):
    secret = "sk-ant-test-secret-value"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    raw = (
        "\x1b[31mred\x1b[0m\x1b]0;owned\x07\x00\rnext\x08\n"
        f"ANTHROPIC_API_KEY={secret}\nAuthorization: Bearer abc.def\n"
        "password='p4ssword' token=plain-token --api-key command-secret Unicode ✓ <script>alert(1)</script>"
    )
    cleaned = sanitize_execution_output(raw, known_secrets=(secret,))
    assert "\x1b" not in cleaned and "\x00" not in cleaned and "\x08" not in cleaned
    assert secret not in cleaned and "abc.def" not in cleaned and "p4ssword" not in cleaned and "plain-token" not in cleaned
    assert "command-secret" not in cleaned
    assert cleaned.count("[REDACTED]") >= 5
    assert "Unicode ✓ <script>alert(1)</script>" in cleaned


def test_known_environment_secret_is_redacted_before_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "known-secret-value")
    writer = ExecutionLogWriter(_query_dir(tmp_path), 1)
    writer.append("error argv=--key known-secret-value")
    text = writer.path.read_text(encoding="utf-8")
    assert "known-secret-value" not in text
    assert "[REDACTED]" in text


def test_known_secret_split_across_incremental_appends_never_reaches_disk(tmp_path):
    secret = "split-secret-value"
    writer = ExecutionLogWriter(_query_dir(tmp_path), 1, environment={"ACS_API_TOKEN": secret})
    writer.append("stderr: split-secret-")
    assert "split-secret-" not in writer.path.read_text(encoding="utf-8")
    writer.append("value\n")
    text = writer.path.read_text(encoding="utf-8")
    assert secret not in text
    assert "stderr: [REDACTED]" in text


def test_size_limit_has_explicit_marker(tmp_path, monkeypatch):
    import DiscoveryAgent.tool_agents.execution_sdk_tracker as tracker_module

    monkeypatch.setattr(tracker_module, "EXECUTION_LOG_MAX_BYTES", 180)
    writer = ExecutionLogWriter(_query_dir(tmp_path), 1)
    writer.append("x" * 500)
    text = writer.path.read_text(encoding="utf-8")
    assert EXECUTION_LOG_TRUNCATION_MARKER.strip() in text
    assert writer.path.stat().st_size <= 180


def test_agent_failure_keeps_public_log(tmp_path):
    query_dir = _query_dir(tmp_path)
    agent = object.__new__(ExecutionAgentCC)
    agent.execute_context_persistence = True
    agent.execution_tracker = ExecutionSDKTracker(logger=_Logger())
    agent.config = SimpleNamespace(dataset_description="")
    agent.logger = _Logger()

    async def fail(*_args, **_kwargs):
        raise RuntimeError("simulated failure")

    agent._execute_with_report_generation = fail
    with pytest.raises(RuntimeError, match="simulated failure"):
        agent.execute("test query", query_num=1, checkpoint_query_dir=str(query_dir))
    text = (query_dir / "execution_log_01.txt").read_text(encoding="utf-8")
    assert "executing autonomous analysis" in text
    assert "RuntimeError: simulated failure" in text


def test_agent_stop_keeps_public_log(tmp_path):
    query_dir = _query_dir(tmp_path)
    agent = object.__new__(ExecutionAgentCC)
    agent.execute_context_persistence = True
    agent.execution_tracker = ExecutionSDKTracker(logger=_Logger())
    agent.config = SimpleNamespace(dataset_description="")
    agent.logger = _Logger()

    async def stop(*_args, **_kwargs):
        raise asyncio.CancelledError("simulated stop")

    agent._execute_with_report_generation = stop
    with pytest.raises(asyncio.CancelledError, match="simulated stop"):
        agent.execute("test query", query_num=1, checkpoint_query_dir=str(query_dir))
    text = (query_dir / "execution_log_01.txt").read_text(encoding="utf-8")
    assert "executing autonomous analysis" in text
    assert "CancelledError: simulated stop" in text


def test_refusal_without_execution_does_not_create_log(tmp_path):
    query_dir = _query_dir(tmp_path)
    assert not (query_dir / "execution_log_01.txt").exists()


def test_agent_success_keeps_completion_and_file_safety_status(tmp_path):
    query_dir = _query_dir(tmp_path)
    agent = object.__new__(ExecutionAgentCC)
    agent.execute_context_persistence = True
    agent.execution_tracker = ExecutionSDKTracker(logger=_Logger())
    agent.config = SimpleNamespace(dataset_description="")
    agent.logger = _Logger()

    async def succeed(*_args, **_kwargs):
        agent._direct_print("stdout: incremental result")
        return {"concise_report": "report"}

    agent._execute_with_report_generation = succeed
    agent._check_file_safety = lambda: {"existing_files_protected": True}
    assert agent.execute("test query", query_num=1, checkpoint_query_dir=str(query_dir)) == "report"
    text = (query_dir / "execution_log_01.txt").read_text(encoding="utf-8")
    assert "stdout: incremental result" in text
    assert "analysis completed for query 1" in text
    assert "All existing files protected" in text


def _message(type_name, **attributes):
    return type(
        type_name, (), {key: value for key, value in attributes.items()}
    )()


def test_report_failure_categories_are_content_free_and_distinguish_refusal():
    refusal = ExecutionAgentCC._report_failure([
        _message("ResultMessage", stop_reason="refusal", subtype="error"),
    ])
    empty = ExecutionAgentCC._report_failure([])
    missing = ExecutionAgentCC._report_failure([_message("AssistantMessage")])
    malformed = ExecutionAgentCC._report_failure([
        _message("ResultMessage", stop_reason="end_turn", subtype="success"),
    ])

    assert isinstance(refusal, ExecutionAgentRefusalError)
    assert isinstance(empty, ExecutionAgentEmptyResponseError)
    assert isinstance(missing, ExecutionAgentMissingFinalResponseError)
    assert isinstance(malformed, ExecutionAgentMalformedReportError)
    assert str(refusal) == "execution_agent_model_refusal"


class _ReportClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def query(self, prompt):
        self.calls.append(prompt)

    async def receive_response(self):
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, Exception):
            raise response
        for message in response:
            yield message


def _report_agent(tmp_path):
    agent = object.__new__(ExecutionAgentCC)
    agent.working_directory = tmp_path
    agent.execution_tracker = None
    agent.logger = _Logger()
    return agent


def test_report_refusal_is_not_retried(tmp_path):
    client = _ReportClient([[
        _message("ResultMessage", stop_reason="refusal", subtype="error"),
    ]])
    agent = _report_agent(tmp_path)

    with pytest.raises(ExecutionAgentRefusalError):
        asyncio.run(agent._generate_concise_summary(client, "query", [], None, 5))

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("responses", "error_type"),
    [
        ([[], []], ExecutionAgentEmptyResponseError),
        (
            [[_message("AssistantMessage")], [_message("AssistantMessage")]],
            ExecutionAgentMissingFinalResponseError,
        ),
        (
            [
                [_message("ResultMessage", stop_reason="end_turn", subtype="success")],
                [_message("ResultMessage", stop_reason="end_turn", subtype="success")],
            ],
            ExecutionAgentMalformedReportError,
        ),
        ([RuntimeError("first"), RuntimeError("second")], ExecutionAgentSDKResponseError),
    ],
)
def test_unusable_report_response_retries_once_then_preserves_category(
    tmp_path, responses, error_type,
):
    client = _ReportClient(responses)
    agent = _report_agent(tmp_path)

    with pytest.raises(error_type):
        asyncio.run(agent._generate_concise_summary(client, "query", [], None, 5))

    assert len(client.calls) == 2
