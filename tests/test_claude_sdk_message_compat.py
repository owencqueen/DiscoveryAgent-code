"""Regression tests for Claude CLI/Agent SDK stream compatibility."""

import pytest

from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message


def test_tool_progress_is_ignored_without_stopping_the_message_stream() -> None:
    raw_messages = [
        {
            "type": "assistant",
            "message": {
                "model": "claude-test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                    }
                ],
            },
        },
        {
            "type": "tool_progress",
            "tool_use_id": "tool-1",
            "tool_name": "Bash",
            "elapsed_time_seconds": 5,
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "completed",
                        "is_error": False,
                    }
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "duration_ms": 10,
            "duration_api_ms": 8,
            "is_error": False,
            "num_turns": 1,
            "session_id": "session-1",
            "result": "done",
        },
    ]

    parsed = [message for raw in raw_messages if (message := parse_message(raw))]

    assert [type(message) for message in parsed] == [
        AssistantMessage,
        UserMessage,
        ResultMessage,
    ]
    assert parsed[-1].is_error is False


def test_task_lifecycle_and_generic_status_messages_are_non_terminal() -> None:
    task_started = parse_message(
        {
            "type": "system",
            "subtype": "task_started",
            "task_id": "task-1",
            "description": "Research",
            "uuid": "uuid-1",
            "session_id": "session-1",
        }
    )
    task_progress = parse_message(
        {
            "type": "system",
            "subtype": "task_progress",
            "task_id": "task-1",
            "description": "Researching",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "uuid": "uuid-2",
            "session_id": "session-1",
        }
    )
    status = parse_message(
        {"type": "system", "subtype": "status", "status": "compacting"}
    )

    assert task_started is not None
    assert task_progress is not None
    assert status is not None


def test_malformed_result_is_not_silently_ignored() -> None:
    with pytest.raises(MessageParseError, match="result message"):
        parse_message({"type": "result", "subtype": "error", "is_error": True})
