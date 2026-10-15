import os
from pathlib import Path

import pytest

# Clear local .env overrides before importing DiscoveryAgent so the tests see
# the default environment, then restore the test shell.
from DiscoveryAgent import config  # noqa: E402
from DiscoveryAgent.tool_agents import execution_agent_cc  # noqa: E402


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_normal_setup_uses_no_override_or_global_lookup(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_CLI_PATH", raising=False)
    monkeypatch.setenv("NODE_PATH", "/developer/node/modules")
    monkeypatch.setattr(
        config.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "normal Claude configuration must not invoke npm or global discovery"
        ),
    )

    assert config.setup_claude_code_environment() is None
    assert os.environ["NODE_PATH"] == "/developer/node/modules"


def test_explicit_absolute_executable_override_is_validated(monkeypatch, tmp_path):
    executable = _make_executable(tmp_path / "developer-claude")
    monkeypatch.setenv("CLAUDE_CODE_CLI_PATH", str(executable))

    assert config.setup_claude_code_environment() == str(executable.resolve())


@pytest.mark.parametrize("kind", ["relative", "missing", "directory", "not-executable"])
def test_invalid_explicit_override_fails_clearly(monkeypatch, tmp_path, kind):
    if kind == "relative":
        configured = "relative/claude"
    elif kind == "missing":
        configured = str(tmp_path / "missing-claude")
    elif kind == "directory":
        configured_path = tmp_path / "claude-directory"
        configured_path.mkdir()
        configured = str(configured_path)
    else:
        configured_path = tmp_path / "claude-not-executable"
        configured_path.write_text("not executable", encoding="utf-8")
        configured_path.chmod(0o644)
        configured = str(configured_path)
    monkeypatch.setenv("CLAUDE_CODE_CLI_PATH", configured)

    with pytest.raises(ValueError, match="CLAUDE_CODE_CLI_PATH"):
        config.setup_claude_code_environment()


def test_execution_options_omit_cli_path_without_override(monkeypatch):
    captured = {}

    def capture_options(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(execution_agent_cc, "ClaudeAgentOptions", capture_options)
    monkeypatch.setattr(execution_agent_cc, "claude_cli_path", None)

    result = execution_agent_cc._create_claude_agent_options({"cwd": "/tmp/work"})

    assert result == {"cwd": "/tmp/work"}
    assert "cli_path" not in captured


def test_execution_options_honor_valid_explicit_override(monkeypatch, tmp_path):
    executable = _make_executable(tmp_path / "developer-claude")
    monkeypatch.setenv("CLAUDE_CODE_CLI_PATH", str(executable))
    validated_override = config.get_explicit_claude_cli_path()
    captured = {}

    def capture_options(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(execution_agent_cc, "ClaudeAgentOptions", capture_options)
    monkeypatch.setattr(execution_agent_cc, "claude_cli_path", validated_override)

    execution_agent_cc._create_claude_agent_options({"cwd": "/tmp/work"})

    assert captured["cli_path"] == str(executable.resolve())
