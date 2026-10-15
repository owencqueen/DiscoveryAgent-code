"""Tests for sandbox filesystem containment.

The claude_agent_sdk only honors ``permissionDecision`` when it is nested
under ``hookSpecificOutput`` (see ``SyncHookJSONOutput`` /
``PreToolUseHookSpecificOutput`` in ``claude_agent_sdk/types.py``); a
top-level key is silently dropped. A hook that returns a deny dict is
therefore not enough on its own, so these tests check the disk directly
(``os.path.exists``) or exercise the installed claude_agent_sdk end-to-end.

Live end-to-end tests (`TestLiveEndToEnd`) spawn the real, installed `claude`
CLI and make model calls. Set `RUN_LIVE_SANDBOX_TESTS=1` to opt in when
an authenticated CLI is available.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from claude_agent_sdk import ClaudeAgentOptions, HookMatcher, query

from DiscoveryAgent.tool_agents.security.sandbox_enforcement import (
    _check_bash_write,
    _is_within,
    create_sandbox_enforcement_hook,
    create_sandbox_posttool_guard,
)
from DiscoveryAgent.tool_agents.execution_sdk_tracker import ExecutionSDKTracker

CLI_PATH = shutil.which("claude")
LIVE_MODEL = os.environ.get("SANDBOX_TEST_MODEL", "claude-haiku-4-5-20251001")

requires_live_cli = pytest.mark.skipif(
    CLI_PATH is None or os.environ.get("RUN_LIVE_SANDBOX_TESTS") != "1",
    reason="live end-to-end tests require RUN_LIVE_SANDBOX_TESTS=1 and an authenticated `claude` CLI",
)


# ---------------------------------------------------------------------------
# Unit level: _is_within edge cases, including the symlink-escape fix
# ---------------------------------------------------------------------------


class TestIsWithin:
    def test_in_sandbox_absolute_path_allowed(self, tmp_path):
        root = str(tmp_path)
        assert _is_within(os.path.join(root, "analysis.py"), [root])

    def test_out_of_sandbox_absolute_path_denied(self, tmp_path):
        root = str(tmp_path / "code")
        os.makedirs(root)
        outside = str(tmp_path / "sibling" / "leak.py")
        assert not _is_within(outside, [root])

    def test_relative_escape_denied(self, tmp_path):
        root = str(tmp_path / "code")
        os.makedirs(root)
        assert not _is_within("../outside.py", [root], cwd=root)

    def test_relative_within_sandbox_allowed(self, tmp_path):
        root = str(tmp_path / "code")
        os.makedirs(root)
        assert _is_within("reports/tables/x.csv", [root], cwd=root)

    def test_symlinked_read_only_data_dir_permitted_by_default(self, tmp_path):
        """Documented, intentional behavior for reads: a symlink that lives
        inside the sandbox and points to an external data directory is
        allowed in the default (non-strict) mode used for Read/Glob/Grep."""
        root = tmp_path / "code"
        root.mkdir()
        outside_target = tmp_path / "outside_data"
        outside_target.mkdir()
        link = root / "data"
        link.symlink_to(outside_target)
        assert _is_within(str(link), [str(root)])

    def test_strict_mode_rejects_write_through_escape_symlink(self, tmp_path):
        """An agent-created symlink pointing outside the sandbox (e.g.
        `sandbox_root/escape -> /`) puts `sandbox_root/escape/anything`
        textually under sandbox_root even though it resolves elsewhere.
        write_edit_hook, the Bash write checks, and the PostToolUse guard
        pass strict=True, which requires both the literal AND resolved path
        to be in-sandbox."""
        root = tmp_path / "code"
        root.mkdir()
        outside_target = tmp_path / "outside"
        outside_target.mkdir()
        escape_link = root / "escape"
        escape_link.symlink_to(outside_target)
        escapee = str(escape_link / "evil.py")

        # Non-strict (read semantics): permissive, as documented.
        assert _is_within(escapee, [str(root)])
        # Strict (write semantics): must reject.
        assert not _is_within(escapee, [str(root)], strict=True)

    def test_strict_mode_still_allows_ordinary_in_sandbox_writes(self, tmp_path):
        root = str(tmp_path / "code")
        os.makedirs(root)
        assert _is_within(os.path.join(root, "reports", "tables", "out.csv"), [root], strict=True)


class TestBashWriteParsing:
    def test_relative_redirect_defaults_to_sandbox_cwd(self, tmp_path):
        root = str(tmp_path / "code")
        os.makedirs(root)

        assert _check_bash_write("python analysis.py > outputs.txt", root) is None
        assert "outside sandbox" in _check_bash_write(
            "python analysis.py > ../outside.txt",
            root,
        )

    @pytest.mark.parametrize("command", ["python analysis.py 2>&1", "cat file >(wc -l)"])
    def test_fd_duplication_and_process_substitution_are_not_file_targets(
        self,
        tmp_path,
        command,
    ):
        root = str(tmp_path / "code")
        os.makedirs(root)

        assert _check_bash_write(command, root) is None


# ---------------------------------------------------------------------------
# PostToolUse guard: exercised directly against the real filesystem
# ---------------------------------------------------------------------------


class TestPostToolUseGuard:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_deletes_newly_created_file_and_aborts(self, tmp_path):
        """Simulates a PreToolUse deny that was
        recorded (escape_attempts has an entry showing the path did not
        pre-exist), but the Write executed anyway and the file landed on
        disk outside the sandbox. The PostToolUse guard must delete it and
        stop the run."""
        sandbox_root = tmp_path / "code"
        sandbox_root.mkdir()
        outside_dir = tmp_path / "Desktop" / "other_project"
        outside_dir.mkdir(parents=True)
        leaked_file = outside_dir / "analysis_targeted.py"
        leaked_file.write_text("# leaked content\n")
        assert leaked_file.exists()

        escape_attempts = {}
        tool_use_id = "toolu_test_1"
        escape_attempts[tool_use_id] = {"path": str(leaked_file), "pre_existed": False}

        guard_matchers = create_sandbox_posttool_guard(
            sandbox_root=str(sandbox_root), escape_attempts=escape_attempts
        )
        write_guard = guard_matchers[0].hooks[0]

        result = self._run(
            write_guard(
                {"tool_input": {"file_path": str(leaked_file)}},
                tool_use_id,
                {"signal": None},
            )
        )

        assert not leaked_file.exists(), "PostToolUse guard must actually delete the escaped file"
        assert result["continue_"] is False
        assert "CRITICAL" in result["reason"]
        assert tool_use_id not in escape_attempts

    def test_does_not_delete_a_pre_existing_file(self, tmp_path):
        """If the out-of-sandbox path already existed before the tool call,
        the guard must not delete it — it could be unrelated user content —
        it must still abort the run."""
        sandbox_root = tmp_path / "code"
        sandbox_root.mkdir()
        outside_dir = tmp_path / "Desktop"
        outside_dir.mkdir()
        pre_existing = outside_dir / "important_unrelated_file.txt"
        pre_existing.write_text("do not delete me\n")

        escape_attempts = {"toolu_test_2": {"path": str(pre_existing), "pre_existed": True}}
        guard_matchers = create_sandbox_posttool_guard(
            sandbox_root=str(sandbox_root), escape_attempts=escape_attempts
        )
        write_guard = guard_matchers[0].hooks[0]

        result = self._run(
            write_guard(
                {"tool_input": {"file_path": str(pre_existing)}},
                "toolu_test_2",
                {"signal": None},
            )
        )

        assert pre_existing.exists()
        assert pre_existing.read_text() == "do not delete me\n"
        assert result["continue_"] is False

    def test_in_sandbox_write_is_a_no_op_for_the_guard(self, tmp_path):
        sandbox_root = tmp_path / "code"
        sandbox_root.mkdir()
        in_sandbox_file = sandbox_root / "analysis.py"
        in_sandbox_file.write_text("print('ok')\n")

        escape_attempts = {}
        guard_matchers = create_sandbox_posttool_guard(
            sandbox_root=str(sandbox_root), escape_attempts=escape_attempts
        )
        write_guard = guard_matchers[0].hooks[0]

        result = self._run(
            write_guard(
                {"tool_input": {"file_path": str(in_sandbox_file)}},
                "toolu_test_3",
                {"signal": None},
            )
        )

        assert in_sandbox_file.exists()
        assert result == {"continue_": True}


# ---------------------------------------------------------------------------
# Live end-to-end: real claude_agent_sdk + real CLI subprocess + real model
# ---------------------------------------------------------------------------


def _build_options(sandbox_root: Path, extra_pre_hooks=None, extra_post_hooks=None, escape_attempts=None):
    escape_attempts = escape_attempts if escape_attempts is not None else {}
    sandbox_hooks = create_sandbox_enforcement_hook(
        sandbox_root=str(sandbox_root), escape_attempts=escape_attempts
    )
    posttool_hooks = create_sandbox_posttool_guard(
        sandbox_root=str(sandbox_root), escape_attempts=escape_attempts
    )
    pre_hooks = list(extra_pre_hooks or []) + sandbox_hooks
    post_hooks = list(extra_post_hooks or []) + posttool_hooks

    return ClaudeAgentOptions(
        cwd=str(sandbox_root),
        cli_path=CLI_PATH,
        model=LIVE_MODEL,
        max_turns=3,
        allowed_tools=["Read", "Write", "Edit", "Bash"],
        permission_mode="acceptEdits",
        system_prompt="You are a test harness. Follow the user's tool-call instructions exactly and do nothing else. Do not explain your actions.",
        hooks={"PreToolUse": pre_hooks, "PostToolUse": post_hooks},
    )


async def _run_prompt(prompt: str, options: ClaudeAgentOptions) -> list:
    """Run a live prompt, tolerating a non-clean end to the conversation.

    These tests deliberately ask the model to do something that gets denied.
    A denied model sometimes retries, asks for clarification, or otherwise
    burns its turn budget instead of stopping cleanly, which the SDK surfaces
    as a "Reached maximum number of turns" exception rather than a graceful
    ResultMessage. That is irrelevant to what these tests actually check
    (the resulting state on disk), so it is swallowed here rather than
    failing the test on an unrelated conversational-flow detail.
    """
    messages = []
    try:
        async for message in query(prompt=prompt, options=options):
            messages.append(message)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see docstring
        messages.append(exc)
    return messages


@requires_live_cli
class TestLiveEndToEnd:
    """A Write-matching sandbox hook plus a catch-all matcher=None execution-tracking hook, both
    registered together, exactly as execution_agent_cc.py does it."""

    def test_out_of_sandbox_write_is_blocked_end_to_end_with_tracker_present(self, tmp_path):
        sandbox_root = tmp_path / "project" / "code"
        sandbox_root.mkdir(parents=True)
        outside_target = tmp_path / "Desktop" / "other_project" / "analysis_targeted.py"

        tracker = ExecutionSDKTracker(logger=None)
        tracker.set_query_log_dir(str(tmp_path / "checkpoint" / "query_001"), query_num=1)
        # The exact catch-all HookMatcher shape from execution_agent_cc.py:
        # HookMatcher(matcher=None, hooks=[self.execution_tracker.pre_tool_use_hook])
        catch_all_pre = HookMatcher(matcher=None, hooks=[tracker.pre_tool_use_hook])
        catch_all_post = HookMatcher(matcher=None, hooks=[tracker.post_tool_use_hook])

        escape_attempts = {}
        options = _build_options(
            sandbox_root,
            extra_pre_hooks=[catch_all_pre],
            extra_post_hooks=[catch_all_post],
            escape_attempts=escape_attempts,
        )

        prompt = (
            f"Call the Write tool with file_path=\"{outside_target}\" and "
            "content=\"leak-test\". That is the entire task."
        )
        asyncio.run(_run_prompt(prompt, options))
        tracker.close()

        assert not outside_target.exists(), (
            "Write outside the sandbox must not create a file on disk, even "
            "with the catch-all execution-tracking HookMatcher registered "
            "alongside the sandbox Write HookMatcher."
        )

        # Execution tracking must still have recorded the attempt: the
        # catch-all matcher=None tracker hook and the sandbox deny must
        # compose safely, not trade off against each other.
        tool_log = (tmp_path / "checkpoint" / "query_001" / "execution_agent" / "tool_calls.jsonl").read_text()
        assert '"tool_name": "Write"' in tool_log
        assert '"event": "tool_call_start"' in tool_log

    def test_in_sandbox_write_still_succeeds(self, tmp_path):
        sandbox_root = tmp_path / "project" / "code"
        sandbox_root.mkdir(parents=True)
        target = sandbox_root / "analysis.py"

        options = _build_options(sandbox_root)
        prompt = (
            f"Call the Write tool with file_path=\"{target}\" and "
            "content=\"print('hello')\". That is the entire task."
        )
        asyncio.run(_run_prompt(prompt, options))

        assert target.exists()
        assert "hello" in target.read_text()

    def test_edit_of_pre_existing_outside_sandbox_file_is_rejected_and_unchanged(self, tmp_path):
        sandbox_root = tmp_path / "project" / "code"
        sandbox_root.mkdir(parents=True)
        outside_dir = tmp_path / "Desktop"
        outside_dir.mkdir()
        target = outside_dir / "pre_existing.txt"
        original_content = "original content — must not change\n"
        target.write_text(original_content)

        options = _build_options(sandbox_root)
        prompt = (
            f"Call the Edit tool on file_path=\"{target}\" to replace "
            '"original content" with "TAMPERED". That is the entire task.'
        )
        asyncio.run(_run_prompt(prompt, options))

        assert target.read_text() == original_content

    def test_relative_escape_write_is_blocked_end_to_end(self, tmp_path):
        sandbox_root = tmp_path / "project" / "code"
        sandbox_root.mkdir(parents=True)
        sibling_escape = tmp_path / "project" / "outside_via_relative.py"

        options = _build_options(sandbox_root)
        prompt = (
            "Call the Write tool with file_path=\"../outside_via_relative.py\" "
            "and content=\"leak\". That is the entire task."
        )
        asyncio.run(_run_prompt(prompt, options))

        assert not sibling_escape.exists()
