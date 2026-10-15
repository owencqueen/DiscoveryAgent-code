"""
Sandbox enforcement hook for Claude Code SDK.

Restricts the execution agent to operate within a designated sandbox directory.
Write/Edit operations are confined to the sandbox root. Read/Glob/Grep operations
are allowed within the sandbox AND a configurable set of additional read paths.
Bash commands are checked on a best-effort basis for common shell patterns.

Symlinks inside the sandbox that resolve to external targets are permitted:
the check passes if EITHER the literal absolute path OR the resolved real path
falls within an allowed root. This lets symlinked data directories work
transparently.
"""

import os
import re
import shlex
import shutil
from typing import List, Optional, Any, Dict

from claude_agent_sdk import HookMatcher


# Commands that destructively modify targets (all non-flag args are targets)
_DESTRUCTIVE_ALL_ARGS = {"rm", "rmdir", "mkdir", "touch", "chmod", "chown", "chgrp"}

# Commands where only the LAST non-flag arg is the write destination
_COPY_LIKE_COMMANDS = {"cp", "mv", "ln", "install"}

# Patterns for in-place edit flags
_INPLACE_EDIT_PATTERN = re.compile(r"\b(sed|perl|awk)\b.*\s-i\b")

# Patterns for output redirection.
#
# This is INTENTIONALLY loose: it catches ``> file``, ``>> file``, and forms
# like ``2> file`` / ``&> file`` because the ``>`` is what we anchor on and
# every literal-file redirect contains it. The downside is that it also
# matches things that aren't file writes:
#   * FD duplication: ``2>&1``, ``>&2`` -> target starts with ``&``
#   * Process substitution: ``>(cmd)``  -> target starts with ``(``
# Those are filtered out below at the use site (see ``_check_bash_write``).
_REDIRECT_PATTERN = re.compile(r"(>{1,2})\s*(\S+)")

# Pattern for tee command writing to a file
_TEE_PATTERN = re.compile(r"\btee\b(?:\s+-a)?\s+(\S+)")

# Read-oriented commands whose arguments are source paths
_READ_COMMANDS = {"cat", "head", "tail", "less", "more", "wc", "file", "stat", "md5sum", "sha256sum"}


def _make_absolute(path: str, cwd: Optional[str] = None) -> str:
    """Return the absolute (but NOT realpath-resolved) form of *path*."""
    if not os.path.isabs(path):
        base = cwd or os.getcwd()
        path = os.path.join(base, path)
    return os.path.normpath(path)


def _is_within(
    path: str,
    allowed_roots: List[str],
    cwd: Optional[str] = None,
    *,
    strict: bool = False,
) -> bool:
    """Check whether *path* falls under any of *allowed_roots*.

    By default (``strict=False``), the check passes if EITHER:
      1. The literal (non-resolved) absolute path is inside an allowed root, OR
      2. The ``os.path.realpath()``-resolved path is inside an allowed root.

    This dual check lets symlinks that live inside the sandbox (literal path
    passes) transparently point to external targets, while also allowing direct
    references to those external targets when they appear in *allowed_roots*.
    This permissive mode is used for reads, where a symlinked external data
    directory placed inside the sandbox is a supported, intentional pattern.

    ``strict=True`` requires BOTH the literal path AND the resolved path to
    fall within an allowed root, rejecting a path that is only "inside" the
    sandbox because it traverses a symlink whose target resolves elsewhere
    (e.g. an agent-created ``sandbox_root/escape -> /`` symlink, then a write
    to ``sandbox_root/escape/Users/x/evil.py``, which is literally under the
    sandbox but resolves outside it). Used for write-path checks, where
    *allowed_roots* is always just the sandbox root and there is never a
    legitimate reason for a write target to resolve outside it.
    """
    abs_path = _make_absolute(path, cwd)
    real_path = os.path.realpath(abs_path)

    def _matches(candidate: str) -> bool:
        for root in allowed_roots:
            root_slash = root + "/"
            if candidate == root or candidate.startswith(root_slash):
                return True
        return False

    literal_ok = _matches(abs_path)
    real_ok = _matches(real_path)

    if strict:
        return literal_ok and real_ok
    return literal_ok or real_ok


def _allow(hook_event_name: str) -> dict:
    """Build a hook response that explicitly allows the tool call.

    Sets ``hookSpecificOutput.permissionDecision`` — the field the CLI's
    PreToolUse permission pipeline actually reads (see ``_deny`` below for why
    this matters). An explicit allow here is deliberate: it is the correct
    decision for in-sandbox operations, not an accidental default.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "permissionDecision": "allow",
        }
    }


def _deny(hook_event_name: str, reason: str) -> dict:
    """Build a hook response that authoritatively denies the tool call.

    A prior version of this module returned ``{"permissionDecision": "deny",
    "permissionDecisionReason": reason}`` as *top-level* keys. Per the
    installed claude_agent_sdk's own type contract (``SyncHookJSONOutput`` /
    ``PreToolUseHookSpecificOutput`` in ``claude_agent_sdk/types.py``),
    ``permissionDecision`` is only recognized when nested under
    ``hookSpecificOutput``; a bare top-level ``permissionDecision`` key is not
    part of the schema and is silently dropped.

    We now set every documented blocking field so the decision does not
    depend on which internal code path a given tool type happens to consult:
    the generic top-level ``decision``/``reason`` (the SDK docs note only
    ``"approve"`` is deprecated for PreToolUse, not ``"block"``) *and* the
    modern ``hookSpecificOutput.permissionDecision``.
    """
    return {
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {
            "hookEventName": hook_event_name,
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def _check_bash_write(command: str, sandbox_root: str, cwd: Optional[str] = None) -> Optional[str]:
    """Check a Bash command for write operations outside the sandbox.

    Returns a denial reason if a violation is found, or None if allowed.
    """
    # Claude Code often emits relative write targets like "outputs" or "src/analysis".
    # If the hook call does not provide the tool cwd, resolve those relative paths
    # against the sandbox root rather than the parent Python process cwd.
    cwd = cwd or sandbox_root
    write_roots = [sandbox_root]
    segments = re.split(r"\s*(?:&&|\|\||;)\s*", command)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # 1. Output redirections (> or >>)
        for redir_match in _REDIRECT_PATTERN.finditer(segment):
            target_file = redir_match.group(2).strip("'\"")
            # FD duplication (``2>&1``, ``>&2``): not a file write.
            if target_file.startswith("&"):
                continue
            # Process substitution (``>(cmd)``): not a file write.
            if target_file.startswith("("):
                continue
            if not _is_within(target_file, write_roots, cwd, strict=True):
                return f"Blocked: redirect to path outside sandbox: {target_file}"

        # 2. tee writing to file
        for tee_match in _TEE_PATTERN.finditer(segment):
            target_file = tee_match.group(1).strip("'\"")
            if not _is_within(target_file, write_roots, cwd, strict=True):
                return f"Blocked: 'tee' writing outside sandbox: {target_file}"

        # 3. In-place edits (sed -i, perl -i)
        if _INPLACE_EDIT_PATTERN.search(segment):
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()
            for token in tokens:
                if token.startswith("-"):
                    continue
                if "/" in token or os.path.sep in token:
                    if not _is_within(token, write_roots, cwd, strict=True):
                        return f"Blocked: in-place edit outside sandbox: {token}"

        # 4. Destructive / copy-like commands
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        if not tokens:
            continue

        cmd_name = os.path.basename(tokens[0])

        if cmd_name in _DESTRUCTIVE_ALL_ARGS:
            for token in tokens[1:]:
                if token.startswith("-"):
                    continue
                if not _is_within(token, write_roots, cwd, strict=True):
                    return f"Blocked: '{cmd_name}' targeting path outside sandbox: {token}"

        elif cmd_name in _COPY_LIKE_COMMANDS:
            non_flag_args = [t for t in tokens[1:] if not t.startswith("-")]
            if non_flag_args:
                dest = non_flag_args[-1]
                if not _is_within(dest, write_roots, cwd, strict=True):
                    return f"Blocked: '{cmd_name}' writing outside sandbox: {dest}"

    return None


def _check_bash_read(command: str, read_roots: List[str], cwd: Optional[str] = None) -> Optional[str]:
    """Check a Bash command for read operations outside allowed roots.

    Returns a denial reason if a violation is found, or None if allowed.
    """
    segments = re.split(r"\s*(?:&&|\|\||;)\s*", command)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # 1. Block cd outside allowed paths
        cd_match = re.match(r"^cd\s+(.+)$", segment)
        if cd_match:
            target = cd_match.group(1).strip().strip("'\"")
            if not _is_within(target, read_roots, cwd):
                return f"Blocked: 'cd' to path outside sandbox: {target}"

        # 2. Read-oriented commands
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        if not tokens:
            continue

        cmd_name = os.path.basename(tokens[0])
        if cmd_name in _READ_COMMANDS:
            for token in tokens[1:]:
                if token.startswith("-"):
                    continue
                # Only check tokens that look like paths
                if "/" in token or os.path.sep in token:
                    if not _is_within(token, read_roots, cwd):
                        return f"Blocked: '{cmd_name}' reading outside sandbox: {token}"

    return None


def create_sandbox_enforcement_hook(
    sandbox_root: str,
    allowed_read_paths: Optional[List[str]] = None,
    logger: Any = None,
    escape_attempts: Optional[Dict[str, dict]] = None,
) -> List[HookMatcher]:
    """Factory that returns PreToolUse HookMatchers enforcing sandbox boundaries.

    Args:
        sandbox_root: Absolute path to the sandbox directory. All writes are
            confined here. Reads are also confined here plus *allowed_read_paths*.
        allowed_read_paths: Optional list of additional absolute directory paths
            that the agent may read from (but not write to).
        logger: Optional logger with ``log_action`` / ``log_error`` methods.
        escape_attempts: Optional dict, shared with
            ``create_sandbox_posttool_guard``, that this hook populates
            (keyed by ``tool_use_id``) whenever it denies a Write/Edit outside
            the sandbox. The PostToolUse guard uses these records to decide
            whether it is safe to delete a file that ended up outside the
            sandbox despite this denial (only if it did not exist before this
            call — never delete something that pre-dates the agent's action).

    Returns:
        List of 6 ``HookMatcher`` objects (Write, Edit, Read, Glob, Grep, Bash).
    """
    # Normalize roots — canonical form without trailing slash
    sandbox_root = os.path.realpath(sandbox_root).rstrip("/")
    read_extras = [os.path.realpath(p).rstrip("/") for p in (allowed_read_paths or [])]
    write_roots = [sandbox_root]
    read_roots = [sandbox_root] + read_extras

    # ------------------------------------------------------------------
    # Write / Edit hook — hard block outside sandbox
    # ------------------------------------------------------------------
    async def write_edit_hook(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        file_path = tool_input.get("file_path", "")

        if not file_path:
            return _allow("PreToolUse")

        if not _is_within(file_path, write_roots, strict=True):
            reason = f"Sandbox enforcement: write denied outside sandbox: {file_path}"
            if logger:
                logger.log_action("sandbox_write_deny", reason)
            if escape_attempts is not None and tool_use_id is not None:
                abs_path = _make_absolute(file_path)
                escape_attempts[tool_use_id] = {
                    "path": abs_path,
                    "pre_existed": os.path.exists(abs_path),
                }
            return _deny("PreToolUse", reason)

        return _allow("PreToolUse")

    # ------------------------------------------------------------------
    # Read hook — hard block outside sandbox + allowed read paths
    # ------------------------------------------------------------------
    async def read_hook(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        file_path = tool_input.get("file_path", "")

        if not file_path:
            return _allow("PreToolUse")

        if not _is_within(file_path, read_roots):
            reason = f"Sandbox enforcement: read denied outside sandbox: {file_path}"
            if logger:
                logger.log_action("sandbox_read_deny", reason)
            return _deny("PreToolUse", reason)

        return _allow("PreToolUse")

    # ------------------------------------------------------------------
    # Glob / Grep hook — hard block searches outside sandbox + allowed reads
    # ------------------------------------------------------------------
    async def search_hook(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        search_path = tool_input.get("path", "")

        # Empty path means cwd, which is set to sandbox root — allow
        if not search_path:
            return _allow("PreToolUse")

        if not _is_within(search_path, read_roots):
            reason = f"Sandbox enforcement: search denied outside sandbox: {search_path}"
            if logger:
                logger.log_action("sandbox_search_deny", reason)
            return _deny("PreToolUse", reason)

        return _allow("PreToolUse")

    # ------------------------------------------------------------------
    # Bash hook — best-effort: check writes AND reads
    # ------------------------------------------------------------------
    async def bash_hook(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command", "")

        if not command:
            return _allow("PreToolUse")

        # Default the cwd used for *relative* path resolution to the sandbox
        # root. The CLI's Bash tool runs with cwd=sandbox (set via
        # ClaudeAgentOptions.cwd), but tool_input rarely echoes that back; if
        # we fell through to os.getcwd() we'd resolve ``data/foo`` against the
        # Python process's cwd (the agent repo) and reject every relative
        # path in commands the CLI actually runs from inside the sandbox.
        cwd = tool_input.get("cwd") or sandbox_root

        # Check writes first (stricter — sandbox only)
        write_violation = _check_bash_write(command, sandbox_root, cwd)
        if write_violation:
            reason = f"Sandbox enforcement: {write_violation}"
            if logger:
                logger.log_action("sandbox_bash_write_deny", write_violation)
            return _deny("PreToolUse", reason)

        # Check reads (sandbox + allowed read paths)
        read_violation = _check_bash_read(command, read_roots, cwd)
        if read_violation:
            reason = f"Sandbox enforcement: {read_violation}"
            if logger:
                logger.log_action("sandbox_bash_read_deny", read_violation)
            return _deny("PreToolUse", reason)

        return _allow("PreToolUse")

    return [
        HookMatcher(matcher="Write", hooks=[write_edit_hook]),
        HookMatcher(matcher="Edit", hooks=[write_edit_hook]),
        HookMatcher(matcher="Read", hooks=[read_hook]),
        HookMatcher(matcher="Glob", hooks=[search_hook]),
        HookMatcher(matcher="Grep", hooks=[search_hook]),
        HookMatcher(matcher="Bash", hooks=[bash_hook]),
    ]


def create_sandbox_posttool_guard(
    sandbox_root: str,
    escape_attempts: Dict[str, dict],
    logger: Any = None,
) -> List[HookMatcher]:
    """Factory that returns PostToolUse HookMatchers backstopping Write/Edit.

    This is defense in depth: a PreToolUse ``write_edit_hook`` deny that is
    not honored by the CLI (for example after an SDK change) must not be
    allowed to silently persist a file outside the sandbox, and
    the run must not continue believing an out-of-sandbox write succeeded.

    This hook does not depend on PreToolUse hook ordering/merging at all —
    PostToolUse is a separate event, evaluated strictly after the tool
    already ran, so it catches an escape regardless of *why* PreToolUse
    failed to block it.

    On detecting a Write/Edit whose ``file_path`` resolves outside
    *sandbox_root*:
      - If ``escape_attempts`` shows the path did not exist before this call
        (i.e. the agent created it), the file/directory is deleted immediately.
      - If the path already existed (the agent modified something that
        predates it — deleting it could destroy unrelated user content), it
        is left alone and only reported; a human must reconcile it.
      - Either way, the hook returns ``continue_: False`` to hard-stop the
        run rather than let it proceed on the assumption the write "worked".

    Args:
        sandbox_root: Absolute path to the sandbox directory (same value
            passed to ``create_sandbox_enforcement_hook``).
        escape_attempts: The same dict instance passed as
            ``create_sandbox_enforcement_hook(..., escape_attempts=...)``.
            Used to look up whether a denied path pre-existed; entries are
            consumed (popped) as they are read.
        logger: Optional logger with ``log_action`` / ``log_error`` methods.

    Returns:
        List of 2 ``HookMatcher`` objects (Write, Edit).
    """
    sandbox_root = os.path.realpath(sandbox_root).rstrip("/")
    write_roots = [sandbox_root]

    async def posttool_guard(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        file_path = tool_input.get("file_path", "")

        if not file_path or _is_within(file_path, write_roots, strict=True):
            escape_attempts.pop(tool_use_id, None)
            return {"continue_": True}

        abs_path = _make_absolute(file_path)
        record = escape_attempts.pop(tool_use_id, None)
        # No PreToolUse record at all means the deny never even ran for this
        # call — treat the target as pre-existing (unknown) so we default to
        # the conservative, non-destructive branch.
        pre_existed = record["pre_existed"] if record else True

        reason = (
            f"CRITICAL: sandbox escape — tool wrote outside sandbox root "
            f"{sandbox_root!r} at {abs_path!r} despite PreToolUse denial. "
        )

        if pre_existed:
            reason += (
                "This path existed before the call, so it was NOT deleted "
                "(auto-deleting could destroy unrelated pre-existing content). "
                "Aborting the run for human review."
            )
        else:
            try:
                if os.path.islink(abs_path) or os.path.isfile(abs_path):
                    os.remove(abs_path)
                    reason += "The newly-created file was deleted. "
                elif os.path.isdir(abs_path):
                    shutil.rmtree(abs_path)
                    reason += "The newly-created directory was deleted. "
                else:
                    reason += "Nothing found on disk to delete. "
            except OSError as exc:
                reason += f"Deletion attempt failed ({exc}). "
            reason += "Aborting the run regardless — this is a security defect, not a recoverable error."

        if logger:
            logger.log_action("sandbox_posttool_escape", reason)
            if hasattr(logger, "log_error"):
                logger.log_error(reason)

        return {
            "continue_": False,
            "stopReason": reason,
            "decision": "block",
            "reason": reason,
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": reason,
            },
        }

    return [
        HookMatcher(matcher="Write", hooks=[posttool_guard]),
        HookMatcher(matcher="Edit", hooks=[posttool_guard]),
    ]
