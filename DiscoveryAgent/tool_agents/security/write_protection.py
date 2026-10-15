"""
Write protection hook for Claude Code SDK.

Prevents the agent from writing to protected paths (e.g. read-only shared data).
Checks Write, Edit, and Bash tool calls for operations targeting protected directories.
"""

import os
import re
import shlex
from typing import List, Optional, Any

from claude_agent_sdk import HookMatcher


# Commands that destructively modify targets (all non-flag args are targets)
_DESTRUCTIVE_ALL_ARGS = {"rm", "rmdir", "mkdir", "touch", "chmod", "chown", "chgrp"}

# Commands where only the LAST non-flag arg is the write destination
_COPY_LIKE_COMMANDS = {"cp", "mv", "ln", "install"}

# Patterns for in-place edit flags
_INPLACE_EDIT_PATTERN = re.compile(r"\b(sed|perl|awk)\b.*\s-i\b")

# Patterns for output redirection
_REDIRECT_PATTERN = re.compile(r"(>{1,2})\s*(\S+)")

# Pattern for tee command writing to a file
_TEE_PATTERN = re.compile(r"\btee\b(?:\s+-a)?\s+(\S+)")


def _resolve_path(path: str, cwd: Optional[str] = None) -> str:
    """Resolve a path to its real absolute form."""
    if not os.path.isabs(path):
        base = cwd or os.getcwd()
        path = os.path.join(base, path)
    return os.path.realpath(path)


def _is_protected(path: str, protected_paths: List[str], cwd: Optional[str] = None) -> bool:
    """Check if a resolved path falls under any protected directory.

    Protected paths are stored WITHOUT a trailing slash.  A resolved path
    matches if it equals a protected path exactly or starts with it followed
    by a ``/`` (i.e. is inside it).
    """
    resolved = _resolve_path(path, cwd)
    for p in protected_paths:
        if resolved == p or resolved.startswith(p + "/"):
            return True
    return False


def _check_bash_command(command: str, protected_paths: List[str], cwd: Optional[str] = None) -> Optional[str]:
    """
    Parse a Bash command string for write operations targeting protected paths.

    Returns a denial reason string if a violation is found, or None if allowed.
    """
    # Split on command chaining operators to check each segment
    segments = re.split(r"\s*(?:&&|\|\||;)\s*", command)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # 1. Block cd into protected paths
        cd_match = re.match(r"^cd\s+(.+)$", segment)
        if cd_match:
            target = cd_match.group(1).strip().strip("'\"")
            if _is_protected(target, protected_paths, cwd):
                return f"Blocked: 'cd' into protected path {target}"

        # 2. Check output redirections (> or >>)
        for redir_match in _REDIRECT_PATTERN.finditer(segment):
            target_file = redir_match.group(2).strip("'\"")
            if _is_protected(target_file, protected_paths, cwd):
                return f"Blocked: redirect to protected path {target_file}"

        # 3. Check tee writing to protected path
        for tee_match in _TEE_PATTERN.finditer(segment):
            target_file = tee_match.group(1).strip("'\"")
            if _is_protected(target_file, protected_paths, cwd):
                return f"Blocked: 'tee' writing to protected path {target_file}"

        # 4. Check in-place edits (sed -i, perl -i)
        if _INPLACE_EDIT_PATTERN.search(segment):
            # Try to extract the file argument (last non-flag token)
            try:
                tokens = shlex.split(segment)
            except ValueError:
                tokens = segment.split()
            # Check all tokens that look like file paths
            for token in tokens:
                if token.startswith("-"):
                    continue
                if "/" in token or os.path.sep in token:
                    if _is_protected(token, protected_paths, cwd):
                        return f"Blocked: in-place edit of protected file {token}"

        # 5. Check file-modifying commands
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()

        if not tokens:
            continue

        cmd_name = os.path.basename(tokens[0])

        if cmd_name in _DESTRUCTIVE_ALL_ARGS:
            # All non-flag arguments are targets (rm, mkdir, touch, etc.)
            for token in tokens[1:]:
                if token.startswith("-"):
                    continue
                if _is_protected(token, protected_paths, cwd):
                    return f"Blocked: '{cmd_name}' targeting protected path {token}"

        elif cmd_name in _COPY_LIKE_COMMANDS:
            # Only the LAST non-flag argument is the write destination
            non_flag_args = [t for t in tokens[1:] if not t.startswith("-")]
            if non_flag_args:
                dest = non_flag_args[-1]
                if _is_protected(dest, protected_paths, cwd):
                    return f"Blocked: '{cmd_name}' writing to protected path {dest}"

    return None


def create_write_protection_hook(protected_paths: List[str], logger: Any = None):
    """
    Factory that returns a PreToolUse hook and HookMatchers for Write, Edit, and Bash.

    Args:
        protected_paths: List of absolute directory paths to protect from writes.
        logger: Optional logger with log_action/log_error methods.

    Returns:
        List of HookMatcher objects to pass to ClaudeAgentOptions hooks["PreToolUse"].
    """
    # Normalize: resolve to canonical form without trailing slash
    resolved_protected = [os.path.realpath(p).rstrip("/") for p in protected_paths]

    async def write_edit_hook(input_data, tool_use_id, context) -> dict:
        """Check Write and Edit tool calls for protected file paths."""
        tool_input = input_data.get("tool_input", {})
        file_path = tool_input.get("file_path", "")

        if not file_path:
            return {"permissionDecision": "allow"}

        if _is_protected(file_path, resolved_protected):
            reason = f"Write protection: cannot modify file in protected path: {file_path}"
            if logger:
                logger.log_action("write_protection_deny", reason)
            return {
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }

        return {"permissionDecision": "allow"}

    async def bash_hook(input_data, tool_use_id, context) -> dict:
        """Check Bash commands for write operations targeting protected paths."""
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command", "")

        if not command:
            return {"permissionDecision": "allow"}

        # Get cwd from tool input if available
        cwd = tool_input.get("cwd", None)

        violation = _check_bash_command(command, resolved_protected, cwd)
        if violation:
            if logger:
                logger.log_action("write_protection_deny", violation)
            return {
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Write protection: {violation}",
            }

        return {"permissionDecision": "allow"}

    return [
        HookMatcher(matcher="Write", hooks=[write_edit_hook]),
        HookMatcher(matcher="Edit", hooks=[write_edit_hook]),
        HookMatcher(matcher="Bash", hooks=[bash_hook]),
    ]
