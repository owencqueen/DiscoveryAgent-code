"""
URL blocker hook for Claude Code SDK.

Denies network fetches to user-configured hosts or URL substrings. Catches:
  - ``Bash`` invocations that embed an HTTP/HTTPS URL (curl, wget, python -c
    "...urlopen", ``huggingface_hub.snapshot_download``, etc.).
  - ``WebFetch`` tool calls.

The block is based on string matching against the literal URL the agent supplies.
Two predicate kinds are supported:

  * ``blocked_hosts``: exact host match OR subdomain of a blocked host. So
    ``marks.hms.harvard.edu`` matches ``https://marks.hms.harvard.edu/...`` and
    ``https://x.marks.hms.harvard.edu/...`` but NOT ``https://example.com/?ref=marks.hms.harvard.edu``.
  * ``blocked_url_substrings``: plain substring match against the full
    command/URL text. Catches both literal URLs and SDK-style references that
    never spell out the URL (``snapshot_download("OATML-Markslab/ProteinGym_v1")``).
    Use this for path-scoped blocks like ``huggingface.co/datasets/OATML-Markslab``
    or repo-id-style strings like ``OATML-Markslab/ProteinGym``.

This is a defense-in-depth layer, not a panacea — an agent that decides to
exfiltrate URL strings via base64, reroute through a proxy, or bounce off a
permissive third-party host can still get out. The point is to make accidental
or naive downloads of known leak vectors fail loudly.
"""

import re
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from claude_agent_sdk import HookMatcher


# Catches http/https URLs anywhere in a string. Stops at typical shell/quote
# boundaries; intentionally lenient about query strings and fragments.
_URL_RE = re.compile(r"""https?://[^\s'"`<>)\\]+""")


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    return _URL_RE.findall(text)


def _host_matches(url_host: str, blocked_host: str) -> bool:
    """True if *url_host* is *blocked_host* or any subdomain of it."""
    url_host = url_host.lower().rstrip(".")
    blocked_host = blocked_host.lower().rstrip(".")
    if not url_host or not blocked_host:
        return False
    return url_host == blocked_host or url_host.endswith("." + blocked_host)


def _check_url(
    url: str,
    blocked_hosts: Sequence[str],
    blocked_url_substrings: Sequence[str],
) -> Optional[str]:
    """Return a human-readable reason if *url* is blocked, else ``None``.

    Host check is applied to the parsed hostname; substring check is applied
    to the URL text.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()

    for bh in blocked_hosts:
        if _host_matches(host, bh):
            return f"URL blocked by host rule '{bh}': {url}"

    folded_url = url.casefold()
    for sub in blocked_url_substrings:
        if sub and sub.casefold() in folded_url:
            return f"URL blocked by substring rule '{sub}': {url}"

    return None


def _check_text_for_blocked_urls(
    text: str,
    blocked_hosts: Sequence[str],
    blocked_url_substrings: Sequence[str],
) -> Optional[str]:
    """Return a deny reason if any URL or blocked substring appears in *text*.

    Two passes: first extract every ``http(s)://`` URL and apply host +
    substring rules; then scan the raw text for substring matches that don't
    require a URL prefix (catches SDK-style repo IDs and inline mentions).
    """
    for url in _extract_urls(text):
        reason = _check_url(url, blocked_hosts, blocked_url_substrings)
        if reason:
            return reason

    folded_text = text.casefold()
    for sub in blocked_url_substrings:
        if sub and sub.casefold() in folded_text:
            return f"Blocked text reference '{sub}' found in command"

    return None


# Default blocklist: ProteinGym test-label / pre-computed-baseline data sources.
# Downloading any of these into a sandboxed run leaks held-out test labels,
# which defeats supervised-split evaluation.
DEFAULT_BLOCKED_HOSTS: List[str] = [
    "marks.hms.harvard.edu",
]

DEFAULT_BLOCKED_URL_SUBSTRINGS: List[str] = [
    # ProteinGym mirrors on HuggingFace (URL form and SDK repo-id form)
    "huggingface.co/datasets/OATML-Markslab",
    "OATML-Markslab/ProteinGym",
    # ProteinGym mirrors on Zenodo (v1.0–v1.3)
    "zenodo.org/records/13932633",
    "zenodo.org/records/13936340",
    "zenodo.org/records/14997691",
    "zenodo.org/records/15293562",
    # GitHub release artifacts and raw blobs of the ProteinGym repo
    "github.com/OATML-Markslab/ProteinGym/releases",
    "github.com/OATML-Markslab/ProteinGym/raw",
    "raw.githubusercontent.com/OATML-Markslab/ProteinGym",
]


def _allow() -> dict:
    """Return an SDK-recognized explicit PreToolUse allow decision."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


def _deny(reason: str) -> dict:
    """Return an authoritative SDK-recognized PreToolUse denial."""
    return {
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def _log_deny(logger, action: str, reason: str) -> None:
    """Keep security decisions independent from best-effort logging."""
    if logger is None:
        return
    try:
        logger.log_action(action, reason)
    except Exception:
        pass


def create_url_blocker_hooks(
    blocked_hosts: Optional[Iterable[str]] = None,
    blocked_url_substrings: Optional[Iterable[str]] = None,
    logger=None,
) -> List[HookMatcher]:
    """Factory that returns PreToolUse HookMatchers blocking specific URLs.

    Args:
        blocked_hosts: Hosts (and their subdomains) to deny. Defaults to
            ``DEFAULT_BLOCKED_HOSTS`` when ``None``; pass an empty list to
            disable host-level blocking entirely.
        blocked_url_substrings: Plain substring matches on URL text. Defaults
            to ``DEFAULT_BLOCKED_URL_SUBSTRINGS`` when ``None``.
        logger: Optional ``Logger`` for ``log_action`` calls on a deny.

    Returns:
        A list of ``HookMatcher`` objects covering ``Bash`` and ``WebFetch``.
    """
    hosts = list(
        DEFAULT_BLOCKED_HOSTS if blocked_hosts is None else blocked_hosts
    )
    subs = list(
        DEFAULT_BLOCKED_URL_SUBSTRINGS
        if blocked_url_substrings is None
        else blocked_url_substrings
    )

    async def bash_hook(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        command = tool_input.get("command", "") or ""
        reason = _check_text_for_blocked_urls(command, hosts, subs)
        if reason:
            reason = f"URL blocker: {reason}"
            _log_deny(logger, "url_blocker_bash_deny", reason)
            return _deny(reason)
        return _allow()

    async def web_fetch_hook(input_data, tool_use_id, context) -> dict:
        tool_input = input_data.get("tool_input", {})
        url = tool_input.get("url", "") or ""
        reason = _check_url(url, hosts, subs) if url else None
        if reason:
            reason = f"URL blocker: {reason}"
            _log_deny(logger, "url_blocker_webfetch_deny", reason)
            return _deny(reason)
        return _allow()

    return [
        HookMatcher(matcher="Bash", hooks=[bash_hook]),
        HookMatcher(matcher="WebFetch", hooks=[web_fetch_hook]),
    ]
