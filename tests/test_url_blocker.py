import asyncio

from DiscoveryAgent.tool_agents.security.url_blocker import create_url_blocker_hooks


def _hook(matchers, matcher):
    return next(item.hooks[0] for item in matchers if item.matcher == matcher)


def _run(hook, tool_input):
    return asyncio.run(hook({"tool_input": tool_input}, "tool-use-id", {}))


def _permission(result):
    return result["hookSpecificOutput"]["permissionDecision"]


def test_bash_hook_authoritatively_blocks_default_host():
    hook = _hook(create_url_blocker_hooks(), "Bash")

    result = _run(
        hook,
        {"command": "curl https://sub.MARKS.HMS.HARVARD.EDU/proteingym.csv"},
    )

    assert result["decision"] == "block"
    assert _permission(result) == "deny"
    assert "marks.hms.harvard.edu" in result["reason"].lower()


def test_bash_hook_blocks_repo_ids_without_literal_url_case_insensitively():
    hook = _hook(create_url_blocker_hooks(), "Bash")

    result = _run(
        hook,
        {"command": "python -c \"snapshot_download('oatml-markslab/proteingym_v1')\""},
    )

    assert _permission(result) == "deny"


def test_webfetch_blocks_subdomain_but_not_hostname_in_query_text():
    hook = _hook(create_url_blocker_hooks(), "WebFetch")

    blocked = _run(hook, {"url": "https://x.marks.hms.harvard.edu/data"})
    allowed = _run(
        hook,
        {"url": "https://example.com/?ref=marks.hms.harvard.edu"},
    )

    assert _permission(blocked) == "deny"
    assert _permission(allowed) == "allow"


def test_empty_custom_lists_disable_both_rule_categories():
    hooks = create_url_blocker_hooks(
        blocked_hosts=[],
        blocked_url_substrings=[],
    )
    bash_hook = _hook(hooks, "Bash")
    webfetch_hook = _hook(hooks, "WebFetch")

    assert _permission(_run(
        bash_hook,
        {"command": "curl https://marks.hms.harvard.edu/data"},
    )) == "allow"
    assert _permission(_run(
        webfetch_hook,
        {"url": "https://marks.hms.harvard.edu/data"},
    )) == "allow"


def test_logging_failure_cannot_bypass_a_deny():
    class BrokenLogger:
        def log_action(self, action, reason):
            raise RuntimeError("logging unavailable")

    hook = _hook(create_url_blocker_hooks(logger=BrokenLogger()), "Bash")

    result = _run(
        hook,
        {"command": "wget https://marks.hms.harvard.edu/data"},
    )

    assert _permission(result) == "deny"
