import asyncio
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from DiscoveryAgent.tool_agents import execution_agent_cc
from DiscoveryAgent.tool_agents.execution_agent_cc import (
    EXECUTION_SKILLS_PLUGIN_NAME,
    ExecutionAgentCC,
    ExecutionSkillsConfigurationError,
    ExecutionSkillsRequiredLoadError,
)
from DiscoveryAgent.utils.agent_config_builders import set_execution_agent_config
from DiscoveryAgent.utils.arg_parsers import create_run_parser
from DiscoveryAgent.tool_agents.execution_sdk_tracker import ExecutionSDKTracker
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport
import claude_agent_sdk


class CaptureLogger:
    def __init__(self):
        self.actions = []
        self.errors = []

    def log_action(self, action, details=None):
        self.actions.append((action, details))

    def log_error(self, message, code=None):
        self.errors.append((message, code))


class SystemMessage:
    def __init__(self, skills):
        self.data = {"type": "system", "subtype": "init", "skills": list(skills)}

    def __str__(self):
        return f"SystemMessage(skills={self.data['skills']!r})"

class ResultMessage:
    subtype = "success"
    stop_reason = None

    def __str__(self):
        return "ResultMessage(success)"


class HookDrivenSDK:
    """Scripted SDK that reaches state only through configured production hooks."""

    def __init__(self, options, scripts):
        self.options = options
        self.scripts = list(scripts)
        self.active_script = []
        self.queries = []
        self.tool_attempts = []
        self.subagent_events = []
        self._tool_counter = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def query(self, prompt):
        self.queries.append(prompt)
        if not self.scripts:
            raise AssertionError(f"No scripted SDK response remains for query: {prompt}")
        self.active_script = self.scripts.pop(0)

    @staticmethod
    def _permission_denied(outputs):
        return any(
            isinstance(output, dict)
            and output.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"
            for output in outputs
        )

    async def _run_hooks(self, event, hook_input, tool_use_id):
        outputs = []
        tool_name = hook_input.get("tool_name")
        for matcher in self.options.get("hooks", {}).get(event, []):
            matcher_name = getattr(matcher, "matcher", None)
            if event in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
                if matcher_name is not None and matcher_name != tool_name:
                    continue
            for callback in getattr(matcher, "hooks", []):
                outputs.append(await callback(hook_input, tool_use_id, None))
        return outputs

    async def _run_tool(self, action, inherited=None):
        self._tool_counter += 1
        tool_use_id = action.get("id", f"tool-{self._tool_counter}")
        hook_input = {
            "tool_name": action["tool"],
            "tool_input": dict(action.get("input", {})),
        }
        hook_input.update(inherited or {})
        pre_outputs = await self._run_hooks("PreToolUse", hook_input, tool_use_id)
        denied = self._permission_denied(pre_outputs)
        for output in pre_outputs:
            hook_output = output.get("hookSpecificOutput", {}) if isinstance(output, dict) else {}
            if "updatedInput" in hook_output:
                hook_input["tool_input"] = hook_output["updatedInput"]

        attempt = {
            "tool": action["tool"],
            "input": hook_input["tool_input"],
            "allowed": not denied,
            "context": dict(inherited or {}),
        }
        self.tool_attempts.append(attempt)
        if denied:
            return

        if action.get("failure"):
            failed_input = dict(hook_input)
            failed_input["error"] = action.get("error", "scripted tool failure")
            await self._run_hooks("PostToolUseFailure", failed_input, tool_use_id)
            return

        effect = action.get("effect")
        if effect is not None:
            effect()
        response = action.get("response", {"success": True})
        post_input = dict(hook_input)
        post_input["tool_response"] = response
        await self._run_hooks("PostToolUse", post_input, tool_use_id)

    async def _run_subagent(self, action):
        inherited = {
            "agent_id": action.get("agent_id", "worker-1"),
            "agent_type": action["subagent"],
        }
        self.subagent_events.append(("start", dict(inherited)))
        start_outputs = await self._run_hooks(
            "SubagentStart", dict(inherited), f"start-{inherited['agent_id']}"
        )
        for nested in action.get("actions", []):
            await self._run_tool(nested, inherited)
        stop_outputs = await self._run_hooks(
            "SubagentStop", dict(inherited), f"stop-{inherited['agent_id']}"
        )
        self.subagent_events.append(("stop", stop_outputs))

    async def receive_response(self):
        for action in self.active_script:
            if action.get("kind") == "system":
                yield SystemMessage(action.get("skills", self.options.get("skills", [])))
            elif "subagent" in action:
                await self._run_subagent(action)
            elif "tool" in action:
                await self._run_tool(action)
            else:
                raise AssertionError(f"Unknown scripted action: {action}")
        yield ResultMessage()


def skill_action(runtime_name, *, response=None, failure=False):
    action = {
        "tool": "Skill",
        "input": {"skill": runtime_name},
        "response": response or {
            "success": True,
            "commandName": runtime_name,
        },
    }
    if failure:
        action["failure"] = True
    return action


def prepare_hook_driven_agent(
    monkeypatch, code_dir, snapshot, scripts, *, sub_agent_architecture=False
):
    agent = bare_agent(code_dir, str(snapshot) if snapshot is not None else None)
    agent.sub_agent_architecture = sub_agent_architecture
    captured, sdk = initialize_with_capture(
        monkeypatch,
        agent,
        client_factory=lambda options: HookDrivenSDK(options, scripts),
    )
    agent._format_message_for_display = lambda _message: ""

    async def fake_summary(*_args, **_kwargs):
        return "Synthetic analysis completed."

    agent._generate_concise_summary = fake_summary
    return agent, captured, sdk


def make_project(tmp_path: Path, name: str = "project with spaces"):
    project_root = tmp_path / name
    code_dir = project_root / "code"
    snapshot = project_root / "checkpoint" / "checkpoints" / "round_1" / ".execution_skills"
    code_dir.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    return project_root, code_dir, snapshot


def make_skill(root: Path, skill_id: str, description: str, *, body: str = "") -> Path:
    skill = root / skill_id
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill


def bare_agent(code_dir: Path, skills_path=None) -> ExecutionAgentCC:
    agent = ExecutionAgentCC.__new__(ExecutionAgentCC)
    agent.config = SimpleNamespace(execution_skills_path=skills_path)
    agent.working_directory = code_dir
    agent.logger = CaptureLogger()
    agent.execution_tracker = None
    agent._execution_skills_runtime_directory = None
    agent._configured_execution_skills = []
    agent._execution_skills_source_paths = []
    agent._execution_skills_discovery_confirmed = False
    agent.allowed_read_paths = []
    agent.sandbox_enforcement = False
    agent.sub_agent_architecture = False
    agent.max_turns_query_execution = None
    agent.codebase_structure_prompt = None
    agent.llm_config = SimpleNamespace(model_name="claude-sonnet-4-6")
    agent.claude_md_generator = False
    agent._direct_print = lambda _message: None
    agent._get_system_prompt = lambda: "BASE EXECUTION SYSTEM PROMPT"
    agent._build_opentargets_mcp_configs = lambda: {}
    return agent


def initialize_with_capture(monkeypatch, agent, client_factory=None):
    captured = {}
    clients = []
    monkeypatch.setattr(
        execution_agent_cc,
        "_create_claude_agent_options",
        lambda options: captured.update(options) or dict(options),
    )
    if client_factory is None:
        client_factory = lambda options: SimpleNamespace(options=options)

    def create_client(options):
        client = client_factory(options)
        clients.append(client)
        return client

    monkeypatch.setattr(execution_agent_cc, "ClaudeSDKClient", create_client)
    agent._init_claude_code_session()
    return captured, clients[-1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("sub_agent_architecture", [False, True])
def test_snapshot_uses_filtered_supported_plugin_options(monkeypatch, tmp_path, sub_agent_architecture):
    project_root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "alpha-skill", "Always applies to alpha analyses.")
    make_skill(snapshot, "beta-skill", "Always applies to beta analyses.")
    make_skill(code_dir / ".claude" / "skills", "unselected-live", "Never expose.")
    other_root, _other_code, other_snapshot = make_project(tmp_path, "other project")
    make_skill(other_snapshot, "other-project", "Never expose.")

    agent = bare_agent(code_dir, str(snapshot))
    agent.sub_agent_architecture = sub_agent_architecture
    captured, _client = initialize_with_capture(monkeypatch, agent)
    expected = [
        f"{EXECUTION_SKILLS_PLUGIN_NAME}:alpha-skill",
        f"{EXECUTION_SKILLS_PLUGIN_NAME}:beta-skill",
    ]
    assert captured["skills"] == expected
    assert captured["setting_sources"] == ["project"]
    plugin_root = Path(captured["plugins"][0]["path"])
    assert captured["plugins"] == [{"type": "local", "path": str(plugin_root)}]
    assert captured["add_dirs"] == [str(plugin_root)]
    assert sorted(path.name for path in (plugin_root / "skills").iterdir()) == [
        "alpha-skill", "beta-skill",
    ]
    assert "unselected-live" not in captured["system_prompt"]
    assert "other-project" not in captured["system_prompt"]
    assert captured["system_prompt"].count("CONFIGURED EXECUTION SKILLS") == 1
    assert "Project selection is an explicit instruction" in captured["system_prompt"]
    assert "receipts, verification tokens" in captured["system_prompt"]
    assert "loaded immutable runtime directory" in captured["system_prompt"]

    transport = SubprocessCLITransport("", ClaudeAgentOptions(**captured))
    transport._cli_path = "/bin/true"
    command = transport._build_command()
    assert command[command.index("--plugin-dir") + 1] == str(plugin_root)
    assert command[command.index("--add-dir") + 1] == str(plugin_root)
    assert "--setting-sources=project" in command
    allowed = command[command.index("--allowedTools") + 1]
    assert all(f"Skill({name})" in allowed for name in expected)

    bundled_cli = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
    validation = subprocess.run(
        [str(bundled_cli), "plugin", "validate", "--strict", str(plugin_root)],
        capture_output=True, text=True, check=False,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert other_root != project_root
    agent._cleanup_execution_skills_runtime()


def run_analysis(agent, query="Analyze the dataset normally."):
    return asyncio.run(
        agent._execute_with_report_generation(
            query, "Synthetic dataset", query_num=1, output_files=[]
        )
    )


def test_discovered_skill_then_agent_is_blocked_and_corrected(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "required-skill", "Always applies.")
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:required-skill"
    scripts = [
        [
            {"kind": "system", "skills": [runtime_name]},
            {
                "tool": "Agent",
                "input": {
                    "subagent_type": "execution_agent",
                    "prompt": "Start computation immediately.",
                },
            },
        ],
        [skill_action(runtime_name)],
        [{"tool": "Bash", "input": {"command": "compute"}}],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts
    )

    result = run_analysis(agent)

    assert result["concise_report"] == "Synthetic analysis completed."
    assert sdk.tool_attempts[0]["tool"] == "Agent"
    assert sdk.tool_attempts[0]["allowed"] is False
    assert sdk.tool_attempts[-1]["tool"] == "Bash"
    assert sdk.tool_attempts[-1]["allowed"] is True
    assert len(sdk.queries) == 3
    agent._cleanup_execution_skills_runtime()


def test_discovery_without_invocation_fails_after_bounded_preflight(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "required-skill", "Always applies.")
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:required-skill"
    scripts = [
        [{"kind": "system", "skills": [runtime_name]}],
        [],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts
    )

    with pytest.raises(ExecutionSkillsRequiredLoadError, match=runtime_name):
        run_analysis(agent)

    assert len(sdk.queries) == 2
    assert agent._execution_skill_loaded_by_context["primary"] == set()
    events = [action for action, _details in agent.logger.actions]
    assert "execution_skills_runtime_discovered" in events
    assert "execution_skill_requested" not in events
    assert "execution_skill_loaded" not in events
    assert events.count("execution_skills_missing_required_load") >= 2
    agent._cleanup_execution_skills_runtime()


@pytest.mark.parametrize("sub_agent_architecture", [False, True])
def test_exact_qualified_load_allows_analysis(
    monkeypatch, tmp_path, sub_agent_architecture
):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "required-skill", "Always applies.")
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:required-skill"
    scripts = [
        [
            {"kind": "system", "skills": [runtime_name]},
            skill_action(runtime_name),
        ],
        [{"tool": "Read", "input": {"file_path": "data.csv"}}],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch,
        code_dir,
        snapshot,
        scripts,
        sub_agent_architecture=sub_agent_architecture,
    )

    run_analysis(agent)

    assert agent._execution_skill_loaded_by_context["primary"] == {runtime_name}
    assert [attempt["allowed"] for attempt in sdk.tool_attempts] == [True, True]
    events = [action for action, _details in agent.logger.actions]
    assert events.count("execution_skill_requested") == 1
    assert events.count("execution_skill_loaded") == 1
    agent._cleanup_execution_skills_runtime()


def test_failed_skill_load_keeps_tools_blocked_and_fails(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "required-skill", "Always applies.")
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:required-skill"
    scripts = [
        [
            {"kind": "system", "skills": [runtime_name]},
            skill_action(runtime_name, failure=True),
            {"tool": "Bash", "input": {"command": "must be blocked"}},
        ],
        [skill_action(runtime_name, failure=True)],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts
    )

    with pytest.raises(ExecutionSkillsRequiredLoadError, match=runtime_name):
        run_analysis(agent)

    bash = [attempt for attempt in sdk.tool_attempts if attempt["tool"] == "Bash"]
    assert bash and bash[0]["allowed"] is False
    assert agent._execution_skill_loaded_by_context["primary"] == set()
    events = [action for action, _details in agent.logger.actions]
    assert events.count("execution_skill_failed") == 2
    agent._cleanup_execution_skills_runtime()


def test_every_exact_frozen_name_is_required(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "alpha-skill", "Always applies.")
    make_skill(snapshot, "beta-skill", "Always applies.")
    alpha = f"{EXECUTION_SKILLS_PLUGIN_NAME}:alpha-skill"
    beta = f"{EXECUTION_SKILLS_PLUGIN_NAME}:beta-skill"
    scripts = [
        [
            {"kind": "system", "skills": [alpha, beta]},
            skill_action(alpha),
            skill_action(alpha),
            skill_action("alpha-skill"),
            skill_action("live-project:alpha-skill"),
            skill_action(f"{EXECUTION_SKILLS_PLUGIN_NAME}:unknown"),
            skill_action(beta),
        ],
        [{"tool": "Bash", "input": {"command": "compute"}}],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts
    )

    run_analysis(agent)

    assert agent._execution_skill_loaded_by_context["primary"] == {alpha, beta}
    denied = [
        attempt["input"]["skill"]
        for attempt in sdk.tool_attempts
        if attempt["tool"] == "Skill" and not attempt["allowed"]
    ]
    assert denied == [
        alpha,
        "alpha-skill",
        "live-project:alpha-skill",
        f"{EXECUTION_SKILLS_PLUGIN_NAME}:unknown",
    ]
    agent._cleanup_execution_skills_runtime()


def test_synthetic_receipt_runs_only_after_production_load_boundary(
    monkeypatch, tmp_path
):
    _root, code_dir, snapshot = make_project(tmp_path)
    skill = make_skill(
        snapshot,
        "always-receipt",
        "Always apply and emit a receipt before computation.",
        body="Run scripts/emit_receipt.py before computation.",
    )
    (skill / "scripts").mkdir()
    emitter = skill / "scripts" / "emit_receipt.py"
    emitter.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps({"
        "'skill':'always-receipt','token':'SYNTHETIC-RECEIPT-OK'}))\n",
        encoding="utf-8",
    )
    receipt = code_dir / "skill_receipt.json"
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:always-receipt"

    def emit_receipt():
        assert not receipt.exists()
        runtime_script = (
            Path(agent.claude_code_options["plugins"][0]["path"])
            / "skills" / "always-receipt" / "scripts" / "emit_receipt.py"
        )
        subprocess.run(
            [sys.executable, str(runtime_script), str(receipt)],
            cwd=code_dir,
            check=True,
        )

    scripts = [
        [
            {"kind": "system", "skills": [runtime_name]},
            skill_action(runtime_name),
        ],
        [{"tool": "Bash", "input": {"command": "emit receipt"}, "effect": emit_receipt}],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts
    )

    run_analysis(agent)

    assert sdk.tool_attempts[0]["tool"] == "Skill"
    assert sdk.tool_attempts[0]["allowed"] is True
    assert sdk.tool_attempts[1]["tool"] == "Bash"
    assert sdk.tool_attempts[1]["allowed"] is True
    assert json.loads(receipt.read_text())["token"] == "SYNTHETIC-RECEIPT-OK"
    agent._cleanup_execution_skills_runtime()


def test_delegated_analytical_context_loads_same_exact_skill(
    monkeypatch, tmp_path
):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "required-skill", "Always applies.")
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:required-skill"
    scripts = [
        [
            {"kind": "system", "skills": [runtime_name]},
            skill_action(runtime_name),
        ],
        [
            {
                "tool": "Agent",
                "input": {
                    "subagent_type": "execution_agent",
                    "prompt": "Perform delegated computation.",
                },
            },
            {
                "subagent": "execution_agent",
                "agent_id": "worker-1",
                "actions": [
                    skill_action(runtime_name),
                    {"tool": "Bash", "input": {"command": "delegated compute"}},
                ],
            },
        ],
    ]
    agent, captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts, sub_agent_architecture=True
    )

    run_analysis(agent)

    execution_def = captured["agents"]["execution_agent"]
    assert execution_def.skills == [runtime_name]
    assert "Skill" in execution_def.tools
    for name in ("documentation_agent", "claude_md_agent"):
        assert captured["agents"][name].skills is None
        assert "Skill" not in captured["agents"][name].tools
    delegated = agent._execution_skill_loaded_by_context["subagent:worker-1"]
    assert delegated == {runtime_name}
    agent_attempt = next(
        attempt for attempt in sdk.tool_attempts if attempt["tool"] == "Agent"
    )
    assert runtime_name in agent_attempt["input"]["prompt"]
    assert agent_attempt["input"]["run_in_background"] is False
    agent._cleanup_execution_skills_runtime()


def test_report_continuation_reuses_primary_load(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "required-skill", "Always applies.")
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:required-skill"
    scripts = [
        [
            {"kind": "system", "skills": [runtime_name]},
            skill_action(runtime_name),
        ],
        [{"tool": "Bash", "input": {"command": "compute"}}],
        [{"tool": "Write", "input": {"file_path": "report.md"}}],
    ]
    agent, _captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, snapshot, scripts
    )

    async def report_in_same_session(client, *_args, **_kwargs):
        await client.query("Continue in the same operative context and write the report.")
        async for _message in client.receive_response():
            pass
        return "Report complete."

    agent._generate_concise_summary = report_in_same_session
    result = run_analysis(agent)

    assert result["concise_report"] == "Report complete."
    assert [attempt["tool"] for attempt in sdk.tool_attempts].count("Skill") == 1
    assert sdk.tool_attempts[-1]["tool"] == "Write"
    assert sdk.tool_attempts[-1]["allowed"] is True
    assert agent._execution_skill_loaded_by_context["primary"] == {runtime_name}
    agent._cleanup_execution_skills_runtime()


def test_round_without_skills_executes_without_preflight(monkeypatch, tmp_path):
    _root, code_dir, _snapshot = make_project(tmp_path)
    scripts = [[{"tool": "Bash", "input": {"command": "ordinary compute"}}]]
    agent, captured, sdk = prepare_hook_driven_agent(
        monkeypatch, code_dir, None, scripts
    )

    run_analysis(agent)

    assert len(sdk.queries) == 1
    assert sdk.tool_attempts == [
        {
            "tool": "Bash",
            "input": {"command": "ordinary compute"},
            "allowed": True,
            "context": {},
        }
    ]
    assert all(
        key not in captured for key in ("skills", "plugins", "setting_sources", "add_dirs")
    )


def test_run_brief_style_tree_stays_intact_and_executable(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    skill = make_skill(snapshot, "run-brief-style", "Always create a linted brief and receipt.")
    (skill / "README.md").write_text("README marker\n")
    (skill / "references").mkdir()
    (skill / "references" / "checks.md").write_text("CHECKS marker\n")
    (skill / "templates").mkdir()
    (skill / "templates" / "brief.md").write_text("NINE SECTION TEMPLATE\n")
    (skill / "scripts").mkdir()
    verifier = skill / "scripts" / "verify_receipt.py"
    verifier.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "template=Path(__file__).parent.parent/'templates'/'brief.md'\n"
        "assert 'NINE SECTION' in template.read_text()\n"
        "Path(sys.argv[1]).write_text(json.dumps({"
        "'token':'RGCRB-1-SYNTHETIC','lint':'PASS'}))\n"
    )
    verifier.chmod(0o755)

    agent = bare_agent(code_dir, str(snapshot))
    captured, _client = initialize_with_capture(monkeypatch, agent)
    runtime_skill = (
        Path(captured["plugins"][0]["path"]) / "skills" / "run-brief-style"
    )
    source_files = sorted(path.relative_to(skill) for path in skill.rglob("*") if path.is_file())
    runtime_files = sorted(
        path.relative_to(runtime_skill) for path in runtime_skill.rglob("*") if path.is_file()
    )
    assert runtime_files == source_files
    assert all(sha256(skill / rel) == sha256(runtime_skill / rel) for rel in source_files)
    assert os.access(runtime_skill / "scripts" / "verify_receipt.py", os.X_OK)

    receipt = code_dir / "skill_receipt.json"
    subprocess.run(
        [sys.executable, str(runtime_skill / "scripts" / "verify_receipt.py"), str(receipt)],
        cwd=code_dir, check=True,
    )
    assert json.loads(receipt.read_text()) == {
        "token": "RGCRB-1-SYNTHETIC", "lint": "PASS",
    }
    agent._cleanup_execution_skills_runtime()


def test_retry_uses_frozen_bytes_without_duplicate_policy(monkeypatch, tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    make_skill(snapshot, "stable-skill", "Always applies.", body="FROZEN INSTRUCTIONS")
    live = make_skill(
        code_dir / ".claude" / "skills", "stable-skill", "Always applies.",
        body="EDITED LIVE INSTRUCTIONS",
    )
    agent = bare_agent(code_dir, str(snapshot))

    first, _client = initialize_with_capture(monkeypatch, agent)
    first_text = (
        Path(first["plugins"][0]["path"]) / "skills" / "stable-skill" / "SKILL.md"
    ).read_text()
    shutil.rmtree(live)
    second, _client = initialize_with_capture(monkeypatch, agent)
    second_text = (
        Path(second["plugins"][0]["path"]) / "skills" / "stable-skill" / "SKILL.md"
    ).read_text()

    assert "FROZEN INSTRUCTIONS" in first_text
    assert "EDITED LIVE INSTRUCTIONS" not in first_text
    assert second_text == first_text
    assert second["system_prompt"].count("CONFIGURED EXECUTION SKILLS") == 1
    assert second["skills"] == first["skills"]
    for variable in (
        "CLAUDE_CODE_SKILLS_PATH", "CLAUDE_CODE_SKILLS_DIR", "CLAUDE_SKILLS_PATH",
    ):
        assert variable not in os.environ
    agent._cleanup_execution_skills_runtime()


def test_unsafe_or_malformed_snapshots_fail_without_fallback(monkeypatch, tmp_path):
    project_root, code_dir, snapshot = make_project(tmp_path)
    with pytest.raises(ExecutionSkillsConfigurationError, match="does not exist"):
        bare_agent(code_dir, str(snapshot / "missing"))._configure_execution_skills()

    real_snapshot = project_root / "real-skills"
    real_snapshot.mkdir()
    make_skill(real_snapshot, "linked-skill", "Always applies.")
    linked_path = snapshot.parent / "linked.execution_skills"
    linked_path.symlink_to(real_snapshot, target_is_directory=True)
    with pytest.raises(ExecutionSkillsConfigurationError, match="symbolic link"):
        bare_agent(code_dir, str(linked_path))._configure_execution_skills()

    _other_root, _other_code, other_snapshot = make_project(tmp_path, "foreign project")
    make_skill(other_snapshot, "foreign-skill", "Always applies.")
    escaped = bare_agent(code_dir, str(other_snapshot))
    with pytest.raises(ExecutionSkillsConfigurationError, match="outside"):
        escaped._configure_execution_skills()
    assert not escaped._configured_execution_skills

    malformed = (
        project_root / "checkpoint" / "checkpoints" / "round_2" / ".execution_skills"
    )
    malformed.mkdir(parents=True)
    (malformed / "not-a-skill.txt").write_text("bad\n")
    with pytest.raises(ExecutionSkillsConfigurationError, match="malformed"):
        bare_agent(code_dir, str(malformed))._configure_execution_skills()

    linked_tree = (
        project_root / "checkpoint" / "checkpoints" / "round_3" / ".execution_skills"
    )
    linked_tree.mkdir(parents=True)
    unsafe = make_skill(linked_tree, "unsafe-skill", "Always applies.")
    (unsafe / "escaped-reference").symlink_to(real_snapshot, target_is_directory=True)
    with pytest.raises(ExecutionSkillsConfigurationError, match="symbolic links"):
        bare_agent(code_dir, str(linked_tree))._configure_execution_skills()


def test_round_without_skills_keeps_existing_sdk_options(monkeypatch, tmp_path):
    _root, code_dir, _snapshot = make_project(tmp_path)
    captured, _client = initialize_with_capture(monkeypatch, bare_agent(code_dir, None))
    assert all(key not in captured for key in ("skills", "plugins", "setting_sources", "add_dirs"))
    assert "CONFIGURED EXECUTION SKILLS" not in captured["system_prompt"]


def test_discovery_and_skill_tool_events_are_authoritative(monkeypatch, tmp_path):
    project_root, code_dir, snapshot = make_project(tmp_path)
    agent = bare_agent(code_dir, str(snapshot))
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:observable-skill"
    agent._configured_execution_skills = [{"runtime_name": runtime_name}]
    observed = []
    agent._reset_execution_skill_enforcement([runtime_name])
    agent.execution_tracker = SimpleNamespace(
        log_skill_discovery=lambda names: observed.extend(names)
    )
    agent._observe_execution_skills_discovery(SystemMessage([runtime_name]))
    assert observed == [runtime_name]
    assert agent._execution_skill_loaded_by_context["primary"] == set()
    assert not any(
        action in {"execution_skill_requested", "execution_skill_loaded"}
        for action, _details in agent.logger.actions
    )

    agent._execution_skills_discovery_confirmed = False
    with pytest.raises(ExecutionSkillsConfigurationError, match="not discovered"):
        agent._observe_execution_skills_discovery(SystemMessage([]))

    agent.execution_tracker = None
    bad_input = {
        "tool_name": "Skill",
        "tool_input": {"skill": runtime_name},
    }
    asyncio.run(agent._execution_skills_pre_tool_hook(bad_input, "tool-1", None))
    asyncio.run(agent._execution_skills_post_tool_hook(
        {
            "tool_name": "Skill",
            "tool_response": {"content": "loaded"},
        },
        "tool-1",
        None,
    ))
    assert agent._execution_skill_loaded_by_context["primary"] == set()

    asyncio.run(agent._execution_skills_pre_tool_hook(bad_input, "tool-2", None))
    asyncio.run(agent._execution_skills_post_tool_hook(
        {
            "tool_name": "Skill",
            "tool_response": {
                "success": True,
                "commandName": runtime_name,
            },
        },
        "tool-2",
        None,
    ))
    assert agent._execution_skill_loaded_by_context["primary"] == {runtime_name}
    events = [action for action, _details in agent.logger.actions]
    assert events.count("execution_skill_requested") == 2
    assert events.count("execution_skill_failed") == 1
    assert events.count("execution_skill_loaded") == 1


    bad_metadata = (
        project_root / "checkpoint" / "checkpoints" / "round_4" / ".execution_skills"
    )
    bad_skill = bad_metadata / "bad-metadata"
    bad_skill.mkdir(parents=True)
    (bad_skill / "SKILL.md").write_text("---\nname: bad-metadata\n---\n")
    with pytest.raises(ExecutionSkillsConfigurationError, match="metadata"):
        bare_agent(code_dir, str(bad_metadata))._configure_execution_skills()

    unreadable = (
        project_root / "checkpoint" / "checkpoints" / "round_5" / ".execution_skills"
    )
    unreadable_skill = make_skill(unreadable, "unreadable-skill", "Always applies.")
    unreadable_file = unreadable_skill / "SKILL.md"
    real_access = execution_agent_cc.os.access
    monkeypatch.setattr(
        execution_agent_cc.os,
        "access",
        lambda path, mode: (
            False if Path(path) == unreadable_file else real_access(path, mode)
        ),
    )
    with pytest.raises(ExecutionSkillsConfigurationError, match="unreadable"):
        bare_agent(code_dir, str(unreadable))._configure_execution_skills()


def test_tracker_persists_six_distinct_skill_diagnostics_without_bodies():
    runtime_name = f"{EXECUTION_SKILLS_PLUGIN_NAME}:observable-skill"
    tracker = ExecutionSDKTracker(logger=CaptureLogger())
    tracker.message_log_file = io.StringIO()
    tracker.transcript_file = io.StringIO()
    observable = []
    tracker.log_observable = observable.append
    tracker.log_skill_configuration(
        ["/frozen/round/.execution_skills"], ["observable-skill"]
    )
    tracker.log_skill_discovery([runtime_name])
    tracker.log_execution_skill_event(
        "execution_skill_requested", runtime_name, "primary"
    )
    tracker.log_execution_skill_event(
        "execution_skill_loaded", runtime_name, "primary"
    )
    tracker.log_execution_skill_event(
        "execution_skill_failed", runtime_name, "primary", "scripted failure"
    )
    tracker.log_missing_required_skill_load([runtime_name], "primary", 2)

    lines = tracker.message_log_file.getvalue().splitlines()
    payloads = [json.loads(line) for line in lines]
    assert [payload["event"] for payload in payloads] == [
        "execution_skills_configured",
        "execution_skills_runtime_discovered",
        "execution_skill_requested",
        "execution_skill_loaded",
        "execution_skill_failed",
        "execution_skills_missing_required_load",
    ]
    assert payloads[2]["runtime_name"] == runtime_name
    assert payloads[3]["runtime_name"] == runtime_name
    assert payloads[-1]["runtime_names"] == [runtime_name]
    assert f"execution_skill_requested: {runtime_name}" in observable
    assert f"execution_skill_loaded: {runtime_name}" in observable
    serialized = tracker.message_log_file.getvalue()
    assert "SKILL.md" not in serialized
    assert "full instruction body" not in serialized


def test_backend_cli_and_config_preserve_execution_skills_path(tmp_path):
    _root, code_dir, snapshot = make_project(tmp_path)
    parsed = create_run_parser().parse_args(
        ["--execution_skills_path", str(snapshot)]
    )
    config = set_execution_agent_config(
        parsed,
        dataset_description="Synthetic dataset",
        working_directory=str(code_dir),
        analysis_name="synthetic-analysis",
    )
    assert config.execution_skills_path == str(snapshot)
