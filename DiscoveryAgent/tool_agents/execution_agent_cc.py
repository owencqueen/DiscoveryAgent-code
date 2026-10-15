"""
Claude Code Execution Agent

This module implements an ExecutionAgent that uses the actual Claude Code SDK for autonomous analysis.
Instead of manually specifying workflow steps, it lets Claude Code handle the entire
analysis process with real coding capabilities.
"""

import os
import sys
import asyncio
import base64
import json
import subprocess
import shutil
import inspect
import re
import stat
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from ..config import LLMConfig, get_slurm_resource_info
from .base import AgentToolConfig
from ..utils.logger import Logger
import datetime
import anthropic
from ..prompts.claude_md_generator_command import CLAUDEMD_prompt
from ..prompts.execution_agent import (
    CODEBASE_STRUCTURE_PROMPT,
    EXECUTION_AGENT_SYSTEM_PROMPT,
    EXECUTION_AGENT_SYSTEM_PROMPT_SUB_AGENTS,
    SANDBOX_PATH_PROMPT,
)
from ..prompts.cc_sub_agents import SUB_AGENT_CONFIGS
from .execution_sdk_tracker import ExecutionSDKTracker

from .base import BaseAgent

# Configure Claude Code environment BEFORE importing SDK
from ..config import setup_claude_code_environment
from .security.kill_blocker import process_protection_hook
from .security.sandbox_enforcement import (
    create_sandbox_enforcement_hook,
    create_sandbox_posttool_guard,
)
from .security.url_blocker import create_url_blocker_hooks

claude_cli_path = setup_claude_code_environment()

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, AgentDefinition
CLAUDE_CODE_AVAILABLE = True


def _create_claude_agent_options(options_kwargs: Dict[str, Any]):
    """Construct SDK options, adding only an intentional validated override."""
    resolved_kwargs = dict(options_kwargs)
    if claude_cli_path is not None:
        resolved_kwargs["cli_path"] = claude_cli_path
    return ClaudeAgentOptions(**resolved_kwargs)


REPORT_SUMMARY_PROMPT = """
You will be shown the raw trace of an agentic coding agent's output.
Your job is to generate a concise summary of the agent's findings through their experimentation, specifically as is relevant to the query.
Your report should focus on the agent's findings and results on successful experiments - ignore the experiments that failed or the retries of experiments.
Report the methods and results that were implemented and found by the agent.
This output will be passed to an upstream agent that will interpret it and use it to guide scientific findings, so ensure that you are comprehensive but also concise and pointed.
Include references to specific image files and their paths when necessary, including an annotation to the properties of that image. 
"""

REPORT_SUMMARY_PROMPT_v2 = \
"""Now that you've performed the analysis, I need you to put together a comprehensive summary of your findings. 
Your report should focus on the work that you just did, don't add any new analyses unless you need clarification on some results.
Your report is going to be used by a downstream agent that will intepret your results, so make sure it's comprehensive and covers all statistics and characterizations that you revealed that is relevant to the original query.
Make sure your report is formatted well, and output it into a file at {report_path}

IMPORTANT: Do not run new analyses, ONLY summarize and describe the analysis that's already been done.
IMPORTANT: If you were not able to complete the analysis as originally requested, you should note the parts that you weren't able to complete.
IMPORTANT: You must write the file to this exact location: {report_path}
"""

class ExecutionAgentResultError(RuntimeError):
    """Content-free terminal category for an unusable ExecutionAgent result."""

    category = "execution_agent_result_invalid"

    def __init__(self):
        super().__init__(self.category)


class ExecutionAgentRefusalError(ExecutionAgentResultError):
    category = "execution_agent_model_refusal"


class ExecutionAgentEmptyResponseError(ExecutionAgentResultError):
    category = "execution_agent_empty_response"


class ExecutionAgentMissingFinalResponseError(ExecutionAgentResultError):
    category = "execution_agent_final_response_missing"


class ExecutionAgentMalformedReportError(ExecutionAgentResultError):
    category = "execution_agent_report_missing_or_malformed"


class ExecutionAgentSDKResponseError(ExecutionAgentResultError):
    category = "execution_agent_sdk_exception"


class ExecutionSkillsConfigurationError(RuntimeError):
    """A declared frozen skill snapshot cannot be loaded safely."""


class ExecutionSkillsRequiredLoadError(RuntimeError):
    """Configured execution skills were not authoritatively loaded."""

    category = "execution_skills_missing_required_load"

    def __init__(self, runtime_names: List[str], context_name: str = "primary"):
        self.runtime_names = list(runtime_names)
        self.context_name = context_name
        super().__init__(
            f"{self.category}: {', '.join(self.runtime_names)} (context={context_name})"
        )


EXECUTION_SKILLS_PLUGIN_NAME = "discoveryagent-round-skills"
_EXECUTION_SKILL_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

@dataclass
class ExecutionAgentCCConfig(AgentToolConfig):
    """Configuration for ExecutionAgentCC."""
    llm_config: LLMConfig = None
    analysis_name: str = None
    dataset_description: str = ""
    log_dir: str = "."
    # Additional parameters for ExecutionAgentCC
    readme_truncate_length: Optional[int] = None
    # None means the agent runs until it decides the task is done. The step cap
    # existed because earlier models were poor at recognising completion.
    max_turns: Optional[int] = None
    working_directory: Optional[str] = None
    manual_file_tracking: bool = False
    max_turns_query_execution: Optional[int] = None
    separate_report_step: bool = False
    disable_file_moves: bool = False
    claude_md_generator: bool = False
    execute_context_persistence: bool = False
    codebase_structure_prompt: Optional[str] = None
    sub_agent_architecture: bool = False
    # Execution logging (tool hooks + message stream) into checkpoint query directories
    log_execution_agent: bool = True
    # Optional path to Claude Code skills for this execution session
    execution_skills_path: Optional[Union[str, List[str]]] = None
    # Sandbox enforcement: restrict file operations to working_directory
    sandbox_enforcement: bool = True
    allowed_read_paths: Optional[List[str]] = None
    # URL blocker: deny network fetches to specific hosts / URL substrings.
    # ``None`` uses the module defaults (ProteinGym test-label sources). Pass
    # ``[]`` to disable that category. Set ``url_blocker_enabled=False`` to skip
    # the hook entirely.
    url_blocker_enabled: bool = True
    blocked_url_hosts: Optional[List[str]] = None
    blocked_url_substrings: Optional[List[str]] = None

class ExecutionAgentCC(BaseAgent):
    """
    Claude Code SDK-based execution agent that provides autonomous analysis with real coding capabilities.
    Inherits from BaseAgent and follows its API.
    """
    def _direct_print(self, message: str):
        """
        Print directly to stdout bypassing the StreamToLogger to avoid duplicate output.
        Used for real-time streaming messages from Claude Code.
        """
        sys.__stdout__.write(message + '\n')
        sys.__stdout__.flush()
        tracker = getattr(self, "execution_tracker", None)
        if tracker is not None:
            tracker.log_observable(message)

    def _resolve_claude_code_model(self) -> Optional[str]:
        """
        Resolve the concrete Claude Code model name from execution config.

        We source this from llm_config so CLI/config-file overrides on
        --execution_model propagate to the Claude Code SDK options.
        """
        model_name = getattr(self.llm_config, "model_name", None)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
        return None

    @staticmethod
    def _report_failure(messages: List[Any]) -> ExecutionAgentResultError:
        """Classify a completed SDK stream without logging response contents."""
        if not messages:
            return ExecutionAgentEmptyResponseError()
        terminal = [message for message in messages if type(message).__name__ == "ResultMessage"]
        if not terminal:
            return ExecutionAgentMissingFinalResponseError()
        result = terminal[-1]
        metadata = " ".join(
            str(getattr(result, field, "") or "").lower()
            for field in ("stop_reason", "subtype")
        )
        if "refusal" in metadata or "refused" in metadata:
            return ExecutionAgentRefusalError()
        return ExecutionAgentMalformedReportError()

    def _step_budget_section(self) -> str:
        """Tell the agent its step budget, but only when it actually has one.

        With no cap there is nothing true to say, and telling a current model to
        ration steps makes it stop early rather than finish the analysis.
        """
        if not self.max_turns_query_execution:
            return ""
        return (
            "⏱️ BUDGET:\n"
            f"You are only allowed to take a total of {self.max_turns_query_execution} steps to "
            "complete this task. Please be efficient and mindful of this budget. Plan your "
            "approach carefully to maximize the value of each step.\n"
        )

    def _resolve_sub_agent_model_alias(self) -> str:
        """
        Resolve the model selector for Claude Code sub-agents.

        Use "inherit" so sub-agents always run on the same model as the main
        Claude Code execution agent configured for this session.
        """
        return "inherit"

    def _parse_skill_frontmatter(self, skill_md_path: Path) -> Dict[str, str]:
        """
        Parse top YAML-like frontmatter from SKILL.md and extract key metadata.

        Expected format:
        ---
        name: ...
        description: ...
        ---
        """
        try:
            content = skill_md_path.read_text(encoding="utf-8")
        except Exception:
            return {}

        match = re.search(r"^\s*---\s*\n(.*?)\n---\s*", content, flags=re.DOTALL)
        if not match:
            return {}

        frontmatter = match.group(1)
        parsed: Dict[str, str] = {}
        for raw_line in frontmatter.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                parsed[key] = value
        return parsed

    @staticmethod
    def _raw_execution_skills_paths(configured: Any) -> List[str]:
        if isinstance(configured, str):
            return [part.strip() for part in configured.split(",") if part.strip()]
        if isinstance(configured, (list, tuple, set)):
            return [str(part).strip() for part in configured if str(part).strip()]
        return [str(configured).strip()] if str(configured).strip() else []

    def _validate_round_snapshot_scope(self, source_path: Path) -> None:
        """Keep frontend round snapshots inside their owning project and round."""
        if source_path.name != ".execution_skills":
            return
        working_directory = Path(self.working_directory).resolve(strict=True)
        round_directory = source_path.parent
        expected_project_root = working_directory.parent
        if not (
            working_directory.name == "code"
            and re.fullmatch(r"round_\d+", round_directory.name)
            and round_directory.parent.name == "checkpoints"
            and round_directory.parent.parent.name == "checkpoint"
            and round_directory.parent.parent.parent == expected_project_root
        ):
            raise ExecutionSkillsConfigurationError(
                "The declared execution skill snapshot is outside the active project's round."
            )

    @staticmethod
    def _validate_skill_tree(source_path: Path) -> None:
        """Reject links, special files, and unreadable entries before copying."""
        if not os.access(source_path, os.R_OK | os.X_OK):
            raise ExecutionSkillsConfigurationError(
                "The declared execution skill snapshot cannot be read safely."
            )
        for root, directory_names, file_names in os.walk(source_path, followlinks=False):
            root_path = Path(root)
            for name in [*directory_names, *file_names]:
                entry = root_path / name
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ExecutionSkillsConfigurationError(
                        "The declared execution skill snapshot cannot be read safely."
                    ) from exc
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ExecutionSkillsConfigurationError(
                        "The declared execution skill snapshot may not contain symbolic links."
                    )
                if not (stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISREG(entry_stat.st_mode)):
                    raise ExecutionSkillsConfigurationError(
                        "The declared execution skill snapshot contains an unsupported file type."
                    )
                required_mode = os.R_OK | os.X_OK if stat.S_ISDIR(entry_stat.st_mode) else os.R_OK
                if not os.access(entry, required_mode):
                    raise ExecutionSkillsConfigurationError(
                        "The declared execution skill snapshot contains an unreadable entry."
                    )

    def _normalize_execution_skills_paths(self) -> List[Path]:
        """
        Normalize execution_skills_path into a list of existing directory paths.

        Supported formats:
        - string path
        - comma-separated string paths
        - list/tuple/set of paths
        """
        configured = getattr(self.config, "execution_skills_path", None)
        if not configured:
            return []

        raw_paths = self._raw_execution_skills_paths(configured)
        if not raw_paths:
            raise ExecutionSkillsConfigurationError(
                "The declared execution skill snapshot path is empty."
            )

        normalized_paths: List[Path] = []
        seen: set[str] = set()
        for raw_path in raw_paths:
            candidate = Path(raw_path).expanduser()
            if os.path.lexists(candidate) and candidate.is_symlink():
                raise ExecutionSkillsConfigurationError(
                    "The declared execution skill snapshot may not be a symbolic link."
                )
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                raise ExecutionSkillsConfigurationError(
                    "The declared execution skill snapshot does not exist or cannot be read."
                ) from exc
            if not resolved.is_dir():
                raise ExecutionSkillsConfigurationError(
                    "The declared execution skill snapshot is not a directory."
                )
            self._validate_round_snapshot_scope(resolved)
            self._validate_skill_tree(resolved)
            resolved_str = str(resolved)
            if resolved_str in seen:
                continue
            seen.add(resolved_str)
            normalized_paths.append(resolved)
        return normalized_paths

    def _discover_execution_skills(self) -> List[Dict[str, str]]:
        """
        Discover skills from execution_skills_path and return metadata list.
        """
        source_paths = self._normalize_execution_skills_paths()
        if not source_paths:
            return []

        skill_dirs: List[Path] = []
        seen_skill_dirs: set[str] = set()
        seen_skill_ids: set[str] = set()
        for source_path in source_paths:
            candidate_skill_dirs: List[Path] = []
            if (source_path / "SKILL.md").is_file():
                candidate_skill_dirs = [source_path]
            else:
                for child in sorted(source_path.iterdir()):
                    if not child.is_dir() or child.is_symlink() or not (child / "SKILL.md").is_file():
                        raise ExecutionSkillsConfigurationError(
                            "The declared execution skill snapshot contains a malformed skill entry."
                        )
                    candidate_skill_dirs.append(child)

            for skill_dir in candidate_skill_dirs:
                resolved_skill_dir = skill_dir.resolve()
                resolved_skill_dir_str = str(resolved_skill_dir)
                if resolved_skill_dir_str in seen_skill_dirs:
                    continue
                if not _EXECUTION_SKILL_ID_RE.fullmatch(skill_dir.name):
                    raise ExecutionSkillsConfigurationError(
                        "Execution skill directory names must be lowercase kebab-case identifiers."
                    )
                if skill_dir.name in seen_skill_ids:
                    raise ExecutionSkillsConfigurationError(
                        f"Execution skill identifier collision: {skill_dir.name}."
                    )
                seen_skill_ids.add(skill_dir.name)
                seen_skill_dirs.add(resolved_skill_dir_str)
                skill_dirs.append(resolved_skill_dir)

        discovered: List[Dict[str, str]] = []
        for skill_dir in skill_dirs:
            skill_md = skill_dir / "SKILL.md"
            frontmatter = self._parse_skill_frontmatter(skill_md)
            name = (frontmatter.get("name") or "").strip()
            description = (frontmatter.get("description") or "").strip()
            if not name or not description:
                raise ExecutionSkillsConfigurationError(
                    f"Execution skill {skill_dir.name!r} needs readable name and description metadata."
                )
            discovered.append({
                "id": skill_dir.name,
                "name": name,
                "description": description,
                "path": str(skill_dir),
                "runtime_name": f"{EXECUTION_SKILLS_PLUGIN_NAME}:{skill_dir.name}",
            })
        if not discovered:
            raise ExecutionSkillsConfigurationError(
                "The declared execution skill snapshot contains no valid skills."
            )
        return discovered

    def _cleanup_execution_skills_runtime(self) -> None:
        runtime_directory = getattr(self, "_execution_skills_runtime_directory", None)
        if runtime_directory is not None:
            runtime_directory.cleanup()
            self._execution_skills_runtime_directory = None

    def _configure_execution_skills(self) -> Optional[Dict[str, Any]]:
        """Build a local plugin view backed only by the declared frozen snapshot."""
        if not getattr(self.config, "execution_skills_path", None):
            self._cleanup_execution_skills_runtime()
            self._configured_execution_skills = []
            self._execution_skills_source_paths = []
            return None
        normalized_paths = self._normalize_execution_skills_paths()
        discovered_skills = self._discover_execution_skills()
        discovered_skill_names = [skill["name"] for skill in discovered_skills]

        required_sdk_fields = {"skills", "plugins", "setting_sources", "add_dirs"}
        option_fields = set(inspect.signature(ClaudeAgentOptions).parameters)
        missing_fields = sorted(required_sdk_fields - option_fields)
        if missing_fields:
            raise ExecutionSkillsConfigurationError(
                "The installed Claude Agent SDK cannot safely load frozen execution skills; "
                f"missing options: {', '.join(missing_fields)}."
            )

        self._cleanup_execution_skills_runtime()
        runtime_directory = tempfile.TemporaryDirectory(prefix="discoveryagent-round-skills-")
        runtime_root = Path(runtime_directory.name) / EXECUTION_SKILLS_PLUGIN_NAME
        try:
            (runtime_root / ".claude-plugin").mkdir(parents=True)
            (runtime_root / ".claude-plugin" / "plugin.json").write_text(
                json.dumps({
                    "name": EXECUTION_SKILLS_PLUGIN_NAME,
                    "version": "1.0.0",
                    "description": "Immutable DiscoveryAgent execution skills for one frozen round.",
                    "author": {"name": "DiscoveryAgent"},
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            skills_root = runtime_root / "skills"
            skills_root.mkdir()
            for skill in discovered_skills:
                shutil.copytree(
                    skill["path"],
                    skills_root / skill["id"],
                    symlinks=False,
                    copy_function=shutil.copy2,
                )
        except Exception as exc:
            runtime_directory.cleanup()
            raise ExecutionSkillsConfigurationError(
                "The frozen execution skill snapshot could not be prepared for the agent runtime."
            ) from exc

        self._execution_skills_runtime_directory = runtime_directory
        self._configured_execution_skills = discovered_skills
        self._execution_skills_source_paths = [str(path) for path in normalized_paths]
        self._execution_skills_discovery_confirmed = False
        self.logger.log_action(
            "execution_skills_configured",
            f"Frozen paths: {[str(path) for path in normalized_paths]} | "
            f"skills={discovered_skill_names}",
        )
        self._direct_print(
            f"🧩 Execution skills configured: {', '.join(discovered_skill_names)}"
        )
        return {
            "source_paths": normalized_paths,
            "plugin_root": runtime_root,
            "skills": discovered_skills,
            "runtime_names": [skill["runtime_name"] for skill in discovered_skills],
        }

    @staticmethod
    def _execution_skills_policy(discovered_skills: List[Dict[str, str]]) -> str:
        if not discovered_skills:
            return ""
        skill_lines = [
            f"- {skill['name']} (invoke as {skill['runtime_name']}): {skill['description']}"
            for skill in discovered_skills
        ]
        return (
            "\n\nCONFIGURED EXECUTION SKILLS (immutable round snapshot):\n"
            + "\n".join(skill_lines)
            + "\n\nProject selection is an explicit instruction to load every listed skill. "
            "The backend will require the Skill tool with each exact qualified invocation name before "
            "computation or delegation; discovery alone is not execution. After loading, follow each "
            "skill completely, including its own relevance rules, scripts, references, templates, "
            "receipts, verification tokens, validation, and required artifacts. Resolve resources and "
            "executable paths from the loaded immutable runtime directory, never a live skill copy. "
            "Propagate the same requirements into delegated analytical work. If a configured skill "
            "cannot be loaded, stop. Never claim a skill ran merely because it was listed or discovered."
        )

    @staticmethod
    def _apply_execution_skills_options(
        options: Dict[str, Any], runtime: Optional[Dict[str, Any]],
    ) -> None:
        if runtime is None:
            return
        plugin_root = str(runtime["plugin_root"])
        options.update({
            "plugins": [{"type": "local", "path": plugin_root}],
            "skills": list(runtime["runtime_names"]),
            "setting_sources": ["project"],
            "add_dirs": [plugin_root],
        })

    def _observe_execution_skills_discovery(self, message: Any) -> None:
        expected = [
            skill["runtime_name"] for skill in getattr(self, "_configured_execution_skills", [])
        ]
        if not expected or type(message).__name__ != "SystemMessage":
            return
        data = getattr(message, "data", None)
        if not isinstance(data, dict) or data.get("subtype") != "init":
            return
        available = [str(name) for name in (data.get("skills") or [])]
        missing = [name for name in expected if name not in available]
        if missing:
            warning = (
                "Configured execution skills were not discovered by the runtime: "
                + ", ".join(missing)
            )
            self.logger.log_error(warning)
            raise ExecutionSkillsConfigurationError(warning)
        if not getattr(self, "_execution_skills_discovery_confirmed", False):
            self._execution_skills_discovery_confirmed = True
            self.logger.log_action(
                "execution_skills_runtime_discovered",
                f"Runtime confirmed configured skills: {expected}",
            )
            if self.execution_tracker is not None:
                self.execution_tracker.log_skill_discovery(expected)

    def _reset_execution_skill_enforcement(self, runtime_names: List[str]) -> None:
        """Reset authoritative load state for a newly constructed SDK session."""
        self._required_execution_skill_names = tuple(runtime_names)
        self._execution_skill_loaded_by_context: Dict[str, set] = {"primary": set()}
        self._execution_skill_pending: Dict[tuple, str] = {}
        self._execution_skill_context_labels: Dict[str, str] = {"primary": "primary"}
        self._execution_skill_analytical_contexts = set()
        self._execution_skill_subagent_stop_attempts: Dict[str, int] = {}

    @staticmethod
    def _execution_skill_context(hook_input: Any) -> tuple:
        if not isinstance(hook_input, dict):
            return "primary", "primary", None
        agent_id = hook_input.get("agent_id")
        agent_type = hook_input.get("agent_type")
        if agent_id:
            context_key = f"subagent:{agent_id}"
            return context_key, f"{agent_type or 'subagent'}:{agent_id}", agent_type
        return "primary", "primary", None

    def _required_execution_skills(self) -> List[str]:
        return list(getattr(self, "_required_execution_skill_names", ()))

    def _missing_execution_skills(self, context_key: str) -> List[str]:
        loaded = self._execution_skill_loaded_by_context.setdefault(context_key, set())
        return [name for name in self._required_execution_skills() if name not in loaded]

    def _log_execution_skill_event(
        self,
        event: str,
        runtime_name: str,
        context_name: str,
        reason: Optional[str] = None,
    ) -> None:
        details = f"skill={runtime_name} context={context_name}"
        if reason:
            details += f" reason={reason}"
        self.logger.log_action(event, details)
        if self.execution_tracker is not None:
            self.execution_tracker.log_execution_skill_event(
                event, runtime_name, context_name, reason
            )

    def _log_missing_execution_skills(
        self, runtime_names: List[str], context_name: str, attempt: int
    ) -> None:
        details = (
            f"skills={runtime_names} context={context_name} attempt={attempt}"
        )
        self.logger.log_action("execution_skills_missing_required_load", details)
        if self.execution_tracker is not None:
            self.execution_tracker.log_missing_required_skill_load(
                runtime_names, context_name, attempt
            )

    @staticmethod
    def _deny_tool_for_missing_execution_skills(reason: str) -> Dict[str, Any]:
        return {
            "continue_": True,
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            },
        }

    def _execution_skill_load_instruction(self, context_name: str) -> str:
        names = self._required_execution_skills()
        return (
            "Before any computation, file access, web access, or delegation in this "
            f"operative context ({context_name}), invoke the Skill tool once for every "
            "exact qualified name below and wait for each successful result:\n"
            + "\n".join(f"- {name}" for name in names)
            + "\nDiscovery or availability is not a load. Do not use unqualified or live-project names."
        )

    async def _execution_skills_pre_tool_hook(self, hook_input, tool_use_id, context):
        required = self._required_execution_skills()
        if not required:
            return {"continue_": True}

        tool_name = hook_input.get("tool_name") if isinstance(hook_input, dict) else None
        tool_input = hook_input.get("tool_input") if isinstance(hook_input, dict) else None
        context_key, context_name, agent_type = self._execution_skill_context(hook_input)
        self._execution_skill_context_labels[context_key] = context_name
        loaded = self._execution_skill_loaded_by_context.setdefault(context_key, set())

        if tool_name == "Skill":
            requested = None
            if isinstance(tool_input, dict):
                requested = tool_input.get("skill")
            requested_name = str(requested or "unknown")
            self._log_execution_skill_event(
                "execution_skill_requested", requested_name, context_name
            )

            if agent_type and agent_type not in {"execution_agent", "skills_agent"}:
                reason = "Project execution skills are not exposed to this delegated agent."
            elif requested_name not in required:
                reason = (
                    "The Skill request does not match a configured frozen qualified name. "
                    f"Required names: {', '.join(required)}"
                )
            elif requested_name in loaded:
                reason = "This exact skill is already loaded in the operative context."
            else:
                self._execution_skill_pending[(context_key, str(tool_use_id))] = requested_name
                return {"continue_": True}

            self._log_execution_skill_event(
                "execution_skill_failed", requested_name, context_name, reason
            )
            return self._deny_tool_for_missing_execution_skills(reason)

        missing = self._missing_execution_skills(context_key)
        if missing:
            reason = (
                "Blocked substantive tool use until every configured execution skill is "
                "authoritatively loaded with the Skill tool. Missing exact names: "
                + ", ".join(missing)
            )
            self._log_missing_execution_skills(missing, context_name, 0)
            return self._deny_tool_for_missing_execution_skills(reason)

        if tool_name in {"Agent", "Task"} and isinstance(tool_input, dict):
            subagent_type = tool_input.get("subagent_type")
            if subagent_type in {"execution_agent", "skills_agent"}:
                updated_input = dict(tool_input)
                prompt = str(updated_input.get("prompt") or "")
                updated_input["prompt"] = (
                    self._execution_skill_load_instruction(str(subagent_type))
                    + "\n\nAfter those loads succeed, perform the delegated task:\n"
                    + prompt
                )
                # Wait for the enforced worker context; do not let background
                # delegation outlive the root completion assertion.
                updated_input["run_in_background"] = False
                return {
                    "continue_": True,
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": updated_input,
                    },
                }
            if (
                self.sub_agent_architecture
                and subagent_type
                and subagent_type not in {"documentation_agent", "claude_md_agent"}
            ):
                return self._deny_tool_for_missing_execution_skills(
                    "Delegated analytical work must use execution_agent or skills_agent "
                    "so configured project-skill enforcement remains active."
                )

        return {"continue_": True}

    async def _execution_skills_post_tool_hook(self, hook_input, tool_use_id, context):
        if not self._required_execution_skills():
            return {"continue_": True}
        tool_name = hook_input.get("tool_name") if isinstance(hook_input, dict) else None
        if tool_name != "Skill":
            return {"continue_": True}

        context_key, context_name, _ = self._execution_skill_context(hook_input)
        pending_key = (context_key, str(tool_use_id))
        requested = self._execution_skill_pending.pop(pending_key, None)
        if requested is None:
            return {"continue_": True}

        response = hook_input.get("tool_response") if isinstance(hook_input, dict) else None
        succeeded = (
            isinstance(response, dict)
            and response.get("success") is True
            and response.get("commandName") == requested
        )
        if succeeded:
            self._execution_skill_loaded_by_context.setdefault(context_key, set()).add(requested)
            self._log_execution_skill_event(
                "execution_skill_loaded", requested, context_name
            )
        else:
            reason = "Skill result did not authoritatively confirm the exact qualified command."
            self._log_execution_skill_event(
                "execution_skill_failed", requested, context_name, reason
            )
        return {"continue_": True}

    async def _execution_skills_post_tool_failure_hook(
        self, hook_input, tool_use_id, context
    ):
        if not self._required_execution_skills():
            return {"continue_": True}
        context_key, context_name, _ = self._execution_skill_context(hook_input)
        requested = self._execution_skill_pending.pop(
            (context_key, str(tool_use_id)), None
        )
        if requested is not None:
            reason = "Skill tool execution failed."
            if isinstance(hook_input, dict) and hook_input.get("error"):
                reason = str(hook_input["error"])
            self._log_execution_skill_event(
                "execution_skill_failed", requested, context_name, reason
            )
        return {"continue_": True}

    async def _execution_skills_subagent_start_hook(
        self, hook_input, tool_use_id, context
    ):
        if not self._required_execution_skills():
            return {"continue_": True}
        context_key, context_name, agent_type = self._execution_skill_context(hook_input)
        if agent_type not in {"execution_agent", "skills_agent"}:
            return {"continue_": True}
        self._execution_skill_loaded_by_context.setdefault(context_key, set())
        self._execution_skill_context_labels[context_key] = context_name
        self._execution_skill_analytical_contexts.add(context_key)
        return {
            "continue_": True,
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "additionalContext": self._execution_skill_load_instruction(context_name),
            },
        }

    async def _execution_skills_subagent_stop_hook(
        self, hook_input, tool_use_id, context
    ):
        if not self._required_execution_skills():
            return {"continue_": True}
        context_key, context_name, agent_type = self._execution_skill_context(hook_input)
        if agent_type not in {"execution_agent", "skills_agent"}:
            return {"continue_": True}
        missing = self._missing_execution_skills(context_key)
        if not missing:
            return {"continue_": True}

        attempt = self._execution_skill_subagent_stop_attempts.get(context_key, 0) + 1
        self._execution_skill_subagent_stop_attempts[context_key] = attempt
        self._log_missing_execution_skills(missing, context_name, attempt)
        reason = (
            "The delegated analytical context cannot finish until it loads: "
            + ", ".join(missing)
        )
        if attempt == 1:
            return {"decision": "block", "reason": reason}
        return {"continue_": False, "stopReason": reason}

    def _assert_required_execution_skills_loaded(
        self, include_subagents: bool = True, attempt: int = 0
    ) -> None:
        if not self._required_execution_skills():
            return
        contexts = ["primary"]
        if include_subagents:
            contexts.extend(sorted(self._execution_skill_analytical_contexts))
        for context_key in contexts:
            missing = self._missing_execution_skills(context_key)
            if missing:
                context_name = self._execution_skill_context_labels.get(
                    context_key, context_key
                )
                self._log_missing_execution_skills(missing, context_name, attempt)
                raise ExecutionSkillsRequiredLoadError(missing, context_name)

    async def _load_required_execution_skills(self, client) -> List[str]:
        """Load selected skills in the same session before sending analytical work."""
        required = self._required_execution_skills()
        if not required or not self._missing_execution_skills("primary"):
            return []

        preflight_messages: List[str] = []
        for attempt in range(1, 3):
            missing = self._missing_execution_skills("primary")
            if not missing:
                return preflight_messages
            prompt = (
                "Mandatory execution-skill preflight. Use only the Skill tool now. "
                "Invoke each exact qualified name below once and wait for its result:\n"
                + "\n".join(f"- {name}" for name in missing)
                + "\nDo not compute, delegate, inspect files, browse, or claim completion."
            )
            if attempt > 1:
                prompt = (
                    "The prior preflight did not authoritatively load every required skill. "
                    + prompt
                )
            if self.execution_tracker is not None:
                self.execution_tracker.set_phase("execution_skill_preflight")
            await client.query(prompt)
            async for message in client.receive_response():
                preflight_messages.append(str(message))
                if self.execution_tracker is not None:
                    self.execution_tracker.log_stream_message(
                        message, phase="execution_skill_preflight"
                    )
                self._observe_execution_skills_discovery(message)

            missing = self._missing_execution_skills("primary")
            if not missing:
                return preflight_messages
            self._log_missing_execution_skills(missing, "primary", attempt)

        raise ExecutionSkillsRequiredLoadError(
            self._missing_execution_skills("primary"), "primary"
        )
    def config_to_prompt(self) -> str:
        """Build the normal helper prompt without exposing execution-only skills."""
        return super().config_to_prompt()
    
    def __init__(self, config: ExecutionAgentCCConfig, logger=None):
        super().__init__(config, logger)
        if not CLAUDE_CODE_AVAILABLE:
            raise ImportError("claude-code-sdk is required. Install with: pip install claude-code-sdk")
        # Use config object for all settings
        llm_config = config.llm_config
        analysis_name = config.analysis_name
        log_dir = config.log_dir
        readme_truncate_length = getattr(config, 'readme_truncate_length', None)
        max_turns = getattr(config, 'max_turns', None)
        working_directory = getattr(config, 'working_directory', None)
        manual_file_tracking = getattr(config, 'manual_file_tracking', False)
        max_turns_query_execution = getattr(config, 'max_turns_query_execution', None)
        separate_report_step = getattr(config, 'separate_report_step', False)
        disable_file_moves = getattr(config, 'disable_file_moves', False)
        claude_md_generator = getattr(config, 'claude_md_generator', False)
        execute_context_persistence = getattr(config, 'execute_context_persistence', True)
        # Set attributes as before
        self.llm_config = llm_config
        self.analysis_name = analysis_name
        self.log_dir = log_dir
        self.readme_truncate_length = readme_truncate_length
        # Resolve per-step max turns with backward-compatible fallback. Both
        # may be None, which means the agent runs until it finishes rather than
        # being cut off at an arbitrary step count.
        self.max_turns_query_execution = max_turns_query_execution if max_turns_query_execution is not None else max_turns
        self.max_turns = self.max_turns_query_execution
        self.separate_report_step = separate_report_step
        self.disable_file_moves = disable_file_moves
        self.claude_md_generator = claude_md_generator
        self.execute_context_persistence = execute_context_persistence
        self.api_provider = self.llm_config.api_provider
        self._api_client = None  # Lazy initialization in _make_api_call
        
        # Initialize execution context for storing variables
        self.execution_context = {}
        self.codebase_structure_prompt = getattr(config, 'codebase_structure_prompt', None)
        self.sub_agent_architecture = getattr(config, 'sub_agent_architecture', False)
        self.log_execution_agent = getattr(config, 'log_execution_agent', True)
        self.sandbox_enforcement = getattr(config, 'sandbox_enforcement', True)
        self.allowed_read_paths = getattr(config, 'allowed_read_paths', None)
        self.url_blocker_enabled = getattr(config, 'url_blocker_enabled', True)
        self.blocked_url_hosts = getattr(config, 'blocked_url_hosts', None)
        self.blocked_url_substrings = getattr(config, 'blocked_url_substrings', None)
        self.execution_tracker: Optional[ExecutionSDKTracker] = None
        self._execution_skills_runtime_directory = None
        self._configured_execution_skills: List[Dict[str, str]] = []
        self._execution_skills_source_paths: List[str] = []
        self._execution_skills_discovery_confirmed = False
        
        # Set up working directory - use project root or specified directory
        if working_directory:
            self.working_directory = Path(working_directory)
        else:
            # Default to the project root directory
            self.working_directory = Path.cwd()
        
        # Ensure working directory exists
        self.working_directory.mkdir(parents=True, exist_ok=True)
        
        # Debug output directory removed: no on-disk hypothesis agent reports will be created
        self.debug_dir = None
        
        # File tracking mode
        self.manual_file_tracking = manual_file_tracking
        
        # Track existing files for safety monitoring
        self.existing_files = self._get_existing_files()
        
        # Use passed logger or create a new one if not provided
        if self.logger is None:
            self.logger = Logger("claude_code_execution_agent", log_dir=log_dir)

        if self.log_execution_agent:
            self.execution_tracker = ExecutionSDKTracker(logger=self.logger)
        
        # Log file safety information
        self.logger.log_action("file_safety_check", f"Tracked {len(self.existing_files)} existing files for protection")
        # Debug setup logging removed since debug reports are disabled
        
        # Execution tracking (for compatibility with workflow reporting)
        self.execution_count = 0

        # CLAUDE.md should be in working directory (where template is copied) not output directory
        if self.claude_md_generator:
            self.claude_md_path = self.working_directory / "CLAUDE.md"
        else:
            self.claude_md_path = None
        
        # Initialize Claude Code client and options
        self.claude_code_options = None
        self.claude_client: Optional[ClaudeSDKClient] = None
        self._init_claude_code_session()
        # Initialization output - use direct print to avoid duplication
        self._direct_print(f"🤖 Claude Code ExecutionAgent initialized")
        self._direct_print(f"📁 Working directory: {self.working_directory}")
        self._direct_print(f"🛡️ File safety: {len(self.existing_files)} existing files protected")
        self._direct_print(f"📋 File tracking: {'Manual' if manual_file_tracking else 'Claude Code self-reporting'}")
        self._direct_print(f"📝 CLAUDE.md generation: {'Enabled' if self.claude_md_generator else 'Disabled'}")
        
        # Check if CLAUDE.md exists for context loading
        claude_md_status = "Not configured"
        if hasattr(self, 'claude_md_path') and self.claude_md_path is not None:
            if self.claude_md_path.exists():
                # Read file to get size for verification
                try:
                    with open(self.claude_md_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    claude_md_status = f"✅ Loaded from {self.claude_md_path} ({len(content)} chars, {len(content.split())} words)"
                except Exception as e:
                    claude_md_status = f"⚠️ Found at {self.claude_md_path} but couldn't read: {str(e)}"
            else:
                claude_md_status = f"❌ Path set to {self.claude_md_path} but file NOT FOUND"
        self._direct_print(f"📄 CLAUDE.md context: {claude_md_status}")
        
        self._direct_print(f"🔄 Context persistence: {'Enabled (maintains session across execute calls)' if self.execute_context_persistence else 'Disabled (resets session each execute call)'}")
        # Debug reports disabled; no report directory to display
        self._direct_print(f"📂 File organization: Disabled (using git/checkpoint managers)")

    def _format_codebase_structure_prompt(self) -> str:
        """Format codebase structure guidance, adding sandbox rules only when enabled."""
        prompt = ""
        if self.codebase_structure_prompt:
            prompt = CODEBASE_STRUCTURE_PROMPT.format(
                codebase_structure_prompt=self.codebase_structure_prompt,
                working_directory=self.working_directory,
            )

        if self.sandbox_enforcement:
            prompt += SANDBOX_PATH_PROMPT.format(
                working_directory=self.working_directory,
            )

        return prompt

    def _init_claude_code_session(self):

        # Set up Claude Code options
        system_prompt = self._get_system_prompt()

        configured_skills_runtime = self._configure_execution_skills()
        discovered_skills = (
            configured_skills_runtime["skills"] if configured_skills_runtime else []
        )
        runtime_skill_names = (
            list(configured_skills_runtime["runtime_names"])
            if configured_skills_runtime else []
        )
        self._reset_execution_skill_enforcement(runtime_skill_names)
        system_prompt += self._execution_skills_policy(discovered_skills)
        if isinstance(self.allowed_read_paths, str):
            runtime_read_paths = [
                path.strip() for path in self.allowed_read_paths.split(",") if path.strip()
            ]
        else:
            runtime_read_paths = list(self.allowed_read_paths or [])
        if configured_skills_runtime is not None:
            runtime_read_paths.append(str(configured_skills_runtime["plugin_root"]))

        pre_tool_hooks = [
            HookMatcher(matcher=None, hooks=[self._execution_skills_pre_tool_hook]),
            HookMatcher(matcher="Bash", hooks=[process_protection_hook]),
        ]
        post_tool_hooks = [
            HookMatcher(matcher=None, hooks=[self._execution_skills_post_tool_hook])
        ]
        post_tool_failure_hooks = [
            HookMatcher(
                matcher=None, hooks=[self._execution_skills_post_tool_failure_hook]
            )
        ]
        subagent_start_hooks = [
            HookMatcher(
                matcher=None, hooks=[self._execution_skills_subagent_start_hook]
            )
        ]
        subagent_stop_hooks = [
            HookMatcher(
                matcher=None, hooks=[self._execution_skills_subagent_stop_hook]
            )
        ]
        if self.execution_tracker is not None:
            pre_tool_hooks.append(
                HookMatcher(matcher=None, hooks=[self.execution_tracker.pre_tool_use_hook])
            )
            post_tool_hooks.append(
                HookMatcher(matcher=None, hooks=[self.execution_tracker.post_tool_use_hook])
            )
            post_tool_failure_hooks.append(
                HookMatcher(
                    matcher=None,
                    hooks=[self.execution_tracker.post_tool_use_failure_hook],
                )
            )

        # Sandbox enforcement hooks — restrict file operations to working directory
        if self.sandbox_enforcement and self.working_directory:
            # Shared between the PreToolUse and PostToolUse sandbox hooks so the
            # PostToolUse guard (defense in depth, see create_sandbox_posttool_guard)
            # knows whether an out-of-sandbox path pre-existed before deciding
            # whether it's safe to auto-delete it.
            sandbox_escape_attempts: Dict[str, dict] = {}
            sandbox_hooks = create_sandbox_enforcement_hook(
                sandbox_root=str(self.working_directory),
                allowed_read_paths=runtime_read_paths,
                logger=self.logger,
                escape_attempts=sandbox_escape_attempts,
            )
            pre_tool_hooks.extend(sandbox_hooks)

            # Backend-owned hard guard: does not depend on PreToolUse hook
            # composition/ordering at all. Re-checks every completed Write/Edit
            # after the fact and hard-stops the run if one landed outside the
            # sandbox regardless of what any PreToolUse hook decided.
            post_tool_hooks.extend(
                create_sandbox_posttool_guard(
                    sandbox_root=str(self.working_directory),
                    escape_attempts=sandbox_escape_attempts,
                    logger=self.logger,
                )
            )

            self.logger.log_action(
                "sandbox_enforcement_init",
                f"Sandbox enforcement enabled: root={self.working_directory}, "
                f"allowed_read_paths={runtime_read_paths or '[]'}"
            )

        # URL blocker hooks — deny network fetches to known leak vectors
        if getattr(self, "url_blocker_enabled", True):
            blocked_hosts = getattr(self, "blocked_url_hosts", None)
            blocked_substrings = getattr(self, "blocked_url_substrings", None)
            url_hooks = create_url_blocker_hooks(
                blocked_hosts=blocked_hosts,
                blocked_url_substrings=blocked_substrings,
                logger=self.logger,
            )
            pre_tool_hooks.extend(url_hooks)
            host_label = "DEFAULT" if blocked_hosts is None else blocked_hosts
            substring_label = (
                "DEFAULT" if blocked_substrings is None else blocked_substrings
            )
            self.logger.log_action(
                "url_blocker_init",
                f"URL blocker enabled: hosts={host_label}, "
                f"substrings={substring_label}"
            )

        # Base prompt for claude code:
        # { # Use default system prompt from Claude Code SDK
        #     "type": "preset",
        #     "preset": "claude_code"  # Use Claude Code's system prompt
        # }
        resolved_model = self._resolve_claude_code_model()
        sub_agent_model_alias = self._resolve_sub_agent_model_alias()
        if resolved_model:
            self.logger.log_action(
                "claude_model_resolved",
                f"Execution model resolved to: {resolved_model} (sub-agents: {sub_agent_model_alias})"
            )

        if self.sub_agent_architecture:
            system_prompt += "\nUse your sub-agents whenever possible to help you complete the analysis."


            # Inject codebase structure prompt into sub-agents so they follow the same rules
            # This is critical because sub-agents do the actual file operations
            modified_sub_agent_configs = {}
            for agent_name, agent_def in SUB_AGENT_CONFIGS.items():
                modified_prompt = agent_def.prompt

                # Inject path/structure rules into the agent that writes files.
                if (
                    agent_name == "execution_agent"
                    and (self.codebase_structure_prompt or self.sandbox_enforcement)
                ):
                    structure_injection = self._format_codebase_structure_prompt()
                    modified_prompt = modified_prompt + "\n\n" + structure_injection
                    self.logger.log_action("sub_agent_structure_injection",
                        f"Injected codebase structure prompt into '{agent_name}' sub-agent ({len(structure_injection)} chars)")

                subagent_tools = list(agent_def.tools or [])
                subagent_skills = None
                if discovered_skills and agent_name == "execution_agent":
                    if "Skill" not in subagent_tools:
                        subagent_tools.insert(0, "Skill")
                    subagent_skills = list(runtime_skill_names)
                    modified_prompt += (
                        "\n\n" + self._execution_skill_load_instruction(agent_name)
                    )

                # Only analytical workers receive the selected skill filter.
                modified_sub_agent_configs[agent_name] = AgentDefinition(
                    description=agent_def.description,
                    prompt=modified_prompt,
                    tools=subagent_tools,
                    model=sub_agent_model_alias,
                    skills=subagent_skills,
                )

            if discovered_skills:
                skill_lines = []
                for skill in discovered_skills:
                    skill_name = skill.get("name", "unknown")
                    skill_description = skill.get("description")
                    if skill_description:
                        skill_lines.append(f"- {skill_name} (invoke as {skill['runtime_name']}): {skill_description}")
                    else:
                        skill_lines.append(f"- {skill_name} (invoke as {skill['runtime_name']})")

                skills_agent_prompt = (
                    "You are the execution agent's helper for the selected immutable round skills. "
                    "Use the Skill tool with the exact qualified name for delegated skill work. "
                    "Follow the loaded instructions completely, including scripts, references, templates, "
                    "receipts, verification tokens, validation, and required artifacts. Resolve referenced "
                    "skill resources and executable paths from the loaded skill's runtime directory, never "
                    "a live skill copy. Keep required output artifacts in the project working directory "
                    "unless the skill specifies another destination. Never claim success from availability "
                    "alone.\n\n"
                    "Available skills:\n"
                    + "\n".join(skill_lines)
                )
                modified_sub_agent_configs["skills_agent"] = AgentDefinition(
                    description="Specialized agent for Claude skill-based external tool access.",
                    prompt=skills_agent_prompt,
                    tools=["Skill", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
                    model=sub_agent_model_alias,
                    skills=list(runtime_skill_names),
                )
                self.logger.log_action(
                    "skills_sub_agent_init",
                    f"Initialized skills_agent with {len(discovered_skills)} discovered skill(s)",
                )

            self.logger.log_action("sub_agent_init",
                f"Initialized {len(modified_sub_agent_configs)} sub-agents with structure constraints")

            # Build allowed_tools list
            allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
            if discovered_skills:
                allowed_tools.insert(0, "Skill")

            claude_options_kwargs = {
                "agents": modified_sub_agent_configs,
                "max_buffer_size": int(os.getenv("CLAUDE_CODE_MAX_BUFFER_SIZE", str(50 * 1024 * 1024))),
                "system_prompt": system_prompt,
                "cwd": self.working_directory,
                "allowed_tools": allowed_tools,
                "permission_mode": "acceptEdits",
                "hooks": {
                    "PreToolUse": pre_tool_hooks,
                    "PostToolUse": post_tool_hooks,
                    "PostToolUseFailure": post_tool_failure_hooks,
                    "SubagentStart": subagent_start_hooks,
                    "SubagentStop": subagent_stop_hooks,
                }
            }
            if resolved_model:
                claude_options_kwargs["model"] = resolved_model
            # Omitted entirely when unset: the SDK only passes --max-turns to
            # the CLI when this is truthy, so leaving it out is what lets the
            # agent run until it decides the task is done.
            if self.max_turns_query_execution:
                claude_options_kwargs["max_turns"] = self.max_turns_query_execution
            self._apply_execution_skills_options(claude_options_kwargs, configured_skills_runtime)

            self.claude_code_options = _create_claude_agent_options(claude_options_kwargs)
        else:
            # Log the system prompt for debugging
            self.logger.log_action("claude_session_init",
                f"Initializing Claude Code session with system prompt ({len(system_prompt)} chars)")
            self.logger.log_action("system_prompt", f"System prompt content:\n{system_prompt}")

            # Build allowed_tools list
            allowed_tools = ["Read", "Write", "Edit", "Bash"]
            if discovered_skills:
                allowed_tools.insert(0, "Skill")

            claude_options_kwargs = {
                "max_buffer_size": int(os.getenv("CLAUDE_CODE_MAX_BUFFER_SIZE", str(50 * 1024 * 1024))),
                "system_prompt": system_prompt,
                "cwd": self.working_directory,
                "allowed_tools": allowed_tools,
                "permission_mode": "acceptEdits",
                "hooks": {
                    "PreToolUse": pre_tool_hooks,
                    "PostToolUse": post_tool_hooks,
                    "PostToolUseFailure": post_tool_failure_hooks,
                    "SubagentStart": subagent_start_hooks,
                    "SubagentStop": subagent_stop_hooks,
                }
            }
            if resolved_model:
                claude_options_kwargs["model"] = resolved_model
            # Omitted entirely when unset: the SDK only passes --max-turns to
            # the CLI when this is truthy, so leaving it out is what lets the
            # agent run until it decides the task is done.
            if self.max_turns_query_execution:
                claude_options_kwargs["max_turns"] = self.max_turns_query_execution
            self._apply_execution_skills_options(claude_options_kwargs, configured_skills_runtime)

            self.claude_code_options = _create_claude_agent_options(claude_options_kwargs)
        
        # Initialize the Claude SDK client with the options
        self.claude_client = ClaudeSDKClient(options=self.claude_code_options)

    def _get_existing_files(self) -> set:
        """
        Get a set of all existing files in the working directory for safety monitoring.
        
        Returns:
            set: Set of file paths that existed before Claude Code started
        """
        existing_files = set()
        # Recursively get all files in the working directory
        for file_path in self.working_directory.rglob("*"):
            if file_path.is_file():
                # Store relative path from working directory
                relative_path = file_path.relative_to(self.working_directory)
                existing_files.add(str(relative_path))

        return existing_files
    
    def _track_generated_images(self, output_files: List[str], query: str, query_num: int, image_tracker) -> List[str]:
        """
        Track images generated by Claude Code and add them to the image tracker.
        
        Args:
            output_files (List[str]): List of generated output files
            query (str): Original query
            query_num (int): Query number
            image_tracker: Object with add_image method for tracking images
            
        Returns:
            List[str]: List of tracked image IDs
        """
        tracked_images = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.gif', '.pdf'}
        
        # Get additional context variables if available
        step_num = self.execution_count
        if hasattr(self, 'execution_context') and 'step_num' in self.execution_context:
            step_num = self.execution_context['step_num']
        
        for file_path in output_files:
            file_path_obj = Path(file_path)

            # Check if it's an image file
            if file_path_obj.suffix.lower() in image_extensions:
                # Generate description based on filename and query
                filename = file_path_obj.name
                description = f"Generated visualization: {filename} (Query: {query[:50]}...)"

                # For non-base64 images, we need to read and encode them
                if file_path_obj.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    # Try to read the image file and convert to base64
                    full_path = self.working_directory / file_path if not file_path_obj.is_absolute() else file_path_obj

                    if full_path.exists():
                        with open(full_path, 'rb') as f:
                            image_bytes = f.read()
                        image_data = base64.b64encode(image_bytes).decode('utf-8')

                        # Add image to tracker
                        image_id = image_tracker.add_image(
                            image_data=image_data,
                            description=description,
                            query_num=query_num or 0,
                            step_num=step_num
                        )

                        if image_id:
                            tracked_images.append(image_id)
                            print(f"📊 Tracked image {image_id}: {description[:50]}...")
                else:
                    # For other formats (SVG, PDF), just note them without base64 encoding
                    print(f"📊 Generated {file_path_obj.suffix.upper()} file: {filename}")

        return tracked_images
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for Claude Code autonomous analysis.
        """

        budget_section = self._step_budget_section()
        if self.sub_agent_architecture:
            base_prompt = EXECUTION_AGENT_SYSTEM_PROMPT_SUB_AGENTS.format(budget_section=budget_section)
        else:
            base_prompt = EXECUTION_AGENT_SYSTEM_PROMPT.format(budget_section=budget_section)
        
        # Add resource information if running on SLURM
        resource_info = get_slurm_resource_info()
        if resource_info:
            base_prompt += f"\n\n{resource_info}\n"
        
        # If codebase_building is enabled, inject session context
        if self.codebase_structure_prompt or self.sandbox_enforcement:
            base_prompt += self._format_codebase_structure_prompt()

        
        return base_prompt
    
    def _get_session_context(self) -> str:
        """
        Get session context from the README file for codebase building continuity.
        
        Returns:
            str: Formatted session context or empty string if not available
        """
        # SESSION_README.md functionality disabled - session context not available
        # Codebase building will rely on Claude Code's native conversation context instead
        return ""
    
    def _format_message_for_display(self, message_str: str) -> str:
        """
        Parse a Claude Code message object and extract meaningful content for display.
        
        Args:
            message_str (str): The raw string representation of the message
            
        Returns:
            str: Formatted message content for display
        """
        try:
            # Convert message to string to work with its representation
            msg_str = str(message_str)
            
            # Handle SystemMessage (initialization)
            if "SystemMessage" in msg_str and "subtype='init'" in msg_str:
                return "🔧 Initializing Claude Code session..."
            
            # Handle AssistantMessage with TextBlock (Claude's responses)
            if "AssistantMessage" in msg_str and "TextBlock" in msg_str:
                # Try multiple patterns to extract text content
                try:
                    import re
                    # Try pattern 1: TextBlock(text="...")
                    text_match = re.search(r'TextBlock\(text="([^"]*)"', msg_str)
                    if not text_match:
                        # Try pattern 2: TextBlock(text='...')
                        text_match = re.search(r"TextBlock\(text='([^']*)'", msg_str)
                    if not text_match:
                        # Try pattern 3: text="..." or text='...'
                        text_match = re.search(r'text=["\']([^"\']*)["\']', msg_str)
                    
                    if text_match:
                        text_content = text_match.group(1)
                        # Unescape common characters
                        text_content = text_content.replace('\\"', '"').replace('\\n', '\n').replace('\\\'', "'")
                        # Truncate very long responses
                        if len(text_content) > 150:
                            text_content = text_content[:150] + "..."
                        return f"💭 {text_content}"
                except Exception as e:
                    # Log parsing error for debugging
                    self.logger.log_error(f"Text parsing error: {str(e)}")
                
                # Fallback: try to find any text content in the message
                try:
                    # Look for any quoted text that might be the content
                    import re
                    quotes_matches = re.findall(r'"([^"]{10,})"', msg_str)
                    if quotes_matches:
                        content = quotes_matches[0]
                        if len(content) > 150:
                            content = content[:150] + "..."
                        return f"💭 {content}"
                except:
                    pass
                
                return "💭 [Claude thinking...]"
            
            # Handle AssistantMessage with ToolUseBlock (tool usage)
            if "AssistantMessage" in msg_str and "ToolUseBlock" in msg_str:
                try:
                    import re
                    # Extract tool name
                    tool_match = re.search(r"name='([^']*)'", msg_str)
                    if not tool_match:
                        tool_match = re.search(r'name="([^"]*)"', msg_str)
                    
                    if tool_match:
                        tool_name = tool_match.group(1)
                        
                        # Extract input description if available
                        desc_match = re.search(r"'description': '([^']*)'", msg_str)
                        if not desc_match:
                            desc_match = re.search(r'"description": "([^"]*)"', msg_str)
                        
                        if desc_match:
                            description = desc_match.group(1)
                            return f"🔧 Using {tool_name}: {description}"
                        else:
                            return f"🔧 Using {tool_name}"
                except Exception as e:
                    self.logger.log_error(f"Tool parsing error: {str(e)}")
                
                return "🔧 [Tool execution]"
            
            # Handle UserMessage with ToolResultBlock (tool results)
            if "UserMessage" in msg_str and "ToolResultBlock" in msg_str:
                try:
                    import re
                    # Try multiple patterns for content extraction
                    content_match = re.search(r"content='([^']*)'", msg_str)
                    if not content_match:
                        content_match = re.search(r'content="([^"]*)"', msg_str)
                    
                    if content_match:
                        content = content_match.group(1)
                        # Clean up content and truncate
                        content = content.replace('\\n', ' ').replace('\\t', ' ')
                        if len(content) > 100:
                            content = content[:100] + "..."
                        return f"📋 Tool result: {content}"
                except Exception as e:
                    self.logger.log_error(f"Result parsing error: {str(e)}")
                
                return "📋 [Tool result received]"
            
            # Fallback for other message types
            if len(msg_str) > 200:
                return f"📨 [Message: {msg_str[:100]}...]"
            else:
                return f"📨 [Message: {msg_str}]"
                
        except Exception as e:
            self.logger.log_error(f"Message parsing error: {str(e)}")
            return f"📨 [Message parsing error: {str(e)}]"

    async def _execute_with_report_generation(self, query_text: str, dataset_description: str = "", query_num: int = None, image_tracker=None, output_files: List[str] = None) -> dict:
        """
        Execute the complete workflow including report generation in a single session.
        
        Args:
            query_text: The analysis query
            dataset_description: Description of the dataset
            query_num: Query number for tracking
            image_tracker: Object for tracking images
            output_files: List of output files generated
            
        Returns:
            dict: Results containing messages and concise_report
        """        
        # Construct the analysis prompt. The step budget appears only when one
        # is configured (see _step_budget_section).
        analysis_prompt = f"""
Here's a description of the dataset you're working with:
{dataset_description}

{query_text}

{self._step_budget_section()}"""

        if self.sub_agent_architecture:
            analysis_prompt += "\nUse your sub-agents whenever possible to help you complete the analysis."

        messages = []
        
        # Log the analysis prompt
        self.logger.log_action("claude_query_input", 
            f"Sending analysis query to Claude Code ({len(analysis_prompt)} chars)")
        self.logger.log_action("analysis_prompt", 
            f"Analysis prompt:\n{analysis_prompt}")
        
        # Log current directory structure for debugging
        if self.codebase_structure_prompt:
            try:
                dir_tree = subprocess.run(['ls', '-la', str(self.working_directory)], 
                                        capture_output=True, text=True, timeout=5)
                self.logger.log_action("pre_execution_directory_state", 
                    f"Working directory contents before execution:\n{dir_tree.stdout}")
            except Exception as e:
                self.logger.log_action("directory_check_failed", f"Could not check directory: {str(e)}")
        
        # Execute Claude Code query with streaming using ClaudeSDKClient
        # Connect client once and maintain session throughout all operations
        async with self.claude_client as client:

            # Required loads happen in this same operative session before analysis.
            await self._load_required_execution_skills(client)
            self._assert_required_execution_skills_loaded(
                include_subagents=False, attempt=2
            )
            # Main query execution
            if self.execution_tracker is not None:
                self.execution_tracker.set_phase("analysis")
            await client.query(analysis_prompt)
            async for message in client.receive_response():
                messages.append(str(message))
                if self.execution_tracker is not None:
                    self.execution_tracker.log_stream_message(message, phase="analysis")
                self._observe_execution_skills_discovery(message)

                # Real-time logging - use direct print to avoid duplicate output
                formatted_msg = self._format_message_for_display(str(message))
                if formatted_msg:
                    self._direct_print(formatted_msg)

            # A discovered-but-unloaded primary or analytical worker is terminal.
            self._assert_required_execution_skills_loaded(
                include_subagents=True, attempt=2
            )
            
            # Update execution count (Claude Code handles multiple turns internally)
            self.execution_count = len(messages)
            
            # Join all messages for full context
            full_output = "\n".join(messages)
            
            result = {
                "messages": messages,
                "full_output": full_output,
                "query_execution_turns": len(messages)
            }
            
            # Run CLAUDE.md generation step if enabled (before codebase building)
            if self.claude_md_generator:
                try:
                    self._direct_print(f"\n📝 Starting CLAUDE.md generation step...")
                    claude_md_result = await self._execute_claude_md_generation(client)
                    result["claude_md_info"] = {
                        "messages": claude_md_result.get("messages", []),
                        "turns": len(claude_md_result.get("messages", []))
                    }
                    self._direct_print(f"✅ CLAUDE.md generation completed with {result['claude_md_info']['turns']} turns")
                except Exception as e:
                    self._direct_print(f"⚠️ Error in CLAUDE.md generation step: {str(e)}")
                    result["claude_md_error"] = str(e)
            
            # Generate the concise report in the same session. A missing final
            # response is a query failure, not report content to persist.
            self._direct_print(f"\n📄 Generating concise summary report...")
            concise_report = await self._generate_concise_summary(
                client, query_text, output_files, image_tracker, query_num
            )
            result["concise_report"] = concise_report
            self._direct_print(f"✅ Concise report generated successfully")
            
            return result
    
    def execute(self, query: str, dataset_description: str = "", query_num: int = None, image_tracker=None, **kwargs) -> str:
        """
        Implements the BaseAgent API. Main entry point for agent execution.
        
        Args:
            query (str): Query from the hypothesis agent
            dataset_description (str): Description of the dataset (optional, defaults to config)
            query_num (int): Query number for tracking (optional)
            image_tracker: Object for tracking images (optional)
            **kwargs: Arbitrary additional variables to be used in execution
            
        Returns:
            str: Concise analysis summary in markdown format
        """
        # Reset conversation if execute_context_persistence is disabled
        if not self.execute_context_persistence:
            self.reset_conversation()
        
        # If dataset_description not provided, use value from config
        if not dataset_description:
            dataset_description = getattr(self.config, 'dataset_description', "")

        # Store any additional variables in a context dictionary for use during execution
        self.execution_context = kwargs
        checkpoint_query_dir = kwargs.get("checkpoint_query_dir")
        if self.execution_tracker is not None and checkpoint_query_dir:
            try:
                self.execution_tracker.set_query_log_dir(
                    checkpoint_query_dir=checkpoint_query_dir,
                    query_num=query_num,
                )
                if self._configured_execution_skills:
                    self.execution_tracker.log_skill_configuration(
                        list(self._execution_skills_source_paths),
                        [skill["name"] for skill in self._configured_execution_skills],
                    )
            except Exception as e:
                self.logger.log_error(f"Failed to initialize execution tracker logs: {str(e)}")
            
        self._direct_print(f"\n🤖 Claude Code SDK executing autonomous analysis for: {query[:100]}...")

        #try:
        # Note: Removed session reset for codebase building to maintain continuity
        # If codebase building is enabled, context will be injected via _get_system_prompt()
        # without breaking the conversation flow

        # Run the comprehensive async function that includes report generation
        # This maintains session continuity throughout the entire process
        try:
            async_result = asyncio.run(self._execute_with_report_generation(query, dataset_description, query_num, image_tracker, []))
            self._direct_print(f"✅ Claude Code SDK analysis completed for query {query_num}")

            # Perform file safety check
            safety_report = self._check_file_safety()
            if safety_report.get('existing_files_protected', True):
                safety_message = "🛡️ File safety: All existing files protected"
            else:
                missing_count = len(safety_report.get('missing_files', []))
                safety_message = f"⚠️ File safety warning: {missing_count} files may have been modified/deleted"
            print(safety_message)
            if self.execution_tracker is not None:
                self.execution_tracker.log_observable(safety_message)

            concise_report = async_result.get("concise_report")
            if not isinstance(concise_report, str) or not concise_report.strip():
                raise ExecutionAgentEmptyResponseError()
            return concise_report
        except BaseException as exc:
            self._direct_print(f"❌ Execution ended before completion: {type(exc).__name__}: {exc}")
            raise
        finally:
            if self.execution_tracker is not None and self.execution_tracker.execution_log is not None:
                self.execution_tracker.execution_log.flush()
            
    
    def _extract_generated_files(self, raw_result: str) -> List[str]:
        """
        Extract file paths from Claude Code's final report.
        
        Args:
            raw_result (str): Raw Claude Code analysis result
            
        Returns:
            List[str]: List of file paths that Claude Code claims to have created
        """
        try:
            import re
            
            # Look for the "## Generated Files" section
            pattern = r"## Generated Files\s*\n((?:- [^\n]+\n?)*)"
            match = re.search(pattern, raw_result, re.IGNORECASE | re.MULTILINE)
            
            if not match:
                # Try alternative patterns
                patterns = [
                    r"Generated Files:\s*\n((?:- [^\n]+\n?)*)",
                    r"Files Created:\s*\n((?:- [^\n]+\n?)*)",
                    r"Output Files:\s*\n((?:- [^\n]+\n?)*)",
                    r"## Files\s*\n((?:- [^\n]+\n?)*)"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, raw_result, re.IGNORECASE | re.MULTILINE)
                    if match:
                        break
            
            if match:
                files_section = match.group(1)
                # Extract file paths from the list
                file_paths = []
                for line in files_section.split('\n'):
                    line = line.strip()
                    if line.startswith('- '):
                        file_path = line[2:].strip()  # Remove "- " prefix
                        if file_path:
                            file_paths.append(file_path)
                return file_paths
            
            return []
            
        except Exception as e:
            self.logger.log_error(f"Error extracting generated files: {str(e)}")
            return []

    async def _execute_claude_md_generation(self, client):
        """
        Execute CLAUDE.md generation step after the main query.
        
        Args:
            client: Connected ClaudeSDKClient instance
        
        Returns:
            dict: Results containing messages and metadata
        """
        self._direct_print(f"\n📝 Starting CLAUDE.md generation step...")
        
        try:
            msgs = []
            
            # Format and log the CLAUDE.md prompt
            claudemd_prompt_formatted = CLAUDEMD_prompt.format(claude_md_path=self.claude_md_path)
            self.logger.log_action("claude_md_query_input", 
                f"Sending CLAUDE.md generation query ({len(claudemd_prompt_formatted)} chars)")
            self.logger.log_action("claudemd_prompt", 
                f"CLAUDE.md prompt:\n{claudemd_prompt_formatted}")
            
            # Use the passed connected client (maintains session continuity)
            if self.execution_tracker is not None:
                self.execution_tracker.set_phase("claude_md_generation")
            await client.query(claudemd_prompt_formatted)
            async for message in client.receive_response():
                msg_str = str(message)
                msgs.append(msg_str)
                if self.execution_tracker is not None:
                    self.execution_tracker.log_stream_message(message, phase="claude_md_generation")
                self._observe_execution_skills_discovery(message)
                # Print the message - use direct print to avoid duplicate output
                formatted_message = self._format_message_for_display(msg_str)
                self._direct_print(formatted_message)
             
            # Log the CLAUDE.md generation attempt
            self.logger.log_action("claude_md_generation", "Generated CLAUDE.md documentation")
            
            return {
                "messages": msgs,
                "full_output": "\n".join(msgs),
                "turns": len(msgs)
            }
            
        except Exception as e:
            self.logger.log_error(f"Error during CLAUDE.md generation: {str(e)}")
            return {
                "messages": [],
                "full_output": f"Error: {str(e)}",
                "turns": 0,
                "error": str(e)
            }


    def _validate_generated_files(self, file_paths: List[str], max_retries: int = 3) -> List[str]:
        """
        Validate that the reported files actually exist.
        
        Args:
            file_paths (List[str]): List of file paths reported by Claude Code
            max_retries (int): Maximum number of retry attempts (currently unused)
            
        Returns:
            List[str]: List of validated file paths that actually exist
        """
        validated_files = []
        missing_files = []
        
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            # Determine which paths to check based on whether the input is absolute
            paths_to_check = []
            if file_path.is_absolute():
                paths_to_check.append(file_path)
            else:
                # If relative, check against working directory
                paths_to_check.append(self.working_directory / file_path)
                # Also check for just the filename in the working directory, a common case
                paths_to_check.append(self.working_directory / file_path.name)

            file_exists = False
            for path in paths_to_check:
                if path.exists() and path.is_file():
                    # Use resolve() to get a canonical, absolute path
                    validated_files.append(str(path.resolve()))
                    file_exists = True
                    break  # Found it, move to the next file path
            
            if not file_exists:
                missing_files.append(file_path_str)
        
        if missing_files:
            # This print is crucial for debugging
            print(f"⚠️ {len(missing_files)} reported files not found on disk: {missing_files}")
            self.logger.log_error(f"Missing files after validation: {missing_files}")
        
        return validated_files

    def _collect_output_files_manual(self) -> List[str]:
        """
        Collect generated output files using manual directory scanning.
        
        Returns:
            List[str]: List of generated file paths
        """
        try:
            output_files = []
            if self.working_directory.exists():
                for file_path in self.working_directory.rglob("*"):
                    if file_path.is_file():
                        # Only include files that didn't exist before
                        relative_path = str(file_path.relative_to(self.working_directory))
                        if relative_path not in self.existing_files:
                            output_files.append(relative_path)
            return output_files
        except Exception as e:
            self.logger.log_error(f"Error in manual file collection: {str(e)}")
            return []

    def _collect_output_files_claude(self, raw_result: str) -> List[str]:
        """
        Collect generated output files by extracting them from Claude Code's report.
        
        Args:
            raw_result (str): Raw Claude Code analysis result to extract files from
            
        Returns:
            List[str]: List of generated file paths
        """
        try:
            if raw_result is None:
                print("📋 No raw result provided for file extraction")
                return []
            
            # Extract file paths from Claude Code's report
            reported_files = self._extract_generated_files(raw_result)
            
            if reported_files:
                print(f"📋 Claude Code reported {len(reported_files)} files: {reported_files}")
                
                # Validate that the files actually exist
                validated_files = self._validate_generated_files(reported_files)
                
                if validated_files:
                    print(f"✅ Validated {len(validated_files)} files exist")
                    return validated_files
                else:
                    print(f"⚠️ No reported files were found to exist")
                    return []
            else:
                print(f"📋 Claude Code did not report any generated files")
                return []
                
        except Exception as e:
            self.logger.log_error(f"Error collecting output files from Claude Code: {str(e)}")
            return []

    def _collect_output_files(self, raw_result: str = None) -> List[str]:
        """
        Collect generated output files using the configured tracking method.
        
        Args:
            raw_result (str, optional): Raw Claude Code analysis result (used for Claude reporting mode)
            
        Returns:
            List[str]: List of generated file paths
        """
        if self.manual_file_tracking:
            return self._collect_output_files_manual()
        else:
            return self._collect_output_files_claude(raw_result)
        
    async def _generate_concise_summary(self, client, query: str, output_files: List[str], image_tracker, query_num) -> str:
        """
        Generate a concise summary report from the raw Claude Code analysis
        Works by 1) trying to get the report at the correct path and
            then 2) retrying querying ClaudeCode to generate the report
        
        Args:
            client: Connected ClaudeSDKClient instance
            query (str): Original query
            output_files (List[str]): List of generated output files
            image_tracker: Object with get_image_references method
            query_num: Query number for tracking
            
        Returns:
            str: Concise summary report
        """
        
        # Pre-create the report directory to ensure Claude Code writes to the right location
        if query_num is not None:
            report_dir = Path(self.working_directory) / "reports" / f"query{query_num}"
            report_dir.mkdir(parents=True, exist_ok=True)
            self._direct_print(f"📁 Created report directory: {report_dir}")

        # Expected report path where Claude should write the report
        # Codebase structure should include this directory already
        expected_report_path = Path(self.working_directory) / "reports" / f"query{query_num}" / "report.md"

        file_request_prompt = REPORT_SUMMARY_PROMPT_v2.format(report_path=str(expected_report_path))
        
        # Log the report generation prompt
        self.logger.log_action("report_query_input", 
            f"Sending report generation query ({len(file_request_prompt)} chars)")
        self.logger.log_action("report_prompt", 
            f"Report generation prompt:\n{file_request_prompt}")
        
        # First attempt
        self._direct_print(f"📄 Attempting to generate report at: {expected_report_path}")
        self._direct_print(f"🔗 Continuing conversation in same session - no new analysis needed")
        
        # Run first attempt
        if self.execution_tracker is not None:
            self.execution_tracker.set_phase("report_generation")
        first_messages = []
        first_failure = None
        try:
            await client.query(file_request_prompt)
            async for message in client.receive_response():
                first_messages.append(message)
                if self.execution_tracker is not None:
                    self.execution_tracker.log_stream_message(message, phase="report_generation")
                self._observe_execution_skills_discovery(message)
                # Print agent interactions as done elsewhere
                formatted_msg = self._format_message_for_display(str(message))
                if formatted_msg:
                    self._direct_print(formatted_msg)
        except Exception:
            first_failure = ExecutionAgentSDKResponseError()
        
        # Check if report was generated
        if expected_report_path.exists():
            try:
                with open(expected_report_path, 'r', encoding='utf-8') as f:
                    report_contents = f.read()
                if not report_contents.strip():
                    raise ExecutionAgentEmptyResponseError()
                self._direct_print(f"✅ Successfully read report from: {expected_report_path}")
                try:
                    shutil.rmtree(report_dir)
                    self._direct_print(f"🗑️ Deleted query directory: {report_dir}")
                except Exception as e:
                    self._direct_print(f"⚠️ Error deleting query directory {report_dir}: {str(e)}")
                return report_contents
            except ExecutionAgentResultError:
                raise
            except Exception as e:
                self._direct_print(f"⚠️ Error reading report file: {str(e)}")
        else:
            first_failure = first_failure or self._report_failure(first_messages)
            if isinstance(first_failure, ExecutionAgentRefusalError):
                raise first_failure
            # Retry once
            self._direct_print(
                f"⚠️ Report unavailable ({first_failure.category}), retrying once..."
            )
            retry_prompt = f"Try to generate this report again as the file path {expected_report_path} cannot be found."
            
            # Log the retry prompt
            self.logger.log_action("report_retry_query_input", 
                f"Sending report retry query ({len(retry_prompt)} chars)")
            self.logger.log_action("report_retry_prompt", 
                f"Report retry prompt:\n{retry_prompt}")
            
            retry_messages = []
            try:
                await client.query(retry_prompt)
                async for message in client.receive_response():
                    retry_messages.append(message)
                    if self.execution_tracker is not None:
                        self.execution_tracker.log_stream_message(message, phase="report_generation_retry")
                    self._observe_execution_skills_discovery(message)
                    # Print agent interactions as done elsewhere
                    formatted_msg = self._format_message_for_display(str(message))
                    if formatted_msg:
                        self._direct_print(formatted_msg)
            except Exception as exc:
                raise ExecutionAgentSDKResponseError() from exc
            
            # Check again after retry
            if expected_report_path.exists():
                try:
                    with open(expected_report_path, 'r', encoding='utf-8') as f:
                        report_contents = f.read()
                    if not report_contents.strip():
                        raise ExecutionAgentEmptyResponseError()
                    self._direct_print(f"✅ Successfully read report from retry: {expected_report_path}")
                    try: # Delete the query directory after reading the report
                        shutil.rmtree(report_dir)
                        self._direct_print(f"🗑️ Deleted query directory: {report_dir}")
                    except Exception as e:
                        self._direct_print(f"⚠️ Error deleting query directory {report_dir}: {str(e)}")
                    return report_contents
                except ExecutionAgentResultError:
                    raise
                except Exception as e:
                    self._direct_print(f"⚠️ Error reading report file after retry: {str(e)}")
            else:
                self._direct_print(f"❌ Report still not found after retry")
                raise self._report_failure(retry_messages)
        
        # A read error or empty artifact must never become a completed report.
        raise ExecutionAgentMalformedReportError()
    
    def _make_api_call(self, messages, response_format=None, max_tokens=None):
        """
        Make an Anthropic API call.
        This is available for direct API calls outside of Claude Code SDK.
        
        Args:
            messages (list): List of messages in conversation format
            max_tokens (int): Maximum tokens for response (optional)
            
        Returns:
            str: Response content from the API
        """
        try:
            # Validate inputs
            if not messages or not isinstance(messages, list):
                raise ValueError("Messages must be a non-empty list")
            
            # Initialize API clients if not already done
            if not hasattr(self, '_api_client') or self._api_client is None:
                if self.llm_config.api_provider == 'anthropic':
                    api_key = self.llm_config.api_key or os.getenv('ANTHROPIC_API_KEY')
                    if not api_key:
                        raise ValueError("ANTHROPIC_API_KEY not found")
                    self._api_client = anthropic.Anthropic(api_key=api_key)
                    print(f"🔧 Initialized Anthropic client for {self.llm_config.model_name}")
                else:
                    raise ValueError(f"Unsupported API provider: {self.llm_config.api_provider}")
            
            # Anthropic API call - use native message format
            # Convert messages to Anthropic's expected format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                elif msg['role'] == 'user':
                    # Handle both string and list content for Anthropic
                    if isinstance(msg['content'], list):
                        # For multimodal content (text + images)
                        anthropic_messages.append({
                            "role": "user",
                            "content": msg['content']
                        })
                    else:
                        # For text-only content
                        anthropic_messages.append({
                            "role": "user", 
                            "content": msg['content']
                        })
                elif msg['role'] == 'assistant':
                    # Add assistant messages as-is
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": msg['content']
                    })
            
            kwargs = {
                'model': self.llm_config.model_name,
                'max_tokens': max_tokens or 4000,
                'messages': anthropic_messages
            }
            
            if system_message:
                kwargs['system'] = system_message
            
            response = self._api_client.messages.create(**kwargs)
            
            if not response or not response.content:
                raise ValueError("Anthropic API returned empty response")
            
            content = response.content[0].text
            if content is None:
                raise ValueError("Anthropic API returned None content")
                
            return content
            
        except Exception as e:
            error_msg = f"API call failed for {self.llm_config.api_provider} ({self.llm_config.model_name}): {str(e)}"
            self.logger.log_error(error_msg)
            print(f"❌ {error_msg}")
            raise e

    def _check_file_safety(self) -> Dict[str, Any]:
        """
        Check for any file safety violations after Claude Code execution.
        
        Returns:
            Dict[str, Any]: File safety report
        """
        try:
            current_files = self._get_existing_files()
            
            # Check for missing files (potential deletions)
            missing_files = self.existing_files - current_files
            
            # Check for new files (should be safe)
            new_files = current_files - self.existing_files
            
            safety_report = {
                'missing_files': list(missing_files),
                'new_files': list(new_files),
                'existing_files_protected': len(missing_files) == 0,
                'total_existing_files': len(self.existing_files),
                'total_current_files': len(current_files)
            }
            
            # Log safety check results
            if missing_files:
                self.logger.log_error(f"FILE SAFETY VIOLATION: {len(missing_files)} files may have been deleted: {list(missing_files)}")
            else:
                self.logger.log_action("file_safety_check", f"File safety confirmed: All {len(self.existing_files)} existing files protected")
            
            if new_files:
                self.logger.log_action("new_files_created", f"Claude Code created {len(new_files)} new files: {list(new_files)[:10]}")  # Log first 10
            
            return safety_report
            
        except Exception as e:
            self.logger.log_error(f"Error checking file safety: {str(e)}")
            return {'error': str(e), 'existing_files_protected': 'unknown'}
    
    def reset_conversation(self):
        """
        Reset conversation state for new queries.
        """
        self.execution_count = 0
        
        # If execute_context_persistence is False, reinitialize the Claude client
        # to start a fresh conversation
        if not self.execute_context_persistence:
            self._direct_print("🔄 Resetting Claude Code session for fresh context...")
            try:
                # Reinitialize the Claude Code session
                self._init_claude_code_session()
                self.logger.log_action("session_reset", "Reinitialized Claude Code session for fresh context")
            except Exception as e:
                self.logger.log_error(f"Error reinitializing Claude Code session: {str(e)}")
                self._direct_print(f"⚠️ Warning: Failed to reset Claude Code session: {str(e)}")
        # Otherwise, session continuity is maintained automatically

    def save_state(self) -> dict:
        """
        Save execution agent state for checkpointing.
        
        Returns:
            dict: Serializable state dictionary
        """
        return {
            'agent_type': type(self).__name__,
            'execution_count': getattr(self, 'execution_count', 0),
            'execution_context': getattr(self, 'execution_context', {}),
            'working_directory': str(getattr(self, 'working_directory', '')),
            'existing_files_count': len(getattr(self, 'existing_files', set())),
            'config_info': {
                'analysis_name': getattr(self, 'analysis_name', ''),
                'max_turns_query_execution': getattr(self, 'max_turns_query_execution', None),
                'separate_report_step': getattr(self, 'separate_report_step', False),
                'claude_md_generator': getattr(self, 'claude_md_generator', False),
                'execute_context_persistence': getattr(self, 'execute_context_persistence', True),
                'log_execution_agent': getattr(self, 'log_execution_agent', True),
            },
            'claude_session_info': {
                'session_active': self.claude_client is not None,
                'max_turns': getattr(self, 'max_turns', None)
            }
        }
    
    def restore_state(self, state: dict) -> bool:
        """
        Restore execution agent state from checkpoint.
        
        Args:
            state: State dictionary from checkpoint
            
        Returns:
            bool: True if restoration was successful
        """
        try:
            # Restore execution context and count
            self.execution_context = state.get('execution_context', {})
            self.execution_count = state.get('execution_count', 0)
            
            # Note: Working directory, output path, and other path-based settings
            # are set during initialization and should not be changed during restoration
            # to avoid breaking file system consistency
            
            # Log restoration details
            if hasattr(self, 'logger'):
                config_info = state.get('config_info', {})
                self.logger.log_action("state_restored", 
                    f"Restored ExecutionAgentCC state: "
                    f"execution_count={self.execution_count}, "
                    f"analysis_name='{config_info.get('analysis_name', 'unknown')}', "
                    f"context_vars={len(self.execution_context)}")
            
            # Verify Claude session is available for continued execution
            if state.get('claude_session_info', {}).get('session_active', False):
                if self.claude_client is None:
                    if hasattr(self, 'logger'):
                        self.logger.log_action("session_reinit", "Reinitializing Claude session after restore")
                    self._init_claude_code_session()
            
            return True
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.log_error(f"Failed to restore execution agent state: {str(e)}")
            return False
    
    def get_state_schema(self) -> dict:
        """
        Get schema description for state validation.
        
        Returns:
            dict: Schema describing state fields
        """
        return {
            'agent_type': 'Class name of the execution agent',
            'execution_count': 'Number of executions performed',
            'execution_context': 'Runtime variables and context',
            'working_directory': 'Working directory path',
            'existing_files_count': 'Number of existing files being protected',
            'config_info': 'Agent configuration details',
            'claude_session_info': 'Claude Code session information'
        }

    def cleanup(self):
        """
        Clean up resources and reset state.
        """
        try:
            # Disconnect Claude client if connected
            if self.claude_client:
                try:
                    # Note: We need to handle this asyncronously, but cleanup is sync
                    # The client will be cleaned up automatically when the object is destroyed
                    pass
                except Exception as e:
                    self.logger.log_error(f"Error disconnecting Claude client: {str(e)}")
                    
        except Exception as e:
            self.logger.log_error(f"Error during Claude client cleanup: {str(e)}")
        
        try:
            # Perform final file safety check
            safety_report = self._check_file_safety()
            
            # Count debug reports created
            # Debug reports disabled; report count is zero
            debug_reports = []

            if self.execution_tracker is not None:
                self.execution_tracker.close()
            self._cleanup_execution_skills_runtime()
            
            # Reset execution count
            self.execution_count = 0
            self.logger.log_action("cleanup", "ExecutionAgentCC cleanup completed")
            
            # Report final safety status
            if safety_report.get('existing_files_protected', True):
                print("🧹 Claude Code ExecutionAgent cleanup completed")
                print("🛡️ Final file safety check: All existing files protected")
            else:
                missing_count = len(safety_report.get('missing_files', []))
                print("🧹 Claude Code ExecutionAgent cleanup completed")
                print(f"⚠️ Final file safety warning: {missing_count} files may have been modified/deleted")
            
            # Report debug information
            # Debug reporting disabled; nothing to print here
            
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {str(e)}")
            print(f"⚠️ Warning: Error during ExecutionAgentCC cleanup: {str(e)}") 
