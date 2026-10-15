"""
OpenTargetsAgent - Claude Code data extraction agent for OpenTargets Platform data.

Reads parquet datasets from a configured read-only directory and writes
CSV/parquet extracts to a fast local output directory for downstream analysis
by the execution agent.

Uses the Claude Agent SDK to spawn a Claude Code session with:
- Bash, Read, Write, Glob, Grep tools for data extraction scripts
- Write-protection hooks preventing writes to the data source directory
- Kill-blocker hook for process safety
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List

from ..base import BaseAgent, AgentToolConfig
from ...utils.logger import Logger
from .prompts.opentargets_agent import (
    OPENTARGETS_TOOL_PROMPT,
    build_opentargets_system_prompt,
)

# Configure Claude Code environment BEFORE importing SDK
from ...config import setup_claude_code_environment

setup_claude_code_environment()

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    AssistantMessage,
    TextBlock,
    ResultMessage,
)

from ..security.kill_blocker import process_protection_hook
from ..security.write_protection import create_write_protection_hook


@dataclass
class OpenTargetsAgentConfig(AgentToolConfig):
    """
    Configuration for OpenTargetsAgent.

    Fields inherited from AgentToolConfig:
        agent_name: registry key used in STRING_TO_AGENT_MAP
        tag: tag used by the Director to invoke this agent (<tag>...</tag>)
        tool_prompt: short description shown to the Director about when to use it
    """
    # Path to the virtualbiotech MCP package (provides data loader library)
    mcp_package_path: Optional[str] = None

    # Path to OpenTargets data directory (read-only).
    open_targets_path: Optional[str] = None

    # Output directory for extracted data files
    output_dir: Optional[str] = None

    # Log directory
    log_dir: str = "logs"

    # Maximum Claude Code turns per query
    max_turns: int = 15

    # Model for the Claude Code session
    model: Optional[str] = "claude-sonnet-4-6"

    # Tool prompt shown to Director (imported from prompts module)
    tool_prompt: str = OPENTARGETS_TOOL_PROMPT


class OpenTargetsAgent(BaseAgent):
    """
    Agent that extracts data from the OpenTargets Platform using Claude Code.

    Instead of MCP servers, this agent spawns a Claude Code session that writes
    and executes Python scripts to read parquet data and save filtered extracts
    (CSV/parquet) to the output directory.
    """

    def __init__(self, config: OpenTargetsAgentConfig, logger: Optional[Any] = None):
        super().__init__(config, logger)
        self.config = config
        self.logger = logger if logger is not None else Logger(
            "opentargets_agent",
            log_dir=config.log_dir,
        )

        if not config.open_targets_path or not os.path.isdir(config.open_targets_path):
            raise ValueError("OpenTargetsAgent requires an existing --opentargets_data_path")
        if not config.mcp_package_path or not os.path.isdir(config.mcp_package_path):
            raise ValueError("OpenTargetsAgent requires an existing --opentargets_mcp_package_path")

        # Ensure output directory exists
        if config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
        else:
            # Default to a temp directory if none provided
            config.output_dir = os.path.join(os.getcwd(), "opentargets_downloads")
            os.makedirs(config.output_dir, exist_ok=True)

        # Build Claude Code session options
        self._init_claude_code_session()

        self.logger.log_action(
            "opentargets_init",
            f"OpenTargetsAgent initialized (Claude Code mode, max_turns={config.max_turns})",
        )
        self.logger.log_action(
            "opentargets_init",
            f"Data path: {config.open_targets_path}",
        )
        self.logger.log_action(
            "opentargets_init",
            f"Output directory: {config.output_dir}",
        )

    def _init_claude_code_session(self):
        """Build ClaudeAgentOptions for the data extraction session."""
        system_prompt = build_opentargets_system_prompt(
            data_path=self.config.open_targets_path,
            mcp_package_path=self.config.mcp_package_path,
            output_dir=self.config.output_dir,
        )

        # Build hooks: write-protection on the data path + kill-blocker
        write_protection_hooks = create_write_protection_hook(
            protected_paths=[self.config.open_targets_path],
            logger=self.logger,
        )

        # Combine all PreToolUse hooks
        pre_tool_hooks = list(write_protection_hooks) + [
            HookMatcher(matcher="Bash", hooks=[process_protection_hook]),
        ]

        options_kwargs = {
            "system_prompt": system_prompt,
            "allowed_tools": ["Bash", "Read", "Write", "Glob", "Grep"],
            "permission_mode": "acceptEdits",
            "cwd": self.config.output_dir,
            "max_turns": self.config.max_turns,
            "max_buffer_size": int(os.getenv("CLAUDE_CODE_MAX_BUFFER_SIZE", str(50 * 1024 * 1024))),
            "hooks": {
                "PreToolUse": pre_tool_hooks,
            },
        }
        if self.config.model:
            options_kwargs["model"] = self.config.model

        self.claude_code_options = ClaudeAgentOptions(**options_kwargs)

        self.claude_client = ClaudeSDKClient(options=self.claude_code_options)

    async def _execute_async(self, query: str) -> str:
        """
        Execute a data extraction query asynchronously using Claude Code.

        Args:
            query: The user's query describing what data to extract.

        Returns:
            The agent's response as a string.
        """
        results = []
        try:
            async with self.claude_client as client:
                await client.query(query)
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                results.append(block.text)
        except Exception as e:
            error_msg = f"Error executing OpenTargets query: {str(e)}"
            self.logger.log_error(error_msg)
            return error_msg

        return "\n".join(results) if results else "No response generated."

    def _run_coro(self, coro):
        """
        Run an async coroutine safely whether or not a loop is already running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return asyncio.run(coro)

    def execute(self, query: str, **kwargs) -> str:
        """
        Execute an OpenTargets data extraction query.

        Args:
            query: The query string describing what data to extract.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The agent's response as a string, including paths to saved files.
        """
        try:
            self.logger.log_action("opentargets_query", f"Query: {query[:200]}...")
            result = self._run_coro(self._execute_async(query))
            self.logger.log_response(result, "opentargets_result")
            return result
        except Exception as e:
            err_msg = f"OpenTargetsAgent error: {str(e)}"
            try:
                self.logger.log_error(err_msg)
            except Exception:
                pass
            raise

    def cleanup(self):
        """Cleanup hook - Claude Code sessions are managed by the SDK context manager."""
        return None

    def save_state(self) -> Dict[str, Any]:
        """Save agent state for checkpointing."""
        state = super().save_state()
        state.update({
            "mcp_package_path": self.config.mcp_package_path,
            "open_targets_path": self.config.open_targets_path,
            "output_dir": self.config.output_dir,
            "max_turns": self.config.max_turns,
        })
        return state

    def restore_state(self, state: Dict[str, Any]) -> bool:
        """Restore agent state from checkpoint."""
        return True

    def get_state_schema(self) -> Dict[str, str]:
        """Get schema description for state fields."""
        schema = super().get_state_schema()
        schema.update({
            "mcp_package_path": "Path to OpenTargets MCP package / data loader",
            "open_targets_path": "Path to OpenTargets parquet data",
            "output_dir": "Directory for saving extracted data files",
            "max_turns": "Maximum Claude Code turns per query",
        })
        return schema
