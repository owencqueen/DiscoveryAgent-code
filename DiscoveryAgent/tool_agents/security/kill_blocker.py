import os
import re
import json
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

async def process_protection_hook(input_data, tool_use_id, context):
    # Only inspect 'Bash' tool calls
    tool_input = input_data.get("tool_input", {})
    command = tool_input.get("command", "")
    
    # Get our own PIDs to protect
    agent_pid = os.getpid()
    parent_pid = os.getppid()
    
    # 1. Detect broad/destructive kill patterns
    # Matches: pkill python, killall python, kill -9 -1, etc.
    dangerous_patterns = [
        r"pkill\s+python", 
        r"killall\s+python", 
        r"kill\s+-\d+\s+-1", 
        r"kill\s+0",
        # Additive coverage for common variants that bypass the simpler patterns.
        r"\bpkill\b(?:\s+-\S+)*\s+(?:-\S+\s+)*python[0-9.]*\b",
        r"\bpkill\b(?:\s+-\S+)*\s+-f\s+['\"]?.*python[0-9.]*.*['\"]?",
        r"\bkillall\b(?:\s+-\S+)*\s+python[0-9.]*\b",
        r"\bpgrep\b(?:\s+-\S+)*\s+.*python[0-9.]*.*\|\s*xargs\s+kill(?:\s+-\d+)?\b",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            # Deterministic Fix: Rewrite the command to exclude the agent's PID
            safe_command = (
                f"pgrep -f python | "
                f"awk '$1 != {agent_pid} && $1 != {parent_pid}' | "
                f"xargs -r kill -9"
            )
            
            return {
                "permissionDecision": "allow",
                "updatedInput": {**tool_input, "command": safe_command},
                "additionalContext": (
                    "NOTE: I intercepted a broad kill command. "
                    f"I have rewritten it to protect process PIDs {agent_pid} and {parent_pid}."
                )
            }

    # 2. Block specific destructive commands entirely
    if "rm -rf /" in command or "chmod -R 000" in command:
        return {
            "permissionDecision": "deny",
            "permissionDecisionReason": "This command is classified as high-risk for system stability."
        }

    return {"permissionDecision": "allow"}