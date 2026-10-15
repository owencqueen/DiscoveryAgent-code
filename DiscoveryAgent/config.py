"""
Configuration classes for DiscoveryAgent agents.

This module contains configuration classes that centralize the configuration
for different DiscoveryAgent agents, making it easier to manage and modify settings.
"""

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict


# Load environment variables from .env file if it exists.
# The launching shell/process is authoritative. A local .env fills missing values.
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from .prompts.hyp_generation import HYP_GEN_SYSTEM_PROMPT_MODULAR


def _run_quick_command(cmd: List[str], timeout_s: float = 2.0) -> Optional[str]:
    """
    Run a small local command with a short timeout.
    Returns stdout (stripped) on success, else None.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=os.environ,
        )
        out = (proc.stdout or "").strip()
        if proc.returncode != 0:
            return None
        return out or None
    except Exception:
        return None


def _parse_cuda_visible_devices(cvd: Optional[str]) -> List[str]:
    """
    Parse CUDA_VISIBLE_DEVICES into tokens. Returns [] if unset/empty/disabled.
    """
    if not cvd:
        return []
    cvd_norm = cvd.strip()
    if not cvd_norm:
        return []
    # Common "no GPU" conventions across schedulers/containers
    if cvd_norm.lower() in {"nodevfiles", "none", "void", "no", "disabled"}:
        return []
    return [t.strip() for t in cvd_norm.split(",") if t.strip() != ""]


def get_visible_gpu_info() -> Optional[str]:
    """
    Return a compact, human-readable summary of GPUs actually visible to this process.

    Primary signal: CUDA_VISIBLE_DEVICES (scheduler/device masking).
    Optional enrichment: nvidia-smi details filtered to the visible set (best-effort).
    """
    cvd = os.getenv("CUDA_VISIBLE_DEVICES")
    tokens = _parse_cuda_visible_devices(cvd)

    # Determine a visible count from CUDA_VISIBLE_DEVICES when present.
    visible_count: Optional[int] = None
    if cvd is not None:
        # If scheduler set it (even empty), trust it as the masking signal.
        visible_count = len(tokens)

    # Best-effort: enrich with nvidia-smi.
    # We *select* GPUs using CUDA_VISIBLE_DEVICES when possible to avoid showing
    # node-wide GPUs you don't actually have access to.
    #
    # We still *present* GPUs in "visible ordering" (GPU 0..N-1) to avoid
    # implying physical indices are meaningful under masking/containers.
    details: List[str] = []
    smi_cmd = ["nvidia-smi"]
    if tokens:
        # nvidia-smi supports -i with comma-separated indices or UUIDs.
        smi_cmd += ["-i", ",".join(tokens)]
    smi_cmd += ["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]

    smi = _run_quick_command(smi_cmd, timeout_s=2.0)

    if smi:
        for i, line in enumerate(smi.splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            name, mem_total = parts[0], parts[1]
            details.append(f"GPU {i}: {name} ({mem_total} MiB)")

    # Build compact string. Keep this prompt block short.
    if visible_count is None and not details and cvd is None:
        return None

    if visible_count is None and details:
        # No masking env var, but we can see GPUs via nvidia-smi.
        visible_count = len(details)

    if visible_count is None:
        # Masking env var exists but parsing didn't produce a count (shouldn't happen).
        visible_count = 0

    if cvd is not None:
        base = f"GPUs (visible via CUDA_VISIBLE_DEVICES): {visible_count} device(s) [{cvd}]"
    else:
        base = f"GPUs (visible): {visible_count} device(s)"

    if details:
        # Include at most a few GPUs to avoid long prompts on big nodes.
        max_items = 4
        shown = details[:max_items]
        suffix = " | ".join(shown)
        if len(details) > max_items:
            suffix += f" | +{len(details) - max_items} more"
        return f"{base} — {suffix}"

    return base


def get_explicit_claude_cli_path() -> Optional[str]:
    """Return a validated developer-supplied Claude CLI override, if any.

    Normal execution deliberately returns ``None`` so ``claude-agent-sdk`` can
    select its bundled executable. No PATH, npm, or repository discovery is
    performed here.
    """
    raw_path = os.getenv("CLAUDE_CODE_CLI_PATH")
    if not raw_path or not raw_path.strip():
        return None

    configured_path = Path(raw_path.strip()).expanduser()
    if not configured_path.is_absolute():
        raise ValueError(
            "CLAUDE_CODE_CLI_PATH must be an absolute path to an executable file"
        )
    try:
        resolved_path = configured_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(
            f"CLAUDE_CODE_CLI_PATH does not exist: {configured_path}"
        ) from exc
    if not resolved_path.is_file():
        raise ValueError(
            f"CLAUDE_CODE_CLI_PATH is not a file: {resolved_path}"
        )
    if not os.access(resolved_path, os.X_OK):
        raise ValueError(
            f"CLAUDE_CODE_CLI_PATH is not executable: {resolved_path}"
        )
    return str(resolved_path)


def get_slurm_resource_info():
    """
    Parse SLURM environment variables to get allocated resource information.
    
    Returns:
        str: Formatted string describing available resources, or None if not on SLURM
    """
    # Check if running on SLURM
    if not os.getenv('SLURM_JOB_ID'):
        return None
    
    resources = []
    
    # Get CPU information
    cpus = os.getenv('SLURM_CPUS_ON_NODE')
    if cpus:
        resources.append(f"CPUs: {cpus} cores")
    
    # Get memory information
    mem = os.getenv('SLURM_MEM_PER_NODE')
    if mem:
        # Convert to GB if it's in MB (SLURM typically provides in MB)
        try:
            mem_mb = int(mem)
            mem_gb = mem_mb / 1024
            resources.append(f"Memory: {mem_gb:.1f} GB ({mem_mb} MB)")
        except (ValueError, TypeError):
            resources.append(f"Memory: {mem}")
    
    # Get GPU information
    # IMPORTANT: SLURM's GPU vars can reflect node totals, not your cgroup/device access.
    # Prefer "visibility" signals (CUDA_VISIBLE_DEVICES) and best-effort nvidia-smi enrichment.
    gpu_info = get_visible_gpu_info()
    if gpu_info:
        resources.append(gpu_info)
    
    if not resources:
        return None
    
    resource_str = "## Allocated Computing Resources\n\n"
    resource_str += "You are running on an HPC cluster with the following allocated resources:\n"
    resource_str += "\n".join(f"- {r}" for r in resources)
    resource_str += "\n\nPlease be mindful of these resource constraints when designing your analysis. "
    resource_str += "Consider parallelization opportunities if multiple CPUs are available, "
    resource_str += "and leverage GPU acceleration if GPUs are available and appropriate for your task."
    
    return resource_str


def setup_claude_code_environment():
    """
    Configure Claude Agent SDK timeouts and validate any explicit CLI override.

    The override is returned to callers instead of being written back into the
    environment. When it is absent, callers must omit ``cli_path`` so the SDK
    uses its bundled executable.
    """
    claude_cli_path = get_explicit_claude_cli_path()

    # Configure bash command timeouts for Claude Agent SDK
    # Default: 30 minutes (1800000ms), Max: 2 hours (7200000ms)
    # Override these by setting BASH_DEFAULT_TIMEOUT_MS and BASH_MAX_TIMEOUT_MS in your .env file
    if not os.getenv('BASH_DEFAULT_TIMEOUT_MS'):
        os.environ['BASH_DEFAULT_TIMEOUT_MS'] = '1800000'  # 30 minutes

    if not os.getenv('BASH_MAX_TIMEOUT_MS'):
        os.environ['BASH_MAX_TIMEOUT_MS'] = '7200000'  # 2 hours

    return claude_cli_path


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    model_name: str
    api_key: Optional[str] = None
    api_provider: Optional[str] = None
    # Model runtime settings
    temperature: float = 0.2
    frequency_penalty: float = 0.0
    max_tokens: Optional[int] = None  # Maximum tokens for output; None means use API default
    # MCP (Model Context Protocol) servers
    mcp_servers: Optional[List[dict]] = None
    system_prompt: Optional[str] = None # May not be used by some agents

    def __post_init__(self):
        """Auto-detect API provider if not specified."""
        if self.api_provider is None:
            self.api_provider = 'anthropic'

        # Get API keys from environment if not provided
        if self.api_key is None:
            if self.api_provider == 'anthropic':
                self.api_key = os.getenv('ANTHROPIC_API_KEY')
            elif self.api_provider == 'perplexity':
                # Perplexity uses PERPLEXITY_API_KEY in the environment
                self.api_key = os.getenv('PERPLEXITY_API_KEY')

        # Validate API provider
        if self.api_provider not in ['anthropic', 'perplexity']:
            raise ValueError("api_provider must be 'anthropic' or 'perplexity'")



@dataclass
class HypothesisGenerationConfig:
    """Configuration for HypothesisGenerationAgent."""
    llm_config: LLMConfig
    log_dir: str = "logs"
    max_tokens: int = 8192
    system_prompt: str = field(default=HYP_GEN_SYSTEM_PROMPT_MODULAR)
    mcp_servers: Optional[List[dict]] = None
    multiturn_prompt: str = None # Is an intermediate prompt to drive the model to find more interesting hypotheses mid-research

    # Human selection mode parameters
    human_selection_mode: bool = False  # Enable human hypothesis selection
    selected_hypothesis_file: Optional[str] = None  # Path to file containing human-selected hypothesis
    no_auto_hypothesis: bool = False    # Disable automatic hypothesis generation at end of rounds

    allow_director_to_stop: bool = False # Condition for automatic stop of directors
    generate_plan: bool = False # Generate a plan for each hypothesis

    # Compaction support (Anthropic only - requires claude-opus-4-6)
    enable_compaction: bool = False
    compaction_trigger_tokens: int = 190000  # Token threshold to trigger compaction (min 50000)
    compaction_instructions: Optional[str] = None  # Custom summarization prompt (replaces default)

# Default configurations
def get_default_hypothesis_config(
    model_name: str = "claude-opus-4-6",
    mcp_servers: Optional[List[dict]] = None
) -> HypothesisGenerationConfig:
    """Get default hypothesis generation configuration."""
    llm_config = LLMConfig(model_name=model_name)
    return HypothesisGenerationConfig(llm_config=llm_config, mcp_servers=mcp_servers)
