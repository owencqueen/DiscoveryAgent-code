# DiscoveryAgent launcher: runs HypothesisBrancher rounds on a dataset under DATA_PATH
# and resumes runs from saved round configs.
import os
import shutil
import json
import sys
from datetime import datetime


from DiscoveryAgent.hypothesis_brancher import HypothesisBrancher, DiscoveryAgentConfig
from DiscoveryAgent.config import HypothesisGenerationConfig, LLMConfig
from DiscoveryAgent.prompts.hyp_generation import *
from DiscoveryAgent.prompts.tools import *
from DiscoveryAgent.prompts.review_agent import *

from DiscoveryAgent.tool_agents.single_search import SingleSearchConfig
from DiscoveryAgent.tool_agents.deep_research import DeepResearchConfig
from DiscoveryAgent.tool_agents.execution_agent_cc import ExecutionAgentCCConfig
from DiscoveryAgent.utils.sandbox import init_sandbox_dir
from DiscoveryAgent.utils.init_codebase_structure import init_from_structure_file
from DiscoveryAgent.utils.arg_parsers import (
    create_run_parser,
    normalize_anthropic_model_configuration,
)
from DiscoveryAgent.utils.config_loader import (
    apply_updates_with_cli_priority,
    explicit_cli_destinations,
    load_yaml_config,
    snapshot_args_to_yaml,
)
from DiscoveryAgent.utils.agent_config_builders import (
    load_data_description,
    load_task_description,
    load_background_report,
    set_deep_research_agent_config,
    set_single_search_agent_config,
    set_census_agent_config,
    set_opentargets_agent_config,
    set_execution_agent_config,
)

from DiscoveryAgent.hyp_agents.hypothesis_gen_agent import HYPOTHESIS_GENERATION_SYSTEM_PROMPT
from DiscoveryAgent.hyp_agents.director_agent import DIRECTOR_AGENT_SYSTEM_PROMPT
from DiscoveryAgent.prompts.hyp_generation import PRESSURE_MULTITURN_PROMPT

DATA_DIR = os.environ.get("DATA_PATH")

os.environ['BASH_DEFAULT_TIMEOUT_MS'] = '600000'

def load_config_from_checkpoint(config_path):
    """
    Load configuration from a checkpoint config.json file.

    Args:
        config_path: Path to config.json file

    Returns:
        Dict containing the configuration, or None if failed
    """
    if not config_path or not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return None

    try:
        import json
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Extract the config section (unwrap the metadata)
        if 'config' in data:
            config = data['config']
            checkpoint_timestamp = data.get('checkpoint_timestamp', 'unknown')
            round_idx = data.get('round_idx', 'unknown')
            print(f"✓ Loaded configuration from checkpoint:")
            print(f"   - File: {config_path}")
            print(f"   - Checkpoint timestamp: {checkpoint_timestamp}")
            print(f"   - Round: {round_idx}")
            return config
        else:
            # Assume the whole file is the config (legacy format)
            print(f"✓ Loaded configuration from: {config_path}")
            return data

    except Exception as e:
        print(f"Error: Could not load config from {config_path}: {e}")
        return None

def print_full_config(config, title="Resolved Configuration"):
    """Pretty-print a configuration dictionary for debugging/resume visibility."""
    print(f"\n{'=' * 80}")
    print(f"📋 {title}")
    print(f"{'=' * 80}")
    print(json.dumps(config, indent=2, sort_keys=True, default=str))
    print(f"{'=' * 80}\n")

def detect_resume_memory_report(config_file_path):
    """
    Detect the required memory report for resume seeding.

    Given a config file path like:
      /path/to/logs/checkpoints/round_1/config.json

    Required behavior:
      - If the config round already has `memory_report.md`, treat it as completed
        and load that report to seed the next round.
      - If the config round is still in progress, load the previous round's
        `memory_report.md` to seed a true mid-round resume.
      - If resuming within round_0 before it is complete, no prior memory report
        is required.

    Args:
        config_file_path: Path to the config.json file

    Returns:
        str: Path to required memory_report.md, or None when no prior report is
        needed
    """
    import re
    from pathlib import Path

    try:
        config_path = Path(config_file_path)

        # Extract round number from path (e.g., "round_1" -> 1)
        round_match = re.search(r'round_(\d+)', str(config_path))
        if not round_match:
            print(f"   ℹ️  Could not detect round number from config path: {config_file_path}")
            return None

        current_round = int(round_match.group(1))
        checkpoints_dir = config_path.parent.parent
        current_round_report = config_path.parent / "memory_report.md"

        print(f"   🔎 Resume memory requirement:")
        print(f"      Current config round: round_{current_round}")

        # If the current round already has a memory report, that round is
        # complete and we should seed the next round from it.
        if current_round_report.exists():
            print("      Resume mode: continuing after a completed round")
            print(f"      Required memory source: round_{current_round}/memory_report.md")
            print(f"      Expected path: {current_round_report}")
            print(f"   ✅ Found required memory report for resume:")
            print(f"      {current_round_report}")
            return str(current_round_report)

        # A genuine mid-round resume in round_0 has no earlier memory report.
        if current_round == 0:
            print("      Resume mode: mid-round in round_0")
            print(f"      Expected path: none")
            print(f"   ℹ️  Resuming within round_0 before completion: no prior memory_report is required.")
            return None

        previous_round = current_round - 1
        required_report = checkpoints_dir / f"round_{previous_round}" / "memory_report.md"

        print("      Resume mode: mid-round in a later round")
        print(f"      Required memory source: round_{previous_round}/memory_report.md")
        print(f"      Expected path: {required_report}")

        if required_report.exists():
            print(f"   ✅ Found required memory report for resume:")
            print(f"      {required_report}")
            return str(required_report)

        raise FileNotFoundError(
            "Required resume memory report is missing. "
            f"For round_{current_round}, expected either the completed-round report "
            f"{current_round_report} or the previous-round report {required_report}"
        )

    except Exception as e:
        print(f"   ❌ Error during auto-detection: {str(e)}")
        raise

def validate_existing_codebase(codebase_path):
    """
    Validate that the existing codebase has expected structure.
    
    Args:
        codebase_path: Path to the existing codebase directory
        
    Returns:
        List of missing required items (empty if all present)
    """
    required_items = ['CLAUDE.md', 'src']  # Minimal requirements
    missing = []
    for item in required_items:
        item_path = os.path.join(codebase_path, item)
        if not os.path.exists(item_path):
            missing.append(item)
    return missing

def ensure_data_access(working_directory, original_data_path):
    """
    Ensure data directory is accessible in existing codebase.
    
    Args:
        working_directory: Path to the working directory (existing codebase)
        original_data_path: Path to the original data directory
        
    Returns:
        True if data is accessible, False otherwise
    """
    data_link = os.path.join(working_directory, "data")
    
    if os.path.exists(data_link):
        # Data already exists
        if os.path.islink(data_link):
            # It's a symlink - verify it points somewhere valid
            if os.path.exists(os.readlink(data_link)):
                print(f"✓ Data symlink already exists and is valid: {data_link}")
                return True
            else:
                # Broken symlink - remove and recreate
                print(f"⚠️  Broken data symlink detected, recreating...")
                os.unlink(data_link)
        elif os.path.isdir(data_link):
            # It's an actual directory
            print(f"✓ Data directory already exists: {data_link}")
            return True
        else:
            # It's a file - this is unexpected
            print(f"⚠️  Warning: 'data' exists but is not a directory or symlink")
            return False
    
    # Data doesn't exist - create symlink
    try:
        os.symlink(os.path.abspath(original_data_path), data_link)
        print(f"✓ Created data symlink: {data_link} -> {original_data_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create data symlink: {e}")
        return False

def set_review_optimization_agent_config(args):
    # Note: If --review_design flag is used, system_prompt will be generated dynamically
    # by ReviewDesigner based on the research task and whether search
    # tools are enabled. Otherwise, you can specify a system_prompt here.
    model_name = getattr(args, "review_optimization_model", None) or "claude-sonnet-4-5-20250929"
    llm_config = LLMConfig(
        model_name=model_name,
        max_tokens = 15000,
        system_prompt=REVIEW_OPTIMIZATION_SEARCH_PROMPT if args.review_search else REVIEW_OPTIMIZATION_PROMPT,
    )
    return llm_config

def set_review_designer_llm_config(args):
    # Config for the LLM used by ReviewDesigner to generate reviewer prompts
    # Only used if --review_design flag is set
    if args.review_design and args.review_designer_model:
        llm_config = LLMConfig(model_name=args.review_designer_model, max_tokens = 16384)
        return llm_config
    return None  # The workflow then uses claude-sonnet-4-5-20250929.

def build_run_config(args, run_datetime, working_directory, log_dir):
    """
    Build a comprehensive configuration dictionary for checkpointing.

    This captures all parameters needed to reproduce the run with run.py.

    Args:
        args: Parsed command-line arguments
        run_datetime: Timestamp for this run
        working_directory: Working directory path
        log_dir: Log directory path

    Returns:
        Dict containing all run configuration
    """
    # Snapshot all resolved argparse values so checkpoint config remains complete
    # as new CLI arguments are added over time.
    config = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith('_')
    }
    # Memory report paths are resume-time derived and should never be checkpoint-persisted.
    config.pop('background_report_path', None)

    # Add runtime metadata and resolved paths needed for robust resumption.
    config.update({
        'run_datetime': run_datetime,
        'working_directory': working_directory,
        'log_dir': log_dir,
        'checkpoint_base_dir': os.path.join(log_dir, 'checkpoints'),
        'data_dir': DATA_DIR,
        'bash_timeout_ms': os.environ.get('BASH_DEFAULT_TIMEOUT_MS', '600000'),
    })

    return config

def normalize_init_code_dir_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [entry.strip() for entry in value.split(",") if entry.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(entry) for entry in value]
    return [str(value)]

def validate_run_environment(args):
    """Check credentials before creating a workspace or writing run artifacts."""
    required = ["ANTHROPIC_API_KEY"]
    if not args.omit_deepresearch or (args.generate_background_report and not args.background_path):
        required.append("PERPLEXITY_API_KEY")
    missing = [name for name in required
               if not os.environ.get(name, "").strip()
               or os.environ[name].strip() in {"replace_with_your_key", "..."}]
    if missing:
        raise ValueError("Set " + ", ".join(missing) + " before starting a run")
    if not DATA_DIR:
        raise ValueError("Set DATA_PATH to the directory for run logs and datasets")


def main(args):
    validate_run_environment(args)

    # Generate timestamp for this run (used for sandbox and logs)
    run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize working directory based on mode
    if args.existing_codebase:
        # Use existing codebase (bypasses sandbox creation)
        print(f"\n{'=' * 80}")
        print(f"📂 Using Existing Codebase Mode")
        print(f"{'=' * 80}\n")
        
        working_directory = os.path.abspath(args.existing_codebase)
        
        # Validate that the directory exists
        if not os.path.exists(working_directory):
            raise FileNotFoundError(f"Existing codebase directory does not exist: {working_directory}")
        
        # Validate that the directory has expected structure
        missing_items = validate_existing_codebase(working_directory)
        if missing_items:
            print(f"⚠️  Warning: Existing codebase is missing some expected items: {missing_items}")
            print(f"   The agent will continue but may need to create these items.")
        else:
            print(f"✓ Existing codebase structure validated")
        
        # Ensure data directory is accessible
        # Check if data already exists in the existing codebase (resumption case)
        data_in_codebase = os.path.join(working_directory, "data")
        if os.path.exists(data_in_codebase):
            print(f"✓ Data directory found in existing codebase: {data_in_codebase}")
        else:
            # Data needs to be linked from original location
            original_data_path = os.path.join(args.working_directory, "data")
            if not ensure_data_access(working_directory, original_data_path):
                raise RuntimeError(f"Failed to ensure data access in existing codebase")
        
        # Load structure prompt for agent instructions (WITHOUT creating directories)
        # The directory structure already exists, we just need to inform the agent
        codebase_structure_prompt = None
        if args.codebase_structure_prompt_path:
            try:
                with open(args.codebase_structure_prompt_path, 'r') as f:
                    codebase_structure_prompt = f.read()
                    # Replace {datapath} placeholder if present
                    codebase_structure_prompt = codebase_structure_prompt.replace("{datapath}", working_directory)
                print(f"✓ Loaded codebase structure prompt from: {args.codebase_structure_prompt_path}")
                print(f"   (Structure prompt will guide agent - directories already exist)")
            except Exception as e:
                print(f"⚠️  Failed to load structure prompt: {e}")
                codebase_structure_prompt = None
        else:
            print(f"ℹ️  No structure prompt provided - agent will work without strict structure constraints")
        
        print(f"📁 Working directory: {working_directory}")
        print(f"📂 Data path: {os.path.join(working_directory, 'data')}")
        print(f"📄 Codebase structure prompt: {'Loaded' if codebase_structure_prompt else 'None'}")
        print(f"\n{'=' * 80}\n")
        
    elif args.expanding_dataset and not args.no_sandbox:
        # Expanding dataset mode: Create sandbox but with real data directory (no symlink)
        # This allows agents to download/create data files that persist in the sandbox
        # Structure: /path/to/dataset/{analysis_name}_{timestamp}/
        sandbox_dir = os.path.join(
            args.working_directory,
            f"{args.analysis_name}_{run_datetime}"
        )

        # Create sandbox directory
        os.makedirs(sandbox_dir, exist_ok=True)
        working_directory = sandbox_dir

        # Create real data/downloads directory (not a symlink)
        # Use expanding_dataset_dir which defaults to "data/downloads"
        expanding_data_path = os.path.join(working_directory, args.expanding_dataset_dir)
        os.makedirs(expanding_data_path, exist_ok=True)

        # Also create the parent data directory if expanding_dataset_dir is a subdirectory
        data_dir = os.path.join(working_directory, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        # Symlink pre-existing data entries from the parent data/ into sandbox data/
        # Skip the expanding subdirectory itself (it stays as a real directory).
        parent_data_dir = os.path.join(args.working_directory, "data")
        if os.path.isdir(parent_data_dir):
            # Determine the expanding subdirectory name to skip
            # e.g. "data/downloads" -> "downloads", "data/results/raw" -> "results"
            expanding_rel = args.expanding_dataset_dir
            if expanding_rel.startswith("data/"):
                expanding_rel = expanding_rel[len("data/"):]
            elif expanding_rel.startswith("data" + os.sep):
                expanding_rel = expanding_rel[len("data" + os.sep):]
            expanding_skip = expanding_rel.split("/")[0].split(os.sep)[0]

            for entry in os.listdir(parent_data_dir):
                if entry == expanding_skip:
                    continue
                source_entry = os.path.join(parent_data_dir, entry)
                dest_entry = os.path.join(data_dir, entry)
                if not os.path.exists(dest_entry) and not os.path.islink(dest_entry):
                    os.symlink(os.path.abspath(source_entry), dest_entry)

            symlinked = [e for e in os.listdir(data_dir) if os.path.islink(os.path.join(data_dir, e))]
            print(f"🔗 Symlinked {len(symlinked)} pre-existing data entries into sandbox: {symlinked}")

        # Ensure data symlink is ignored by git in sandbox
        gitignore_path = os.path.join(working_directory, ".gitignore")
        if not os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, "w") as f:
                    f.write("data/*\n")
            except Exception as e:
                print(f"Warning: could not write .gitignore in sandbox: {e}")

        # Copy CLAUDE.md template if provided
        if args.claude_md_template_path and os.path.exists(args.claude_md_template_path):
            shutil.copy(args.claude_md_template_path, os.path.join(working_directory, "CLAUDE.md"))

        if args.codebase_structure_prompt_path:
            codebase_structure_prompt = init_from_structure_file(args.codebase_structure_prompt_path, working_directory)
        else:
            codebase_structure_prompt = None

        print(f"🔒 Sandbox initialized at: {sandbox_dir}")
        print(f"📂 Expanding dataset mode: data directory is real (not symlinked)")
        print(f"📁 Data directory: {expanding_data_path}")
        print(f"📁 Working directory: {working_directory}")
        print(f"📄 Codebase structure prompt: {codebase_structure_prompt if codebase_structure_prompt else 'None being used'}")

    elif not args.no_sandbox:
        # Normal sandbox mode: Create unique sandbox directory with timestamp
        # Sandbox provides isolated execution environment with symlinked data
        original_data_path = os.path.join(args.working_directory, "data")

        # Structure: /path/to/dataset/{analysis_name}_{timestamp}/
        sandbox_dir = os.path.join(
            args.working_directory,  # Dataset directory
            f"{args.analysis_name}_{run_datetime}"  # Timestamped run folder
        )

        # Initialize sandbox with symlinked data directory
        working_directory = init_sandbox_dir(
            data_path=original_data_path,
            destination_path=sandbox_dir,
            claude_md_template_path=args.claude_md_template_path,
            init_code_dir_list=args.init_code_dir_list
        )

        if args.codebase_structure_prompt_path:
            # init codebase structure:
            codebase_structure_prompt = init_from_structure_file(args.codebase_structure_prompt_path, working_directory)
        else:
            codebase_structure_prompt = None

        print(f"🔒 Sandbox initialized at: {sandbox_dir}")
        print(f"📂 Data symlinked from: {original_data_path}")
        print(f"📁 Working directory: {working_directory}")
        print(f"📄 Codebase structure prompt: {codebase_structure_prompt if codebase_structure_prompt else 'None being used'}")
    else:
        # Non-sandbox mode (legacy behavior)
        working_directory = args.working_directory
        codebase_structure_prompt = None
        print(f"⚠️  Running in non-sandboxed mode")
        print(f"📁 Working directory: {working_directory}")

    data_description_file = os.path.join(working_directory, "data", "README.txt")

    if args.readme_path:
        print(f"ℹ️  Using dataset README from: {args.readme_path}")
        if os.path.exists(args.readme_path):
            os.makedirs(os.path.dirname(data_description_file), exist_ok=True)
            if os.path.exists(data_description_file) and os.path.samefile(args.readme_path, data_description_file):
                print(f"ℹ️  README already in place: {data_description_file}")
            else:
                shutil.copyfile(args.readme_path, data_description_file)
        else:
            print(f"⚠️  WARNING: Specified readme_path does not exist: {args.readme_path}")
            print(f"   Dataset description will be empty!")

    dataset_description = load_data_description(data_description_file)
    if dataset_description:
        dataset_description = dataset_description.replace("{datapath}", os.path.join(working_directory, "data"))
        print(f"✓ Loaded dataset description ({len(dataset_description)} chars)")
    else:
        dataset_description = ""
        print(f"⚠️  No dataset description loaded (file missing or empty)")

    if args.hint_txt_path and os.path.exists(args.hint_txt_path):
        with open(args.hint_txt_path, 'r') as f:
            hint = f.read()
        dataset_description = dataset_description + f"\n\n{hint}"

    # Load task description if provided
    task_description = None
    if args.task_description_file:
        # Task description file path should be relative to the working directory
        if os.path.isabs(args.task_description_file):
            task_description_file = args.task_description_file
        else:
            # Relative path: resolve relative to the original working directory (not sandbox)
            task_description_file = os.path.join(args.working_directory, args.task_description_file)

        task_description = load_task_description(task_description_file)

        if task_description:
            # Replace datapath placeholder in task description as well
            task_description = task_description.replace("{datapath}", os.path.join(working_directory, "data"))
            print(f"📋 Task description loaded from: {task_description_file}")
        else:
            print(f"⚠️  Warning: Task description file not found or empty: {task_description_file}")
    else:
        inline_task_description = getattr(args, "task_description", None)
        if inline_task_description and str(inline_task_description).strip():
            task_description = str(inline_task_description).strip()
            task_description = task_description.replace("{datapath}", os.path.join(working_directory, "data"))
            print("📋 Task description loaded from inline config")

    # Load background report for resume seeding (auto-detected only)
    background_report = None
    if getattr(args, '_resume_memory_report_path', None):
        background_report = load_background_report(args._resume_memory_report_path)
        if background_report:
            print(f"\n{'=' * 80}")
            print(f"📚 Background Knowledge Loaded")
            print(f"{'=' * 80}")
            print(f"Source: {args._resume_memory_report_path}")
            print(f"Size: {len(background_report)} characters")
            print(f"\nThe agent will have access to this accumulated knowledge when:")
            print(f"  • Generating hypotheses")
            print(f"  • Planning research queries")
            print(f"  • Providing review feedback")
            print(f"{'=' * 80}\n")
        else:
            print(f"⚠️  Warning: Background report file not found or empty: {args._resume_memory_report_path}")

    # Prepare shared log directory for this dataset
    # If resuming from config, reuse the existing log_dir instead of creating a new one
    if args.config_file and hasattr(args, 'log_dir') and args.log_dir:
        # Resuming: use explicit log_dir from config (saved from previous run)
        RUN_LOG_DIR = args.log_dir
        print(f"\n📂 Resuming - using log directory from config:")
        print(f"   {RUN_LOG_DIR}")

        # Also show checkpoint directory for clarity
        if hasattr(args, 'checkpoint_base_dir') and args.checkpoint_base_dir:
            print(f"   Checkpoints: {args.checkpoint_base_dir}")
        else:
            print(f"   Checkpoints: {os.path.join(RUN_LOG_DIR, 'checkpoints')}")
        print()

        if not os.path.exists(RUN_LOG_DIR):
            print(f"⚠️  Warning: Log directory from config doesn't exist, will create it")
            os.makedirs(RUN_LOG_DIR, exist_ok=True)
    else:
        # New run: create fresh log directory with timestamp
        run_dir_name = f"{args.analysis_name}_{run_datetime}"
        # Make log dir if it doesn't exist:
        PARENT_LOG = os.path.join(DATA_DIR, "LOGS")
        os.makedirs(PARENT_LOG, exist_ok=True)

        # Now make sub log-dir
        DATASET_LOG_DIR = os.path.join(PARENT_LOG, args.dataset_name)
        os.makedirs(DATASET_LOG_DIR, exist_ok=True)

        RUN_LOG_DIR = os.path.join(DATASET_LOG_DIR, run_dir_name)
        os.makedirs(RUN_LOG_DIR, exist_ok=True)
        print(f"\n📂 Created new log directory:")
        print(f"   {RUN_LOG_DIR}\n")

    # Setup agent configs
    deep_research_agent = set_deep_research_agent_config(args)
    single_search_agent = set_single_search_agent_config(args)
    census_agent = set_census_agent_config(args, working_directory) if args.use_census_agent else None
    opentargets_agent = set_opentargets_agent_config(args, working_directory) if args.use_opentargets_agent else None

    # Point agent log_dirs to the shared dataset log directory
    deep_research_agent.log_dir = RUN_LOG_DIR
    single_search_agent.log_dir = RUN_LOG_DIR
    if census_agent:
        census_agent.log_dir = RUN_LOG_DIR
    if opentargets_agent:
        opentargets_agent.log_dir = RUN_LOG_DIR

    execution_agent = set_execution_agent_config(
        args,
        dataset_description,
        working_directory,
        args.analysis_name,
        codebase_structure_prompt=codebase_structure_prompt
    )
    # Ensure execution agent logs are written to the shared dataset log directory
    execution_agent.log_dir = RUN_LOG_DIR

    agent_config_list = [execution_agent]
    if not args.omit_deepresearch:
        agent_config_list.append(deep_research_agent)
    if not args.omit_singlesearch:
        agent_config_list.append(single_search_agent)
    if census_agent:
        agent_config_list.append(census_agent)
    if opentargets_agent:
        agent_config_list.append(opentargets_agent)

    multiturn_prompt = PRESSURE_MULTITURN_PROMPT

    # Core hypothesis configuration used by HypothesisGenAgent (hypothesis generation)
    hypothesis_config = HypothesisGenerationConfig(
        llm_config=LLMConfig(model_name=args.hypothesis_model, max_tokens = 16384), # Large number of tokens
        log_dir=RUN_LOG_DIR,
        max_tokens=16384,
        human_selection_mode=args.human_selection,
        selected_hypothesis_file=args.selected_hypothesis_file,
        no_auto_hypothesis=args.no_auto_hypothesis,
        system_prompt=HYPOTHESIS_GENERATION_SYSTEM_PROMPT,
        generate_plan=args.generate_plan,
    )

    # Separate configuration for DirectorAgent (query orchestration).
    # This allows using different prompts or even a different model for directing research
    # while keeping hypothesis generation settings independent.
    director_hypothesis_config = HypothesisGenerationConfig(
        llm_config=LLMConfig(
            model_name=(getattr(args, "director_model", None) or args.hypothesis_model),
            max_tokens = 16384
        ),
        log_dir=RUN_LOG_DIR,
        max_tokens=16384,
        multiturn_prompt=multiturn_prompt,
        system_prompt = DIRECTOR_AGENT_SYSTEM_PROMPT,
        allow_director_to_stop=args.allow_director_to_stop,
        enable_compaction=getattr(args, 'enable_compaction', False),
        compaction_trigger_tokens=getattr(args, 'compaction_trigger_tokens', 150000),
    )

    review_optimization_llm_config = set_review_optimization_agent_config(args)
    review_designer_llm_config = set_review_designer_llm_config(args)

    # Determine expanding data directory
    expanding_data_dir = None
    if args.expanding_dataset:
        if args.expanding_dataset_dir:
            expanding_data_dir = (
                os.path.join(working_directory, args.expanding_dataset_dir)
                if working_directory and not os.path.isabs(args.expanding_dataset_dir)
                else args.expanding_dataset_dir
            )
        else:
            # Default to {working_directory}/data
            expanding_data_dir = os.path.join(execution_agent.working_directory, "data")
        print(f"📁 Expanding dataset mode enabled")
        print(f"   Data directory: {expanding_data_dir}")

    # Create DiscoveryAgentConfig for the HypothesisBrancher
    workflow_config = DiscoveryAgentConfig(
        hypothesis_config=hypothesis_config,
        director_hypothesis_config=director_hypothesis_config,
        dataset_description=dataset_description,
        task_description=task_description,
        agent_config_list=agent_config_list,
        max_queries=args.num_queries,
        log_dir=RUN_LOG_DIR,
        images_dir=RUN_LOG_DIR,
        output_home=execution_agent.working_directory,
        multiturn_hypothesis=True,
        review_optimization_steps=args.review_optimization_steps,
        review_optimization_llm_config=review_optimization_llm_config,
        review_search=args.review_search,
        review_design=args.review_design,
        review_designer_llm_config=review_designer_llm_config,
        explicit_hypothesis_step=args.explicit_hypothesis_step,
        exploratory_phase_warmup_steps=args.exploratory_phase_warmup_steps,
        hold_reviews_in_exploratory_phase=args.hold_reviews_in_exploratory_phase,
        background_path=args.background_path,
        background_report=background_report,
        git_tracking=args.git_tracking,
        # Background report generation
        generate_background_report=args.generate_background_report,
        # Expanding dataset mode settings
        expanding_dataset_mode=args.expanding_dataset,
        expanding_data_directory=expanding_data_dir,
        # file_listing_placeholder uses default "{file_listing}"
    )

    # Build comprehensive run configuration for checkpointing
    run_config = build_run_config(args, run_datetime, working_directory, RUN_LOG_DIR)

    # Persist resolved arguments to YAML alongside checkpoints for reuse
    if run_config.get('checkpoint_base_dir'):
        os.makedirs(run_config['checkpoint_base_dir'], exist_ok=True)
        snapshot_args_to_yaml(args, os.path.join(run_config['checkpoint_base_dir'], "config.yaml"))

    # Show paths being tracked (for new runs only)
    if not args.config_file:
        print(f"\n{'=' * 80}")
        print(f"📝 Configuration Tracking")
        print(f"{'=' * 80}")
        print(f"The following paths will be saved to config.json for resumption:")
        print(f"   • Working directory: {working_directory}")
        print(f"   • Log directory: {RUN_LOG_DIR}")
        print(f"   • Checkpoint directory: {run_config['checkpoint_base_dir']}")
        print(f"{'=' * 80}\n")

    # Create and run HypothesisBrancher
    print(f"\n{'=' * 80}")
    print(f"🌳 Initializing HypothesisBrancher")
    print(f"   - Rounds: {args.num_rounds}")
    print(f"   - Branches per round: {args.num_branches}")
    print(f"   - Queries per branch: {args.num_queries}")
    print(f"   - Total executions: {args.num_rounds * args.num_branches}")
    print(f"{'=' * 80}\n")

    brancher = HypothesisBrancher(
        workflow_config=workflow_config,
        num_rounds=args.num_rounds,
        num_branches=args.num_branches,
        num_parallel_workflows=args.num_parallel_workflows,
        num_exploration_rounds=args.num_exploration_rounds,
        memory_llm_config=LLMConfig(model_name=args.memory_llm_model, max_tokens = 18000),
        run_config=run_config,
    )

    # Run all branches
    results = brancher.run_all_branches()

    # Print final statistics
    print(f"\n{'=' * 80}")
    print(f"📊 HypothesisBrancher Execution Complete")
    print(f"{'=' * 80}")
    print(f"Total executions: {len(results)}")
    print(f"Successful: {len([r for r in results if r['status'] == 'completed'])}")
    print(f"Failed: {len([r for r in results if r['status'] == 'failed'])}")

    # Show paused count for human selection mode
    paused_count = len([r for r in results if r['status'] == 'paused_for_selection'])
    if paused_count > 0:
        print(f"Paused (awaiting hypothesis selection): {paused_count}")

        # Show checkpoint locations for paused executions
        for r in results:
            if r['status'] == 'paused_for_selection':
                round_id = r.get('round_id', '?')
                candidates_path = r.get('candidates_path', '')
                if candidates_path:
                    # Extract checkpoint directory from candidates path
                    # candidates_path is like: .../checkpoints/round_N/next_candidate_hypotheses.md
                    from pathlib import Path
                    checkpoint_dir = Path(candidates_path).parent
                    print(f"\n📂 Round {round_id} checkpoint directory:")
                    print(f"   {checkpoint_dir}")
                    print(f"   • Candidates: {Path(candidates_path).name}")
                    print(f"   • Config: config.json")
                    print(f"   • Write the next hypothesis to: {checkpoint_dir / 'selected_hypothesis.md'}")

    # Get and display final report paths
    report_paths = brancher.get_final_report_paths()
    if report_paths:
        print(f"\n📄 Final Reports Generated ({len(report_paths)}):")
        for path in report_paths:
            print(f"   - {path}")

    # Display ResultsMemoryManager statistics
    if hasattr(brancher, 'results_memory'):
        stats = brancher.results_memory.get_statistics()
        if stats['has_report']:
            print(f"\n💾 Results Memory Summary:")
            print(f"   - Mode: {stats['mode']}")
            print(f"   - Total updates: {stats['update_count']}")
            print(f"   - Accumulated report size: {stats['report_length']} characters")
            print(f"   - Total LLM cost: ${stats['total_cost']:.4f}")
            print(f"   - Model: {stats['model_name']}")

    print(f"\n{'=' * 80}\n")

if __name__ == "__main__":
    parser = create_run_parser()
    raw_cli_args = sys.argv[1:]
    args = parser.parse_args(raw_cli_args)
    cli_overrides = explicit_cli_destinations(parser, raw_cli_args)

    # Deprecated input: background report paths are now always derived at resume time.
    if args.background_report_path is not None:
        print("ℹ️  Ignoring --background_report_path (deprecated). Resume memory is auto-detected from round checkpoints.")
    args.background_report_path = None
    args._resume_memory_report_path = None

    # Load configuration from file if provided
    if args.config_file:
        print(f"\n{'=' * 80}")
        print(f"📋 Loading Configuration from Checkpoint")
        print(f"{'=' * 80}\n")

        config = load_config_from_checkpoint(args.config_file)
        if config:
            # Never load persisted background_report_path from checkpoint configs.
            config.pop('background_report_path', None)
            args = apply_updates_with_cli_priority(
                args,
                config,
                cli_overrides,
                parser=parser,
                source_label=f"config file: {args.config_file}",
            )
            print_full_config(config, title="Full Config Loaded From Checkpoint")
            print()
        else:
            print(f"❌ Failed to load configuration from {args.config_file}")
            print(f"   Continuing with command-line arguments only\n")

        print(f"🔍 Auto-detecting memory report for resume...")
        auto_detected_path = detect_resume_memory_report(args.config_file)
        if auto_detected_path:
            args._resume_memory_report_path = auto_detected_path
            print(f"   ✅ Will load memory report from:")
            print(f"      {auto_detected_path}")
        else:
            print(f"   ℹ️  No memory report loaded for resume (round_0).")
        print()

    # Load YAML configuration (higher priority than checkpoint config, lower than CLI)
    if args.config_yaml:
        yaml_config = load_yaml_config(args.config_yaml)
        if yaml_config:
            yaml_config.pop('background_report_path', None)
            args = apply_updates_with_cli_priority(
                args,
                yaml_config,
                cli_overrides,
                parser=parser,
                source_label=f"YAML config: {args.config_yaml}",
            )
            print()

    try:
        args = normalize_anthropic_model_configuration(args)
    except ValueError as exc:
        parser.error(str(exc))

    # Apply fallback defaults for args whose default was set to None to enable
    # CLI-override detection (e.g. --num_rounds 1 would be indistinguishable from
    # "not provided" if the default were also 1, causing the config file value to win).
    if args.num_rounds is None:
        args.num_rounds = 1

    # Normalize init_code_dir_list so downstream code always sees a list
    args.init_code_dir_list = normalize_init_code_dir_list(args.init_code_dir_list)

    # Validation: analysis_name is required
    if not args.analysis_name:
        parser.error("--analysis_name is required (provide via command line or --config-file)")

    try:
        validate_run_environment(args)
    except ValueError as exc:
        parser.error(str(exc))

    # Auto-detect existing codebase on resumption
    # If loading from config, use working_directory as existing_codebase (unless explicitly overridden)
    if args.config_file and not args.existing_codebase and args.working_directory:
        # When resuming from config, working_directory points to the previous run's directory
        # Use it as existing_codebase to continue in the same location
        if os.path.exists(args.working_directory):
            print(f"\n🔍 Resumption detected - using working_directory from config:")
            print(f"   {args.working_directory}")
            print(f"   Setting as existing codebase for resumption\n")
            args.existing_codebase = args.working_directory
            args.no_sandbox = False  # existing_codebase supersedes no_sandbox on auto-resume
        else:
            print(f"\n⚠️  Warning: working_directory from config doesn't exist:")
            print(f"   {args.working_directory}")
            print(f"   Will create new sandbox instead\n")

    # Validation: Ensure mutually exclusive flags
    if args.existing_codebase and args.no_sandbox:
        parser.error("--existing_codebase and --no_sandbox are mutually exclusive. Use only one.")

    # Validation: working_directory is required for new runs (unless using existing_codebase)
    if not args.existing_codebase:
        if args.dataset_name is not None:
            # MLEBench datasets don't use the _data suffix
            if not args.dataset_name.startswith("MLEBench/") and not args.dataset_name.endswith("_data"):
                args.dataset_name = f"{args.dataset_name}_data"
            args.working_directory = os.path.join(DATA_DIR, args.dataset_name)
            assert os.path.exists(args.working_directory), f"Dataset directory does not exist: {args.working_directory}"
        else:
            assert args.working_directory is not None, "Either --dataset_name or --working_directory must be provided"
            assert os.path.exists(args.working_directory), f"Working directory does not exist: {args.working_directory}"
    else:
        # When using existing_codebase, we still need working_directory for the data path
        # If not provided via dataset_name or working_directory, try to infer from existing_codebase parent
        if args.dataset_name is None and args.working_directory is None:
            # Try to infer the dataset directory from the existing codebase path
            # Assume the existing codebase is in the dataset directory
            existing_codebase_parent = os.path.dirname(os.path.abspath(args.existing_codebase))
            if os.path.exists(os.path.join(existing_codebase_parent, "data")):
                args.working_directory = existing_codebase_parent
                print(f"ℹ️  Inferred working_directory from existing_codebase: {args.working_directory}")
            else:
                parser.error("When using --existing_codebase, you must also provide --dataset_name or --working_directory for data access")
        elif args.dataset_name is not None:
            # MLEBench datasets don't use the _data suffix
            if not args.dataset_name.startswith("MLEBench/") and not args.dataset_name.endswith("_data"):
                args.dataset_name = f"{args.dataset_name}_data"
            args.working_directory = os.path.join(DATA_DIR, args.dataset_name)

    if args.dataset_name is None:
        args.dataset_name = os.path.basename(os.path.abspath(args.working_directory))

    main(args)
