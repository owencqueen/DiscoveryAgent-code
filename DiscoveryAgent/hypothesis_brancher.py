"""
Hypothesis Brancher - Wrapper for running DirectorWorkflow multiple times

This module provides a wrapper class that can execute the DirectorWorkflow
workflow multiple times in succession, potentially with different configurations
or parameters for each run.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import datetime
import copy
import re
from dataclasses import dataclass, field

from .director_workflow import DirectorWorkflow
from .config import HypothesisGenerationConfig
from .memory.results_memory import ResultsMemoryManager
from .utils.data_directory_manager import DataDirectoryManager
from .utils import progress_recorder


@dataclass
class DiscoveryAgentConfig:
    """Configuration for DirectorWorkflow instances."""
    
    hypothesis_config: HypothesisGenerationConfig
    dataset_description: str
    # Optional separate config for DirectorAgent (query orchestration).
    # If None, `hypothesis_config` will be used for both DirectorAgent and HypothesisGenAgent.
    director_hypothesis_config: Optional[HypothesisGenerationConfig] = None
    task_description: Optional[str] = None
    agent_config_list: Optional[List[Any]] = None
    max_queries: int = 10
    log_dir: str = "logs"
    images_dir: str = "logs"
    output_home: Optional[str] = None
    multiturn_hypothesis: bool = True
    review_optimization_steps: Optional[int] = None
    review_optimization_llm_config: Optional[Any] = None
    review_search: bool = False
    review_design: bool = False
    review_designer_llm_config: Optional[Any] = None
    explicit_hypothesis_step: Optional[int] = None
    exploratory_phase_warmup_steps: Optional[int] = None
    # Hold reviews in the exploratory phase (reviews deferred until explicit hypothesis phase).
    hold_reviews_in_exploratory_phase: bool = True
    # Optional directory of prior-work PDFs. When background report generation is enabled,
    # these PDFs are summarized into deep_research_background.md instead of being injected
    # directly into agent context.
    background_path: Optional[str] = None
    background_report: Optional[str] = None
    git_tracking: bool = False
    
    # Background report generation
    # When enabled, a deep research background report is generated before the workflow starts
    generate_background_report: bool = False

    # Expanding Dataset Mode settings
    # When enabled, the data directory is treated as a growing directory and the
    # dataset_description is refreshed after each query to reflect new files.
    expanding_dataset_mode: bool = False
    # Path to the data directory to monitor for new files.
    # If None, defaults to "{working_directory}/data" where working_directory
    # is determined from the first ExecutionAgentCC in agent_config_list.
    expanding_data_directory: Optional[str] = None
    # Placeholder string in dataset_description that will be replaced with file listing.
    # Set to None to append file listing at the end instead of replacing a placeholder.
    file_listing_placeholder: Optional[str] = "{file_listing}"

class HypothesisBrancher:
    """
    Wrapper class that executes DirectorWorkflow.run() multiple times.

    This allows for running multiple research branches, iterations, or
    variations of the same workflow.
    """

    def __init__(
        self,
        workflow_config: DiscoveryAgentConfig,
        num_rounds: int,
        num_parallel_workflows: int = 1,
        num_branches: int = 1,
        num_exploration_rounds = None,
        memory_llm_config: Optional[Any] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the HypothesisBrancher.

        Args:
            workflow_config: Configuration for the DirectorWorkflow instances
            num_rounds: Number of rounds to run
            num_parallel_workflows: Number of parallel workflows (not yet implemented)
            num_branches: Number of branches per round (default: 1)
            num_exploration_rounds: Number of exploration rounds (optional)
                          If False, uses simple (1-agent) mode for faster operation.
                          ACE mode: INTEGRATOR → ANALYZER → REFINER (more expensive, better quality)
                          Simple mode: INTEGRATOR only (cheaper, faster)
            memory_llm_config: LLM configuration for ResultsMemoryManager. If None, uses
                             default (claude-sonnet-4-5-20250929).
            run_config: Optional dictionary containing all run configuration metadata for
                       checkpointing. This should include all parameters needed to reproduce
                       the run (model names, flags, file paths, etc.)
        """
        self.workflow_config = workflow_config
        self.num_branches = num_branches
        self.num_rounds = num_rounds
        self.num_parallel_workflows = num_parallel_workflows
        self.memory_llm_config = memory_llm_config
        self.num_exploration_rounds = num_exploration_rounds
        self.run_config = run_config or {}

        if self.num_parallel_workflows > 1:
            raise NotImplementedError("Parallel workflow execution is not implemented yet")

        # Store results from each branch
        self.branch_results: List[Dict[str, Any]] = []
        self.workflows: List[DirectorWorkflow] = []

        # Initialize ResultsMemoryManager with ACE or simple mode
        self.results_memory = ResultsMemoryManager(
            llm_config=self.memory_llm_config,
        )
        self._seed_results_memory()

        # Log the memory configuration
        print("💾 ResultsMemoryManager initialized in ACE (3-agent) mode")

        # Initialize DataDirectoryManager for expanding dataset mode
        self.data_manager: Optional[DataDirectoryManager] = None
        if self.workflow_config.expanding_dataset_mode:
            data_dir = self._resolve_data_directory()
            if data_dir:
                self.data_manager = DataDirectoryManager(
                    data_path=data_dir,
                    base_description=self.workflow_config.dataset_description,
                    file_listing_placeholder=self.workflow_config.file_listing_placeholder or "{file_listing}"
                )
                print(f"📁 Expanding dataset mode enabled")
                print(f"   Data directory: {data_dir}")

    def _seed_results_memory(self):
        """
        Load existing accumulated knowledge into ResultsMemoryManager if available.
        """
        seed_content = None
        seed_source = None

        if getattr(self.workflow_config, 'background_report', None):
            seed_content = self.workflow_config.background_report.strip()
            seed_source = "workflow_config.background_report"

        if seed_content:
            self.results_memory.current_report = seed_content
            print(f"   ✓ Seeded ResultsMemoryManager with {seed_source} ({len(seed_content)} characters)")

    def _resolve_data_directory(self) -> Optional[str]:
        """
        Resolve the data directory path for expanding dataset mode.
        
        If expanding_data_directory is set in config, use that.
        Otherwise, try to determine from the first ExecutionAgentCC's working_directory.
        
        Returns:
            str: Path to data directory, or None if cannot be determined
        """
        # Use explicit path if provided
        if self.workflow_config.expanding_data_directory:
            return self.workflow_config.expanding_data_directory
        
        # Try to determine from agent config
        if self.workflow_config.agent_config_list:
            for agent_config in self.workflow_config.agent_config_list:
                working_dir = getattr(agent_config, 'working_directory', None)
                if working_dir:
                    return str(Path(working_dir) / "data")
        
        # Fall back to output_home if available
        if self.workflow_config.output_home:
            return str(Path(self.workflow_config.output_home) / "data")
        
        print("⚠️ Warning: Could not determine data directory for expanding dataset mode")
        return None

    def refresh_dataset_description(self) -> str:
        """
        Refresh the dataset description with current data directory contents.
        
        This is called before each query when expanding_dataset_mode is enabled.
        
        Returns:
            str: Updated dataset description with current file listing
        """
        if not self.data_manager:
            return self.workflow_config.dataset_description
        
        # Scan for changes
        self.data_manager.scan()
        
        # Check for new files and log them
        new_files = self.data_manager.detect_new_files()
        if new_files:
            print(f"📁 Detected {len(new_files)} new data file(s):")
            for f in new_files[:5]:  # Show first 5
                print(f"   + {f}")
            if len(new_files) > 5:
                print(f"   + ... and {len(new_files) - 5} more")
        
        # Generate updated description
        return self.data_manager.update_dataset_description()

    @staticmethod
    def _extract_accumulated_report(raw_content: str) -> str:
        """
        Extract the accumulated report section from a memory_report.md file.
        """
        match = re.search(r'## Accumulated Report\s*\n+(.*)', raw_content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_content.strip()

    def _detect_last_completed_round(self, checkpoint_base: Path) -> int:
        """
        Detect the last completed round from checkpoint directory structure.
        A round is considered completed if it has a memory_report.md file.

        Args:
            checkpoint_base: Path to checkpoints directory

        Returns:
            int: Last completed round number, or -1 if no rounds completed
        """
        try:
            if not checkpoint_base.exists():
                return -1

            completed_rounds = []
            for item in checkpoint_base.iterdir():
                if item.is_dir() and item.name.startswith('round_'):
                    try:
                        round_num = int(item.name.split('_')[1])
                        # Check if round has memory_report.md (indicates completion)
                        memory_report = item / "memory_report.md"
                        if memory_report.exists():
                            completed_rounds.append(round_num)
                    except (ValueError, IndexError):
                        continue

            if completed_rounds:
                return max(completed_rounds)
            return -1

        except Exception as e:
            print(f"⚠️ Warning: Could not detect completed rounds: {str(e)}")
            return -1

    def _detect_incomplete_round_state(self, checkpoint_base: Path) -> Optional[Dict[str, Any]]:
        """
        Detect if there's an incomplete round that can be resumed from mid-round.

        An incomplete round is one that:
        - Has a round directory (round_N)
        - Has query checkpoints (query_XXX directories with checkpoint.json)
        - Does NOT have memory_report.md (which indicates completion)

        Args:
            checkpoint_base: Path to checkpoints directory

        Returns:
            Optional[Dict]: Information about the incomplete round, or None if no incomplete round found
                - 'round_num': The incomplete round number
                - 'last_query_num': The last completed query number
                - 'checkpoint_path': Path to the last checkpoint.json
                - 'config_path': Path to the round's config.json
        """
        try:
            if not checkpoint_base.exists():
                return None

            # Find incomplete rounds (have checkpoints but no memory_report.md)
            incomplete_rounds = []
            for item in checkpoint_base.iterdir():
                if item.is_dir() and item.name.startswith('round_'):
                    try:
                        round_num = int(item.name.split('_')[1])
                        memory_report = item / "memory_report.md"
                        config_file = item / "config.json"

                        # Check if round is incomplete (has config but no memory_report)
                        if config_file.exists() and not memory_report.exists():
                            # Find the highest query checkpoint
                            query_dirs = []
                            for query_item in item.iterdir():
                                if query_item.is_dir() and query_item.name.startswith('query_'):
                                    checkpoint_file = query_item / "checkpoint.json"
                                    if checkpoint_file.exists():
                                        try:
                                            query_num = int(query_item.name.split('_')[1])
                                            query_dirs.append((query_num, checkpoint_file))
                                        except (ValueError, IndexError):
                                            continue

                            if query_dirs:
                                # Sort by query number and get the highest
                                query_dirs.sort(key=lambda x: x[0], reverse=True)
                                last_query_num, last_checkpoint = query_dirs[0]

                                incomplete_rounds.append({
                                    'round_num': round_num,
                                    'last_query_num': last_query_num,
                                    'checkpoint_path': str(last_checkpoint),
                                    'config_path': str(config_file),
                                    'round_dir': str(item),
                                })
                    except (ValueError, IndexError):
                        continue

            if not incomplete_rounds:
                return None

            # Return the incomplete round with the highest round number
            incomplete_rounds.sort(key=lambda x: x['round_num'], reverse=True)
            return incomplete_rounds[0]

        except Exception as e:
            print(f"⚠️ Warning: Could not detect incomplete round state: {str(e)}")
            return None

    def _restore_workflow_from_checkpoint(
        self,
        checkpoint_path: str,
        round_idx: int,
    ) -> Optional[DirectorWorkflow]:
        """
        Restore a DirectorWorkflow from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint.json file
            round_idx: Round index for the restored workflow

        Returns:
            Optional[DirectorWorkflow]: Restored workflow, or None if restoration failed
        """
        try:
            print(f"🔄 Restoring workflow from checkpoint: {checkpoint_path}")

            # Use DirectorWorkflow.from_checkpoint to restore
            workflow = DirectorWorkflow.from_checkpoint(
                checkpoint_path_or_query=checkpoint_path,
                agent_config_list=self.workflow_config.agent_config_list,
                hypothesis_config=self.workflow_config.hypothesis_config,
                director_hypothesis_config=self.workflow_config.director_hypothesis_config,
                log_dir=self.workflow_config.log_dir,
                output_home=self.workflow_config.output_home,
                review_optimization_steps=self.workflow_config.review_optimization_steps,
                review_optimization_llm_config=self.workflow_config.review_optimization_llm_config,
                review_search=self.workflow_config.review_search,
                review_design=self.workflow_config.review_design,
                review_designer_llm_config=self.workflow_config.review_designer_llm_config,
                hold_reviews_in_exploratory_phase=self.workflow_config.hold_reviews_in_exploratory_phase,
                git_tracking=self.workflow_config.git_tracking,
                background_report=self.workflow_config.background_report,
            )

            # Set round number
            workflow.round_num = round_idx
            workflow.report_filename = None  # Don't save report to working dir

            # Update checkpoint manager's round number
            if hasattr(workflow, 'checkpoint_manager') and hasattr(workflow.checkpoint_manager, 'set_round_num'):
                workflow.checkpoint_manager.set_round_num(round_idx)


            # Pass data_manager for expanding dataset mode
            if self.data_manager:
                workflow.data_manager = self.data_manager

            print(f"   ✓ Workflow restored from query {workflow.query_num}")
            return workflow

        except Exception as e:
            print(f"❌ Failed to restore workflow from checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _init_workflows(
        self, 
        n_workflows: int,
        round_idx: int = 0,
        custom_task_description: Optional[str] = None
    ) -> List[DirectorWorkflow]:
        """
        Initialize multiple workflow instances with the same parameters.
        
        Args:
            n_workflows: Number of workflow instances to create
            round_idx: Current round index for output directory naming
            custom_task_description: Optional custom task description to override config
            
        Returns:
            List[DirectorWorkflow]: List of initialized workflow instances
        """
        print(f"🔧 Initializing {n_workflows} workflow instances for round {round_idx + 1}...")
        
        # Use custom task description if provided, otherwise use config
        task_desc = custom_task_description if custom_task_description is not None else self.workflow_config.task_description
        
        workflows = []
        for branch_idx in range(n_workflows):
            # DirectorWorkflow can optionally use a separate config for the DirectorAgent
            workflow = DirectorWorkflow(
                hypothesis_config=self.workflow_config.hypothesis_config,
                dataset_description=self.workflow_config.dataset_description,
                task_description=task_desc,
                agent_config_list=self.workflow_config.agent_config_list,
                max_queries=self.workflow_config.max_queries,
                log_dir=self.workflow_config.log_dir,
                images_dir=self.workflow_config.images_dir,
                output_home=self.workflow_config.output_home,
                multiturn_hypothesis=self.workflow_config.multiturn_hypothesis,
                review_optimization_steps=self.workflow_config.review_optimization_steps,
                review_optimization_llm_config=self.workflow_config.review_optimization_llm_config,
                review_search=self.workflow_config.review_search,
                review_design=self.workflow_config.review_design,
                review_designer_llm_config=self.workflow_config.review_designer_llm_config,
                explicit_hypothesis_step=self.workflow_config.explicit_hypothesis_step,
                exploratory_phase_warmup_steps=self.workflow_config.exploratory_phase_warmup_steps,
                hold_reviews_in_exploratory_phase=self.workflow_config.hold_reviews_in_exploratory_phase,
                background_path=self.workflow_config.background_path,
                background_report=self.workflow_config.background_report,
                git_tracking=self.workflow_config.git_tracking,
                report_filename=None,  # Don't save report to working dir (it's saved in checkpoints)
                director_hypothesis_config=self.workflow_config.director_hypothesis_config,
                generate_background_report=self.workflow_config.generate_background_report,
            )

            # Set round number for git tracking and checkpointing
            workflow.round_num = round_idx

            # Update checkpoint manager's round number for round-aware checkpointing
            if hasattr(workflow, 'checkpoint_manager') and hasattr(workflow.checkpoint_manager, 'set_round_num'):
                workflow.checkpoint_manager.set_round_num(round_idx)

            # Pass data_manager for expanding dataset mode
            # This allows the workflow to refresh dataset description after each query
            if self.data_manager:
                workflow.data_manager = self.data_manager

            workflows.append(workflow)
            print(f"   ✓ Workflow {branch_idx + 1}/{n_workflows} initialized")

        print(f"✅ All {n_workflows} workflows initialized for round {round_idx + 1}\n")
        
        return workflows

    def generate_next_round_prompts(self, round_idx: int, previous_results: List[Dict[str, Any]]) -> str:
        """
        Generate a modified task description for the next round based on previous results.
        
        Args:
            round_idx: Current round index
            previous_results: Results from previous rounds
            
        Returns:
            str: Modified task description for this round
        """
        # Default implementation: append round information to task description
        base_task = self.workflow_config.task_description or ""
        
        if round_idx == 0:
            return base_task
        
        # For subsequent rounds, you can incorporate insights from previous results
        round_info = f"\n\n--- Round {round_idx + 1} ---\n"
        round_info += f"Building upon insights from {len(previous_results)} previous workflow runs.\n"
        
        return base_task + round_info
    
    
    def run_all_branches(self) -> List[Dict[str, Any]]:
        """
        Run the DirectorWorkflow workflow multiple times across rounds and branches.

        For multiple rounds: creates workflow instances once and uses reset_for_new_round()
        between rounds to clear state while preserving configuration.

        In human_selection_mode:
        - Ignores num_rounds parameter
        - Runs one round at a time in step-by-step fashion
        - Pauses after exploratory round for human hypothesis selection
        - Resumes with selected hypothesis for next round

        Returns:
            List[Dict[str, Any]]: List of results from each workflow run
        """

        # Check if we're in human selection mode
        is_human_selection = self.workflow_config.hypothesis_config.human_selection_mode

        # Detect starting round from checkpoint structure (for resumption)
        start_round = 0
        mid_round_resume_state = None  # Will hold info if resuming mid-round

        if self.workflow_config.log_dir:
            checkpoint_base = Path(self.workflow_config.log_dir) / "checkpoints"
            last_completed = self._detect_last_completed_round(checkpoint_base)

            if last_completed >= 0:
                start_round = last_completed + 1
                print(f"📂 Detected {last_completed + 1} completed round(s)")

                # Check for incomplete round that can be resumed mid-round
                incomplete_state = self._detect_incomplete_round_state(checkpoint_base)
                if incomplete_state and incomplete_state['round_num'] == start_round:
                    # There's an incomplete round matching the next expected round
                    # This means the round was started but not completed
                    mid_round_resume_state = incomplete_state
                    print(f"🔄 Detected incomplete round {start_round} - will resume from mid-round")
                    print(f"   Last checkpoint: query {incomplete_state['last_query_num']}")
                    print(f"   Checkpoint path: {incomplete_state['checkpoint_path']}")
                else:
                    print(f"   Resuming from round {start_round} (fresh start)")
            else:
                # No completed rounds - check if there's an incomplete round 0
                incomplete_state = self._detect_incomplete_round_state(checkpoint_base)
                if incomplete_state and incomplete_state['round_num'] == 0:
                    mid_round_resume_state = incomplete_state
                    print(f"🔄 Detected incomplete round 0 - will resume from mid-round")
                    print(f"   Last checkpoint: query {incomplete_state['last_query_num']}")
                    print(f"   Checkpoint path: {incomplete_state['checkpoint_path']}")

        # Calculate end round (relative to start)
        end_round = start_round + self.num_rounds

        if is_human_selection:
            print(f"🧪 Starting HypothesisBrancher in HUMAN SELECTION MODE")
            print(f"   Will run {self.num_rounds} round(s): {start_round} to {end_round - 1}")
            print(f"   Branches: {self.num_branches}")
            print("=" * 80)
        else:
            print(f"🌳 Starting HypothesisBrancher with {self.num_branches} branches")
            print(f"   Will run {self.num_rounds} round(s): {start_round} to {end_round - 1}")
            print("=" * 80)

        # Initialize workflows - either restore from checkpoint or create fresh
        if mid_round_resume_state:
            # Restore workflow from mid-round checkpoint
            print(f"\n{'=' * 80}")
            print(f"🔄 MID-ROUND RESUMPTION")
            print(f"{'=' * 80}")
            print(f"Restoring workflow state from checkpoint...")

            restored_workflow = self._restore_workflow_from_checkpoint(
                checkpoint_path=mid_round_resume_state['checkpoint_path'],
                round_idx=mid_round_resume_state['round_num'],
            )

            if restored_workflow:
                workflows = [restored_workflow]
                print(f"✓ Workflow restored successfully")
                print(f"   Will resume from query {restored_workflow.query_num + 1}")
                print(f"{'=' * 80}\n")
            else:
                # Fallback to fresh start if restoration fails
                print(f"⚠️ Failed to restore from checkpoint, starting fresh")
                workflows = self._init_workflows(self.num_branches, round_idx=start_round)
                mid_round_resume_state = None  # Clear so we don't try special handling
        else:
            # Normal initialization - create fresh workflows
            workflows = self._init_workflows(self.num_branches, round_idx=start_round)

        for round_idx in range(start_round, end_round):
            print(f"\n🔄 Round {round_idx} (batch: {round_idx - start_round + 1}/{self.num_rounds})")
            print("=" * 80)

            # Expanding dataset mode: ensure data directory exists at round start
            if self.data_manager:
                if self.data_manager.ensure_data_directory_exists():
                    print(f"📁 Data directory ready: {self.data_manager.data_path}")
                    # Perform initial scan at round start
                    self.data_manager.scan()
                else:
                    print(f"⚠️ Warning: Could not create data directory: {self.data_manager.data_path}")

            # Save configuration checkpoint at the start of each round
            if hasattr(workflows[0], 'checkpoint_manager'):
                try:
                    # Merge important workflow_config flags into run_config for complete checkpointing
                    # This ensures these flags are preserved and can be used during resume
                    config_to_save = dict(self.run_config) if self.run_config else {}

                    config_to_save.update({
                        'review_optimization_steps': self.workflow_config.review_optimization_steps,
                        'hold_reviews_in_exploratory_phase': self.workflow_config.hold_reviews_in_exploratory_phase,
                    })

                    config_to_save.setdefault('workflow_config_flags', {})
                    config_to_save['workflow_config_flags'].update({
                        'git_tracking': self.workflow_config.git_tracking,
                        'multiturn_hypothesis': self.workflow_config.multiturn_hypothesis,
                        'review_search': self.workflow_config.review_search,
                        'review_design': self.workflow_config.review_design,
                        'hold_reviews_in_exploratory_phase': self.workflow_config.hold_reviews_in_exploratory_phase,
                        'expanding_dataset_mode': self.workflow_config.expanding_dataset_mode,
                        'generate_background_report': self.workflow_config.generate_background_report,
                        'review_optimization_steps': self.workflow_config.review_optimization_steps,
                    })

                    config_path = workflows[0].checkpoint_manager.save_round_config_checkpoint(
                        round_idx=round_idx,
                        config=config_to_save
                    )
                    if config_path:
                        print(f"💾 Round {round_idx} configuration checkpoint saved: {config_path}")
                    else:
                        print(f"⚠️ Warning: Failed to save configuration checkpoint for round {round_idx}")
                except Exception as e:
                    print(f"⚠️ Warning: Error saving configuration checkpoint: {str(e)}")

            # If not the first round of this batch, reset workflows and inject accumulated knowledge
            if round_idx > start_round:
                print(f"🔄 Resetting {self.num_branches} workflows for round {round_idx + 1}...")
                
                # Get accumulated knowledge from results memory
                accumulated_report = None
                if hasattr(self, 'results_memory') and self.results_memory.get_report() is not None:
                    accumulated_report = self.results_memory.get_report()
                    print(f"📚 Retrieved accumulated knowledge from results memory ({len(accumulated_report)} characters)")
                    # DEBUG: Detailed report inspection
                    print(f"DEBUG: accumulated_report type: {type(accumulated_report)}")
                    print(f"DEBUG: accumulated_report length: {len(accumulated_report) if accumulated_report else 0}")
                    print(f"DEBUG: accumulated_report preview: {accumulated_report[:200] if accumulated_report else 'NONE'}...")
                else:
                    print(f"DEBUG: No accumulated_report retrieved - hasattr(results_memory)={hasattr(self, 'results_memory')}, get_report()={self.results_memory.get_report() if hasattr(self, 'results_memory') else 'N/A'}")
                
                for branch_idx, workflow in enumerate(workflows):
                    # Update round number for git tracking (output_home stays the same - no branch directories)
                    workflow.report_filename = None  # Don't save report to working dir (it's saved in checkpoints)
                    workflow.round_num = round_idx  # Update round number for git tracking

                    # Update checkpoint manager's round number for round-aware checkpointing
                    if hasattr(workflow, 'checkpoint_manager') and hasattr(workflow.checkpoint_manager, 'set_round_num'):
                        workflow.checkpoint_manager.set_round_num(round_idx)

                    # Reset workflow state using new method
                    if hasattr(workflow, 'reset_for_new_round'):
                        workflow.reset_for_new_round()

                    # Inject accumulated knowledge into the workflow via background_report
                    # This keeps report clean for the current round's exploration only
                    if accumulated_report:
                        # Set background_report with accumulated knowledge from previous rounds
                        workflow.background_report = accumulated_report
                        
                        # Keep report and cumulative_report clean for this round
                        # (they were already reset by reset_for_new_round())
                        
                        print(f"   ✓ Branch {branch_idx + 1}: Injected accumulated knowledge into background_report ({len(accumulated_report)} chars)")

                print(f"✅ All workflows reset for round {round_idx + 1}\n")

            # Run each branch in this round
            assert len(workflows) == 1, "Only one workflow is supported for now"
            for branch_idx, workflow in enumerate(workflows):
                print(f"\n🌿 Round {round_idx}, Branch {branch_idx + 1}/{self.num_branches}")
                print("-" * 80)

                # In human selection mode, default to previous round's selected hypothesis if not provided
                if (is_human_selection and round_idx > 0 and
                        hasattr(workflow, 'hypothesis_config') and
                        getattr(workflow.hypothesis_config, 'selected_hypothesis_file', None) is None and
                        hasattr(workflow, 'checkpoint_manager')):
                    prev_round_path = (
                        Path(workflow.checkpoint_manager.checkpoint_base) /
                        f"round_{round_idx - 1}" / "selected_hypothesis.md"
                    )
                    if prev_round_path.exists():
                        workflow.hypothesis_config.selected_hypothesis_file = str(prev_round_path)
                        print(f"   📄 Using previous round selected hypothesis: {prev_round_path}")
                    else:
                        print(f"   ⚠️ No selected_hypothesis.md found for round {round_idx - 1}")

                # Get the output directory from the workflow
                branch_output = workflow.output_home

                # Resolve round directory for progress monitoring and live-log handler.
                round_dir: Optional[Path] = None
                try:
                    if hasattr(workflow, "checkpoint_manager"):
                        round_dir = workflow.checkpoint_manager.checkpoint_base / f"round_{round_idx}"
                        round_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                # Run the workflow
                try:
                    # NOTE: the live_round.log handler writes every log line from this round
                    # into a per-round file that the frontend polls for real-time visibility.
                    # It is always detached in the finally block regardless of success/failure.
                    if hasattr(workflow, "logger") and round_dir is not None:
                        workflow.logger.attach_round_log_handler(
                            round_dir, resume_marker=mid_round_resume_state is not None
                        )
                    progress_recorder.write_round_event(
                        round_dir,
                        "round_resumed" if mid_round_resume_state is not None else "round_started",
                    )
                    # DirectorWorkflow supports exploratory_mode for exploration rounds
                    # BUT: disable exploratory mode if we have a selected hypothesis file (resuming)
                    has_selected_hypothesis = (
                        hasattr(workflow, 'hypothesis_config') and
                        hasattr(workflow.hypothesis_config, 'selected_hypothesis_file') and
                        workflow.hypothesis_config.selected_hypothesis_file is not None
                    )

                    is_exploratory_round = (
                        self.num_exploration_rounds is not None and
                        round_idx < self.num_exploration_rounds and
                        not has_selected_hypothesis  # Don't explore if we have a selected hypothesis
                    )

                    # Log mode for clarity
                    if has_selected_hypothesis:
                        print(f"   Using selected hypothesis (exploratory mode disabled)")
                    elif is_exploratory_round:
                        print(f"   Running in exploratory mode (round {round_idx} < {self.num_exploration_rounds})")

                    results = workflow.run(
                        final_report_step=True,
                        exploratory_mode=is_exploratory_round
                    )

                    # Extract and verify final report (do this before checking pause status)
                    final_report = results.get('report', '')
                    report_path = results.get('report_path', '')

                    # Verify report was generated
                    if not final_report:
                        print(f"⚠️ Warning: No report content generated for Round {round_idx + 1}, Branch {branch_idx + 1}")
                    if not report_path:
                        # Expected when report_filename is None (reports saved via checkpoints)
                        if getattr(workflow, "report_filename", None) is None:
                            print(f"ℹ️ Info: Report path not returned for Round {round_idx + 1}, Branch {branch_idx + 1} (report saved in checkpoints)")
                        else:
                            print(f"⚠️ Warning: No report path returned for Round {round_idx + 1}, Branch {branch_idx + 1}")
                    else:
                        # Verify file exists
                        if Path(report_path).exists():
                            file_size = Path(report_path).stat().st_size
                            print(f"📄 Final report saved: {report_path} ({file_size} bytes)")
                        else:
                            print(f"⚠️ Warning: Report path {report_path} does not exist")

                    # Save the current round's report before memory integration
                    round_report_path = ""
                    memory_report_path = ""
                    if round_dir is not None:
                        progress_recorder.write_round_event(round_dir, "round_report_started")
                    if final_report and hasattr(workflow, 'checkpoint_manager'):
                        try:
                            round_report_path = workflow.checkpoint_manager.save_round_report_checkpoint(
                                round_idx=round_idx,
                                round_report=final_report
                            )
                            if round_report_path:
                                print(f"💾 Round {round_idx} report checkpoint saved: {round_report_path}")
                                if round_dir is not None:
                                    progress_recorder.write_round_event(round_dir, "round_report_saved")
                            else:
                                print(f"⚠️ Warning: Failed to save round report checkpoint")
                        except Exception as e:
                            print(f"⚠️ Warning: Error saving round report checkpoint: {str(e)}")

                    # Save the round log (snapshot of workflow log for this round)
                    if hasattr(workflow, 'checkpoint_manager'):
                        try:
                            round_log_path = workflow.checkpoint_manager.save_round_log(round_idx=round_idx)
                            if round_log_path:
                                if hasattr(workflow, "logger"):
                                    workflow.logger.log_action("round_log_checkpoint_saved",
                                        f"Round {round_idx} log saved to {round_log_path}")
                                print(f"💾 Round {round_idx} log checkpoint saved: {round_log_path}")
                        except Exception as e:
                            print(f"⚠️ Warning: Error saving round log: {str(e)}")

                    # Update results memory with this branch's report
                    # This accumulates knowledge across rounds
                    if hasattr(self, 'results_memory') and final_report:
                        print(f"💾 Updating results memory with report from Round {round_idx + 1}, Branch {branch_idx + 1}...")

                        # Update with verbose=False to avoid cluttering the output
                        # Set to True if you want detailed integration logs
                        self.results_memory.update_report(final_report, verbose=False)

                        # Get updated accumulated report stats
                        accumulated = self.results_memory.get_report()
                        if accumulated:
                            print(f"   ✓ Results memory updated (total: {len(accumulated)} characters)")

                            # Save the memory-integrated report to checkpoint
                            try:
                                if hasattr(workflow, 'checkpoint_manager'):
                                    memory_report_path = workflow.checkpoint_manager.save_memory_report_checkpoint(
                                        round_idx=round_idx,
                                        memory_report=accumulated
                                    )
                                    if memory_report_path:
                                        print(f"   💾 Memory report checkpoint saved: {memory_report_path}")
                                        if round_dir is not None:
                                            progress_recorder.write_round_event(round_dir, "memory_report_saved")
                                    else:
                                        print(f"   ⚠️ Warning: Failed to save memory report checkpoint")
                            except Exception as e:
                                print(f"   ⚠️ Warning: Error saving memory report checkpoint: {str(e)}")
                        else:
                            print(f"   ⚠️ Results memory update may have failed")

                    # Check if workflow paused for hypothesis selection
                    if results.get('status') == 'paused_for_selection':
                        # Only stop if this is the LAST round in the batch
                        is_last_round_in_batch = (round_idx == end_round - 1)

                        if is_last_round_in_batch:
                            print(f"\n{'='*80}")
                            print(f"⏸️  BRANCHER PAUSED: Waiting for hypothesis selection")
                            print(f"{'='*80}")
                            print(f"\n📋 Next steps:")
                            print(f"   1. Review candidates at: {results['candidates_path']}")
                            print(f"   2. Create a new file with your selected hypothesis (and optional feedback)")
                            print(f"   3. Resume execution with:")
                            print(f"      python <your_script>.py --resume --selected-hypothesis-file <your_file>.md")
                            print(f"\n💡 The workflow will automatically load your selection and continue research.")
                            print(f"{'='*80}\n")

                            # Save brancher state for this paused execution
                            self.branch_results.append({
                                'round_id': round_idx,
                                'branch_id': branch_idx,
                                'status': 'paused_for_selection',
                                'results': results,
                                'candidates_path': results['candidates_path'],
                                'n_candidates': results['n_candidates'],
                                'output_dir': branch_output
                            })

                            # Stop execution here - user must manually resume with selected hypothesis
                            print(f"✋ Execution stopped. Resume with your selected hypothesis to continue.\n")
                            return self.branch_results
                        else:
                            # Not the last round - this shouldn't happen in the new design
                            # But if it does, log it and continue
                            print(f"⚠️ Warning: Workflow generated candidates mid-batch (round {round_idx})")
                            print(f"   Continuing to next round instead of stopping...")
                            print(f"   Candidates saved at: {results['candidates_path']}\n")

                    # Append results to branch_results
                    self.branch_results.append({
                        'round_id': round_idx,
                        'branch_id': branch_idx,
                        'results': results,
                        'final_report': final_report,
                        'report_path': report_path,
                        'output_dir': branch_output,
                        'status': 'completed'
                    })
                    print(f"✅ Round {round_idx + 1}, Branch {branch_idx + 1} completed successfully")
                except Exception as e:
                    print(f"❌ Round {round_idx + 1}, Branch {branch_idx + 1} failed with error: {str(e)}")
                    self.branch_results.append({
                        'round_id': round_idx,
                        'branch_id': branch_idx,
                        'error': str(e),
                        'output_dir': branch_output,
                        'status': 'failed'
                    })
                finally:
                    if hasattr(workflow, "logger"):
                        workflow.logger.detach_round_log_handler()
                print("-" * 80)

        print("\n" + "=" * 80)
        print(f"🏁 HypothesisBrancher completed all rounds and branches")
        self._print_summary()

        # Save consolidated report if there were successful executions
        # Skip in human selection mode (user will handle reporting manually)
        if not is_human_selection and any(r['status'] == 'completed' for r in self.branch_results):
            try:
                consolidated_path = self.save_consolidated_report()
                print(f"\n📋 Consolidated report available at: {consolidated_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save consolidated report: {str(e)}")
        elif is_human_selection:
            print(f"\n📋 Skipping final report generation in human selection mode")

        return self.branch_results
    
    def _print_summary(self):
        """Print a summary of all branch results."""
        successful = sum(1 for r in self.branch_results if r['status'] == 'completed')
        failed = sum(1 for r in self.branch_results if r['status'] == 'failed')

        print(f"\n📊 Summary:")
        print(f"   - Total rounds: {self.num_rounds}")
        print(f"   - Total branches per round: {self.num_branches}")
        print(f"   - Total executions: {self.num_rounds * self.num_branches}")
        print(f"   - Successful: {successful}")
        print(f"   - Failed: {failed}")

        # Report information
        report_paths = self.get_final_report_paths()
        if report_paths:
            print(f"\n📄 Final Reports Generated: {len(report_paths)}")
            for path in report_paths:
                print(f"   - {path}")

        # ResultsMemoryManager status
        if hasattr(self, 'results_memory'):
            stats = self.results_memory.get_statistics()
            if stats['has_report']:
                print(f"\n💾 Results Memory:")
                print(f"   - Mode: {stats['mode']}")
                print(f"   - Total updates: {stats['update_count']}")
                print(f"   - Accumulated report size: {stats['report_length']} characters")
                print(f"   - Total cost: ${stats['total_cost']:.4f}")
                if 'total_redundancies_found' in stats:
                    print(f"   - Total redundancies found: {stats['total_redundancies_found']}")

        if failed > 0:
            print(f"\n❌ Failed executions:")
            for result in self.branch_results:
                if result['status'] == 'failed':
                    round_id = result.get('round_id', '?')
                    branch_id = result.get('branch_id', '?')
                    print(f"   - Round {round_id}, Branch {branch_id}: {result.get('error', 'Unknown error')}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get the results from all branches.
        
        Returns:
            List[Dict[str, Any]]: List of results from each branch
        """
        return self.branch_results
    
    def get_successful_results(self) -> List[Dict[str, Any]]:
        """
        Get only the successful results.

        Returns:
            List[Dict[str, Any]]: List of successful branch results
        """
        return [r for r in self.branch_results if r['status'] == 'completed']

    def get_final_reports(self) -> List[Dict[str, Any]]:
        """
        Get all final reports from successful executions.

        Returns:
            List[Dict[str, Any]]: List of dicts with report metadata:
                - round_id: Round number
                - branch_id: Branch number
                - report_path: Path to saved report file
                - final_report: Report content
                - output_dir: Output directory
        """
        reports = []
        for result in self.branch_results:
            if result['status'] == 'completed' and 'final_report' in result:
                reports.append({
                    'round_id': result.get('round_id'),
                    'branch_id': result.get('branch_id'),
                    'report_path': result.get('report_path'),
                    'final_report': result.get('final_report'),
                    'output_dir': result.get('output_dir')
                })
        return reports

    def get_final_report_paths(self) -> List[str]:
        """
        Get paths to all final report files from successful executions.

        Returns:
            List[str]: List of file paths to final reports
        """
        paths = []
        for result in self.branch_results:
            if result['status'] == 'completed' and 'report_path' in result:
                report_path = result.get('report_path')
                if report_path:
                    paths.append(report_path)
        return paths

    def save_consolidated_report(self, output_path: Optional[str] = None) -> str:
        """
        Save a consolidated report containing all branch reports.

        Args:
            output_path: Path to save consolidated report. If None, saves to
                        base output directory as 'consolidated_report.md'

        Returns:
            str: Path to the saved consolidated report
        """
        import os
        from pathlib import Path

        if output_path is None:
            base_dir = self.workflow_config.output_home or "outputs"
            output_path = os.path.join(base_dir, "consolidated_report.md")

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Build consolidated report
        content = f"""# Consolidated Report - HypothesisBrancher

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration
- Rounds: {self.num_rounds}
- Branches per round: {self.num_branches}
- Total executions: {self.num_rounds * self.num_branches}

## Dataset Description
{self.workflow_config.dataset_description}
"""

        if self.workflow_config.task_description:
            content += f"\n## Task Description\n{self.workflow_config.task_description}\n"

        content += "\n---\n\n"

        # Add each branch report
        successful_results = [r for r in self.branch_results if r['status'] == 'completed']

        for i, result in enumerate(successful_results, 1):
            round_id = result.get('round_id', '?')
            branch_id = result.get('branch_id', '?')
            final_report = result.get('final_report', 'No report available')
            report_path = result.get('report_path', 'Unknown')

            content += f"## Execution {i}: Round {round_id}, Branch {branch_id}\n\n"
            content += f"**Report File:** `{report_path}`\n\n"
            content += f"### Report Content\n\n{final_report}\n\n"
            content += "---\n\n"

        # Add accumulated knowledge from ResultsMemoryManager if available
        if hasattr(self, 'results_memory') and self.results_memory.get_report():
            content += "\n## Accumulated Knowledge (ResultsMemoryManager)\n\n"
            content += self.results_memory.get_report()
            content += "\n\n"

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"📄 Consolidated report saved: {output_path}")
        return output_path
