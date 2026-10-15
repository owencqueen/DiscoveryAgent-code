"""
Improved Checkpoint System for DiscoveryAgent Workflow

This module implements a robust, multi-layer checkpoint system that provides:
- Agent-aware state serialization/deserialization
- Query-based checkpoint organization
- Graceful error handling and validation
- Full conversation history preservation
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, is_dataclass
import datetime

from .utils.query_artifacts import query_directory


@dataclass
class ValidationResult:
    """Result of checkpoint validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class CheckpointManager:
    """
    Manages checkpoint creation, storage, and restoration for DiscoveryAgent workflows.
    
    Provides query-based checkpoint organization with full state preservation.
    """
    
    def __init__(self, log_dir: str, workflow, round_num: Optional[int] = None):
        """
        Initialize the checkpoint manager.

        Args:
            log_dir: Base logging directory
            workflow: DirectorWorkflow instance to checkpoint
            round_num: Optional round number for multi-round workflows (from HypothesisBrancher)
        """
        self.log_dir = Path(log_dir)
        self.checkpoint_base = self.log_dir / "checkpoints"
        self.checkpoint_base.mkdir(parents=True, exist_ok=True)

        self.workflow = workflow
        self.logger = getattr(workflow, 'logger', None)
        self.round_num = round_num  # Store round number for round-aware checkpointing

    def set_round_num(self, round_num: Optional[int]):
        """
        Update the round number for subsequent checkpoints.

        This should be called when the workflow moves to a new round
        (e.g., in HypothesisBrancher).

        Args:
            round_num: New round number (can be None to disable round-aware checkpointing)
        """
        self.round_num = round_num
        if self.logger:
            if round_num is not None:
                self.logger.log_action("checkpoint_round_updated", f"Checkpoint round number set to {round_num}")
            else:
                self.logger.log_action("checkpoint_round_updated", "Checkpoint round number cleared (legacy mode)")
    
    def _log_action(self, action: str, message: str):
        """Helper to log actions."""
        if self.logger:
            self.logger.log_action(action, message)
    
    def _log_error(self, message: str):
        """Helper to log errors."""
        if self.logger:
            self.logger.log_error(message)
    
    def get_checkpoint_dir(self, query_num: int, round_num: Optional[int] = None) -> Path:
        """
        Get checkpoint directory for specific query number and optional round number.

        Args:
            query_num: Query number
            round_num: Optional round number (defaults to self.round_num)

        Returns:
            Path: Checkpoint directory path in format:
                - With round: checkpoints/round_X/query_YY/
                - Without round: checkpoints/query_YY/ (legacy root layout)
        """
        # Use provided round_num or fall back to instance round_num
        effective_round = round_num if round_num is not None else self.round_num

        if effective_round is not None:
            return query_directory(self.checkpoint_base / f"round_{effective_round}", query_num)
        else:
            return query_directory(self.checkpoint_base, query_num)
    
    def _recursive_serialize(self, obj) -> Any:
        """Recursively convert objects into JSON-serializable structures."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
            
        if is_dataclass(obj):
            return self._recursive_serialize(asdict(obj))
            
        if isinstance(obj, Path):
            return str(obj)
            
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
            
        if isinstance(obj, dict):
            return {k: self._recursive_serialize(v) for k, v in obj.items()}
            
        if isinstance(obj, (list, tuple, set)):
            return [self._recursive_serialize(v) for v in obj]
        
        # Try JSON serialization
        try:
            json.dumps(obj)
            return obj
        except Exception:
            try:
                return str(obj)
            except Exception:
                return None
    
    def _create_checkpoint_data(self, query_num: int) -> Dict[str, Any]:
        """
        Create complete checkpoint data structure.

        Args:
            query_num: Current query number

        Returns:
            Dict containing all checkpoint data
        """
        # Core workflow state
        workflow_state = {
            'dataset_description': getattr(self.workflow, 'dataset_description', ''),
            'task_description': getattr(self.workflow, 'task_description', None),
            'max_queries': getattr(self.workflow, 'max_queries', 10),
            'query_num': query_num,
            'round_num': self.round_num,  # Include round number in checkpoint data
            'results': self._recursive_serialize(getattr(self.workflow, 'results', [])),
            'report': getattr(self.workflow, 'report', ''),
            'cumulative_report': getattr(self.workflow, 'cumulative_report', ''),
            'images': self._recursive_serialize(getattr(self.workflow, 'images', {})),
            'image_counter': getattr(self.workflow, 'image_counter', 0),
            'review_history': self._recursive_serialize(getattr(self.workflow, 'review_history', [])),
            'generated_reviewer_prompt': getattr(self.workflow, 'generated_reviewer_prompt', None),
            # Workflow configuration flags needed for proper restoration
            'git_tracking': getattr(self.workflow, 'git_tracking', False),
            'multiturn_hypothesis': getattr(self.workflow, 'multiturn_hypothesis', True),
            'review_search': getattr(self.workflow, 'review_search', False),
            'review_design': getattr(self.workflow, 'review_design', False),
        }
        
        # Agent states using new protocol
        agents_state = {}
        agents = getattr(self.workflow, 'agents', {})

        # Director agent (new workflow structure - handles query generation)
        if hasattr(self.workflow, 'director_agent'):
            dir_agent = self.workflow.director_agent
            if hasattr(dir_agent, 'save_state'):
                agents_state['director_agent'] = {
                    'type': type(dir_agent).__name__,
                    'state': dir_agent.save_state(),
                    'state_hash': self._compute_state_hash(dir_agent.save_state())
                }

        # Hypothesis generation agent (new workflow structure - handles hypothesis generation)
        if hasattr(self.workflow, 'hypothesis_gen_agent'):
            hyp_gen_agent = self.workflow.hypothesis_gen_agent
            if hasattr(hyp_gen_agent, 'save_state'):
                agents_state['hypothesis_gen_agent'] = {
                    'type': type(hyp_gen_agent).__name__,
                    'state': hyp_gen_agent.save_state(),
                    'state_hash': self._compute_state_hash(hyp_gen_agent.save_state())
                }

        # Legacy: Hypothesis agent (old workflow structure - for backward compatibility)
        if hasattr(self.workflow, 'hypothesis_agent'):
            hyp_agent = self.workflow.hypothesis_agent
            if hasattr(hyp_agent, 'save_state'):
                agents_state['hypothesis_agent'] = {
                    'type': type(hyp_agent).__name__,
                    'state': hyp_agent.save_state(),
                    'state_hash': self._compute_state_hash(hyp_agent.save_state())
                }
        
        # Tool agents
        for tag, agent in agents.items():
            if hasattr(agent, 'save_state'):
                agent_state = agent.save_state()
                agents_state[tag] = {
                    'type': type(agent).__name__,
                    'state': agent_state,
                    'state_hash': self._compute_state_hash(agent_state)
                }

        # Validation data
        validation_data = {
            'state_hashes': {k: v.get('state_hash', '') for k, v in agents_state.items()},
            'checkpoint_hash': '',  # Will be computed after serialization
            'timestamp': datetime.datetime.now().isoformat()
        }

        checkpoint = {
            'checkpoint_version': '2.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'query_num': query_num,
            'workflow_state': workflow_state,
            'agents': agents_state,
            'validation': validation_data
        }
        
        # Compute overall checkpoint hash
        checkpoint_json = json.dumps(checkpoint, sort_keys=True, default=str)
        checkpoint_hash = hashlib.sha256(checkpoint_json.encode()).hexdigest()
        checkpoint['validation']['checkpoint_hash'] = checkpoint_hash
        
        return checkpoint
    
    def _compute_state_hash(self, state: Dict[str, Any]) -> str:
        """Compute hash of agent state for validation."""
        try:
            state_json = json.dumps(state, sort_keys=True, default=str)
            return hashlib.sha256(state_json.encode()).hexdigest()
        except Exception:
            return ""
    
    def save_checkpoint(self, query_num: int, force_full: bool = False) -> str:
        """
        Save checkpoint for specific query iteration.
        
        Args:
            query_num: Current query number
            force_full: Force full checkpoint (not incremental)
            
        Returns:
            str: Path to saved checkpoint file
        """
        try:
            checkpoint_dir = self.get_checkpoint_dir(query_num)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create checkpoint data
            checkpoint_data = self._create_checkpoint_data(query_num)
            
            # Save main checkpoint file
            checkpoint_path = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Update latest symlink
            self._update_latest_link(query_num)
            
            self._log_action("checkpoint_saved", 
                f"Query {query_num} checkpoint saved to {checkpoint_path}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            self._log_error(f"Failed to save checkpoint for query {query_num}: {str(e)}")
            return ""
    
    def save_hypothesis_checkpoint(self, hypothesis_output: str) -> str:
        """
        Save initial hypothesis generation as hypothesis_000 checkpoint.

        This is called when init_generate_hypothesis=True and saves the initial
        hypothesis generation output before the main query loop begins.

        Args:
            hypothesis_output: The generated hypothesis text

        Returns:
            str: Path to saved hypothesis checkpoint file
        """
        try:
            # Use query_num = 0 for the initial hypothesis
            checkpoint_dir = self.get_checkpoint_dir(0)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Create minimal checkpoint data for hypothesis
            checkpoint_data = {
                'checkpoint_version': '2.0',
                'timestamp': datetime.datetime.now().isoformat(),
                'query_num': 0,
                'checkpoint_type': 'initial_hypothesis',
                'workflow_state': {
                    'dataset_description': getattr(self.workflow, 'dataset_description', ''),
                    'task_description': getattr(self.workflow, 'task_description', None),
                    'max_queries': getattr(self.workflow, 'max_queries', 10),
                    'query_num': 0,
                    'round_num': self.round_num,
                },
                'hypothesis_output': hypothesis_output
            }

            # Save main checkpoint file
            checkpoint_path = checkpoint_dir / "checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            # Save hypothesis as markdown file
            markdown_content = f"# Initial Hypothesis Generation\n\n"
            markdown_content += f"**Timestamp:** {datetime.datetime.now().isoformat()}\n\n"
            markdown_content += f"## Dataset Description\n\n{getattr(self.workflow, 'dataset_description', '')}\n\n"

            task_desc = getattr(self.workflow, 'task_description', None)
            if task_desc:
                markdown_content += f"## Task Description\n\n{task_desc}\n\n"

            background_report = getattr(self.workflow, 'background_report', None)
            if background_report:
                markdown_content += f"## Background Report\n\n{background_report}\n\n"

            markdown_content += f"## Generated Hypotheses\n\n{hypothesis_output}\n"

            markdown_path = checkpoint_dir / "hypothesis_000.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self._log_action("hypothesis_checkpoint_saved",
                f"Initial hypothesis checkpoint saved to {checkpoint_path}")

            return str(checkpoint_path)

        except Exception as e:
            self._log_error(f"Failed to save hypothesis checkpoint: {str(e)}")
            return ""

    def load_selected_hypothesis(self, hypothesis_file_path: str) -> Optional[str]:
        """
        Load the human-selected hypothesis from a markdown file.

        The file should contain the hypothesis text and optionally scientist feedback.
        This method reads the entire file contents.

        Args:
            hypothesis_file_path: Path to markdown file containing selected hypothesis

        Returns:
            Optional[str]: Selected hypothesis text (entire file contents), or None if failed
        """
        try:
            path = Path(hypothesis_file_path)
            if not path.exists():
                self._log_error(f"Selected hypothesis file not found: {hypothesis_file_path}")
                return None

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                self._log_error("Selected hypothesis file is empty")
                return None

            self._log_action("selected_hypothesis_loaded",
                f"Loaded selected hypothesis from {hypothesis_file_path} ({len(content)} chars)")

            return content

        except Exception as e:
            self._log_error(f"Failed to load selected hypothesis: {str(e)}")
            return None

    def save_reviewer_prompt_checkpoint(self, reviewer_prompt: str, designer_model: str = None) -> str:
        """
        Save the designed reviewer prompt at the top level of checkpointing.

        This is called once at workflow initialization when review_design=True.
        The prompt is saved at the checkpoint base level (or round level if using rounds)
        so it's available for the entire run/round.

        Args:
            reviewer_prompt: The generated reviewer system prompt
            designer_model: Optional name of the model used to design the prompt

        Returns:
            str: Path to saved reviewer prompt markdown file
        """
        try:
            # Determine where to save based on round awareness
            if self.round_num is not None:
                # Save in round directory for round-aware checkpointing
                save_dir = self.checkpoint_base / f"round_{self.round_num}"
            else:
                # Save at checkpoint base for legacy mode
                save_dir = self.checkpoint_base

            save_dir.mkdir(parents=True, exist_ok=True)

            # Create markdown content
            markdown_content = f"# Reviewer System Prompt\n\n"
            markdown_content += f"**Timestamp:** {datetime.datetime.now().isoformat()}\n\n"

            if self.round_num is not None:
                markdown_content += f"**Round:** {self.round_num}\n\n"

            if designer_model:
                markdown_content += f"**Designer Model:** {designer_model}\n\n"

            markdown_content += f"**Dataset:** {getattr(self.workflow, 'dataset_description', 'N/A')[:100]}...\n\n"

            task_desc = getattr(self.workflow, 'task_description', None)
            if task_desc:
                markdown_content += f"**Task:** {task_desc[:100]}...\n\n"

            markdown_content += "---\n\n"
            markdown_content += "## Generated Reviewer System Prompt\n\n"
            markdown_content += f"{reviewer_prompt}\n"

            # Save markdown file
            markdown_path = save_dir / "reviewer_prompt.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Also save as JSON for easy programmatic access
            json_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'round_num': self.round_num,
                'designer_model': designer_model,
                'reviewer_prompt': reviewer_prompt,
                'dataset_description': getattr(self.workflow, 'dataset_description', ''),
                'task_description': getattr(self.workflow, 'task_description', None)
            }

            json_path = save_dir / "reviewer_prompt.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            self._log_action("reviewer_prompt_checkpoint_saved",
                f"Reviewer prompt saved to {markdown_path}")

            return str(markdown_path)

        except Exception as e:
            self._log_error(f"Failed to save reviewer prompt checkpoint: {str(e)}")
            return ""

    def save_round_config_checkpoint(self, round_idx: int, config: Dict[str, Any]) -> str:
        """
        Save the run configuration metadata for a specific round.

        This saves all parameters necessary to reproduce the run, including model
        configurations, feature flags, file paths, and execution parameters.

        Args:
            round_idx: Round index for which to save configuration
            config: Dictionary containing all run configuration parameters

        Returns:
            str: Path to saved config JSON file
        """
        try:
            # Determine save directory based on round awareness
            if self.round_num is not None:
                # Save in the round directory for round-aware checkpointing
                save_dir = self.checkpoint_base / f"round_{round_idx}"
            else:
                # Save at checkpoint base for legacy mode
                save_dir = self.checkpoint_base

            save_dir.mkdir(parents=True, exist_ok=True)

            # Add metadata about when this config was saved
            config_with_metadata = {
                'checkpoint_timestamp': datetime.datetime.now().isoformat(),
                'round_idx': round_idx,
                'config': config
            }

            # Save as JSON
            config_path = save_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_with_metadata, f, indent=2, ensure_ascii=False, default=str)

            self._log_action("round_config_checkpoint_saved",
                f"Round {round_idx} config saved to {config_path}")

            return str(config_path)

        except Exception as e:
            self._log_error(f"Failed to save round config checkpoint for round {round_idx}: {str(e)}")
            return ""

    def save_round_report_checkpoint(self, round_idx: int, round_report: str) -> str:
        """
        Save the current round's report before ResultsMemoryManager integration.

        This method saves the report generated by the current round before it's
        passed to ResultsMemoryManager for integration with accumulated knowledge.
        The report is saved at the round level as round_report.md.

        Args:
            round_idx: Round index for which the report was generated
            round_report: The report content from the current round

        Returns:
            str: Path to saved round report markdown file
        """
        try:
            # Determine save directory based on round awareness
            if self.round_num is not None:
                # Save in the round directory for round-aware checkpointing
                save_dir = self.checkpoint_base / f"round_{round_idx}"
            else:
                # Save at checkpoint base for legacy mode
                save_dir = self.checkpoint_base

            save_dir.mkdir(parents=True, exist_ok=True)

            # Create markdown content
            markdown_content = f"# Round {round_idx} Report\n\n"
            markdown_content += f"**Timestamp:** {datetime.datetime.now().isoformat()}\n\n"
            markdown_content += f"**Round:** {round_idx}\n\n"
            markdown_content += "---\n\n"
            markdown_content += "## Description\n\n"
            markdown_content += f"This report contains the findings from Round {round_idx} only, "
            markdown_content += "before integration with ResultsMemoryManager. This represents the "
            markdown_content += "raw output from this round's research queries.\n\n"
            markdown_content += "---\n\n"
            markdown_content += "## Round Report\n\n"
            markdown_content += f"{round_report}\n"

            # Save markdown file
            markdown_path = save_dir / "round_report.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self._log_action("round_report_checkpoint_saved",
                f"Round {round_idx} report saved to {markdown_path}")

            return str(markdown_path)

        except Exception as e:
            self._log_error(f"Failed to save round report checkpoint for round {round_idx}: {str(e)}")
            return ""

    def save_memory_report_checkpoint(self, round_idx: int, memory_report: str) -> str:
        """
        Save the memory-updated report after ResultsMemoryManager integration.

        This method saves the accumulated report from ResultsMemoryManager after it has
        integrated the round's findings. The report is saved at the round level as
        memory_report.md.

        Args:
            round_idx: Round index for which the memory report was generated
            memory_report: The accumulated report content from ResultsMemoryManager

        Returns:
            str: Path to saved memory report markdown file
        """
        try:
            # Determine save directory based on round awareness
            if self.round_num is not None:
                # Save in the round directory for round-aware checkpointing
                save_dir = self.checkpoint_base / f"round_{round_idx}"
            else:
                # Save at checkpoint base for legacy mode
                save_dir = self.checkpoint_base

            save_dir.mkdir(parents=True, exist_ok=True)

            # Create markdown content
            markdown_content = f"# Memory-Integrated Report - Round {round_idx}\n\n"
            markdown_content += f"**Timestamp:** {datetime.datetime.now().isoformat()}\n\n"
            markdown_content += f"**Round:** {round_idx}\n\n"
            markdown_content += "---\n\n"
            markdown_content += "## Description\n\n"
            markdown_content += "This report represents the accumulated knowledge from ResultsMemoryManager "
            markdown_content += f"after integrating findings from Round {round_idx}. It contains synthesized "
            markdown_content += "insights across all rounds completed so far.\n\n"
            markdown_content += "---\n\n"
            markdown_content += "## Accumulated Report\n\n"
            markdown_content += f"{memory_report}\n"

            # Save markdown file
            markdown_path = save_dir / "memory_report.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self._log_action("memory_report_checkpoint_saved",
                f"Memory report for round {round_idx} saved to {markdown_path}")

            return str(markdown_path)

        except Exception as e:
            self._log_error(f"Failed to save memory report checkpoint for round {round_idx}: {str(e)}")
            return ""

    def save_round_log(self, round_idx: int) -> str:
        """
        Save the current workflow log as round.log in the round directory.

        This method copies the current state of the workflow log file to the round
        directory, providing a snapshot of all logging for that round. This replaces
        the previous per-query log copying which created redundant duplicate files.

        Args:
            round_idx: Round index for which to save the log

        Returns:
            str: Path to saved round.log file, or empty string on failure
        """
        try:
            # Determine save directory based on round awareness
            if self.round_num is not None:
                save_dir = self.checkpoint_base / f"round_{round_idx}"
            else:
                save_dir = self.checkpoint_base

            save_dir.mkdir(parents=True, exist_ok=True)

            # Get the workflow log path
            if hasattr(self.workflow, 'logger') and hasattr(self.workflow.logger, 'get_log_path'):
                source_log = self.workflow.logger.get_log_path()
                if source_log and Path(source_log).exists():
                    dest_log = save_dir / "round.log"
                    shutil.copy2(source_log, dest_log)

                    self._log_action("round_log_saved",
                        f"Round {round_idx} log saved to {dest_log}")

                    return str(dest_log)
                else:
                    self._log_error(f"Source log file does not exist: {source_log}")
                    return ""
            else:
                self._log_error("Workflow logger not available for round log save")
                return ""

        except Exception as e:
            self._log_error(f"Failed to save round log for round {round_idx}: {str(e)}")
            return ""

    def save_review_checkpoint(self, query_num: int, reviewer_input: str, reviewer_output: str) -> str:
        """
        Save a review checkpoint to the round-level reviewer_logs directory.

        This method saves review data as:
        - checkpoints/round_i/reviewer_logs/review_{round}.md

        Reviews are appended to the same file for each round, creating a complete
        log of all reviewer feedback within that round.

        Args:
            query_num: Query number for which this review was generated
            reviewer_input: The exact prompt/input sent to the reviewer
            reviewer_output: The feedback/output generated by the reviewer

        Returns:
            str: Path to saved review markdown file
        """
        try:
            # Determine the round directory
            if self.round_num is not None:
                round_dir = self.checkpoint_base / f"round_{self.round_num}"
            else:
                round_dir = self.checkpoint_base

            # Create reviewer_logs directory within the round
            reviewer_logs_dir = round_dir / "reviewer_logs"
            reviewer_logs_dir.mkdir(parents=True, exist_ok=True)

            # Determine the review file path
            round_num_for_file = self.round_num if self.round_num is not None else 0
            review_file_path = reviewer_logs_dir / f"review_{round_num_for_file}.md"

            # Create review content for this query
            review_content = f"\n## Review for Query {query_num}\n\n"
            review_content += f"**Timestamp:** {datetime.datetime.now().isoformat()}\n\n"
            review_content += "### Reviewer Output\n\n"
            review_content += f"{reviewer_output}\n\n"
            review_content += "---\n"

            # Check if file exists - if so, append; if not, create with header
            if review_file_path.exists():
                # Append to existing file
                with open(review_file_path, 'a', encoding='utf-8') as f:
                    f.write(review_content)
            else:
                # Create new file with header
                header = f"# Reviewer Logs - Round {round_num_for_file}\n\n"
                header += f"**Created:** {datetime.datetime.now().isoformat()}\n\n"
                header += "This file contains all reviewer feedback for this round.\n\n"
                header += "---\n"
                with open(review_file_path, 'w', encoding='utf-8') as f:
                    f.write(header + review_content)

            self._log_action("review_checkpoint_saved",
                f"Review for query {query_num} saved to {review_file_path}")

            return str(review_file_path)

        except Exception as e:
            self._log_error(f"Failed to save review checkpoint for query {query_num}: {str(e)}")
            return ""
    
    def _save_queries_and_reports_markdown(self, checkpoint_dir: Path, query_num: int):
        """
        Save queries and reports as easy-to-read markdown files in the checkpoint directory.
        
        Args:
            checkpoint_dir: Directory to save markdown files
            query_num: Current query number for context
        """
        try:
            # Get the current query result from workflow results
            results = getattr(self.workflow, 'results', [])
            current_result = None
            
            # Find the result for this query number
            for result in results:
                if result.get('query_num') == query_num:
                    current_result = result
                    break
            
            if not current_result:
                self._log_error(f"No result found for query {query_num}")
                return
            
            # Create markdown content
            markdown_content = f"# Query {query_num} Details\n\n"
            markdown_content += f"**Timestamp:** {current_result.get('timestamp', 'Unknown')}\n\n"
            
            # Add hypothesis input if available
            hypothesis_input = self._extract_hypothesis_input(query_num)
            if hypothesis_input:
                markdown_content += f"# Hypothesis Input\n\n{hypothesis_input}\n\n"
            
            # Add query content
            markdown_content += "# Query\n\n"
            query_raw = current_result.get('query_raw', '')
            query_parsed = current_result.get('query_parsed', '')
            
            if query_raw:
                markdown_content += f"**Raw Query:**\n```\n{query_raw}\n```\n\n"
            
            if query_parsed and query_parsed != query_raw:
                markdown_content += f"**Parsed Query:**\n```\n{query_parsed}\n```\n\n"
            
            # Add report/result content with agent name detection
            result_content = current_result.get('result', '')
            agent_name = self._extract_agent_name_from_query(query_raw)
            
            if agent_name:
                markdown_content += f"# {agent_name} Report\n\n"
            else:
                markdown_content += "# Agent Report\n\n"
            
            markdown_content += f"{result_content}\n\n"
            
            # Save to markdown file
            markdown_path = checkpoint_dir / f"query_{query_num:03d}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            self._log_action("markdown_saved", f"Query {query_num} markdown saved to {markdown_path}")
            
        except Exception as e:
            self._log_error(f"Failed to save queries and reports markdown for query {query_num}: {str(e)}")
    
    def _extract_hypothesis_input(self, query_num: int) -> str:
        """
        Extract the hypothesis agent input for a specific query.
        This looks at the hypothesis agent's conversation history.
        
        Args:
            query_num: Query number to extract input for
            
        Returns:
            str: The hypothesis input content, or empty string if not found
        """
        try:
            # Get hypothesis agent from workflow
            hyp_agent = getattr(self.workflow, 'hypothesis_agent', None)
            if not hyp_agent or not hasattr(hyp_agent, 'messages'):
                return ""
            
            # The hypothesis agent maintains a conversation history
            # We want the user message that led to generating this query
            messages = hyp_agent.messages
            
            # Find user messages that might correspond to this query
            # Since it's multiturn, we need to find the relevant user input
            for i, message in enumerate(messages):
                if message.get('role') == 'user':
                    content = message.get('content', '')
                    # Check if this looks like it could be for this query
                    if any(keyword in content.lower() for keyword in ['query', 'dataset', 'task', 'report']):
                        return content
            
            # If no specific match, return the last user message
            user_messages = [msg for msg in messages if msg.get('role') == 'user']
            if user_messages:
                return user_messages[-1].get('content', '')
                
            return ""
            
        except Exception as e:
            self._log_error(f"Failed to extract hypothesis input for query {query_num}: {str(e)}")
            return ""
    
    def _extract_agent_name_from_query(self, query_raw: str) -> str:
        """
        Extract the agent name from a query string by looking for XML tags.
        
        Args:
            query_raw: Raw query string
            
        Returns:
            str: Agent name if found, empty string otherwise
        """
        try:
            import re
            # Look for XML-style tags like <execution_agent>...</execution_agent>
            match = re.search(r'<(\w+)>', query_raw)
            if match:
                tag = match.group(1)
                # Convert tag to a more readable name
                return tag.replace('_', ' ').title()
            return ""
        except Exception:
            return ""

    def _update_latest_link(self, query_num: int):
        """Update the 'latest' symlink to point to the newest checkpoint."""
        try:
            if self.round_num is not None:
                # Round-aware: create latest link inside round directory
                round_dir = self.checkpoint_base / f"round_{self.round_num}"
                latest_link = round_dir / "latest"
                target = self.get_checkpoint_dir(query_num).name
            else:
                # Legacy: latest link at checkpoint base
                latest_link = self.checkpoint_base / "latest"
                target = self.get_checkpoint_dir(query_num).name

            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(target)
        except Exception as e:
            self._log_error(f"Failed to update latest symlink: {str(e)}")
    
    def validate_checkpoint(self, checkpoint_path: str) -> ValidationResult:
        """
        Validate a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            ValidationResult: Validation results
        """
        errors = []
        warnings = []
        
        try:
            path = Path(checkpoint_path)
            if not path.exists():
                errors.append(f"Checkpoint file not found: {checkpoint_path}")
                return ValidationResult(False, errors, warnings)
            
            # Load and validate JSON structure
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in checkpoint: {str(e)}")
                return ValidationResult(False, errors, warnings)
            
            # Check required fields
            required_fields = ['checkpoint_version', 'query_num', 'workflow_state', 'agents']
            for field in required_fields:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
            
            # Validate version
            if data.get('checkpoint_version') != '2.0':
                warnings.append(f"Unexpected checkpoint version: {data.get('checkpoint_version')}")
            
            # Validate state hashes if present
            if 'validation' in data and 'state_hashes' in data['validation']:
                for agent_name, expected_hash in data['validation']['state_hashes'].items():
                    if agent_name in data['agents']:
                        agent_state = data['agents'][agent_name]['state']
                        computed_hash = self._compute_state_hash(agent_state)
                        if computed_hash != expected_hash:
                            warnings.append(f"State hash mismatch for {agent_name}")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid, errors, warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return ValidationResult(False, errors, warnings)
    
    def load_checkpoint(self, query_num: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data.
        
        Args:
            query_num: Specific query number to load (None for latest)
            
        Returns:
            Optional[Dict]: Checkpoint data or None if failed
        """
        try:
            if query_num is None:
                # Load latest
                latest_link = self.checkpoint_base / "latest"
                if latest_link.exists():
                    checkpoint_path = latest_link / "checkpoint.json"
                else:
                    # Find highest query number
                    query_dirs = []
                    for entry in self.checkpoint_base.iterdir():
                        if entry.is_dir() and entry.name.startswith('query_'):
                            try:
                                num = int(entry.name.split('_')[1])
                                query_dirs.append(num)
                            except ValueError:
                                continue
                    if not query_dirs:
                        self._log_error("No checkpoints found")
                        return None
                    query_num = max(query_dirs)
                    checkpoint_path = self.get_checkpoint_dir(query_num) / "checkpoint.json"
            else:
                checkpoint_path = self.get_checkpoint_dir(query_num) / "checkpoint.json"
            
            # Validate before loading
            validation = self.validate_checkpoint(str(checkpoint_path))
            if not validation.is_valid:
                self._log_error(f"Checkpoint validation failed: {', '.join(validation.errors)}")
                return None
            
            if validation.warnings:
                for warning in validation.warnings:
                    self._log_error(f"Checkpoint warning: {warning}")
            
            # Load checkpoint data
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._log_action("checkpoint_loaded", f"Loaded checkpoint from query {query_num}")
            return data
            
        except Exception as e:
            self._log_error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def list_checkpoints(self, round_num: Optional[int] = None) -> List[Tuple[int, Optional[int], str, str]]:
        """
        List all available checkpoints, optionally filtered by round number.

        Args:
            round_num: Optional round number to filter by (None = all rounds)

        Returns:
            List[Tuple[int, Optional[int], str, str]]: List of (query_num, round_num, path, timestamp) tuples
        """
        checkpoints = []
        try:
            # Check for round-aware structure
            round_dirs = [d for d in self.checkpoint_base.iterdir() if d.is_dir() and d.name.startswith('round_')]

            if round_dirs:
                # Round-aware structure exists
                for round_dir in round_dirs:
                    try:
                        round_number = int(round_dir.name.split('_')[1])

                        # Skip if filtering by specific round
                        if round_num is not None and round_number != round_num:
                            continue

                        # List queries in this round
                        for entry in round_dir.iterdir():
                            if entry.is_dir() and entry.name.startswith('query_'):
                                try:
                                    query_num = int(entry.name.split('_')[1])
                                    checkpoint_file = entry / "checkpoint.json"
                                    if checkpoint_file.exists():
                                        # Get timestamp from checkpoint
                                        try:
                                            with open(checkpoint_file, 'r') as f:
                                                data = json.load(f)
                                            timestamp = data.get('timestamp', 'unknown')
                                        except Exception:
                                            timestamp = 'unknown'

                                        checkpoints.append((query_num, round_number, str(entry), timestamp))
                                except ValueError:
                                    continue
                    except ValueError:
                        continue
            else:
                # Legacy structure (no rounds)
                for entry in self.checkpoint_base.iterdir():
                    if entry.is_dir() and entry.name.startswith('query_'):
                        try:
                            query_num = int(entry.name.split('_')[1])
                            checkpoint_file = entry / "checkpoint.json"
                            if checkpoint_file.exists():
                                # Get timestamp from checkpoint
                                try:
                                    with open(checkpoint_file, 'r') as f:
                                        data = json.load(f)
                                    timestamp = data.get('timestamp', 'unknown')
                                except Exception:
                                    timestamp = 'unknown'

                                checkpoints.append((query_num, None, str(entry), timestamp))
                        except ValueError:
                            continue

            # Sort by round number (if present) then query number
            checkpoints.sort(key=lambda x: (x[1] if x[1] is not None else -1, x[0]))
            return checkpoints

        except Exception as e:
            self._log_error(f"Failed to list checkpoints: {str(e)}")
            return []
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5, round_num: Optional[int] = None) -> int:
        """
        Remove old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
            round_num: Optional round number to clean up (None = clean all rounds)

        Returns:
            int: Number of checkpoints removed
        """
        try:
            checkpoints = self.list_checkpoints(round_num=round_num)
            if len(checkpoints) <= keep_last_n:
                return 0

            # Remove oldest checkpoints
            to_remove = checkpoints[:-keep_last_n]
            removed_count = 0

            for query_num, checkpoint_round, checkpoint_path, _ in to_remove:
                try:
                    shutil.rmtree(checkpoint_path)
                    removed_count += 1
                    if checkpoint_round is not None:
                        self._log_action("checkpoint_cleanup",
                            f"Removed old checkpoint round_{checkpoint_round}/query_{query_num:03d}")
                    else:
                        self._log_action("checkpoint_cleanup", f"Removed old checkpoint query_{query_num:03d}")
                except Exception as e:
                    self._log_error(f"Failed to remove checkpoint {checkpoint_path}: {str(e)}")

            return removed_count

        except Exception as e:
            self._log_error(f"Checkpoint cleanup failed: {str(e)}")
            return 0
