"""
Director-Based Workflow

This workflow implements a hypothesis-directed research approach where:
1. HypothesisGenAgent generates an initial hypothesis at the start (always, not optional)
2. DirectorAgent receives the hypothesis and directs research to test it
3. Tool agents execute the research queries
4. The workflow focuses on testing and validating the proposed hypothesis

HypothesisGenAgent generates a hypothesis, and DirectorAgent tests it through
queries to the registered tool agents.

UNCERTAINTIES & NOTES:
1. Hypothesis Presentation: Currently using feedback mechanism to show hypothesis
   to director. Alternative would be to add it to dataset_description or as a
   separate parameter to generate_query(). Current approach keeps it visible
   in logs and allows for easy modification.

2. Resume Behavior: When resuming, we assume director agent state contains
   hypothesis context. Could alternatively load hypothesis from checkpoint
   and re-inject it. Need to test resume behavior.

3. Hypothesis Storage: hypothesis_output is stored in local variable but not
   as instance variable. Should we store self.hypothesis_output for access
   later (e.g., in final report)?

4. Multiturn Context: Both director and hypothesis agents maintain their own
   conversation contexts. The hypothesis context after generation is not
   explicitly transferred to director (only via the feedback text). This
   seems intentional but worth noting.

5. Final Report: Currently uses director_agent.generate_final_report() which
   will see its own conversation history. Should final report mention the
   original hypothesis? Currently implicit in director's context.
"""

import re
import inspect
import anthropic
import os
import json
import datetime
import pathlib
import time
import threading
import itertools
import copy
import dataclasses
from pathlib import Path
import shutil

from typing import Optional, List, Dict, Any
from termcolor import colored
from .utils.logger import Logger
from .utils import progress_recorder
from .utils.query_artifacts import (
    artifact_filename,
    persist_query_input,
    persist_query_report,
    query_directory,
    read_query_artifact,
)
from .utils.llm_diagnostics import REFUSAL_ADVICE, first_refusal
from .config import LLMConfig, HypothesisGenerationConfig, get_default_hypothesis_config
from .tool_agents.execution_agent_cc import ExecutionAgentCC
from .checkpoint_manager import CheckpointManager
from .tool_agents.deep_research import DeepResearchAgent, DeepResearchConfig
from .prompts.tools import DEEP_RESEARCH_TAG
from .hyp_agents.director_agent import DirectorAgent
from .hyp_agents.hypothesis_gen_agent import HypothesisGenAgent
from .prompts.hyp_generation import (
    EVIDENCE_DRIVEN_HYPOTHESIS_GENERATION_SYSTEM_PROMPT,
    DATASET_DESCRIPTION_PROMPT,
    REPORT_PROMPT
)

from .tool_agents.base_llm_with_search import BaseLLMWithSearch
from .tool_agents.agent_directory import STRING_TO_AGENT_MAP
from .tool_agents.execution_agent_cc import ExecutionAgentCC

from .base_llm import OnePassLLM
from .meta_agents.review_designer import ReviewDesigner
from .utils.pdf_loader import load_pdf_texts_from_directory
from .utils.git_version_control import GitVersionControl

BACKGROUND_PDF_SUMMARY_SYSTEM_PROMPT = """
You are an expert scientific reader. Summarize prior work using only the PDF text provided by the user.
Do not perform web search. Do not rely on outside knowledge. If the extracted text is incomplete or noisy,
state that limitation explicitly instead of filling gaps from memory.
"""

def tool_agent_prompt_generator(agent_list: List[Any]) -> str:
    """
    Generate a prompt for a list of tool agents.
    """
    prompt = "You are given access to helper agents which can be used to complete your task."
    for i, a in enumerate(agent_list):
        a_config = a.config_to_prompt()
        prompt += f"\n\n**Helper Agent {i+1}**:\n{a_config}"
    return prompt


def _start_spinner(message: str = "Working..."):
    """
    Lightweight CLI spinner to show progress during long-running LLM calls.
    """
    stop_event = threading.Event()

    def _spin():
        for ch in itertools.cycle("|/-\\"):
            if stop_event.is_set():
                break
            print(f"\r{message} {ch}", end="", flush=True)
            time.sleep(0.1)
        # Clear the spinner line when finished
        print("\r", end="", flush=True)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    return stop_event, t


def _generate_background_report(
    dataset_description: str,
    task_description: Optional[str] = None,
    background_path: Optional[str] = None,
    log_dir: str = "logs",
    summary_llm_config: Optional[LLMConfig] = None,
    max_output_tokens: int = 8192,
) -> str:
    """
    Create a background report.

    Args:
        dataset_description: Description of the dataset being explored.
        task_description: Task/goal description for the run.
        background_path: Optional directory of PDFs to summarize into the report.
        log_dir: Directory for agent logging/output.
        summary_llm_config: Optional LLM config used for PDF-only summarization mode.
        max_output_tokens: Maximum output tokens for the generated report.

    Returns:
        The generated Markdown report as a string.
    """
    if background_path:
        pdf_documents = load_pdf_texts_from_directory(
            background_path,
            max_chars_per_pdf=20000,
        )
        if not pdf_documents:
            raise ValueError(f"No readable PDF files found in {background_path}")

        llm_config = summary_llm_config
        if llm_config is None:
            if os.getenv("ANTHROPIC_API_KEY"):
                llm_config = LLMConfig(
                    model_name="claude-sonnet-4-5-20250929",
                    api_provider="anthropic",
                    max_tokens=max_output_tokens,
                )
            else:
                raise ValueError(
                    "background_path now requires a standard LLM to summarize the supplied PDFs, "
                    "but no summary_llm_config was provided and no ANTHROPIC_API_KEY was found."
                )

        pdf_sections = []
        total_chars = 0
        max_total_chars = 120000
        truncated = False
        for pdf_doc in pdf_documents:
            text = (pdf_doc.get("text") or "").strip()
            if not text:
                text = "[No extractable text was found in this PDF.]"

            section = (
                f"## PDF: {pdf_doc['filename']}\n"
                f"File size: {pdf_doc['size_bytes'] / (1024 * 1024):.1f} MB\n\n"
                f"{text}"
            )

            if total_chars + len(section) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining > 0:
                    pdf_sections.append(section[:remaining])
                truncated = True
                break

            pdf_sections.append(section)
            total_chars += len(section)

        prompt = f"""Summarize the prior work described in the provided PDFs.

Only use the document text below. Do not use web search, README context, task context, or any outside knowledge.

Goals:
1) Summarize each paper or document's main question, methods, and findings when the text supports it.
2) Synthesize cross-paper themes, recurring methods, and important disagreements.
3) Identify limitations or ambiguities caused by incomplete PDF text extraction.
4) Conclude with actionable takeaways and open questions grounded only in these PDFs.

Return a detailed Markdown report with sections for:
- Executive Summary
- Per-Document Summaries
- Cross-Document Themes
- Limitations of the Provided Documents
- Actionable Takeaways

Provided PDF text:

{chr(10).join(pdf_sections)}
"""

        if truncated:
            prompt += "\n\nNote: the provided PDF excerpts were truncated to fit the summarization context window."

        llm = OnePassLLM(
            system_prompt=BACKGROUND_PDF_SUMMARY_SYSTEM_PROMPT,
            llm_config=llm_config,
        )
        return llm(prompt, max_tokens=max_output_tokens)

    llm_config = LLMConfig(model_name="sonar-deep-research", api_provider="perplexity")
    deep_research_config = DeepResearchConfig(
        llm_config=llm_config,
        log_dir=log_dir,
        max_output_tokens=max_output_tokens,
    )
    agent = DeepResearchAgent(config=deep_research_config)

    tdstr = f"\n## Task description\n{task_description}\n" if task_description else ""

    prompt = f"""Generate a thorough background report that encompasses previous work that's been done related to the following dataset and task:

## Dataset description
{dataset_description}
{tdstr}
Goals:
1) Summarize the work that's been done on this dataset and related datasets in the past.
2) Highlight considerations and methods that have been successful for analyzing this type of data.
3) Research some specific research questions that can be investigated with this type of data.
4) Keep it actionable and tightly scoped to the dataset/task.
5) Be precise and scientific. This report will be used for scientific decision-making, so be transparent and rigorous in your reporting.
6) Focus on published, peer-reviewed research articles when gathering sources.

Report your findings in Markdown format in a detailed, research-style report.
"""

    report = agent.execute(prompt)

    return report


class DirectorWorkflow:
    """
    Director-based workflow that follows a hypothesis-directed research pattern.

    This workflow always generates a hypothesis first using HypothesisGenAgent, then
    uses DirectorAgent to direct research queries aimed at testing and validating
    that hypothesis.
    """
    def __init__(self,
                 hypothesis_config: HypothesisGenerationConfig,
                 dataset_description: str,
                 task_description: str = None,
                 agent_config_list: Optional[List[Any]] = None,
                 max_queries: int = 10,
                 log_dir: str = "logs",
                 images_dir: str = "logs",
                 output_home: Optional[str] = None,
                 multiturn_hypothesis: bool = True,
                 review_optimization_steps = None,
                 review_optimization_llm_config = None,
                 review_search = False,
                 review_design = False,
                 review_designer_llm_config = None,
                 explicit_hypothesis_step = None,
                 exploratory_phase_warmup_steps = None,
                 hold_reviews_in_exploratory_phase = True,
                 background_path: Optional[str] = None,
                 background_report: Optional[str] = None,
                 git_tracking: bool = False,
                 report_filename: Optional[str] = "final_report.md",
                 max_tool_invocation_retries: int = 3,
                 director_hypothesis_config: Optional[HypothesisGenerationConfig] = None,
                 generate_background_report: bool = False,
        ):
        """
        Initialize the DirectorWorkflow.

        Args:
            hypothesis_config (HypothesisGenerationConfig): Configuration for the hypothesis agent
            dataset_description (str): Description of the dataset
            agent_config_list (Optional[List[Any]]): List of agent configurations to use
            max_queries (int): Maximum number of queries to execute
            log_dir (str): Directory for logging
            images_dir (str): Directory for storing images
            output_home (Optional[str]): Output directory for results
            multiturn_hypothesis (bool): Enable multi-turn hypothesis generation
            review_optimization_steps (int): Number of steps between review optimizations
            review_optimization_llm_config (LLMConfig): Configuration for review optimization LLM
            review_search (bool): Enable web search tools for the reviewer
            review_design (bool): Use ReviewDesigner to generate custom reviewer prompts based on research task
            review_designer_llm_config (Optional[LLMConfig]): Configuration for the ReviewDesigner LLM.
                If None and review_design=True, claude-sonnet-4-5-20250929 is used.
            explicit_hypothesis_step (int): Step at which to start explicit hypothesis generation
            exploratory_phase_warmup_steps (int): Number of warmup steps for exploratory phase
            hold_reviews_in_exploratory_phase (bool): Hold reviews during exploratory phase
            background_path (Optional[str]): Path to directory containing prior-work PDFs to summarize
            background_report (Optional[str]): Accumulated knowledge/report from previous rounds.
                This is kept separate from the current round's report to cleanly track accumulated knowledge.
            git_tracking (bool): Enable git version control tracking of workflow changes
            report_filename (str): Filename for the final report (default: "final_report.md"). 
                Can include subdirectories (e.g., "reports/report_round1.md")
            max_tool_invocation_retries (int): Maximum number of retries for tool invocation format errors (default: 3)
        """
        self.agent_config_list = agent_config_list if agent_config_list is not None else []
        # Core hypothesis configuration used for HypothesisGenAgent and human-selection settings
        self.hypothesis_config = hypothesis_config
        self.dataset_description = dataset_description
        self.task_description = task_description
        self.max_queries = max_queries
        self.multiturn_hypothesis = multiturn_hypothesis
        self.review_search = review_search
        self.review_design = review_design
        self.git_tracking = git_tracking
        self.report_filename = report_filename
        self.max_tool_invocation_retries = max_tool_invocation_retries
        self.round_num = None  # Will be set by HypothesisBrancher if used
        self.background_report = background_report  # Accumulated knowledge from previous rounds
        self.log_dir = log_dir

        self.review_optimization_steps = review_optimization_steps
        self.review_optimization_llm_config = review_optimization_llm_config
        self.review_designer_llm_config = review_designer_llm_config

        self.explicit_hypothesis_step = explicit_hypothesis_step
        self.exploratory_phase_warmup_steps = exploratory_phase_warmup_steps
        self.hold_reviews_in_exploratory_phase = hold_reviews_in_exploratory_phase

        self.generate_background_report = generate_background_report

        # Determine output_home: explicit arg -> default "outputs"
        self.output_home = output_home if output_home is not None else "outputs"

        # Initialize logger first so it can be passed to agents
        self.logger = Logger("director_workflow", log_dir=log_dir)

        if background_path and not self.generate_background_report:
            self.logger.log_action(
                "background_path_unused",
                "background_path was provided without generate_background_report; PDFs will not be injected directly into agents.",
            )
            print("ℹ️ background_path is now only used when generating `deep_research_background.md`.")

        if self.generate_background_report:
            self.background_report_path = Path(self.log_dir) / "deep_research_background.md"
            try:
                if self.background_report_path.exists():
                    # Re-use existing background report to avoid unnecessary regeneration
                    self.logger.log_action("background_report_exists", f"Using cached report at: {self.background_report_path}")
                    print(f"✅ Using existing background report at: {self.background_report_path}")
                else:
                    spinner_message = (
                        "🔎 Summarizing research background..."
                        if background_path
                        else "🔎 Generating background report..."
                    )
                    spinner_stop, spinner_thread = _start_spinner(spinner_message)
                    try:
                        generated_report = _generate_background_report(
                            dataset_description=self.dataset_description,
                            task_description=self.task_description,
                            background_path=background_path,
                            log_dir=self.log_dir,
                            summary_llm_config=(
                                director_hypothesis_config.llm_config
                                if director_hypothesis_config is not None
                                else hypothesis_config.llm_config
                            ),
                        )
                    finally:
                        spinner_stop.set()
                        spinner_thread.join()

                    self.background_report_path.write_text(generated_report)
                    self.logger.log_action("background_report_generated", f"Wrote report to: {self.background_report_path}")
                    print(f"📝 Generated background report at: {self.background_report_path}")
            except Exception as e:
                self.logger.log_error(f"Failed to generate background report: {e}")
                print(f"⚠️ Failed to generate background report: {e}")
        else:
            self.background_report_path = None

        self._initialize_agents()

        self.hypothesis_agent_tool_prompt = tool_agent_prompt_generator(list(self.agents.values()))
        self.allow_director_to_stop = getattr(director_hypothesis_config, "allow_director_to_stop", False)

        # Initialize DirectorAgent for query generation and research orchestration.
        # If a separate director_hypothesis_config is provided, use it; otherwise fall back
        # to the main hypothesis_config so existing behavior is preserved.
        director_config = director_hypothesis_config if (director_hypothesis_config is not None) else hypothesis_config
        self.director_agent = DirectorAgent(
            config=director_config,
            tool_agent_prompt=self.hypothesis_agent_tool_prompt,
            logger=self.logger
        )

        # Initialize HypothesisGenAgent for hypothesis generation
        self.hypothesis_gen_agent = HypothesisGenAgent(
            config=self.hypothesis_config,
            background_report_path=self.background_report_path,
            logger=self.logger,
            helper_agents_prompt=self.hypothesis_agent_tool_prompt,
        )

        # Keep backward compatibility reference
        self.hypothesis_agent = self.director_agent
        
        # Store results
        self.results = []
        self.report = ""
        self.cumulative_report = ""

        # Initialize review tracking
        self.review_history = []  # List to track all reviews: [{query_num, round_num, timestamp, input_length, output_length}]
        self.generated_reviewer_prompt = None  # Store the generated reviewer prompt (if review_design=True)

        # Initialize image tracking
        self.images = {}  # Dictionary to track all images: {image_id: {filename, description, query_num, step_num}}
        self.image_counter = 0
        self.images_dir = os.path.join(images_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        if self.review_optimization_llm_config is not None:
            # Conditionally use ReviewDesigner based on review_design flag
            if self.review_design:
                # Use ReviewDesigner to generate a custom reviewer prompt based on the research task
                designer_llm_config = self.review_designer_llm_config or LLMConfig(model_name="claude-sonnet-4-5-20250929", max_tokens = 8192)
                review_designer = ReviewDesigner(
                    llm_config=designer_llm_config
                )
                
                # Add context about search tool availability
                additional_context = None
                if self.review_search:
                    additional_context = (
                        "The reviewer will have access to web search tools to evaluate the novelty "
                        "and impact of the agent's findings by searching for similar studies and relevant literature."
                    )

                if self.task_description is not None:
                    research_task = self.dataset_description +"\n\n## Task Description\n" + self.task_description
                else:
                    research_task = self.dataset_description
                
                # Generate the custom reviewer prompt
                generated_reviewer_prompt = review_designer.design_reviewer_prompt(
                    research_task=research_task,
                    additional_context=additional_context
                )

                # Store the generated prompt in the workflow instance
                self.generated_reviewer_prompt = generated_reviewer_prompt

                self.logger.log_action("reviewer_prompt_generated",
                    f"Generated custom reviewer prompt using {designer_llm_config.model_name} ({len(generated_reviewer_prompt)} chars)")

                self.logger.log_action("generated_reviewer_prompt", f"Generated reviewer prompt: {generated_reviewer_prompt}")
                self.logger.log_prompt("reviewer", generated_reviewer_prompt, "generated_reviewer_prompt")

                # Create a copy of the config with the generated system prompt
                review_config_with_prompt = LLMConfig(
                    model_name=self.review_optimization_llm_config.model_name,
                    api_provider=self.review_optimization_llm_config.api_provider,
                    api_key=self.review_optimization_llm_config.api_key,
                    temperature=self.review_optimization_llm_config.temperature,
                    frequency_penalty=self.review_optimization_llm_config.frequency_penalty,
                    max_tokens=self.review_optimization_llm_config.max_tokens,
                    system_prompt=generated_reviewer_prompt
                )
            else:
                # Use the provided config as-is (with its existing system_prompt if any)
                review_config_with_prompt = self.review_optimization_llm_config
                self.logger.log_action("reviewer_config_used", 
                    "Using provided review optimization config without ReviewDesigner")
            
            if self.review_search:
                self.review_optimization_llm = BaseLLMWithSearch(
                    llm_config=review_config_with_prompt,
                    logger=self.logger # Passed for redundancy
                )
            else:
                self.review_optimization_llm = OnePassLLM(
                    llm_config=review_config_with_prompt
                )
        # Initialize checkpoint manager with round number (if set)
        self.checkpoint_manager = CheckpointManager(
            log_dir=self.log_dir,
            workflow=self,
            round_num=self.round_num
        )

        # Save reviewer prompt checkpoint if it was generated
        if self.generated_reviewer_prompt is not None:
            try:
                # Determine the designer model name
                designer_model_name = None
                if hasattr(self, 'review_designer_llm_config') and self.review_designer_llm_config:
                    designer_model_name = self.review_designer_llm_config.model_name
                elif self.review_design:
                    designer_model_name = "claude-sonnet-4-5-20250929"  # Default from line 218

                reviewer_prompt_path = self.checkpoint_manager.save_reviewer_prompt_checkpoint(
                    reviewer_prompt=self.generated_reviewer_prompt,
                    designer_model=designer_model_name
                )
                if reviewer_prompt_path:
                    self.logger.log_action("reviewer_prompt_checkpointed",
                        f"Reviewer prompt saved to checkpoint at {reviewer_prompt_path}")
                    print(f"💾 Reviewer prompt checkpointed: {os.path.basename(reviewer_prompt_path)}")
                else:
                    self.logger.log_error("Failed to checkpoint reviewer prompt (empty path returned)")
            except Exception as e:
                error_msg = f"Error checkpointing reviewer prompt: {str(e)}"
                self.logger.log_error("reviewer_prompt_checkpoint_failed", error_msg)
                print(f"⚠️ {error_msg}")

        # Initialize git version control if enabled
        self.git_vc = None
        if self.git_tracking:
            # Get working directory from ExecutionAgentCC if available
            working_dir = None
            for agent in self.agents.values(): # Loops to find the ExecutionAgent - should be a small loop
                if isinstance(agent, ExecutionAgentCC) and hasattr(agent, 'working_directory'):
                    working_dir = agent.working_directory
                    break

            if working_dir:
                try:
                    self.git_vc = GitVersionControl(working_dir, logger=self.logger)
                    if self.git_vc.init_repo():
                        self.logger.log_action("git_tracking_enabled",
                            f"Git version control enabled for {working_dir}")
                        print(f"🔧 Git version control enabled")
                    else:
                        self.logger.log_error("Failed to initialize git repository")
                        self.git_vc = None
                except Exception as e:
                    self.logger.log_error(f"Failed to initialize git version control: {str(e)}")
                    self.git_vc = None
            else:
                self.logger.log_action("git_tracking_skipped",
                    "Git tracking enabled but no ExecutionAgentCC with working_directory found")
                print(f"⚠️ Git tracking enabled but no working directory available")

    def _initialize_agents(self):
        self.agents = {}
        for agent_config in self.agent_config_list:
            self.agents[agent_config.tag] = STRING_TO_AGENT_MAP[agent_config.agent_name](config=agent_config, logger=self.logger)

    def _test_iteration_condition_with_warmup(self, query_num, frequency) -> bool:
        """
        Test if we should perform an action (like review feedback) based on warmup period.

        This is used when hold_reviews_in_exploratory_phase is True to delay reviews
        until after the warmup period defined by exploratory_phase_warmup_steps.

        Args:
            query_num: Current query number
            frequency: How often to perform the action (e.g., every N queries)

        Returns:
            bool: True if the action should be performed, False otherwise
        """
        if self.explicit_hypothesis_step is None:
            return False # We're not using explicit hypotheses

        # Treat None as 0 (no warmup period)
        warmup_steps = self.exploratory_phase_warmup_steps if self.exploratory_phase_warmup_steps is not None else 0

        if query_num < warmup_steps:
            return False # We're before the exploratory phase
        elif query_num == warmup_steps:
            # We're right at the exploratory phase (or immediately if no warmup)
            return True
        elif ((query_num - warmup_steps) % frequency) == 0:
            # We're after the exploratory phase and now trying to generate explicit hypotheses
            return True
        else:
            return False # Else, should be false

    def _progress_round_dir(self, query_num: int) -> Optional[Path]:
        try:
            return self.checkpoint_manager.get_checkpoint_dir(query_num).parent
        except Exception:
            return None

    def _partial_query_agent_tag(self, round_dir: Path, query_num: int) -> Optional[str]:
        """Recover the helper identity without trusting arbitrary persisted paths."""
        query_dir = query_directory(round_dir, query_num)
        status_path = query_dir / "query_status.json"
        try:
            if (
                status_path.is_file()
                and not status_path.is_symlink()
                and status_path.stat().st_size <= 1024 * 1024
            ):
                status = json.loads(status_path.read_text(encoding="utf-8"))
                tag = status.get("agent_tag") if isinstance(status, dict) else None
                if isinstance(tag, str) and tag in self.agents:
                    return tag
        except (OSError, ValueError, TypeError):
            pass

        # Status records without an agent_tag are identified as ExecutionAgent
        # work by their execution log.
        execution_log = query_dir / artifact_filename(query_num, "execution_log")
        try:
            has_execution_log = (
                execution_log.is_file()
                and not execution_log.is_symlink()
                and execution_log.stat().st_size > 0
            )
        except OSError:
            has_execution_log = False
        if has_execution_log:
            execution_tags = [
                tag for tag, agent in self.agents.items()
                if isinstance(agent, ExecutionAgentCC)
            ]
            if len(execution_tags) == 1:
                return execution_tags[0]
        return None

    def _load_partial_query_for_resume(self, query_num: int) -> Optional[Dict[str, Any]]:
        """Return a safe, persisted unfinished-query handoff for exact retry.

        A completed checkpoint remains the sole authority for advancing the
        query number. This helper only handles artifacts for that next number:
        it reuses a valid report for local checkpoint handoff, or retries the
        exact saved input. It never asks the Director to invent a replacement.
        """
        round_dir = self._progress_round_dir(query_num)
        if round_dir is None:
            return None
        query_input = read_query_artifact(round_dir, query_num, "input")
        if query_input is None:
            return None
        query_report = read_query_artifact(round_dir, query_num, "report")
        tag = self._partial_query_agent_tag(round_dir, query_num)
        if query_report is None and tag is None:
            raise RuntimeError(
                f"query {query_num} has persisted input but its helper identity cannot be recovered"
            )
        query_raw = f"<{tag}>{query_input}</{tag}>" if tag else query_input
        return {
            "query_raw": query_raw,
            "query_parsed": query_input,
            "report": query_report,
            "agent_tag": tag,
        }

    def execute_agent(self, hypothesis_agent_out: str) -> str:
        """
        Execute the agent based on the tool call output from the hypothesis agent.
        The output is expected to be in the form <tag>[query to agent]</tag>.
        """
        import re
        # Match <tag>...</tag> and extract tag and query
        # Use search so the XML-style tool call can appear anywhere in the text (not only at the start)
        match = re.search(r'<(\w+)>([\s\S]*?)</\1>', hypothesis_agent_out, re.DOTALL)
        if not match:
            raise ValueError(f"Invalid tool call format: {hypothesis_agent_out}")
        tag, query = match.group(1), match.group(2).strip()
        print(f"🔧 Executing agent with tag '{tag}'")
        if tag not in self.agents:
            raise ValueError(f"Agent with tag '{tag}' not found in self.agents")

        current_query_num_for_progress = getattr(self, "query_num", None)
        progress_round_dir = None
        if current_query_num_for_progress is not None:
            progress_round_dir = self._progress_round_dir(current_query_num_for_progress)
            if progress_round_dir is not None:
                persist_query_input(progress_round_dir, current_query_num_for_progress, query)
            progress_recorder.write_query_started(
                progress_round_dir,
                current_query_num_for_progress,
                objective=query,
                agent_tag=tag,
            )
            progress_step = "running_literature_check" if tag == DEEP_RESEARCH_TAG else "executing_code"
            progress_recorder.write_query_step(progress_round_dir, current_query_num_for_progress, progress_step)

        agent = self.agents[tag]
        current_query_num = self.query_num if hasattr(self, "query_num") else None
        self.logger.log_action(
            "helper_agent_called",
            (
                f"tag={tag}\n"
                f"agent_type={agent.__class__.__name__}\n"
                f"query_num={current_query_num}\n"
                f"query_chars={len(query)}"
            ),
        )
        # Persist search outputs under the run log directory to avoid polluting
        # the execution working directory. When available, scope by round.
        searches_dir = None
        try:
            if getattr(self, "log_dir", None):
                round_num = getattr(self, "round_num", None)
                if isinstance(round_num, int) and round_num >= 0:
                    searches_dir = str(Path(self.log_dir) / "searches" / f"round_{round_num}")
                else:
                    searches_dir = str(Path(self.log_dir) / "searches")
            else:
                exec_agent = next((a for a in self.agents.values() if isinstance(a, ExecutionAgentCC)), None)
                if exec_agent and hasattr(exec_agent, 'working_directory'):
                    searches_dir = str(Path(exec_agent.working_directory) / "searches")
        except Exception:
            searches_dir = None

        # If this is an ExecutionAgentCC, pass filesystem context and variables directly
        if isinstance(agent, ExecutionAgentCC):
            # Pass variables directly to the execute method instead of updating config
            checkpoint_query_dir = None
            try:
                if hasattr(self, "checkpoint_manager") and self.query_num is not None:
                    checkpoint_dir = self.checkpoint_manager.get_checkpoint_dir(self.query_num)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_query_dir = str(checkpoint_dir)
            except Exception:
                checkpoint_query_dir = None

            kwargs = {
                'query_num': self.query_num if hasattr(self, 'query_num') else None,
                'image_tracker': self  # Pass the workflow as image tracker
            }
            if searches_dir:
                kwargs['searches_dir'] = searches_dir
            if checkpoint_query_dir:
                kwargs['checkpoint_query_dir'] = checkpoint_query_dir
            agent_output = agent.execute(query, **kwargs)
            if progress_round_dir is not None:
                persist_query_report(progress_round_dir, current_query_num_for_progress, agent_output)
            self.logger.log_action(
                "helper_agent_completed",
                (
                    f"tag={tag}\n"
                    f"agent_type={agent.__class__.__name__}\n"
                    f"query_num={current_query_num}\n"
                    f"output_chars={len(agent_output) if isinstance(agent_output, str) else len(str(agent_output))}"
                ),
            )
            return agent_output, query

        # For other agents, call execute and pass searches_dir when the method supports it
        try:
            sig = inspect.signature(agent.execute)
            if 'searches_dir' in sig.parameters:
                agent_output = agent.execute(query, searches_dir=searches_dir)
            else:
                agent_output = agent.execute(query)
        except Exception:
            # Fallback to calling without searches_dir
            agent_output = agent.execute(query)

        if progress_round_dir is not None:
            persist_query_report(progress_round_dir, current_query_num_for_progress, agent_output)

        self.logger.log_action(
            "helper_agent_completed",
            (
                f"tag={tag}\n"
                f"agent_type={agent.__class__.__name__}\n"
                f"query_num={current_query_num}\n"
                f"output_chars={len(agent_output) if isinstance(agent_output, str) else len(str(agent_output))}"
            ),
        )

        try:
            agent.cleanup()
        except Exception:
            pass
        return agent_output, query

    def run(self, final_report_step: bool = False, exploratory_mode: bool = False) -> Dict[str, Any]:
        """
        Run the Director workflow.

        This workflow behavior depends on the mode:
        - Normal mode: Generates hypothesis first, then tests it
        - Exploratory mode: Skips hypothesis generation, explores data freely

        Args:
            final_report_step: Whether to generate a structured final report at the end
            exploratory_mode: If True, skip hypothesis generation and run in exploration mode

        Returns:
            Dict[str, Any]: Results of the workflow
        """
        # Determine starting point (for resume support)
        # Start from the next query number after the last successful query
        start_query = getattr(self, 'query_num', 0) + 1
        mid_round_resuming = start_query > 1

        if mid_round_resuming:
            print(f"🔄 Resuming Director Workflow from query {start_query}")
            print(f"📊 Previous queries completed: {start_query - 1}")
        else:
            print(f"🚀 Starting Director Workflow")
            print(f"📊 Dataset: {self.dataset_description[:100]}...")

        if self.task_description is not None:
            print(f"📝 Task: {self.task_description[:100]}...")
        print(f"🔍 Max queries: {self.max_queries}")

        # Log workflow start
        if mid_round_resuming:
            self.logger.log_action("workflow_resumed", f"Resuming from query {start_query} | max_queries={self.max_queries}")
        else:
            self.logger.log_action("workflow_started", f"Dataset: {self.dataset_description[:100]}... | max_queries={self.max_queries}")

        # Initialize feedback - use refinement_feedback if resuming and provided
        feedback = None
        if mid_round_resuming and hasattr(self, 'refinement_feedback') and self.refinement_feedback is not None:
            feedback = self.refinement_feedback # Apply refinement feedback to the first query after resume
            self.logger.log_action("refinement_feedback_loaded",
                f"Using refinement feedback for first query after resume: {len(feedback)} characters")
            print(f"📝 Applying refinement feedback to guide next query")

        tool_invocation_retry_count = 0

        # Store the generated hypothesis for reference
        hypothesis_output = None

        # Determine if we should skip hypothesis generation
        # Skip if: resuming, exploratory_mode=True, or human_selection_mode=True
        skip_hypothesis_generation = mid_round_resuming or exploratory_mode or self.hypothesis_config.human_selection_mode

        if not skip_hypothesis_generation:
            # Normal mode: generate single hypothesis
            print(f"\n🧠 Generating hypothesis before directing research...")
            self.logger.log_action("hypothesis_generation",
                "Calling hypothesis_gen_agent.generate_hypothesis() to establish research direction")

            # Call hypothesis generation agent to generate hypothesis
            hypothesis_output = self.hypothesis_gen_agent.generate_hypothesis(
                dataset_description=self.dataset_description,
                task_description=self.task_description,
                background_report=self.background_report,  # Consolidated knowledge
            )

            if hypothesis_output:
                self.logger.log_action("hypothesis_generated",
                    f"Generated {len(hypothesis_output)} characters of hypothesis")
                print(f"✓ Hypothesis generated ({len(hypothesis_output)} characters)")

                # Save hypothesis checkpoint as hypothesis_000
                try:
                    hypothesis_checkpoint_path = self.checkpoint_manager.save_hypothesis_checkpoint(hypothesis_output)
                    if hypothesis_checkpoint_path:
                        self.logger.log_action("hypothesis_checkpoint_saved",
                            f"Hypothesis saved to checkpoint at {hypothesis_checkpoint_path}")
                        print(f"💾 Hypothesis checkpoint saved")
                    else:
                        self.logger.log_error("Failed to save hypothesis checkpoint (empty path returned)")
                except Exception as e:
                    self.logger.log_error(f"Failed to save hypothesis checkpoint: {str(e)}")
            else:
                self.logger.log_action("hypothesis_failed", "No hypothesis output generated")
                print(f"⚠️ Warning: Hypothesis generation returned no output")

            print()  # Add spacing before main loop
        elif exploratory_mode:
            # Exploratory mode: skip hypothesis generation
            print(f"\n🔍 Running in EXPLORATORY MODE")
            print(f"   Director will explore the dataset without a specific hypothesis\n")
            hypothesis_output = None
            self.logger.log_action("exploratory_mode",
                "Exploratory mode enabled - skipping hypothesis generation")
        else:
            # RESUMING: Only load selected hypothesis when not mid-round resuming.
            if mid_round_resuming:
                # Mid-round resume: hypothesis should already be in director agent state
                self.logger.log_action("hypothesis_resume",
                    "Mid-round resume - hypothesis context should be in director agent state")
                print(f"📝 Resuming with hypothesis context from checkpoint")
                hypothesis_output = None
            elif self.hypothesis_config.selected_hypothesis_file:
                # Load human-selected hypothesis (round resume)
                hypothesis_output = self.checkpoint_manager.load_selected_hypothesis(
                    self.hypothesis_config.selected_hypothesis_file
                )

                if hypothesis_output:
                    self.logger.log_action("selected_hypothesis_loaded",
                        f"Loaded human-selected hypothesis ({len(hypothesis_output)} chars)")
                    print(f"\n✅ Loaded human-selected hypothesis ({len(hypothesis_output)} characters)")
                    print(f"📄 From: {self.hypothesis_config.selected_hypothesis_file}\n")
                else:
                    raise ValueError(
                        f"Failed to load selected hypothesis from {self.hypothesis_config.selected_hypothesis_file}")
            else:
                # Normal resume: hypothesis context should be in director state
                self.logger.log_action("hypothesis_resume",
                    "Resuming - hypothesis context should be in director agent state")
                print(f"📝 Resuming with hypothesis context from checkpoint")
                hypothesis_output = None

        # Use query_num for successful queries, attempt_count for loop control
        query_num = start_query
        attempt_count = 0
        director_requested_stop = False

        while query_num <= self.max_queries:
            attempt_count += 1
            self.query_num = query_num
            self.logger.log_action("query_start", f"Query {query_num}/{self.max_queries} (attempt {attempt_count})")

            # Estimate and print director context token usage before generation
            if self.multiturn_hypothesis:
                dir_tokens = self.director_agent.estimate_context_tokens()
            else:
                dir_tokens = 0
            self.logger.log_action("director_context_tokens", f"query={query_num}, tokens={dir_tokens}")

            # Generate query from director agent
            # Log feedback if provided
            if feedback is not None:
                self.logger.log_action("feedback_provided", f"Providing feedback to director agent for query {query_num}: {len(feedback)} characters")

            try:
                partial_query = (
                    self._load_partial_query_for_resume(query_num)
                    if mid_round_resuming and query_num == start_query and attempt_count == 1
                    else None
                )
            except Exception as recovery_exc:
                progress_recorder.write_query_failed(
                    self._progress_round_dir(query_num), query_num, error=str(recovery_exc)
                )
                raise
            if partial_query is not None:
                query = partial_query["query_raw"]
                recovery_kind = (
                    "persisted_report_handoff"
                    if partial_query["report"] is not None
                    else "persisted_input_retry"
                )
                self.logger.log_action(
                    "partial_query_recovered",
                    f"query={query_num}\nrecovery={recovery_kind}",
                )
                print(f"🔄 Recovering query {query_num} from its persisted {recovery_kind}")
            else:
                query = self.director_agent.generate_query(
                    dataset_description=self.dataset_description,
                    report=self.report,
                    query_count=query_num,
                    total_queries=self.max_queries,
                    feedback=feedback,
                    after_hypothesis_step=False,  # No mid-workflow hypothesis generation in Director mode
                    background_report=self.background_report,
                    target_hypothesis=hypothesis_output,  # Passed on every call; director handles first-time injection
                    human_selection_mode=self.hypothesis_config.human_selection_mode,
                    exploratory_mode=exploratory_mode,  # Tells director to explore data without hypothesis
                )

            if feedback is not None:
                feedback = None # Reset so it's not added next time
            
            # Check if no query was generated (agent decided to end iterations)
            if query is None:
                print(f"🛑 No query generated by hypothesis agent - ending iterations at query {query_num}")
                self.logger.log_action("query_generation", "No query generated - ending iterations")
                progress_recorder.write_round_event(
                    self._progress_round_dir(query_num),
                    "early_stopping",
                    message=f"Query {query_num}: director returned no query",
                )
                break  # Exit while loop

            # Director can opt to stop early by emitting the sentinel
            if self.allow_director_to_stop and ("<<STOP>>" in query):
                print(f"🛑 Director requested stop via <<STOP>> at query {query_num}")
                self.logger.log_action("director_stop_requested", f"Query {query_num}: Director emitted <<STOP>>")
                director_requested_stop = True
                break  # Exit query loop early
            
            if partial_query is None:
                self.logger.log_response(f"Generated query: {query}", "hypothesis_agent")
            
            # Check query for "<STOP>" pattern
            
            # Execute query with CodeAct execution agent
            self.logger.log_action("execute_query", f"Executing query {query_num} with CodeAct framework")
            # Accept XML-style tool call appearing anywhere in the response
            xml_match = re.search(r'<(\w+)>([\s\S]*?)</\1>', query, re.DOTALL)
            if partial_query is not None and partial_query["report"] is not None:
                # The helper result was durably written before the prior process
                # exited. Replay only the local Director/checkpoint handoff.
                agent_output = partial_query["report"]
                parsed_query = partial_query["query_parsed"]
                tool_invocation_retry_count = 0
            elif xml_match:
                # Use the general agent execution logic
                try:
                    agent_output, parsed_query = self.execute_agent(query)
                except Exception as exec_agent_exc:
                    progress_recorder.write_query_failed(
                        self._progress_round_dir(query_num), query_num, error=str(exec_agent_exc)
                    )
                    raise
                # Reset retry counter on success
                tool_invocation_retry_count = 0
            else:
                # An API refusal, an empty completion, and a genuinely malformed
                # one all land here looking identical. Only the agent knows which.
                refusal = first_refusal(self.director_agent)
                self.logger.log_error(
                    f"Query {query_num}: "
                    f"{refusal or 'no XML tool invocation found in director output.'}\n"
                    f"raw_output_chars={len(query)}\n"
                    f"raw_output={query[:1000]!r}"
                )
                if refusal:
                    progress_recorder.write_round_event(
                        self._progress_round_dir(query_num),
                        "model_refusal",
                        message=f"Query {query_num}: {refusal}",
                    )
                    print(f"⚠️ {refusal}")
                # If multiturn, prompt the hypothesis agent to try again and restart loop
                if self.multiturn_hypothesis:
                    tool_invocation_retry_count += 1

                    if tool_invocation_retry_count > self.max_tool_invocation_retries:
                        error_msg = (
                            f"Refused by the API on {self.max_tool_invocation_retries} "
                            f"consecutive attempts. Gracefully exiting query loop (early "
                            f"stopping). {REFUSAL_ADVICE}"
                        ) if refusal else (
                            f"Max tool invocation retries ({self.max_tool_invocation_retries}) exceeded. Gracefully exiting query loop (early stopping)."
                        )
                        self.logger.log_error(error_msg)
                        self.logger.log_action("early_stopping", f"Query {query_num}: Max tool invocation retries exceeded - proceeding to summary generation")
                        # Surface the abort in round_activity.jsonl — otherwise the round
                        # reports "completed" with no record of why nothing ran.
                        progress_recorder.write_round_event(
                            self._progress_round_dir(query_num),
                            "early_stopping",
                            message=f"Query {query_num}: {error_msg}",
                        )
                        print(f"⚠️ Max tool invocation retries exceeded - ending iterations early at query {query_num}")
                        break  # Exit while loop

                    if refusal:
                        # Nothing was malformed, so asking for a shorter query is
                        # noise. Refusals are non-deterministic; just ask again.
                        self.logger.log_error(
                            f"Retrying after refusal (attempt {tool_invocation_retry_count}/"
                            f"{self.max_tool_invocation_retries}). {REFUSAL_ADVICE}"
                        )
                        self.logger.log_action(
                            "model_refusal",
                            f"Query {query_num}: declined by the API. Retry attempt {tool_invocation_retry_count}",
                        )
                        feedback = None
                        continue

                    self.logger.log_error(f"Tool invocation did not match expected format (attempt {tool_invocation_retry_count}/{self.max_tool_invocation_retries}). Asking hypothesis agent to try again with shorter query.")
                    self.logger.log_action("tool_invocation_error", f"Query {query_num}: Tool invocation did not match expected format. Retry attempt {tool_invocation_retry_count}")

                    # Add feedback to hypothesis agent for next iteration
                    feedback = """Your previous query did not contain a properly formatted tool invocation. This is likely because your query was too long and got cut off before completing the XML tags.

Please try again with a SHORTER and more concise query. Remember to use the format:
**Interpretation**: [Brief interpretation]
**Thought**: [Brief thought]
<[tag]>
[Concise query to helper agent]
</[tag]>

Make sure your query is complete and the closing XML tag is present. Keep your interpretation and thought sections brief (1-2 sentences each), and make your actual query to the helper agent concise and focused on the most important aspects."""

                    # Continue to next iteration (retry with same query_num, don't increment)
                    continue  # Retry without incrementing query_num
                else:
                    error_msg = f"Tool invocation did not match expected format. Gracefully exiting query loop (early stopping)."
                    self.logger.log_error(error_msg)
                    self.logger.log_action("early_stopping", f"Query {query_num}: Invalid tool invocation format - proceeding to summary generation")
                    print(f"⚠️ Tool invocation format error - ending iterations early at query {query_num}")
                    break  # Exit while loop
            
            self.logger.log_action("query_executed", f"Query {query_num} executed successfully")
            progress_recorder.write_query_step(
                self._progress_round_dir(query_num), query_num, "analyzing_results"
            )

            # Store result (keep both raw and parsed queries)
            # If parsed_query is not defined (shouldn't happen when xml_match), fall back to raw query
            self.results.append({
                'query_num': query_num,
                'query_raw': query,
                'query_parsed': parsed_query,
                'result': agent_output,
                'timestamp': datetime.datetime.now().isoformat()
            })

            # REPORT UPDATE - pass parsed query so reports see the cleaned/inner query text
            self.report_update(agent_output, parsed_query, query_num)
            progress_recorder.write_query_step(
                self._progress_round_dir(query_num), query_num, "writing_query_report"
            )
            # Save per-iteration checkpoint using new system
            try:
                checkpoint_path = self.checkpoint_manager.save_checkpoint(query_num)
                if checkpoint_path:
                    self.logger.log_action("checkpoint_saved", f"Query {query_num} checkpoint saved")
                else:
                    self.logger.log_error("Checkpoint save returned empty path")
            except Exception as e:
                self.logger.log_error(f"Failed to save iteration checkpoint: {str(e)}")

            self.logger.log_action("query_completed", f"Query {query_num} completed successfully")
            progress_recorder.write_query_completed(
                self._progress_round_dir(query_num), query_num, summary=agent_output
            )
            # Git version control: commit changes after query completion (before incrementing query_num)
            if self.git_vc is not None:
                try:
                    if self.git_vc.add_all():
                        commit_message = f"Query {query_num}"
                        self.git_vc.commit(commit_message, round_num=self.round_num)
                except Exception as e:
                    self.logger.log_error(f"Git commit failed for query {query_num}: {str(e)}")

            # Increment query_num only after successful completion and git commit
            query_num += 1
            attempt_count = 0  # Reset attempt counter for next query

            # Expanding dataset mode: refresh dataset description after each query
            # This allows agents to see newly added data files
            if hasattr(self, 'data_manager') and self.data_manager is not None:
                self._refresh_dataset_description()

            if (self.review_optimization_steps is not None):
                perform_review = False
                # Note: query_num was just incremented, so we need to check the previous query number
                previous_query_num = query_num - 1

                # Skip review on the last query - feedback won't be used since the loop will exit
                if previous_query_num >= self.max_queries:
                    perform_review = False
                elif self.hold_reviews_in_exploratory_phase:
                    # Sets to true if frequency is met
                    perform_review = self._test_iteration_condition_with_warmup(previous_query_num, self.review_optimization_steps)
                elif (previous_query_num % self.review_optimization_steps == 0):
                    perform_review = True

                if perform_review:
                    self.logger.log_action("feedback_triggered", f"Query {previous_query_num} completed, triggering feedback generation (every {self.review_optimization_steps} queries)")
                    feedback = self.get_feedback()
        
        # Save the final report
        final_report_content = self.cumulative_report  # Default to cumulative report
        
        if final_report_step:
            # Generate a structured final report using the director agent
            self.logger.log_action("final_report_generation_start",
                "Calling director agent to generate structured final report")
            print(f"\n📝 Generating structured final report...")

            try:
                report_to_pass = self.report if self.multiturn_hypothesis else self.cumulative_report
                structured_report = self.director_agent.generate_final_report(
                    report=report_to_pass # This should have been updated in report_update() call above
                )
                
                if structured_report:
                    self.logger.log_action("final_report_generated", 
                        f"Generated structured report ({len(structured_report)} characters)")
                    print(f"✓ Structured final report generated ({len(structured_report)} characters)")
                    
                    # Save the structured report instead of the raw cumulative report
                    report_path = self._save_final_report(report=structured_report)
                    final_report_content = structured_report  # Use structured report for return
                else:
                    refusal = first_refusal(self.director_agent)
                    self.logger.log_error(
                        "final_report_generation_failed",
                        f"{refusal or 'No structured report generated'}, falling back to cumulative report",
                    )
                    if refusal:
                        progress_recorder.write_round_event(
                            self._progress_round_dir(getattr(self, "query_num", 1)),
                            "model_refusal",
                            message=f"Final report: {refusal}",
                        )
                    print(
                        f"⚠️ Warning: {refusal or 'Final report generation returned no output'}, "
                        "using cumulative report"
                    )
                    report_path = self._save_final_report()
                    final_report_content = self.cumulative_report  # Fallback to cumulative
            except Exception as e:
                error_msg = f"Error generating final report: {str(e)}"
                self.logger.log_error("final_report_generation_error", error_msg)
                print(f"⚠️ {error_msg}, falling back to cumulative report")
                report_path = self._save_final_report()
                final_report_content = self.cumulative_report  # Fallback to cumulative
        else:
            report_path = self._save_final_report() # Uses internal self.cumulative_report
        
        # Log final status if director stopped early
        if director_requested_stop:
            self.logger.log_action("workflow_stopped_early",
                f"Director emitted <<STOP>> at query {query_num}")

        self.logger.log_action("workflow_completed", f"Total queries executed: {len([r for r in self.results if not r.get('error', False)])}")
        self.logger.log_action("final_report_saved", f"Report path: {report_path}")

        return {
            'dataset_description': self.dataset_description,
            'max_queries': self.max_queries,
            'results': self.results,
            'report': final_report_content,  # Return actual report content, not path
            'report_path': report_path,
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'stopped_by_director' if director_requested_stop else 'completed'
        }
    
    def report_update(self, result: str, query: str, query_num: int):
        if self.multiturn_hypothesis:
            self.report = f"\n\n## Query {query_num}\n\n**Query**: {query}\n\n**Result**:\n{result}"
        else:
            self.report += f"\n\n## Query {query_num}\n\n**Query**: {query}\n\n**Result**:\n{result}"
        
        self.cumulative_report += f"\n\n## Query {query_num}\n\n**Query**: {query}\n\n**Result**:\n{result}"
        
        return 0

    def _refresh_dataset_description(self):
        """
        Refresh the dataset description with current data directory contents.
        
        This is called after each query when expanding_dataset_mode is enabled
        (data_manager is set by HypothesisBrancher).
        """
        if not hasattr(self, 'data_manager') or self.data_manager is None:
            return
        
        # Scan for changes
        self.data_manager.scan()
        
        # Check for new files and log them
        new_files = self.data_manager.detect_new_files()
        if new_files:
            self.logger.log_action("data_files_added", 
                f"Detected {len(new_files)} new data file(s): {new_files[:5]}")
            print(f"📁 Detected {len(new_files)} new data file(s):")
            for f in new_files[:5]:  # Show first 5
                print(f"   + {f}")
            if len(new_files) > 5:
                print(f"   + ... and {len(new_files) - 5} more")
        
        # Update dataset description
        old_len = len(self.dataset_description)
        self.dataset_description = self.data_manager.update_dataset_description()
        new_len = len(self.dataset_description)
        
        if new_len != old_len:
            self.logger.log_action("dataset_description_refreshed",
                f"Dataset description updated: {old_len} -> {new_len} chars")

    def make_feedback_prompt(self):
        prompt = \
f"""Please generate a thorough review of the following report.

First, here is some information about the dataset:

{self.dataset_description}

And here's a background report on what we've previously explored and learned on this dataset and task:

{self.background_report}

And here is the most recent analysis. You should focus your critiques on these findings and approaches:

{self.cumulative_report}

Now generate a thorough review of these findings.
"""
        return prompt

    def get_feedback(self):
        """
        Get feedback from the LLM on the current report.
        """
        if self.review_optimization_llm is None:
            return None

        self.logger.log_action("feedback_generation_start", "Starting feedback generation from review optimization LLM")

        report_summary = self.make_feedback_prompt()

        if report_summary is None:
            self.logger.log_error("feedback_generation_failed", "Failed to generate feedback: report summary is None")
            return None

        # Store the input for checkpointing
        reviewer_input = report_summary

        feedback = self.review_optimization_llm.forward(report_summary)

        self.logger.log_action("feedback_generated", f"Feedback generated successfully: {len(feedback)} characters")
        self.logger.log_response(f"Generated feedback: {feedback[:200]}...", "review_optimization_llm")

        # Save review using CheckpointManager (new mechanism)
        try:
            review_checkpoint_path = self.checkpoint_manager.save_review_checkpoint(
                query_num=self.query_num,
                reviewer_input=reviewer_input,
                reviewer_output=feedback
            )
            if review_checkpoint_path:
                # Track review in history
                review_entry = {
                    'query_num': self.query_num,
                    'round_num': self.round_num,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'input_length': len(reviewer_input),
                    'output_length': len(feedback),
                    'checkpoint_path': review_checkpoint_path
                }
                self.review_history.append(review_entry)

                self.logger.log_action("review_checkpoint_saved",
                    f"Review for query {self.query_num} saved to checkpoint at {review_checkpoint_path}")
                print(f"📝 Review checkpoint saved: {os.path.basename(review_checkpoint_path)}")
            else:
                self.logger.log_error("Failed to save review checkpoint (empty path returned)")
        except Exception as e:
            error_msg = f"Error saving review checkpoint: {str(e)}"
            self.logger.log_error("review_checkpoint_save_failed", error_msg)
            print(f"⚠️ {error_msg}")

        return feedback

    def _save_final_report(self, report: Optional[str] = None) -> str:
        """
        Save the final report to a file. Only saves the final `self.report` content.

        Args:
            final_report (str): The final report content to save.

        Returns:
            str: Path to the saved report file, or empty string if saving is skipped
        """
        # Skip saving if report_filename is None (report will be saved in checkpoints instead)
        if self.report_filename is None:
            return ""

        # Ensure output directory exists
        os.makedirs(self.output_home, exist_ok=True)

        # Use the configured filename for the final report (may include subdirectories)
        report_path = os.path.join(self.output_home, self.report_filename)
        
        # Ensure the subdirectory exists if the filename includes a path
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        # Create the full report content - include dataset description and the final report content
        if report is None:
            dd_report = self.dataset_description +"\n\n## Task Description\n" + self.task_description if self.task_description is not None else self.dataset_description

            report_content = f"""# Scientific Analysis Report

Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Description
{dd_report}

## Final Report
{self.cumulative_report}

{self.get_image_references()}
"""
        else:
            report_content = report # Just copy what was passed
            
        # Save the report
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            try:
                self.logger.log_action("report_saved", f"Report saved to {report_path}")
            except Exception:
                pass
            return report_path
            
        except Exception as e:
            error_msg = f"Error saving report: {str(e)}"
            try:
                self.logger.log_error(error_msg)
            except Exception:
                print(f"⚠️ Warning: {error_msg}")
            return ""

    def add_image(self, image_data: str, description: str, query_num: int, step_num: int) -> str:
        """
        Add an image to the tracking system and save it to disk.
        
        Args:
            image_data (str): Base64 encoded image data
            description (str): Description of the image
            query_num (int): Query number when image was generated
            step_num (int): Step number when image was generated
            
        Returns:
            str: Image reference ID for use in reports
        """
        self.image_counter += 1
        image_id = f"img_{self.image_counter:03d}"
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{image_id}_{timestamp}.png"
        filepath = os.path.join(self.images_dir, filename)
        
        # Save image to disk
        try:
            import base64
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode and save
            image_bytes = base64.b64decode(image_data)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            # Track the image
            self.images[image_id] = {
                'filename': filename,
                'filepath': filepath,
                'description': description,
                'query_num': query_num,
                'step_num': step_num,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.logger.log_action("image_saved", f"Image {image_id} saved as {filename}")
            print(f"📊 Image tracked: {image_id} -> {filename} ({description[:50]}...)")
            return image_id
            
        except Exception as e:
            error_msg = f"Error saving image {image_id}: {str(e)}"
            self.logger.log_error(error_msg)
            print(f"⚠️ {error_msg}")
            return None

    def get_image_references(self) -> str:
        """
        Get a formatted string of all image references for use in reports.
        
        Returns:
            str: Formatted image references
        """
        if not self.images:
            return ""
        
        references = "\n\n## Generated Visualizations\n\n"
        for image_id, image_info in self.images.items():
            references += f"**{image_id}** (Query {image_info['query_num']}, Step {image_info['step_num']}): {image_info['description']}\n"
            references += f"*File: {image_info['filename']}*\n\n"
        
        return references

    def get_image_citation_instructions(self) -> str:
        """
        Get instructions for the execution agent on how to cite images in reports.
        
        Returns:
            str: Citation instructions
        """
        if not self.images:
            return ""
        
        instructions = "\n\n**IMAGE CITATION INSTRUCTIONS:**\n"
        instructions += "When writing your final response, reference the generated images using their IDs. "
        instructions += "For example: 'As shown in img_001, the data exhibits a clear trend...' or 'Figure img_002 demonstrates...'\n\n"
        instructions += "Available images:\n"
        
        for image_id, image_info in self.images.items():
            instructions += f"- {image_id}: {image_info['description']} (Query {image_info['query_num']}, Step {image_info['step_num']})\n"
        
        return instructions

    @staticmethod
    def load_checkpoint_metadata(checkpoint_path_or_query: str, log_dir: str = 'logs') -> Dict[str, Any]:
        """
        Load checkpoint metadata without creating a workflow instance.
        Useful for extracting configuration details before resuming.
        
        Args:
            checkpoint_path_or_query: Path to checkpoint file/directory or query number
            log_dir: Log directory (used if checkpoint_path_or_query is a query number)
            
        Returns:
            Dict containing checkpoint metadata including workflow_state and filesystem info
        """
        from pathlib import Path
        import json
        
        # Determine checkpoint path
        path = Path(checkpoint_path_or_query)
        
        if path.is_dir():
            if path.name == "checkpoints":
                checkpoint_file = path / "latest" / "checkpoint.json"
            elif path.name.startswith("query_"):
                checkpoint_file = path / "checkpoint.json"
            else:
                checkpoint_file = path / "checkpoints" / "latest" / "checkpoint.json"
        elif path.suffix == '.json':
            checkpoint_file = path
        else:
            try:
                query_num = int(checkpoint_path_or_query)
                checkpoint_file = query_directory(Path(log_dir) / "checkpoints", query_num) / "checkpoint.json"
            except ValueError:
                raise ValueError(f"Invalid checkpoint path or query number: {checkpoint_path_or_query}")
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        return {
            'checkpoint_path': str(checkpoint_file),
            'checkpoint_dir': str(checkpoint_file.parent),
            'workflow_state': checkpoint_data.get('workflow_state', {}),
            'filesystem': checkpoint_data.get('filesystem', {}),
            'agents': checkpoint_data.get('agents', {}),
            'timestamp': checkpoint_data.get('timestamp', ''),
            'query_num': checkpoint_data.get('query_num', 0),
            'checkpoint_version': checkpoint_data.get('checkpoint_version', '1.0')
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path_or_query: str, **override_kwargs) -> 'DirectorWorkflow':
        """
        Load a DirectorWorkflow instance from a checkpoint.

        Args:
            checkpoint_path_or_query: Path to checkpoint file/directory or query number
            **override_kwargs: Optional parameters to override during restoration
                Required: agent_config_list (List[AgentConfig])
                Optional: log_dir, output_home, hypothesis_config, refinement_feedback, and other workflow parameters

        Returns:
            DirectorWorkflow: Restored workflow instance
        """
        from pathlib import Path
        
        # Determine checkpoint path
        path = Path(checkpoint_path_or_query)
        
        if path.is_dir():
            # Directory provided - look for checkpoint file
            if path.name == "checkpoints":
                # Checkpoints directory - use latest
                checkpoint_file = path / "latest" / "checkpoint.json"
            elif path.name.startswith("query_"):
                # Specific query directory
                checkpoint_file = path / "checkpoint.json"
            else:
                # Look for checkpoints subdirectory
                checkpoint_file = path / "checkpoints" / "latest" / "checkpoint.json"
        elif path.suffix == '.json':
            # Direct checkpoint file path
            checkpoint_file = path
        else:
            # Try to interpret as query number
            try:
                query_num = int(checkpoint_path_or_query)
                # Need to find the log directory - use current directory as fallback
                log_dir = override_kwargs.get('log_dir', 'logs')
                checkpoint_file = query_directory(Path(log_dir) / "checkpoints", query_num) / "checkpoint.json"
            except ValueError:
                raise ValueError(f"Invalid checkpoint path or query number: {checkpoint_path_or_query}")
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
        
        # Load checkpoint data
        import json
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        workflow_state = checkpoint_data.get('workflow_state', {})

        # Reconstruct configuration objects
        def _reconstruct_llm_config(config_dict):
            if isinstance(config_dict, dict):
                # Ignore saved fields that LLMConfig does not define.
                known = {f.name for f in dataclasses.fields(LLMConfig)}
                try:
                    return LLMConfig(**{k: v for k, v in config_dict.items() if k in known})
                except Exception:
                    return LLMConfig(model_name=config_dict.get('model_name', 'claude-opus-4-6'))
            return LLMConfig(model_name='claude-opus-4-6')

        # Callers often provide hypothesis_config to restore model and prompt
        # settings, so its dataclass default must not silently override checkpoint
        # behavior. A dedicated behavior override is required for that.
        supplied_hypothesis_config = override_kwargs.get('hypothesis_config')
        hypothesis_config = copy.copy(
            supplied_hypothesis_config or get_default_hypothesis_config()
        )
        # Extract workflow configuration flags from the checkpoint. Flags missing
        # from a checkpoint fall back to the values below; new runs use the
        # constructor defaults.
        checkpoint_git_tracking = workflow_state.get('git_tracking', False)
        checkpoint_multiturn = workflow_state.get('multiturn_hypothesis', True)
        checkpoint_review_search = workflow_state.get('review_search', False)
        checkpoint_review_design = workflow_state.get('review_design', False)

        # Override with any provided parameters
        init_kwargs = {
            'hypothesis_config': hypothesis_config,
            'dataset_description': workflow_state.get('dataset_description', ''),
            'task_description': workflow_state.get('task_description', None),
            'agent_config_list': override_kwargs.get('agent_config_list', []),
            'max_queries': workflow_state.get('max_queries', 10),
            'log_dir': override_kwargs.get('log_dir', 'logs'),
            'output_home': override_kwargs.get('output_home', 'outputs'),
            # Use checkpoint values as defaults, can be overridden by override_kwargs below
            'git_tracking': checkpoint_git_tracking,
            'multiturn_hypothesis': checkpoint_multiturn,
            'review_search': checkpoint_review_search,
            'review_design': checkpoint_review_design,
        }

        # Apply any additional overrides (exclude parameters not in __init__)
        # This allows explicit override_kwargs to take precedence over checkpoint values
        excluded_override_keys = {
            'agent_config_list',
            'log_dir',
            'output_home',
            'refinement_feedback',
            'hypothesis_config',
        }
        for key, value in override_kwargs.items():
            if key not in excluded_override_keys:
                init_kwargs[key] = value
        
        # Create workflow instance
        workflow = cls(**init_kwargs)

        # Restore workflow state
        workflow.query_num = workflow_state.get('query_num', 0)
        workflow.results = workflow_state.get('results', [])
        workflow.report = workflow_state.get('report', '')
        workflow.cumulative_report = workflow_state.get('cumulative_report', '')
        workflow.images = workflow_state.get('images', {})
        workflow.image_counter = workflow_state.get('image_counter', 0)
        workflow.review_history = workflow_state.get('review_history', [])
        workflow.generated_reviewer_prompt = workflow_state.get('generated_reviewer_prompt', None)

        # If reviewer prompt was restored, recreate the review_optimization_llm with it
        if workflow.generated_reviewer_prompt and hasattr(workflow, 'review_optimization_llm_config'):
            review_config_with_prompt = LLMConfig(
                model_name=workflow.review_optimization_llm_config.model_name,
                api_provider=workflow.review_optimization_llm_config.api_provider,
                api_key=workflow.review_optimization_llm_config.api_key,
                temperature=workflow.review_optimization_llm_config.temperature,
                frequency_penalty=workflow.review_optimization_llm_config.frequency_penalty,
                max_tokens=workflow.review_optimization_llm_config.max_tokens,
                system_prompt=workflow.generated_reviewer_prompt
            )

            if workflow.review_search:
                workflow.review_optimization_llm = BaseLLMWithSearch(
                    llm_config=review_config_with_prompt,
                    logger=workflow.logger
                )
            else:
                workflow.review_optimization_llm = OnePassLLM(
                    llm_config=review_config_with_prompt
                )

            workflow.logger.log_action("reviewer_prompt_restored",
                f"Restored reviewer prompt from checkpoint ({len(workflow.generated_reviewer_prompt)} chars)")

        # Store refinement feedback if provided
        workflow.refinement_feedback = override_kwargs.get('refinement_feedback', None)
        
        # Restore agent states
        agents_data = checkpoint_data.get('agents', {})

        # Restore director agent state (new workflow structure)
        if 'director_agent' in agents_data and hasattr(workflow, 'director_agent'):
            agent_data = agents_data['director_agent']
            if hasattr(workflow.director_agent, 'restore_state'):
                success = workflow.director_agent.restore_state(agent_data.get('state', {}))
                if not success:
                    workflow.logger.log_error("Failed to restore director agent state")

        # Restore hypothesis generation agent state (new workflow structure)
        if 'hypothesis_gen_agent' in agents_data and hasattr(workflow, 'hypothesis_gen_agent'):
            agent_data = agents_data['hypothesis_gen_agent']
            if hasattr(workflow.hypothesis_gen_agent, 'restore_state'):
                success = workflow.hypothesis_gen_agent.restore_state(agent_data.get('state', {}))
                if not success:
                    workflow.logger.log_error("Failed to restore hypothesis generation agent state")

        # Legacy: Restore old hypothesis_agent state for backward compatibility
        if 'hypothesis_agent' in agents_data and hasattr(workflow, 'hypothesis_agent'):
            agent_data = agents_data['hypothesis_agent']
            if hasattr(workflow.hypothesis_agent, 'restore_state'):
                success = workflow.hypothesis_agent.restore_state(agent_data.get('state', {}))
                if not success:
                    workflow.logger.log_error("Failed to restore hypothesis agent state (legacy)")
        
        # Restore tool agents states
        for tag, agent in getattr(workflow, 'agents', {}).items():
            if tag in agents_data and hasattr(agent, 'restore_state'):
                agent_data = agents_data[tag]
                success = agent.restore_state(agent_data.get('state', {}))
                if not success:
                    workflow.logger.log_error(f"Failed to restore {tag} agent state")
        
        workflow.logger.log_action("workflow_restored", 
            f"Restored workflow from query {workflow.query_num} checkpoint")
        return workflow

    def save_checkpoint(self, query_num: Optional[int] = None) -> str:
        """
        Save a checkpoint manually.
        
        Args:
            query_num: Query number (defaults to current query_num)
            
        Returns:
            str: Path to saved checkpoint
        """
        if query_num is None:
            query_num = getattr(self, 'query_num', 1)
        
        return self.checkpoint_manager.save_checkpoint(query_num, force_full=True)

    def list_checkpoints(self):
        """List all available checkpoints."""
        return self.checkpoint_manager.list_checkpoints()

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the most recent N."""
        return self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n)

    def reset_for_new_round(self):
        """
        Reset workflow state for a new round while keeping configuration intact.

        This method clears accumulated state (results, reports, query numbers, etc.)
        but preserves all configuration and agent instances. Useful when running
        multiple rounds with the same workflow instance.

        Call this before running run() again if you want a fresh round.
        
        Note: background_report is NOT reset here as it contains accumulated knowledge
        from previous rounds that should persist. It should be updated externally
        (e.g., by HypothesisBrancher) when new accumulated knowledge is available.
        """
        # Reset query tracking
        if hasattr(self, 'query_num'):
            delattr(self, 'query_num')  # Remove so it defaults to 0 in run()

        # Reset results and reports (current round only)
        self.results = []
        self.report = ""
        self.cumulative_report = ""

        # NOTE: background_report is intentionally NOT reset - it should persist across rounds
        # It will be updated externally when new accumulated knowledge is available

        # Reset review tracking
        self.review_history = []

        # Reset image tracking
        self.images = {}
        self.image_counter = 0

        # Reset director agent conversation history
        if hasattr(self.director_agent, 'reset_conversation'):
            self.director_agent.reset_conversation()

        # Reset hypothesis generation agent conversation history
        if hasattr(self.hypothesis_gen_agent, 'reset_conversation'):
            self.hypothesis_gen_agent.reset_conversation()

        # Reset refinement feedback if it exists
        if hasattr(self, 'refinement_feedback'):
            self.refinement_feedback = None

        # Log the reset
        try:
            self.logger.log_action("workflow_reset", "Workflow state reset for new round")
        except Exception:
            pass



    def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up director agent
            if hasattr(self, 'director_agent'):
                self.director_agent.cleanup()

            # Clean up hypothesis generation agent
            if hasattr(self, 'hypothesis_gen_agent'):
                self.hypothesis_gen_agent.cleanup()

            # Clean up tool agents
            if hasattr(self, 'agents'):
                for agent in self.agents.values():
                    if hasattr(agent, 'cleanup'):
                        agent.cleanup()

            self.logger.log_action("cleanup", "DirectorWorkflow cleanup completed")
        except Exception as e:
            self.logger.log_error(f"Warning: Error during cleanup: {str(e)}")
