"""Single-hypothesis generation for DiscoveryAgent research rounds."""

import time
from typing import Optional, List, Dict, Any

import anthropic

from ..config import HypothesisGenerationConfig, get_slurm_resource_info
from ..utils.logger import Logger
from ..prompts.hyp_generation import (
    DATASET_DESCRIPTION_PROMPT,
    DATASET_DESCRIPTION_TASK_DESCRIPTION_PROMPT,
    REPORT_MULTITURN_PROMPT,
    BACKGROUND_REPORT_PROMPT,
    HYPOTHESIS_GENERATION_PROMPT,
)
from ..utils.token_utils import (
    count_anthropic_tokens,
)
from ..utils.llm_diagnostics import (
    describe_response,
    log_api_exception,
    log_api_exchange,
    refusal_notice,
)

HYPOTHESIS_GENERATION_SYSTEM_PROMPT = f"""
You are a scientific expert who is highly skilled in analyzing scientific data, results, and making judgements about future work.
You're creative and think outside the box, but you also recognize what hypotheses are feasible and interesting to the scientitic community.

Your job is to take in a dataset description, a description of a research question you're tasked with answering, and a report from some explorations on this dataset and task.
Using this information and your broad knowledge of the scientific literature in this field, you should generate hypotheses that are interesting and feasible to pursue in a scientific investigation.
Testing such hypotheses should require only computational methods; do not propose hypotheses that require experimental work or generating new data.
This hypothesis will be tested by a computational agent that is generally capable of running computational analyses and experiments and is an expert at writing code and running analyses.

You will be asked to generate a list of hypotheses that are proposals for future research on this dataset and task.
For each hypothesis, you must clearly explain why it is interesting, plausible, and how it could be tested with computational methods only.
You MUST ground your hypotheses in specific literature or in the evidence found from previous explorations on this dataset.
Provide specific citations (literature, sources, etc.) and evidence from the background report.
You may decide to continue exploring a hypothesis that was previously proposed or explored in the background report, but make sure to propose some hypotheses that are novel and that haven't been explored.

Importantly, your hypotheses are proposals; a scientist who is an expert in the field will choose from the list.
The scientist expects a data-driven, literature-grounded justification for why you would want to explore this hypothesis, so be rigorous in your description and reasoning.
"""

class HypothesisGenAgent:
    """
    Hypothesis generation agent that specializes in creating scientifically sound
    hypotheses based on evidence, prior knowledge, and accumulated findings.

    This agent focuses purely on hypothesis generation and can be augmented with
    deep research tools (literature review, web search) in the future.

    For query generation and research orchestration, use DirectorAgent.
    """

    def __init__(
        self,
        config: HypothesisGenerationConfig,
        background_report_path: str = None,
        logger=None,
        helper_agents_prompt: Optional[str] = None,
    ):
        """
        Initialize the hypothesis generation agent.

        Args:
            config (HypothesisGenerationConfig): Full configuration object
            logger (Logger, optional): Shared logger instance. If None, creates a new one.
            helper_agents_prompt (str, optional): Description of helper agents available to
                the DirectorAgent; used to inform plan generation when generate_plan is enabled.
        """
        self.config = config

        self.llm_config = getattr(self.config, "llm_config", None)
        self.log_dir = getattr(self.config, "log_dir", None)
        self.system_prompt = getattr(self.config, "system_prompt", None)
        self.background_report_path = background_report_path
        self.background_literature_content = None
        self.generate_plan = getattr(self.config, "generate_plan", False)
        self.helper_agents_prompt = helper_agents_prompt
        # Build system prompt focused on hypothesis generation
        hypothesis_system_prompt = self.system_prompt

        # Add resource information if running on SLURM
        resource_info = get_slurm_resource_info()
        if resource_info:
            hypothesis_system_prompt += f"\n\n{resource_info}"

        # Add hypothesis-specific instructions
        hypothesis_system_prompt += """

# Hypothesis Generation Guidelines

When generating hypotheses, you should:
1. Base hypotheses on evidence from the data and previous findings
2. Ensure hypotheses are testable and falsifiable
3. Consider novelty - look for unexplored aspects or novel insights
4. Ground hypotheses in scientific reasoning and domain knowledge
5. Provide clear rationale for each hypothesis
6. Consider multiple competing hypotheses when appropriate

Your hypotheses should advance scientific understanding and be actionable through further experimentation or analysis.
"""

        self.system_prompt = hypothesis_system_prompt

        # Use passed logger or create a new one if not provided
        self.logger = logger if logger is not None else Logger("hypothesis_gen_agent", log_dir=self.log_dir)

        # Log the system prompt for debugging and transparency
        try:
            self.logger.log_action("system_prompt_initialized", f"System prompt length: {len(self.system_prompt)} characters")
            self.logger.log_prompt("system", self.system_prompt, "hypothesis_gen_agent_system_prompt")
        except Exception:
            print(f"📝 Hypothesis generation agent system prompt initialized: {len(self.system_prompt)} characters")

        # Initialize client based on API provider
        if not self.llm_config.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = anthropic.Anthropic(api_key=self.llm_config.api_key)
        self.logger.log_action("llm_client_init", f"Using Anthropic API with model: {self.llm_config.model_name}")

        # Persistent conversation messages for hypothesis refinement
        # The system prompt is passed separately from the messages.
        self.messages: List[Dict[str, Any]] = []

        self.background_report_read = False # Relates to background report used internally
        if self.background_report_path:
            try:
                with open(self.background_report_path, 'r') as f:
                    self.background_literature_content = f.read()
                self.logger.log_action(
                    "background_report_loaded",
                    f"Loaded background report from {self.background_report_path}"
                )
            except Exception as e:
                self.logger.log_error(
                    f"Failed to load background report from {self.background_report_path}: {e}"
                )
                self.background_literature_content = None

    def estimate_context_tokens(self) -> int:
        """Estimate input tokens for the current conversation state."""
        try:
            count_info = count_anthropic_tokens(
                client=self.client,
                model_name=self.llm_config.model_name,
                messages=self.messages,
                system=self.system_prompt,
            )
            tokens = int(count_info.get("input_tokens", 0))
            self.logger.log_action("token_estimate", f"Estimated input tokens: {tokens}")
            return tokens
        except Exception as e:
            self.logger.log_error(f"Token estimation failed: {str(e)}")
            return 0

    def _extract_text_from_anthropic_response(self, response) -> str:
        """Safely extract concatenated text blocks from an Anthropic response, even when tools are used."""
        text_parts = []
        try:
            for block in getattr(response, "content", []):
                if getattr(block, "type", None) == "text" and hasattr(block, "text"):
                    text_parts.append(block.text)
            return "\n".join(part for part in text_parts if part)
        except Exception as e:
            self.logger.log_error(
                f"Failed to extract text from response: {type(e).__name__}: {e}\n"
                f"response: {describe_response(response)}"
            )
            return ""

    def _call_api_with_retry(self, api_params: dict, max_retries: int = 5, wait_time: int = 60) -> str:
        """
        Call the LLM API with retry logic for handling transient errors.

        Args:
            api_params: Parameters to pass to the API
            max_retries: Maximum number of retry attempts (default: 5)
            wait_time: Time to wait between retries in seconds (default: 60)

        Returns:
            str: The response text from the API

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None
        # Set when Anthropic declines this call, so callers can tell a refusal
        # apart from a malformed completion (they look identical downstream).
        self.last_refusal = None

        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                response = self.client.messages.create(**api_params)
                text = self._extract_text_from_anthropic_response(response)
                log_api_exchange(self.logger, "HypothesisGenAgent", api_params, response, text)
                self.last_refusal = refusal_notice(response)
                return text

            except (anthropic.InternalServerError, anthropic.APIError) as e:
                last_error = e
                error_msg = str(e)

                # Check if this is a retryable error
                if "Overloaded" in error_msg or "500" in error_msg or "503" in error_msg or "429" in error_msg:
                    if attempt < max_retries:
                        self.logger.log_action(
                            "api_retry",
                            f"API error (attempt {attempt + 1}/{max_retries + 1}): {error_msg}. "
                            f"Waiting {wait_time} seconds before retry..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        self.logger.log_error(f"Max retries ({max_retries}) reached.")
                        log_api_exception(
                            self.logger,
                            e,
                            source="HypothesisGenAgent",
                            api_params=api_params,
                            retryable=False,
                        )
                        raise Exception(f"API call failed after {max_retries} retries: {error_msg}")
                else:
                    # Non-retryable error, raise immediately
                    log_api_exception(
                        self.logger,
                        e,
                        source="HypothesisGenAgent",
                        api_params=api_params,
                        retryable=False,
                    )
                    raise

            except Exception as e:
                # For unexpected errors, raise immediately without retry
                last_error = e
                log_api_exception(
                    self.logger,
                    e,
                    source="HypothesisGenAgent",
                    api_params=api_params,
                    retryable=False,
                )
                raise

        # If we somehow exit the loop without returning or raising
        if last_error:
            raise Exception(f"API call failed: {str(last_error)}")
        raise Exception("API call failed for unknown reason")

    def add_user_message(self, content) -> None:
        """Append a user message (string or structured content) to the conversation and log it."""
        self.messages.append({"role": "user", "content": content})
        # Log stringified preview to avoid huge dumps
        try:
            preview = content if isinstance(content, str) else str(content)[:500]
            self.logger.log_prompt("user", preview, "Hypothesis Generation")
        except Exception:
            pass

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the conversation and log it."""
        self.messages.append({"role": "assistant", "content": content})
        self.logger.log_response(content, "hypothesis_generation")

    def generate_hypothesis(
        self,
        dataset_description: str = None,
        task_description: str = None,
        background_report: str = None,
    ) -> Optional[str]:
        """Generate one hypothesis using accumulated research context."""
        return self._generate_single_hypothesis(
            dataset_description=dataset_description,
            task_description=task_description,
            background_report=background_report,
        )

    def _build_hypothesis_prefix(
        self,
        dataset_description: str = None,
        task_description: str = None,
        background_report: str = None,
    ) -> str:
        """
        Build the common prefix prompt used for hypothesis generation.

        This includes:
        - Dataset / task description OR prior report
        - Optional background_report (only once per agent instance)
        - Instructions to first produce an **Interpretation** of the results
        """
        prefix_prompt = ""

        # Add dataset/task description (if provided)
        if dataset_description is not None:
            if task_description is not None:
                prefix_prompt += DATASET_DESCRIPTION_TASK_DESCRIPTION_PROMPT.format(
                    dataset_description=dataset_description,
                    task_description=task_description
                )
            else:
                prefix_prompt += DATASET_DESCRIPTION_PROMPT.format(
                    dataset_description=dataset_description
                )

        # Add background report if available (accumulated knowledge from previous rounds)
        if background_report and (not self.background_report_read):
            prefix_prompt += "\n\n" + BACKGROUND_REPORT_PROMPT.format(
                background_report=background_report
            ) + "\n\n"
            self.background_report_read = True

        if prefix_prompt != "":
            prefix_prompt += \
"""
Produce first an interpretation of these results, which should look like this:

**Interpretation**: [Your interpretation of the report from the previous query]

Then you should focus on generating a hypothesis.

"""

        return prefix_prompt

    def _generate_single_hypothesis(
        self,
        dataset_description: str = None,
        task_description: str = None,
        background_report: str = None,
    ) -> Optional[str]:
        """
        Generate a single hypothesis.

        Args:
            dataset_description (str): Description of the dataset
            task_description (str): Description of the task
            background_report (str): Consolidated report / accumulated knowledge

        Returns:
            Optional[str]: Generated hypothesis text from the response
        """

        prefix_prompt = self._build_hypothesis_prefix(
            dataset_description=dataset_description,
            task_description=task_description,
            background_report=background_report,
        )

        hypothesis_prompt = prefix_prompt + HYPOTHESIS_GENERATION_PROMPT

        self.add_user_message(hypothesis_prompt)

        try:
            # Prepare API call parameters
            api_params = {
                "model": self.llm_config.model_name,
                "max_tokens": self.config.max_tokens,
                "system": self.system_prompt,
                "messages": self.messages,
            }

            result_text = self._call_api_with_retry(api_params)

            # Add assistant turn to the conversation and log
            self.add_assistant_message(result_text)

            return result_text

        except Exception as e:
            error_msg = f"Error generating hypothesis: {str(e)}"
            self.logger.log_error(error_msg)
            raise Exception(error_msg)

    def reset_conversation(self) -> None:
        """Reset the conversation to the initial state (preserving the system prompt)."""
        self.messages = []
        self.background_report_read = False
        self.logger.log_action("conversation_reset", "Conversation reset for hypothesis generation agent")

    def save_state(self) -> dict:
        """
        Save complete conversation state for checkpointing.

        Returns:
            dict: Complete agent state including full conversation history
        """
        return {
            'agent_type': type(self).__name__,
            'messages': self.messages.copy(),  # Full conversation history
            'system_prompt': self.system_prompt,
            'conversation_length': len(self.messages),
            'background_report_read': self.background_report_read,
            'estimated_tokens': self.estimate_context_tokens(),
            'llm_config': {
                'model_name': self.llm_config.model_name,
                'api_provider': self.llm_config.api_provider,
                'temperature': getattr(self.llm_config, 'temperature', None),
                'frequency_penalty': getattr(self.llm_config, 'frequency_penalty', None),
                'api_key_hash': hash(self.llm_config.api_key) if self.llm_config.api_key else None
            },
            'config_data': {
                'log_dir': self.config.log_dir,
                'max_tokens': self.config.max_tokens,
                'system_prompt': self.config.system_prompt,
            },
        }

    def restore_state(self, state: dict) -> bool:
        """
        Restore complete conversation state from checkpoint.

        Args:
            state: State dictionary from checkpoint

        Returns:
            bool: True if restoration was successful
        """
        try:
            # Restore conversation messages
            self.messages = state.get('messages', []).copy()


            # Validate message structure
            for i, msg in enumerate(self.messages):
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    self.logger.log_error(f"Invalid message structure at index {i}")
                    return False

            # Restore other state if needed (most is already set during initialization)
            self.background_report_read = state.get('background_report_read', False)
            restored_tokens = state.get('estimated_tokens', 0)
            current_tokens = self.estimate_context_tokens()

            # Log restoration details
            self.logger.log_action("state_restored",
                f"Restored {len(self.messages)} messages, "
                f"tokens: {restored_tokens} -> {current_tokens}")

            # Verify conversation makes sense
            conversation_length = state.get('conversation_length', 0)
            if len(self.messages) != conversation_length:
                self.logger.log_action("conversation_length_mismatch",
                    f"Expected {conversation_length}, got {len(self.messages)}")

            return True

        except Exception as e:
            self.logger.log_error(f"Failed to restore hypothesis generation agent state: {str(e)}")
            return False

    def get_state_schema(self) -> dict:
        """
        Get schema description for state validation.

        Returns:
            dict: Schema describing state fields
        """
        return {
            'agent_type': 'Class name of the hypothesis generation agent',
            'messages': 'Complete conversation history (list of message dicts)',
            'system_prompt': 'System prompt used for the agent',
            'conversation_length': 'Number of messages in conversation',
            'estimated_tokens': 'Estimated token count for conversation',
            'llm_config': 'LLM configuration details',
            'config_data': 'Agent configuration data',
        }

    def cleanup(self):
        """Clean up resources."""
        try:
            self.logger.log_action("cleanup", "HypothesisGenAgent cleanup completed")
        except Exception as e:
            # Use centralized logger for cleanup errors
            try:
                self.logger.log_error(f"Warning: Error during cleanup: {str(e)}")
            except Exception:
                import logging
                logging.getLogger("discoveryagent").error(f"Warning: Error during cleanup: {str(e)}")
