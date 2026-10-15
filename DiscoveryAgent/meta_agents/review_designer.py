"""
Review Designer Meta-Agent

This module contains an LLM-powered agent that generates effective reviewer prompts
for use in an agentic research system. The generated prompts are designed to provide
critical, actionable feedback that drives research towards specific goals.
"""

from typing import Optional
from DiscoveryAgent.base_llm import OnePassLLM
from DiscoveryAgent.config import LLMConfig


REVIEW_DESIGNER_SYSTEM_PROMPT = """
You are an expert at designing effective reviewer prompts for AI research systems.

Your task is to create a reviewer prompt that will be used by an LLM critic within an agentic 
research system. This reviewer will evaluate research progress and provide feedback to guide 
the research towards specific goals.

The reviewer prompt you design should:
1. Encourage the reviewer to provide highly specific, actionable feedback
2. Drive the research towards the stated research task and objectives
3. Promote critical evaluation of methods, findings, and scientific rigor
4. Help the research agent self-evolve and improve its approach iteratively
5. Focus on computational analyses (not data collection or real-world interventions)
6. Steer towards novel, scientifically impactful directions
7. Be clear about what constitutes good progress vs. what needs improvement

Your generated reviewer prompt should be comprehensive yet focused, providing clear guidance 
on how to evaluate the research work effectively within the constraints of a computational 
research system.

IMPORTANT: Avoid putting steering the reviewer to be biased or prefer any methods or approaches. 
It should merely evaluate the quality of the research work, aligned towards whatever task the research agent is tasked with doing.

IMPORTANT: The reviewer prompt must instruct the reviewer to keep the research agent on task. 
If a specific task is given to the research agent, then the reviewer should be instructed to ensure the research agent is accomplishing this task and not straying too far from what the user requested.
"""


class ReviewDesigner:
    """
    A meta-agent that generates effective reviewer prompts for agentic research systems.
    
    This agent takes a research task description and generates a tailored prompt that
    will be used by a reviewer LLM to provide critical feedback on research progress.
    The generated prompts are optimized to drive research towards specific objectives
    while encouraging rigorous, novel, and impactful work.
    
    Attributes:
        llm: OnePassLLM instance for generating reviewer prompts
        system_prompt: The system prompt that guides prompt generation
    """
    
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the ReviewDesigner.
        
        Args:
            llm_config: Configuration for the LLM. If None, uses default config.
            system_prompt: Custom system prompt for the designer. If None, uses default.
        """
        if llm_config is None:
            llm_config = LLMConfig(model_name="claude-sonnet-4-6")
        
        if system_prompt is None:
            system_prompt = REVIEW_DESIGNER_SYSTEM_PROMPT
        
        self.system_prompt = system_prompt
        self.llm = OnePassLLM(
            system_prompt=system_prompt,
            llm_config=llm_config
        )
    
    def design_reviewer_prompt(
        self,
        research_task: str,
        max_tokens: int = 2000,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Generate a reviewer prompt tailored to a specific research task.
        
        Args:
            research_task: Description of the research task/objective
            max_tokens: Maximum tokens for the generated prompt
            additional_context: Optional additional context about the research system
        
        Returns:
            A generated reviewer prompt string
        """
        user_prompt = self._construct_user_prompt(research_task, additional_context)
        
        reviewer_prompt = self.llm.forward(
            input=user_prompt,
            max_tokens=max_tokens
        )
        
        return reviewer_prompt
    
    def _construct_user_prompt(
        self,
        research_task: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Construct the user prompt for the LLM.
        
        Args:
            research_task: Description of the research task
            additional_context: Optional additional context
            
        Returns:
            Formatted user prompt string
        """
        prompt = f"""Generate a reviewer prompt for an agentic research system working on the following research task:

RESEARCH TASK:
{research_task}
"""
        
        if additional_context:
            prompt += f"""

ADDITIONAL CONTEXT:
{additional_context}
"""
        
        prompt += """

Please generate a comprehensive reviewer prompt that will help an LLM critic provide effective 
feedback to guide this research. The prompt should:

- Be specific to this research task and its objectives
- Encourage the reviewer to provide actionable, concrete feedback
- Help steer the research towards novel and scientifically impactful findings
- Emphasize critical evaluation of methods and rigor
- Remind the reviewer that the research agent is computational (can't collect data or do user studies)
- Guide the reviewer to help the research agent self-evolve and improve iteratively
- Include specific criteria for what constitutes good progress

Your output should be the reviewer prompt itself (not meta-commentary about the prompt).
"""
        
        return prompt
    
    def __call__(
        self,
        research_task: str,
        max_tokens: int = 2000,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Callable interface for the ReviewDesigner.
        
        Args:
            research_task: Description of the research task/objective
            max_tokens: Maximum tokens for the generated prompt
            additional_context: Optional additional context
            
        Returns:
            A generated reviewer prompt string
        """
        return self.design_reviewer_prompt(
            research_task=research_task,
            max_tokens=max_tokens,
            additional_context=additional_context
        )


def create_review_designer(
    model_name: str = "claude-sonnet-4-6",
    api_provider: Optional[str] = None
) -> ReviewDesigner:
    """
    Convenience function to create a ReviewDesigner with custom model settings.
    
    Args:
        model_name: Name of the LLM model to use
        api_provider: API provider (e.g. anthropic)
        
    Returns:
        Configured ReviewDesigner instance
    """
    llm_config = LLMConfig(
        model_name=model_name,
        api_provider=api_provider
    )
    return ReviewDesigner(llm_config=llm_config)


# Example usage
if __name__ == "__main__":
    # Example: Create a review designer and generate a reviewer prompt
    designer = create_review_designer(model_name = "claude-opus-4-1-20250805")
    
    research_task = """
    Analyze a dataset of EEG recordings to identify patterns associated with different 
    cognitive states. The goal is to develop novel methods for classifying mental states 
    and understanding the neural correlates of cognition.
    """
    
    reviewer_prompt = designer.design_reviewer_prompt(research_task)
    
    print("Generated Reviewer Prompt:")
    print("=" * 80)
    print(reviewer_prompt)
    print("=" * 80)

