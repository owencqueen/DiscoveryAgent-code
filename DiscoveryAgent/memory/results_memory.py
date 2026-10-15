"""
Results Memory Manager for DiscoveryAgent package.

This module provides a memory manager that iteratively updates reports by
integrating new information with previously stored reports using an LLM.

Implements a three-agent system following the ACE (Atomic Context Extension) paper:
1. INTEGRATOR: Combines base + delta comprehensively (errs on inclusion)
2. ANALYZER: Identifies redundancy and consolidation opportunities
3. REFINER: Performs careful consolidation with verification
"""

import os
import json
from typing import Optional, Dict, List, Any
from ..base_llm import OnePassLLM
from ..config import LLMConfig


# ============================================================================
# INTEGRATOR AGENT - Combines base + delta comprehensively
# ============================================================================

INTEGRATOR_SYSTEM_PROMPT = """You are an expert research integrator. Your ONLY job is to combine base and delta reports comprehensively.

CRITICAL RULES:
- Do NOT worry about length or redundancy at this stage
- Include EVERYTHING from both reports
- When in doubt, keep both versions
- Preserve ALL specific examples, data points, and technical details
- Never drop information to save space
- Maintain clear section structure

Your goal is COMPLETENESS, not conciseness."""

INTEGRATOR_PROMPT = """You are combining two research reports into one comprehensive integrated report.

BASE REPORT (existing information):
{existing_report}

REPORT TO INTEGRATE:
{new_report}

Your task: Create a COMPREHENSIVE integrated report that includes ALL information from both reports.

Integration Guidelines:
1. Preserve every specific finding, data point, example, and technical detail from both reports
2. Organize information logically with clear sections
3. If the same topic appears in both reports:
   - Keep both versions if they have ANY differences in detail, perspective, or examples
   - Only merge if they are EXACTLY identical
4. Resolve contradictions by presenting both perspectives and noting the difference
5. Maintain chronological flow when relevant
6. You must adhere to the base structure of the report, which is described like this:

# Background
- This section contains background information about the dataset, including any literature review or searching.

# Data Characteristics
- This section should convey a comprehensive view of the data and it's characteristics, including important statistics or qualities that should be taken into consideration when using it.

# Experiments and Analyses
- You should put all of your experiments and analyses into this section, making sure to structure them in a logical and coherent way.
- Format it with sub-sections with this exact format:

## [Experiment Name]
Hypothesis: [The original hypothesis]
Methods: [Any methodologies used to test the hypothesis]
Results: [The results: Was the hypothesis proven or disproven? Were there any additional insights or findings from this experiment?]

# Discussion
- This section should summarize overall findings. Focus on things that are interesting and scientifically valuable.
- What oddities or surprising findings should I know about this dataset? What was found that was most surprising?
- What are avenues for future research or investigation?
- What should I or other researchers know about your work if we want to extend or evaluate it? Focus on limitations and caveats.

IMPORTANT: You must adhere to this structure, don't deviate from it.
CRITICAL: Do NOT consolidate or summarize at this stage. When in doubt, include more rather than less."""

# ============================================================================
# ANALYZER AGENT - Identifies redundancy
# ============================================================================

ANALYZER_SYSTEM_PROMPT = """You are a redundancy analyst reviewing integrated research reports.

Your task: Identify ONLY truly redundant content where consolidation is safe.

CRITICAL RULES:
- Only flag content that is EXACTLY or nearly identical
- If consolidating would lose ANY detail, example, or nuance, mark as HIGH RISK
- Be conservative: when in doubt, do NOT recommend consolidation
- Focus on true duplicates, not related but distinct information"""

ANALYZER_PROMPT = """Review this integrated research report and identify redundancy that could be safely consolidated.

INTEGRATED REPORT:
{integrated_report}

Your task: Identify redundant content where:
- The exact same information appears multiple times
- One section fully subsumes another with no additional detail
- Content can be merged WITHOUT losing any specifics, examples, or nuance

Output a JSON object with this structure:
{{
  "redundancies": [
    {{
      "sections": ["Section identifier 1", "Section identifier 2"],
      "reason": "Brief explanation of why these are redundant",
      "risk_level": "low|medium|high",
      "key_information_to_preserve": ["specific detail 1", "specific detail 2", ...]
    }}
  ]
}}

Risk level guidelines:
- LOW: Exact duplicates with no loss of information if merged
- MEDIUM: Very similar but small details differ - consolidation possible with care
- HIGH: Related content but consolidating would lose domain-specific insights

CRITICAL: If you're uncertain about whether consolidation is safe, mark it as HIGH risk.

If no safe consolidation opportunities exist, return: {{"redundancies": []}}

Provide your analysis as valid JSON:"""

# ============================================================================
# REFINER AGENT - Careful consolidation with verification
# ============================================================================

REFINER_SYSTEM_PROMPT = """You are a research refiner who performs careful consolidation with verification.
You are to refine the proposed report by integrating and organizing results, removing any unnecessary details and trimming text to only the main results.
You should consider if the results may be important for future researchers to know when developing hypotheses on this dataset.

CRITICAL RULES:
- Only consolidate LOW-RISK redundancies
- Skip consolidations where verification fails
- Preserve ALL key information identified in the analysis
- When in doubt, keep content separate
- Never generalize specific findings into abstract statements
- Document all changes in the verification log"""

REFINER_PROMPT = """You will selectively consolidate redundancies in an integrated report while maintaining verification.

INTEGRATED REPORT:
{integrated_report}

REDUNDANCY ANALYSIS:
{redundancy_analysis}

Your task: Carefully consolidate ONLY the low-risk redundancies.

Process:
1. Review the redundancy analysis
2. For each LOW-RISK redundancy:
   - Merge the sections while ensuring ALL key information is retained
   - Verify every important detail is preserved
3. For MEDIUM and HIGH-RISK redundancies:
   - Skip consolidation entirely
   - Keep content as-is
4. Remove unnecessary verbosity and trim to essential results
5. Keep content that would be valuable for future researchers developing hypotheses

CRITICAL: If consolidation feels forced or you're unsure, keep the content separate. Preserve all specific findings, data points, and examples.

IMPORTANT: You must adhere to this structure for the final report:

# Background and Literature Review
- This section contains background information about the dataset and a literature review or searching you've done on it.
- Focus on including the most relevant papers or other sources that are useful to understand this dataset and the methods you've used.

# Data Characteristics
- This section should convey a comprehensive view of the data and it's characteristics, including important statistics or qualities that should be taken into consideration when using it.
- Include both new analyses you ran on the data and overall characteristics that are already known (through the background provided to you, for example).

# Experiments and Analyses
- You should put all of your experiments and analyses into this section, making sure to structure them in a logical and coherent way.
- Format it with sub-sections with this exact format:

## [Experiment Name]
Hypothesis: [The original hypothesis you sought to test]
Methods: [Any methodologies you used to test the hypothesis]
Results: [The results of your experiment. Was the hypothesis proven or disproven? Were there any additional insights or findings from this experiment?]
Figures: [Any figures or images that are relevant to the experiment and results]
- For figures, you should use standard markdown image syntax to include them in the report. Use the `![Figure Description](path/to/figure)` syntax.
- It is important that you include figures EXACTLY AS THEY ORIGINALLY APPEARED IN THE REPORT. Do not change the figures paths.

# Discussion
- This section should summarize overall findings. Focus on things that are interesting and scientifically valuable.
- What oddities or surprising findings should I know about this dataset? What was found that was most surprising?
- What are avenues for future research or investigation?
- What should I or other researchers know about your work if we want to extend or evaluate it? Focus on limitations and caveats.

Please provide the refined report as plain text."""


# ============================================================================
# AGENT CLASSES
# ============================================================================

class IntegratorAgent:
    """
    INTEGRATOR Agent - Combines base and delta reports comprehensively.
    Following ACE paper's Generator role - errs on the side of inclusion.
    """

    def __init__(self, llm_config: LLMConfig, 
            system_prompt: Optional[str] = INTEGRATOR_SYSTEM_PROMPT,
            turn_prompt: Optional[str] = INTEGRATOR_PROMPT):
        """
        Initialize the Integrator agent.

        Args:
            llm_config: Configuration for the LLM
        """
        self.llm = OnePassLLM(
            system_prompt=system_prompt,
            llm_config=llm_config
        )

        self.turn_prompt = turn_prompt

    def integrate(
        self,
        existing_report: str,
        new_report: str,
        max_tokens: Optional[int] = None
    ) -> tuple[str, float]:
        """
        Integrate base and delta reports comprehensively.

        Args:
            existing_report: The base report
            new_report: The delta report to integrate
            max_tokens: Maximum tokens for the LLM response

        Returns:
            Tuple of (integrated_report, cost)
        """
        prompt = self.turn_prompt.format(
            existing_report=existing_report,
            new_report=new_report
        )

        integrated_report, cost = self.llm(
            prompt,
            max_tokens=max_tokens,
            get_cost=True
        )

        return integrated_report, cost


class AnalyzerAgent:
    """
    ANALYZER Agent - Identifies redundancy and consolidation opportunities.
    Following ACE paper's Reflector role - provides quality feedback.
    """

    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the Analyzer agent.

        Args:
            llm_config: Configuration for the LLM
        """
        self.llm = OnePassLLM(
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            llm_config=llm_config
        )

    def analyze(
        self,
        integrated_report: str,
        max_tokens: Optional[int] = None
    ) -> tuple[Dict[str, Any], float]:
        """
        Analyze the integrated report for redundancy.

        Args:
            integrated_report: The report to analyze
            max_tokens: Maximum tokens for the LLM response

        Returns:
            Tuple of (redundancy_analysis_dict, cost)
        """
        prompt = ANALYZER_PROMPT.format(
            integrated_report=integrated_report
        )

        analysis_text, cost = self.llm(
            prompt,
            max_tokens=max_tokens,
            get_cost=True
        )

        # Parse JSON response
        try:
            # Clean the response to extract JSON
            analysis_text = analysis_text.strip()
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text[7:]
            if analysis_text.startswith("```"):
                analysis_text = analysis_text[3:]
            if analysis_text.endswith("```"):
                analysis_text = analysis_text[:-3]
            analysis_text = analysis_text.strip()

            analysis_dict = json.loads(analysis_text)
        except json.JSONDecodeError as e:
            # If parsing fails, return empty redundancies
            print(f"Warning: Failed to parse analyzer output as JSON: {e}")
            print(f"Raw output: {analysis_text[:500]}...")
            analysis_dict = {"redundancies": []}

        return analysis_dict, cost


class RefinerAgent:
    """
    REFINER Agent - Performs careful consolidation with verification.
    Following ACE paper's Curator role - makes selective, verified changes.
    """

    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the Refiner agent.

        Args:
            llm_config: Configuration for the LLM
        """
        self.llm = OnePassLLM(
            system_prompt=REFINER_SYSTEM_PROMPT,
            llm_config=llm_config
        )

    def refine(
        self,
        integrated_report: str,
        redundancy_analysis: Dict[str, Any],
        max_tokens: Optional[int] = None
    ) -> tuple[str, float]:
        """
        Refine the integrated report by carefully consolidating redundancies.

        Args:
            integrated_report: The report to refine
            redundancy_analysis: The analysis from the Analyzer agent
            max_tokens: Maximum tokens for the LLM response

        Returns:
            Tuple of (refined_report, cost)
        """
        prompt = REFINER_PROMPT.format(
            integrated_report=integrated_report,
            redundancy_analysis=json.dumps(redundancy_analysis, indent=2)
        )

        refined_report, cost = self.llm(
            prompt,
            max_tokens=max_tokens,
            get_cost=True
        )

        return refined_report, cost


# ============================================================================
# RESULTS MEMORY MANAGER
# ============================================================================

class ResultsMemoryManager:
    """
    Manages iterative report updates by storing and integrating reports using a multi-agent system.

    This class implements the ACE (Atomic Context Extension) paper's approach with three agents:
    1. INTEGRATOR: Combines base + delta comprehensively (errs on inclusion)
    2. ANALYZER: Identifies redundancy and consolidation opportunities
    3. REFINER: Performs careful consolidation with verification

    The manager uses the three-agent ACE memory pipeline.

    Attributes:
        llm_config (LLMConfig): Configuration for the LLM used in report integration
        current_report (Optional[str]): The currently stored report, or None if no report exists
        integrator (IntegratorAgent): Agent for comprehensive integration
        analyzer (AnalyzerAgent): Agent for redundancy analysis
        refiner (RefinerAgent): Agent for careful consolidation
        total_cost (float): Total cost of all LLM calls made by this manager
        update_count (int): Number of times update_report has been called
        refinement_logs (List[Dict]): Logs of all refinement operations
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize the ResultsMemoryManager.

        Args:
            llm_config: Configuration for the LLM. If None, uses Claude Sonnet 4.5
                       (recommended for the ACE approach).
        """
        # Set up LLM configuration - default to Claude for better performance
        if llm_config is None:
            llm_config = LLMConfig(model_name="claude-sonnet-4-5-20250929", max_tokens = 18000)

        self.llm_config = llm_config
        self.current_report: Optional[str] = None
        self.total_cost: float = 0.0
        self.update_count: int = 0
        self.refinement_logs: List[Dict[str, Any]] = []

        # Initialize agents

        self.integrator = IntegratorAgent(llm_config)
        self.analyzer = AnalyzerAgent(llm_config)
        self.refiner = RefinerAgent(llm_config)

    def update_report(
        self,
        new_report: str,
        max_tokens: Optional[int] = None,
        verbose: bool = True
    ) -> str:
        """
        Update the stored report with new information using the multi-agent approach.

        If no report currently exists, stores the new report directly.
        If a report exists:
        Runs INTEGRATOR → ANALYZER → REFINER.

        Args:
            new_report: The new report to add or integrate
            max_tokens: Maximum tokens for the LLM response. If None, uses config default.
            verbose: If True, prints status messages during the update

        Returns:
            The updated report (either the new report if first, or integrated/refined report)
        """
        self.update_count += 1

        if self.current_report is None:
            # First report - just store it
            if verbose:
                print(f"[ResultsMemoryManager] Storing initial report (update #{self.update_count})")
            self.current_report = new_report
            return self.current_report

        # === STEP 1: INTEGRATION ===
        if verbose:
            print(f"\n[ResultsMemoryManager] Update #{self.update_count}")
            print("  Mode: ACE (3-agent)")
            print(f"  - Existing report length: {len(self.current_report)} characters")
            print(f"  - New report length: {len(new_report)} characters")
            print("\n[STEP 1/3] INTEGRATOR: Combining reports comprehensively...")

        integrated_report, integration_cost = self.integrator.integrate(
            existing_report=self.current_report,
            new_report=new_report,
            max_tokens=max_tokens
        )
        self.total_cost += integration_cost

        if verbose:
            print(f"  - Integrated report length: {len(integrated_report)} characters")
            print(f"  - Integration cost: ${integration_cost:.4f}")

        # === STEP 2: ANALYSIS ===
        if verbose:
            print(f"\n[STEP 2/3] ANALYZER: Identifying redundancy...")

        redundancy_analysis, analysis_cost = self.analyzer.analyze(
            integrated_report=integrated_report,
            max_tokens=max_tokens
        )
        self.total_cost += analysis_cost

        num_redundancies = len(redundancy_analysis.get("redundancies", []))
        if verbose:
            print(f"  - Found {num_redundancies} potential redundancies")
            if num_redundancies > 0:
                # Count by risk level
                risk_counts = {"low": 0, "medium": 0, "high": 0}
                for r in redundancy_analysis.get("redundancies", []):
                    risk_level = r.get("risk_level", "high").lower()
                    risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
                print(f"    - Low risk: {risk_counts['low']}, Medium risk: {risk_counts['medium']}, High risk: {risk_counts['high']}")
            print(f"  - Analysis cost: ${analysis_cost:.4f}")

        # === STEP 3: REFINEMENT ===
        if verbose:
            print(f"\n[STEP 3/3] REFINER: Consolidating low-risk redundancies and trimming verbosity...")

        refined_report, refinement_cost = self.refiner.refine(
            integrated_report=integrated_report,
            redundancy_analysis=redundancy_analysis,
            max_tokens=max_tokens
        )
        self.total_cost += refinement_cost

        # Store refinement metadata for inspection
        self.refinement_logs.append({
            "update_number": self.update_count,
            "redundancies_found": num_redundancies,
            "integrated_length": len(integrated_report),
            "refined_length": len(refined_report)
        })

        if verbose:
            print(f"  - Refined report length: {len(refined_report)} characters")
            print(f"  - Size reduction: {len(integrated_report) - len(refined_report)} characters")
            print(f"  - Refinement cost: ${refinement_cost:.4f}")
            print(f"\n  ===== Summary =====")
            print(f"  - Total cost for this update: ${integration_cost + analysis_cost + refinement_cost:.4f}")
            print(f"  - Total cost so far: ${self.total_cost:.4f}")
            print(f"  - Report size change: {len(self.current_report)} → {len(refined_report)} characters "
                  f"({len(refined_report) - len(self.current_report):+d})\n")

        # Update the stored report
        self.current_report = refined_report
        return self.current_report

    def get_report(self) -> Optional[str]:
        """
        Get the current stored report.

        Returns:
            The current report, or None if no report has been stored yet
        """
        return self.current_report

    def clear_report(self):
        """
        Clear the stored report and reset statistics.
        """
        self.current_report = None
        self.total_cost = 0.0
        self.update_count = 0

    def save_report(self, filepath: str):
        """
        Save the current report to a file.

        Args:
            filepath: Path where the report should be saved

        Raises:
            ValueError: If no report exists to save
        """
        if self.current_report is None:
            raise ValueError("No report exists to save")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.current_report)

    def load_report(self, filepath: str):
        """
        Load a report from a file and set it as the current report.

        Args:
            filepath: Path to the report file to load

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        with open(filepath, 'r') as f:
            self.current_report = f.read()

        # Don't increment update_count for loads
        # This allows loading a checkpoint without affecting statistics

    def get_statistics(self) -> dict:
        """
        Get statistics about the memory manager's usage.

        Returns:
            Dictionary containing statistics like update_count, total_cost, etc.
        """
        stats = {
            "update_count": self.update_count,
            "total_cost": self.total_cost,
            "has_report": self.current_report is not None,
            "report_length": len(self.current_report) if self.current_report else 0,
            "model_name": self.llm_config.model_name,
            "mode": "ACE (3-agent)"
        }

        stats["total_refinements"] = len(self.refinement_logs)
        stats["total_redundancies_found"] = sum(
            log["redundancies_found"] for log in self.refinement_logs
        )
        stats["refinement_logs"] = self.refinement_logs

        return stats

    def get_last_refinement_log(self) -> Optional[Dict[str, Any]]:
        """
        Get the refinement metadata from the most recent update (ACE mode only).

        Returns:
            Dictionary containing update number, redundancies found, and size changes,
            or None if no refinements have been made
        """
        if not self.refinement_logs:
            return None
        return self.refinement_logs[-1]

    def export_refinement_logs(self, filepath: str):
        """
        Export all refinement logs to a JSON file (ACE mode only).

        Args:
            filepath: Path where the logs should be saved

        Raises:
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.refinement_logs, f, indent=2)
