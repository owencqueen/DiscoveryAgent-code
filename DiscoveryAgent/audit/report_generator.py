"""
Audit Report Generator Module

This module generates structured audit reports from query summaries,
maintaining clear tracking of rounds and query numbers.
"""

from typing import List, Optional, Dict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from .summarizer import QuerySummary


class AuditReportGenerator:
    """
    Generates structured audit reports from query summaries.
    """

    def __init__(self, include_hypotheses: bool = False, include_reviews: bool = False,
                 per_round_hypotheses: Optional[Dict[int, str]] = None):
        """
        Initialize report generator.

        Args:
            include_hypotheses: Whether to include hypotheses section in report
            include_reviews: Whether to include reviews section in report
            per_round_hypotheses: Dict mapping round_num -> selected hypothesis text
        """
        self.include_hypotheses = include_hypotheses
        self.include_reviews = include_reviews
        self.per_round_hypotheses = per_round_hypotheses or {}

    def generate_report(
        self,
        summaries: List[QuerySummary],
        checkpoint_path: str,
        output_path: str,
        statistics: Optional[Dict] = None
    ) -> str:
        """
        Generate a comprehensive audit report.

        Args:
            summaries: List of query summaries
            checkpoint_path: Path to checkpoints directory (for reference)
            output_path: Path where report will be saved
            statistics: Optional checkpoint statistics

        Returns:
            str: The generated report content
        """
        # Organize summaries by round
        summaries_by_round = self._organize_by_round(summaries)

        # Build report sections
        report_parts = []

        # Header
        report_parts.append(self._generate_header(checkpoint_path, summaries, statistics))

        # Executive Summary
        report_parts.append(self._generate_executive_summary(summaries, summaries_by_round))

        # Hypotheses Section (if enabled)
        if self.include_hypotheses:
            hypotheses_section = self._generate_hypotheses_section(summaries_by_round)
            if hypotheses_section:
                report_parts.append(hypotheses_section)

        # Research Timeline (main content)
        report_parts.append(self._generate_timeline(summaries_by_round))

        # Reviews Section (if enabled)
        if self.include_reviews:
            reviews_section = self._generate_reviews_section(summaries_by_round)
            if reviews_section:
                report_parts.append(reviews_section)

        # Key Insights
        report_parts.append(self._generate_key_insights(summaries))

        # Metadata
        report_parts.append(self._generate_metadata(summaries, statistics))

        # Combine all parts
        report = "\n\n".join(report_parts)

        return report

    def _organize_by_round(self, summaries: List[QuerySummary]) -> Dict[Optional[int], List[QuerySummary]]:
        """Organize summaries by round number."""
        by_round = defaultdict(list)
        for summary in summaries:
            by_round[summary.round_num].append(summary)

        # Sort queries within each round
        for round_num in by_round:
            by_round[round_num].sort(key=lambda s: s.query_num)

        return dict(by_round)

    def _generate_header(self, checkpoint_path: str, summaries: List[QuerySummary], statistics: Optional[Dict]) -> str:
        """Generate report header."""
        total_queries = len(summaries)
        rounds = set(s.round_num for s in summaries if s.round_num is not None)
        total_rounds = len(rounds) if rounds else 1

        # Get time span
        if summaries:
            timestamps = [s.timestamp for s in summaries if s.timestamp != "unknown"]
            if timestamps:
                time_span = f"{timestamps[0]} to {timestamps[-1]}"
            else:
                time_span = "Unknown"
        else:
            time_span = "Unknown"

        header = f"""# DiscoveryAgent Workflow Audit Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Checkpoint Path:** `{checkpoint_path}`
**Total Queries:** {total_queries}
**Total Rounds:** {total_rounds}
**Time Span:** {time_span}

---"""

        return header

    def _generate_executive_summary(
        self,
        summaries: List[QuerySummary],
        summaries_by_round: Dict[Optional[int], List[QuerySummary]]
    ) -> str:
        """Generate executive summary section."""
        total_queries = len(summaries)
        has_rounds = any(r is not None for r in summaries_by_round.keys())

        summary = "## Executive Summary\n\n"

        if has_rounds:
            summary += f"This research workflow consisted of {len(summaries_by_round)} rounds with a total of {total_queries} queries. "
            summary += "The workflow followed a structured multi-round approach, building upon findings from previous rounds.\n\n"

            # Summarize each round briefly
            for round_num in sorted(k for k in summaries_by_round.keys() if k is not None):
                round_summaries = summaries_by_round[round_num]
                summary += f"**Round {round_num}:** {len(round_summaries)} queries\n"
        else:
            summary += f"This research workflow consisted of {total_queries} queries in a single execution. "
            summary += "The queries progressively explored the research questions through iterative investigation.\n\n"

        return summary

    def _generate_timeline(self, summaries_by_round: Dict[Optional[int], List[QuerySummary]]) -> str:
        """Generate detailed timeline section."""
        timeline = "## Detailed Research Timeline\n\n"
        timeline += "This section provides a detailed chronological view of all queries and findings, "
        timeline += "organized by round and query number.\n\n"
        timeline += "---\n"

        # Sort rounds (None should come first if present)
        round_keys = sorted(summaries_by_round.keys(), key=lambda x: (x is not None, x if x is not None else -1))

        for round_num in round_keys:
            round_summaries = summaries_by_round[round_num]

            # Round header
            if round_num is not None:
                timeline += f"\n## Round {round_num}\n\n"
                timeline += f"*{len(round_summaries)} queries in this round*\n\n"

                # Include selected hypothesis that drove this round
                if round_num in self.per_round_hypotheses:
                    timeline += "### Research Directive\n\n"
                    timeline += "The following selected hypothesis guided this round's research:\n\n"
                    timeline += f"> {self.per_round_hypotheses[round_num]}\n\n"
                    timeline += "---\n\n"
            else:
                timeline += f"\n## Research Queries\n\n"
                timeline += f"*{len(round_summaries)} queries total*\n\n"

            # Individual queries
            for summary in round_summaries:
                timeline += self._format_query_summary(summary)
                timeline += "\n---\n"

        return timeline

    def _format_query_summary(self, summary: QuerySummary) -> str:
        """Format a single query summary."""
        # Query header
        if summary.round_num is not None:
            header = f"\n### Query {summary.query_num} (Round {summary.round_num})\n\n"
        else:
            header = f"\n### Query {summary.query_num}\n\n"

        # Metadata
        metadata = f"**Timestamp:** {summary.timestamp}  \n"
        metadata += f"**Agent:** {summary.agent_name}  \n"
        if summary.tokens_used > 0:
            metadata += f"**Summary Tokens:** {summary.tokens_used}\n"
        metadata += "\n"

        # Input query — show full raw query (includes thought/interpretation)
        query_section = ""
        if summary.query_raw:
            query_section = "**Full Query (Thought + Interpretation + Input):**\n"
            query_section += f"```\n{summary.query_raw.strip()}\n```\n\n"
        elif summary.parsed_query:
            query_section = "**Input Query:**\n"
            query_section += f"```\n{summary.parsed_query}\n```\n\n"

        # Summary content (just the raw summary from LLM)
        content = f"{summary.summary}\n\n"

        return header + metadata + query_section + content

    def _generate_key_insights(self, summaries: List[QuerySummary]) -> str:
        """Generate key insights section."""
        insights = "## Query Summary Overview\n\n"
        insights += "This section provides a brief overview of what each query investigated:\n\n"

        # List each query with a brief excerpt
        for i, summary in enumerate(summaries, 1):
            # Get first sentence or first 150 chars of summary
            first_line = summary.summary.split('\n')[0][:150]
            if len(summary.summary.split('\n')[0]) > 150:
                first_line += "..."

            round_label = f" (Round {summary.round_num})" if summary.round_num is not None else ""
            insights += f"{i}. **Query {summary.query_num}{round_label}**: {first_line}\n"

        return insights

    def _generate_metadata(self, summaries: List[QuerySummary], statistics: Optional[Dict]) -> str:
        """Generate metadata section."""
        metadata = "## Workflow Metadata\n\n"

        # Query statistics
        total_queries = len(summaries)
        rounds = set(s.round_num for s in summaries if s.round_num is not None)
        has_rounds = len(rounds) > 0

        metadata += f"- **Total Queries:** {total_queries}\n"

        if has_rounds:
            metadata += f"- **Total Rounds:** {len(rounds)}\n"

            # Queries per round
            if statistics and 'queries_per_round' in statistics:
                metadata += "- **Queries per Round:**\n"
                for round_num, count in sorted(statistics['queries_per_round'].items()):
                    metadata += f"  - Round {round_num}: {count} queries\n"
        else:
            metadata += f"- **Rounds:** Single-round workflow\n"

        # Agent usage
        agents = {}
        for summary in summaries:
            agents[summary.agent_name] = agents.get(summary.agent_name, 0) + 1

        metadata += "- **Agents Used:**\n"
        for agent, count in sorted(agents.items(), key=lambda x: x[1], reverse=True):
            metadata += f"  - {agent}: {count} queries\n"

        # Token usage
        total_tokens = sum(s.tokens_used for s in summaries)
        if total_tokens > 0:
            metadata += f"- **Total Summary Tokens:** {total_tokens:,}\n"

        # Timestamps
        timestamps = [s.timestamp for s in summaries if s.timestamp != "unknown"]
        if timestamps:
            metadata += f"- **First Query:** {timestamps[0]}\n"
            metadata += f"- **Last Query:** {timestamps[-1]}\n"

        return metadata

    def _generate_hypotheses_section(self, summaries_by_round: Dict[Optional[int], List[QuerySummary]]) -> Optional[str]:
        """
        Generate hypotheses section showing all generated hypotheses.

        Args:
            summaries_by_round: Summaries organized by round

        Returns:
            str: Hypotheses section content, or None if no hypotheses found
        """
        # Check if any summaries have hypothesis data
        has_hypotheses = any(
            any(s.hypothesis_output for s in summaries)
            for summaries in summaries_by_round.values()
        )

        if not has_hypotheses:
            return None

        section = "## Generated Hypotheses\n\n"
        section += "This section shows the hypotheses generated during the research workflow. "
        section += "Hypotheses help guide the research direction and are typically generated at the start of each round "
        section += "or at key points during investigation.\n\n"
        section += "---\n"

        # Sort rounds
        round_keys = sorted(summaries_by_round.keys(), key=lambda x: (x is not None, x if x is not None else -1))

        for round_num in round_keys:
            round_summaries = summaries_by_round[round_num]

            # Find summaries with hypotheses in this round
            summaries_with_hyp = [s for s in round_summaries if s.hypothesis_output]

            if not summaries_with_hyp:
                continue

            # Round header
            if round_num is not None:
                section += f"\n### Round {round_num} Hypotheses\n\n"
            else:
                section += f"\n### Generated Hypotheses\n\n"

            for summary in summaries_with_hyp:
                section += f"\n#### Query {summary.query_num} Hypotheses\n\n"
                section += f"**Timestamp:** {summary.timestamp}\n\n"
                section += f"{summary.hypothesis_output}\n\n"
                section += "---\n"

        return section

    def _generate_reviews_section(self, summaries_by_round: Dict[Optional[int], List[QuerySummary]]) -> Optional[str]:
        """
        Generate reviews section showing all reviewer feedback.

        Args:
            summaries_by_round: Summaries organized by round

        Returns:
            str: Reviews section content, or None if no reviews found
        """
        # Check if any summaries have review data
        has_reviews = any(
            any(s.review_output for s in summaries)
            for summaries in summaries_by_round.values()
        )

        if not has_reviews:
            return None

        section = "## Reviewer Feedback\n\n"
        section += "This section contains feedback from the reviewer agent that critiques and guides the research process. "
        section += "Reviews typically occur at regular intervals to assess progress and suggest improvements.\n\n"
        section += "---\n"

        # Sort rounds
        round_keys = sorted(summaries_by_round.keys(), key=lambda x: (x is not None, x if x is not None else -1))

        for round_num in round_keys:
            round_summaries = summaries_by_round[round_num]

            # Find summaries with reviews in this round
            summaries_with_reviews = [s for s in round_summaries if s.review_output]

            if not summaries_with_reviews:
                continue

            # Round header
            if round_num is not None:
                section += f"\n### Round {round_num} Reviews\n\n"
            else:
                section += f"\n### Reviews\n\n"

            for summary in summaries_with_reviews:
                section += f"\n#### Review for Query {summary.query_num}\n\n"
                section += f"**Timestamp:** {summary.timestamp}\n\n"

                # Show a brief excerpt of the review input if available
                if summary.review_input:
                    # Get first 200 chars of review input
                    input_excerpt = summary.review_input[:200]
                    if len(summary.review_input) > 200:
                        input_excerpt += "..."
                    section += f"**Review Prompt Excerpt:** {input_excerpt}\n\n"

                section += f"**Reviewer Feedback:**\n\n{summary.review_output}\n\n"
                section += "---\n"

        return section

    def save_report(self, report_content: str, output_path: str) -> str:
        """
        Save the report to a file.

        Args:
            report_content: Generated report content
            output_path: Path to save the report

        Returns:
            str: Path where report was saved
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return str(output_path)
