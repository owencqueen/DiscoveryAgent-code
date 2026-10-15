"""
Audit Summarizer Module

This module uses Claude Haiku to generate summaries of query-result pairs,
making the research process more understandable.

Supports both synchronous and asynchronous (parallel) processing for speed.
"""

from typing import List, Optional
from dataclasses import dataclass
import anthropic
import asyncio
import os

from .data_extractor import QueryResultPair


SUMMARIZATION_SYSTEM_PROMPT = """You are an expert research analyst tasked with summarizing scientific research processes.
Your job is to analyze queries and results from an AI research workflow and create clear, concise summaries
that help humans understand the research trajectory.

Focus on:
1. What research question or hypothesis was being explored
2. What method or analysis approach was used
3. What were the key findings and results
4. How this advances the overall research objectives

Be concise but comprehensive. Use clear scientific language."""


SUMMARIZATION_USER_PROMPT = """Please provide a clear, concise summary of the following research result from an AI workflow.

Focus on what was discovered, what methods were used, and why it matters. Keep it readable and informative (2-4 paragraphs).
{hypothesis_section}
## Query Context
{query}

## Research Result
{result}

Your summary:"""


@dataclass
class QuerySummary:
    """Summary of a single query-result pair."""
    query_num: int
    round_num: Optional[int]
    timestamp: str
    agent_name: str
    summary: str  # Full summary text from LLM
    query_raw: str = ""  # Full raw query including thought/interpretation
    parsed_query: str = ""  # Parsed query (agent input only)
    tokens_used: int = 0
    hypothesis_output: Optional[str] = None  # Generated hypothesis (if available)
    review_input: Optional[str] = None  # Reviewer input (if available)
    review_output: Optional[str] = None  # Reviewer output (if available)


class AuditSummarizer:
    """
    Generates summaries of query-result pairs using Claude Haiku.

    Supports both synchronous and asynchronous processing.
    Async processing allows parallel summarization for significant speedup.
    """

    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        max_concurrent: int = 10
    ):
        """
        Initialize the audit summarizer.

        Args:
            model_name: Name of the Claude model to use
            api_key: Anthropic API key (or None to use ANTHROPIC_API_KEY env var)
            max_concurrent: Maximum number of concurrent API requests (for async mode)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_concurrent = max_concurrent

        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable")

        # Synchronous client
        self.client = anthropic.Anthropic(api_key=self.api_key)

        # Async client (created lazily when needed)
        self._async_client = None
        self._token_lock = None  # Created lazily in async context

        self.total_tokens = 0

    def _format_hypothesis_section(self, pair: QueryResultPair) -> str:
        """Format the hypothesis section for the summarization prompt."""
        if pair.selected_hypothesis:
            return f"\n## Selected Hypothesis Driving This Round\n{pair.selected_hypothesis[:3000]}\n"
        return ""

    def summarize_pair(self, pair: QueryResultPair) -> QuerySummary:
        """
        Summarize a single query-result pair.

        Args:
            pair: QueryResultPair to summarize

        Returns:
            QuerySummary: Structured summary
        """
        # Prepare the prompt
        user_prompt = SUMMARIZATION_USER_PROMPT.format(
            query=pair.query_parsed or pair.query_raw,
            result=pair.result[:10000],  # Limit result length to avoid token limits
            hypothesis_section=self._format_hypothesis_section(pair)
        )

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.3,  # Lower temperature for more consistent summaries
                system=SUMMARIZATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract summary text
            summary_text = response.content[0].text

            # Track token usage
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            self.total_tokens += tokens_used

            return QuerySummary(
                query_num=pair.query_num,
                round_num=pair.round_num,
                timestamp=pair.timestamp,
                agent_name=pair.agent_name,
                summary=summary_text,
                query_raw=pair.query_raw,
                parsed_query=pair.query_parsed or pair.query_raw,
                tokens_used=tokens_used,
                hypothesis_output=pair.hypothesis_output,
                review_input=pair.review_input,
                review_output=pair.review_output
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n❌ Error summarizing query {pair.query_num} (sync):")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            if os.environ.get("DEBUG"):
                print(f"   Full traceback:\n{error_details}")
            # Return a basic summary with error info
            return QuerySummary(
                query_num=pair.query_num,
                round_num=pair.round_num,
                timestamp=pair.timestamp,
                agent_name=pair.agent_name,
                summary=f"Error summarizing: {type(e).__name__}: {str(e)}\n\nQuery length: {len(pair.query_parsed)} chars\nResult length: {len(pair.result)} chars",
                query_raw=pair.query_raw,
                parsed_query=pair.query_parsed or pair.query_raw,
                tokens_used=0
            )

    def _parse_summary_sections(self, summary_text: str) -> dict:
        """
        Parse structured sections from the LLM summary.

        Handles both markdown header format (## Section) and bold format (**Section:**).

        Args:
            summary_text: Raw summary text from LLM

        Returns:
            dict: Parsed sections
        """
        import re

        sections = {}

        # Try multiple patterns for each section
        # Pattern 1: Markdown headers with optional numbers (## Research Question or ## 1. Research Question)
        # Pattern 2: Bold format (**Research Question:**)
        patterns = {
            'research_question': [
                r'##\s*\d*\.?\s*Research Question\s*\n+(.+?)(?=\n##|\Z)',
                r'\*\*Research Question\*\*:?\s*(.+?)(?=\n\*\*|\n\n|\Z)'
            ],
            'method': [
                r'##\s*\d*\.?\s*Method\s*\n+(.+?)(?=\n##|\Z)',
                r'\*\*Method\*\*:?\s*(.+?)(?=\n\*\*|\n\n|\Z)'
            ],
            'key_findings': [
                r'##\s*\d*\.?\s*Key Findings\s*\n+(.+?)(?=\n##|\Z)',
                r'\*\*Key Findings\*\*:?\s*(.+?)(?=\n\*\*|\n\n|\Z)'
            ],
            'research_advancement': [
                r'##\s*\d*\.?\s*Research Advancement\s*\n+(.+?)(?=\n##|\Z)',
                r'\*\*Research Advancement\*\*:?\s*(.+?)(?=\n\*\*|\n\n|\Z)'
            ]
        }

        for key, pattern_list in patterns.items():
            found = False
            for pattern in pattern_list:
                match = re.search(pattern, summary_text, re.DOTALL | re.IGNORECASE)
                if match:
                    sections[key] = match.group(1).strip()
                    found = True
                    break

            if not found:
                # If section not found with any pattern, try fallback
                sections[key] = self._extract_section_fallback(summary_text, key)

        return sections

    def _extract_section_fallback(self, text: str, section_name: str) -> str:
        """
        Fallback method to extract section if structured parsing fails.

        Args:
            text: Full summary text
            section_name: Name of section to extract

        Returns:
            str: Extracted section or default text
        """
        # If structured extraction failed, return the full text or a portion
        # This is a fallback in case the LLM doesn't follow the exact format
        if section_name == 'research_question':
            # Try to find first sentence or paragraph
            lines = text.split('\n')
            for line in lines:
                if line.strip() and not line.strip().startswith('**'):
                    return line.strip()

        return "See full summary below"

    def summarize_all(self, pairs: List[QueryResultPair], verbose: bool = True) -> List[QuerySummary]:
        """
        Summarize all query-result pairs (synchronous version).

        Args:
            pairs: List of QueryResultPair objects to summarize
            verbose: Whether to print progress

        Returns:
            List[QuerySummary]: List of summaries
        """
        summaries = []
        total = len(pairs)

        for i, pair in enumerate(pairs, 1):
            if verbose:
                round_label = f"Round {pair.round_num}, " if pair.round_num is not None else ""
                print(f"Summarizing {round_label}Query {pair.query_num} ({i}/{total})...")

            summary = self.summarize_pair(pair)
            summaries.append(summary)

        if verbose:
            print(f"\n✓ Summarization complete. Total tokens used: {self.total_tokens}")

        return summaries

    async def _get_async_client(self):
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client

    async def _get_token_lock(self):
        """Get or create token lock (must be created in async context)."""
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        return self._token_lock

    async def summarize_pair_async(self, pair: QueryResultPair) -> QuerySummary:
        """
        Asynchronously summarize a single query-result pair.

        Args:
            pair: QueryResultPair to summarize

        Returns:
            QuerySummary: Structured summary
        """
        # Prepare the prompt
        user_prompt = SUMMARIZATION_USER_PROMPT.format(
            query=pair.query_parsed or pair.query_raw,
            result=pair.result[:10000],  # Limit result length to avoid token limits
            hypothesis_section=self._format_hypothesis_section(pair)
        )

        try:
            # Get async client
            client = await self._get_async_client()

            # Call Claude API asynchronously
            response = await client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.3,  # Lower temperature for more consistent summaries
                system=SUMMARIZATION_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )

            # Extract summary text
            summary_text = response.content[0].text

            # Track token usage (thread-safe)
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            lock = await self._get_token_lock()
            async with lock:
                self.total_tokens += tokens_used

            return QuerySummary(
                query_num=pair.query_num,
                round_num=pair.round_num,
                timestamp=pair.timestamp,
                agent_name=pair.agent_name,
                summary=summary_text,
                query_raw=pair.query_raw,
                parsed_query=pair.query_parsed or pair.query_raw,
                tokens_used=tokens_used,
                hypothesis_output=pair.hypothesis_output,
                review_input=pair.review_input,
                review_output=pair.review_output
            )

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"\n❌ Error summarizing query {pair.query_num} (async):")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            if os.environ.get("DEBUG"):
                print(f"   Full traceback:\n{error_details}")
            # Return a basic summary with error info
            return QuerySummary(
                query_num=pair.query_num,
                round_num=pair.round_num,
                timestamp=pair.timestamp,
                agent_name=pair.agent_name,
                summary=f"Error summarizing: {type(e).__name__}: {str(e)}\n\nQuery length: {len(pair.query_parsed)} chars\nResult length: {len(pair.result)} chars",
                query_raw=pair.query_raw,
                parsed_query=pair.query_parsed or pair.query_raw,
                tokens_used=0
            )

    async def summarize_all_async(
        self,
        pairs: List[QueryResultPair],
        verbose: bool = True
    ) -> List[QuerySummary]:
        """
        Asynchronously summarize all query-result pairs in parallel.

        This method processes multiple summaries concurrently for significant speedup.

        Args:
            pairs: List of QueryResultPair objects to summarize
            verbose: Whether to print progress

        Returns:
            List[QuerySummary]: List of summaries (in original order)
        """
        total = len(pairs)

        if verbose:
            print(f"Starting parallel summarization of {total} queries...")
            print(f"Max concurrent requests: {self.max_concurrent}")
            print()

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def summarize_with_semaphore(pair: QueryResultPair, index: int):
            """Summarize with semaphore control."""
            async with semaphore:
                if verbose:
                    round_label = f"Round {pair.round_num}, " if pair.round_num is not None else ""
                    print(f"  [{index+1}/{total}] Summarizing {round_label}Query {pair.query_num}...")
                return await self.summarize_pair_async(pair)

        # Create tasks for all pairs
        tasks = [summarize_with_semaphore(pair, i) for i, pair in enumerate(pairs)]

        # Run all tasks and gather results
        summaries = await asyncio.gather(*tasks)

        if verbose:
            print(f"\n✓ Parallel summarization complete!")
            print(f"  Total tokens used: {self.total_tokens:,}")

        return list(summaries)

    def summarize_all_parallel(
        self,
        pairs: List[QueryResultPair],
        verbose: bool = True
    ) -> List[QuerySummary]:
        """
        Convenience wrapper to run async summarization from sync code.

        This is the recommended method for parallel processing.

        Args:
            pairs: List of QueryResultPair objects to summarize
            verbose: Whether to print progress

        Returns:
            List[QuerySummary]: List of summaries
        """
        # Run the async function in an event loop
        return asyncio.run(self.summarize_all_async(pairs, verbose=verbose))

    def get_total_cost_estimate(self) -> float:
        """
        Estimate the total cost based on tokens used.

        Uses approximate pricing for Claude Haiku:
        - Input: $0.25 per 1M tokens
        - Output: $1.25 per 1M tokens

        Returns:
            float: Estimated cost in USD
        """
        # Rough estimate: assume 30% input, 70% output
        input_tokens = self.total_tokens * 0.3
        output_tokens = self.total_tokens * 0.7

        input_cost = (input_tokens / 1_000_000) * 0.25
        output_cost = (output_tokens / 1_000_000) * 1.25

        return input_cost + output_cost
