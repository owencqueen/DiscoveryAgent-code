"""
Data Extractor Module

This module extracts query and result data from checkpoints, supporting both
checkpoint.json and markdown file formats.
"""

import json
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from .checkpoint_discovery import CheckpointInfo


@dataclass
class QueryResultPair:
    """A single query-result pair from a checkpoint."""
    query_num: int
    round_num: Optional[int]
    timestamp: str
    query_raw: str  # Full query including any wrapper text
    query_parsed: str  # Extracted/parsed query (the actual agent input)
    agent_name: str  # Name/tag of the agent used
    result: str  # Agent output/report
    hypothesis_input: Optional[str] = None  # Input from hypothesis agent (if available)
    hypothesis_output: Optional[str] = None  # Generated hypothesis for this query (if available)
    review_input: Optional[str] = None  # Input to reviewer (if available)
    review_output: Optional[str] = None  # Reviewer feedback (if available)
    selected_hypothesis: Optional[str] = None  # Selected hypothesis driving this round (from previous round)

    def get_round_label(self) -> str:
        """Get a display label for the round."""
        if self.round_num is not None:
            return f"Round {self.round_num}"
        return "Single Round"


class DataExtractor:
    """
    Extracts query and result data from checkpoint files.
    """

    def __init__(self, extract_hypotheses: bool = False, extract_reviews: bool = False):
        """
        Initialize data extractor.

        Args:
            extract_hypotheses: Whether to extract hypothesis files
            extract_reviews: Whether to extract review files
        """
        self.extract_hypotheses = extract_hypotheses
        self.extract_reviews = extract_reviews

    def extract_from_checkpoint(self, checkpoint_info: CheckpointInfo) -> Optional[QueryResultPair]:
        """
        Extract data from a checkpoint.

        Args:
            checkpoint_info: Checkpoint information

        Returns:
            QueryResultPair if extraction successful, None otherwise
        """
        # Try markdown first (easier to parse, more readable)
        if checkpoint_info.has_markdown and checkpoint_info.markdown_path:
            result = self._extract_from_markdown(checkpoint_info)
            if result:
                return result

        # Fall back to checkpoint.json
        return self._extract_from_json(checkpoint_info)

    def _extract_from_markdown(self, checkpoint_info: CheckpointInfo) -> Optional[QueryResultPair]:
        """Extract data from markdown file."""
        try:
            with open(checkpoint_info.markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract timestamp
            timestamp_match = re.search(r'\*\*Timestamp:\*\*\s*(.+)', content)
            timestamp = timestamp_match.group(1).strip() if timestamp_match else checkpoint_info.timestamp

            # Extract hypothesis input (if present)
            hypothesis_input = None
            hyp_match = re.search(r'# Hypothesis Input\n\n(.+?)\n\n#', content, re.DOTALL)
            if hyp_match:
                hypothesis_input = hyp_match.group(1).strip()

            # Extract raw query
            query_raw = ""
            raw_query_match = re.search(r'\*\*Raw Query:\*\*\n```\n(.+?)\n```', content, re.DOTALL)
            if raw_query_match:
                query_raw = raw_query_match.group(1).strip()

            # Extract parsed query
            query_parsed = ""
            parsed_query_match = re.search(r'\*\*Parsed Query:\*\*\n```\n(.+?)\n```', content, re.DOTALL)
            if parsed_query_match:
                query_parsed = parsed_query_match.group(1).strip()
            elif query_raw:
                # If no parsed query, try to extract from raw query
                query_parsed = self._extract_query_from_xml(query_raw)
                if not query_parsed:
                    query_parsed = query_raw

            # Extract agent name from section header
            agent_name = "Unknown Agent"
            agent_match = re.search(r'# (.+?) Report\n\n', content)
            if agent_match:
                agent_name = agent_match.group(1).strip()

            # Extract result/report
            result = ""
            # Look for the report section - everything after the agent report header
            report_match = re.search(r'# .+? Report\n\n(.+)', content, re.DOTALL)
            if report_match:
                result = report_match.group(1).strip()

            return QueryResultPair(
                query_num=checkpoint_info.query_num,
                round_num=checkpoint_info.round_num,
                timestamp=timestamp,
                query_raw=query_raw,
                query_parsed=query_parsed,
                agent_name=agent_name,
                result=result,
                hypothesis_input=hypothesis_input
            )

        except Exception as e:
            print(f"Warning: Failed to extract from markdown {checkpoint_info.markdown_path}: {str(e)}")
            return None

    def _extract_from_json(self, checkpoint_info: CheckpointInfo) -> Optional[QueryResultPair]:
        """Extract data from checkpoint.json."""
        try:
            with open(checkpoint_info.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            workflow_state = data.get('workflow_state', {})
            results = workflow_state.get('results', [])

            # Find the result for this query number
            query_num = checkpoint_info.query_num
            current_result = None
            for result in results:
                if result.get('query_num') == query_num:
                    current_result = result
                    break

            if not current_result:
                print(f"Warning: No result found for query {query_num} in {checkpoint_info.checkpoint_file}")
                return None

            # Extract fields
            timestamp = current_result.get('timestamp', checkpoint_info.timestamp)
            query_raw = current_result.get('query_raw', '')
            query_parsed = current_result.get('query_parsed', '')
            result_text = current_result.get('result', '')

            # If no parsed query, try to extract from raw
            if not query_parsed and query_raw:
                query_parsed = self._extract_query_from_xml(query_raw)
                if not query_parsed:
                    query_parsed = query_raw

            # Extract agent name from query
            agent_name = self._extract_agent_name(query_raw)

            # Try to extract hypothesis input from agents state (if available)
            hypothesis_input = None
            agents = data.get('agents', {})
            if 'hypothesis_agent' in agents:
                hyp_state = agents['hypothesis_agent'].get('state', {})
                messages = hyp_state.get('messages', [])
                # Find last user message before this query
                if messages:
                    user_messages = [msg for msg in messages if msg.get('role') == 'user']
                    if user_messages:
                        hypothesis_input = user_messages[-1].get('content', '')

            return QueryResultPair(
                query_num=query_num,
                round_num=checkpoint_info.round_num,
                timestamp=timestamp,
                query_raw=query_raw,
                query_parsed=query_parsed,
                agent_name=agent_name,
                result=result_text,
                hypothesis_input=hypothesis_input
            )

        except Exception as e:
            print(f"Warning: Failed to extract from JSON {checkpoint_info.checkpoint_file}: {str(e)}")
            return None

    def _extract_query_from_xml(self, query_raw: str) -> str:
        """Extract the query content from XML tags."""
        # Match <tag>...</tag> and extract the content
        match = re.search(r'<(\w+)>(.*?)</\1>', query_raw, re.DOTALL)
        if match:
            return match.group(2).strip()
        return ""

    def _extract_agent_name(self, query_raw: str) -> str:
        """Extract agent name from XML tags in query."""
        match = re.search(r'<(\w+)>', query_raw)
        if match:
            tag = match.group(1)
            # Convert tag to readable name
            return tag.replace('_', ' ').title()
        return "Unknown Agent"

    def extract_all(
        self,
        checkpoints: List[CheckpointInfo],
        per_round_hypotheses: Optional[dict] = None
    ) -> List[QueryResultPair]:
        """
        Extract data from all checkpoints.

        Args:
            checkpoints: List of checkpoints to extract from
            per_round_hypotheses: Optional dict mapping round_num -> selected hypothesis text
                                  that drove research in that round

        Returns:
            List[QueryResultPair]: List of extracted query-result pairs
        """
        pairs = []
        for checkpoint in checkpoints:
            pair = self.extract_from_checkpoint(checkpoint)
            if pair:
                # Attach selected hypothesis context for this round
                if per_round_hypotheses and pair.round_num is not None:
                    pair.selected_hypothesis = per_round_hypotheses.get(pair.round_num)

                # Optionally extract hypothesis data
                if self.extract_hypotheses:
                    pair = self._add_hypothesis_data(pair, checkpoint)

                # Optionally extract review data
                if self.extract_reviews:
                    pair = self._add_review_data(pair, checkpoint)

                pairs.append(pair)
            else:
                print(f"Warning: Could not extract data from checkpoint {checkpoint.checkpoint_path}")

        return pairs

    def _add_hypothesis_data(self, pair: QueryResultPair, checkpoint_info: CheckpointInfo) -> QueryResultPair:
        """
        Extract and add hypothesis data to a QueryResultPair.

        Args:
            pair: Existing QueryResultPair
            checkpoint_info: Checkpoint information

        Returns:
            QueryResultPair with hypothesis data added
        """
        try:
            checkpoint_dir = Path(checkpoint_info.checkpoint_path).parent
            hypothesis_file = checkpoint_dir / f"hypothesis_{pair.query_num:03d}.md"

            # Also check for initial hypothesis at query_000
            if pair.query_num == 0:
                hypothesis_file = checkpoint_dir / "hypothesis_000.md"

            if hypothesis_file.exists():
                with open(hypothesis_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract the generated hypotheses section
                hyp_match = re.search(r'## Generated Hypotheses\n\n(.+)', content, re.DOTALL)
                if hyp_match:
                    pair.hypothesis_output = hyp_match.group(1).strip()

        except Exception as e:
            # Silently fail - hypotheses are optional
            pass

        return pair

    def _add_review_data(self, pair: QueryResultPair, checkpoint_info: CheckpointInfo) -> QueryResultPair:
        """
        Extract and add review data to a QueryResultPair.

        Args:
            pair: Existing QueryResultPair
            checkpoint_info: Checkpoint information

        Returns:
            QueryResultPair with review data added
        """
        try:
            checkpoint_dir = Path(checkpoint_info.checkpoint_path).parent
            review_file = checkpoint_dir / f"review_{pair.query_num:03d}.md"

            if review_file.exists():
                with open(review_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract reviewer input
                input_match = re.search(r'## Reviewer Input \(Prompt\)\n\n(.+?)\n\n---', content, re.DOTALL)
                if input_match:
                    pair.review_input = input_match.group(1).strip()

                # Extract reviewer output
                output_match = re.search(r'## Reviewer Output \(Feedback\)\n\n(.+)', content, re.DOTALL)
                if output_match:
                    pair.review_output = output_match.group(1).strip()

        except Exception as e:
            # Silently fail - reviews are optional
            pass

        return pair
