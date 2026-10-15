"""
Checkpoint Discovery Module

This module discovers and lists all checkpoints in a DiscoveryAgent workflow run.
Supports both round-aware (checkpoints/round_X/query_YYY/) and legacy
(checkpoints/query_YYY/) checkpoint structures.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ..utils.query_artifacts import artifact_filename


@dataclass
class CheckpointInfo:
    """Information about a single checkpoint."""
    query_num: int
    round_num: Optional[int]  # None for legacy checkpoints without rounds
    checkpoint_path: Path
    checkpoint_file: Path
    timestamp: str
    has_markdown: bool
    markdown_path: Optional[Path] = None

    def __lt__(self, other):
        """Enable sorting by round_num (None comes first) then query_num."""
        # Sort by round (None = -1 for sorting), then by query
        self_round = -1 if self.round_num is None else self.round_num
        other_round = -1 if other.round_num is None else other.round_num
        return (self_round, self.query_num) < (other_round, other.query_num)


class CheckpointDiscovery:
    """
    Discovers checkpoints in a DiscoveryAgent workflow run directory.
    """

    def __init__(self, base_path: str):
        """
        Initialize checkpoint discovery.

        Args:
            base_path: Path to logs directory or checkpoints directory
        """
        self.base_path = Path(base_path).resolve()

        # Determine if this is a logs directory or checkpoints directory
        if self.base_path.name == "checkpoints":
            self.checkpoint_base = self.base_path
        elif (self.base_path / "checkpoints").exists():
            self.checkpoint_base = self.base_path / "checkpoints"
        else:
            # Assume the provided path is the checkpoint directory itself
            self.checkpoint_base = self.base_path

    def discover_checkpoints(self, round_filter: Optional[int] = None) -> List[CheckpointInfo]:
        """
        Discover all checkpoints in the directory structure.

        Args:
            round_filter: If set, only discover checkpoints from this specific round

        Returns:
            List[CheckpointInfo]: List of discovered checkpoints, sorted by round/query
        """
        checkpoints = []

        if not self.checkpoint_base.exists():
            return checkpoints

        # Check for round-aware structure
        round_dirs = [d for d in self.checkpoint_base.iterdir()
                      if d.is_dir() and d.name.startswith('round_')]

        if round_dirs:
            # Round-aware structure
            checkpoints.extend(self._discover_round_aware(round_filter=round_filter))
        else:
            # Legacy structure or mixed
            checkpoints.extend(self._discover_legacy())

        # Sort by round (if present) then query number
        checkpoints.sort()

        return checkpoints

    def get_available_rounds(self) -> List[int]:
        """
        Get list of available round numbers in the checkpoint directory.

        Returns:
            List[int]: Sorted list of round numbers found
        """
        rounds = []
        if not self.checkpoint_base.exists():
            return rounds

        for d in self.checkpoint_base.iterdir():
            if d.is_dir() and d.name.startswith('round_'):
                try:
                    rounds.append(int(d.name.split('_')[1]))
                except (ValueError, IndexError):
                    continue

        return sorted(rounds)

    def _discover_round_aware(self, round_filter: Optional[int] = None) -> List[CheckpointInfo]:
        """Discover checkpoints in round-aware structure."""
        checkpoints = []

        for round_dir in self.checkpoint_base.iterdir():
            if not round_dir.is_dir() or not round_dir.name.startswith('round_'):
                continue

            try:
                round_num = int(round_dir.name.split('_')[1])
            except (ValueError, IndexError):
                continue

            # Skip rounds that don't match the filter
            if round_filter is not None and round_num != round_filter:
                continue

            # Find query directories in this round
            for query_dir in round_dir.iterdir():
                if not query_dir.is_dir() or not query_dir.name.startswith('query_'):
                    continue

                checkpoint_info = self._parse_checkpoint_dir(query_dir, round_num)
                if checkpoint_info:
                    checkpoints.append(checkpoint_info)

        return checkpoints

    def _discover_legacy(self) -> List[CheckpointInfo]:
        """Discover checkpoints in legacy structure (no rounds)."""
        checkpoints = []

        for query_dir in self.checkpoint_base.iterdir():
            if not query_dir.is_dir() or not query_dir.name.startswith('query_'):
                continue

            checkpoint_info = self._parse_checkpoint_dir(query_dir, round_num=None)
            if checkpoint_info:
                checkpoints.append(checkpoint_info)

        return checkpoints

    def _parse_checkpoint_dir(self, query_dir: Path, round_num: Optional[int]) -> Optional[CheckpointInfo]:
        """
        Parse a single checkpoint directory.

        Args:
            query_dir: Path to query directory
            round_num: Round number or None for legacy

        Returns:
            CheckpointInfo if valid checkpoint found, None otherwise
        """
        try:
            query_num = int(query_dir.name.split('_')[1])
        except (ValueError, IndexError):
            return None

        checkpoint_file = query_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return None

        # Read timestamp from checkpoint
        timestamp = "unknown"
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            timestamp = data.get('timestamp', 'unknown')
        except Exception:
            pass

        # Check for markdown file
        markdown_path = query_dir / artifact_filename(query_num, "report")
        if not markdown_path.exists():
            markdown_path = query_dir / f"query_{query_num:03d}.md"
        has_markdown = markdown_path.exists()

        return CheckpointInfo(
            query_num=query_num,
            round_num=round_num,
            checkpoint_path=query_dir,
            checkpoint_file=checkpoint_file,
            timestamp=timestamp,
            has_markdown=has_markdown,
            markdown_path=markdown_path if has_markdown else None
        )

    def get_statistics(self, checkpoints: List[CheckpointInfo]) -> dict:
        """
        Get statistics about discovered checkpoints.

        Args:
            checkpoints: List of discovered checkpoints

        Returns:
            dict: Statistics about the checkpoints
        """
        if not checkpoints:
            return {
                'total_checkpoints': 0,
                'total_queries': 0,
                'total_rounds': 0,
                'has_rounds': False,
                'queries_per_round': {}
            }

        has_rounds = any(cp.round_num is not None for cp in checkpoints)

        if has_rounds:
            rounds = set(cp.round_num for cp in checkpoints if cp.round_num is not None)
            queries_per_round = {}
            for round_num in sorted(rounds):
                queries_per_round[round_num] = len([cp for cp in checkpoints if cp.round_num == round_num])
        else:
            rounds = set()
            queries_per_round = {}

        return {
            'total_checkpoints': len(checkpoints),
            'total_queries': len(checkpoints),
            'total_rounds': len(rounds),
            'has_rounds': has_rounds,
            'queries_per_round': queries_per_round
        }
