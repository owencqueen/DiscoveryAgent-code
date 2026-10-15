"""
Prompts Package

This package contains the prompt templates and system prompts used by the DiscoveryAgent framework.
"""

from .hyp_generation import (
    EVIDENCE_DRIVEN_HYPOTHESIS_GENERATION_SYSTEM_PROMPT,
    DATASET_DESCRIPTION_PROMPT,
    REPORT_PROMPT,
    QUERY_EXECUTION_SYSTEM_PROMPT
)

__all__ = [
    "EVIDENCE_DRIVEN_HYPOTHESIS_GENERATION_SYSTEM_PROMPT",
    "DATASET_DESCRIPTION_PROMPT", 
    "REPORT_PROMPT",
    "QUERY_EXECUTION_SYSTEM_PROMPT"
] 