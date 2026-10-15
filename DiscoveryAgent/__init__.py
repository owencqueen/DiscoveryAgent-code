"""
DiscoveryAgent (AI Researcher) Package

A modular AI research framework for automated hypothesis generation,
data analysis, and report generation.
"""

__version__ = "0.1.0"

from .utils.logger import Logger
from .config import (
    LLMConfig,
    HypothesisGenerationConfig,
    get_default_hypothesis_config,
)

__all__ = [
    "Logger",
    "LLMConfig",
    "HypothesisGenerationConfig",
    "get_default_hypothesis_config",
]