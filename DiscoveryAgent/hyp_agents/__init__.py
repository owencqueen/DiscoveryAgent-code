"""
Hypothesis Agents Module

Main Agents:
- DirectorAgent: Orchestrates research workflow and generates queries for tool agents
- HypothesisGenAgent: Specialized agent for generating scientific hypotheses

Usage:
    from DiscoveryAgent.hyp_agents import DirectorAgent, HypothesisGenAgent
"""

from .director_agent import DirectorAgent
from .hypothesis_gen_agent import HypothesisGenAgent

__all__ = [
    'DirectorAgent',
    'HypothesisGenAgent',
]
