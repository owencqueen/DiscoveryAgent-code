"""Checks for per-agent model settings and the deep-research tool."""

from unittest.mock import Mock, patch

import pytest

from DiscoveryAgent.config import LLMConfig
from DiscoveryAgent.tool_agents.deep_research import DeepResearchAgent, DeepResearchConfig
from DiscoveryAgent.tool_agents.specialized.opentargets_agent import OpenTargetsAgent, OpenTargetsAgentConfig
from DiscoveryAgent.utils.arg_parsers import create_run_parser


def test_independent_paper_models_and_local_workspace_spec():
    args = create_run_parser().parse_args([
        "--director_model", "claude-opus-4-6",
        "--execution_model", "claude-sonnet-4-6",
        "--codebase_structure_prompt_path", "configs/workspace_base_early.txt",
    ])
    assert args.director_model == "claude-opus-4-6"
    assert args.execution_model == "claude-sonnet-4-6"
    assert args.codebase_structure_prompt_path == "configs/workspace_base_early.txt"


def test_deep_research_uses_fixed_paper_model(tmp_path):
    logger = Mock()
    config = DeepResearchConfig(llm_config=LLMConfig(model_name="sonar-deep-research", api_key="test"))
    agent = DeepResearchAgent(config, logger=logger)
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "Finding with source"}}]}
    with patch("DiscoveryAgent.tool_agents.deep_research.requests.post", return_value=response) as post:
        report = agent.execute("Research question", searches_dir=str(tmp_path))
    assert report.startswith("# Research Report")
    assert post.call_args.kwargs["json"]["model"] == "sonar-deep-research"
    assert len(list(tmp_path.glob("deep_research_*.md"))) == 1


def test_opentargets_requires_explicit_domain_resources(tmp_path):
    config = OpenTargetsAgentConfig(open_targets_path=str(tmp_path), mcp_package_path=None)
    with pytest.raises(ValueError, match="opentargets_mcp_package_path"):
        OpenTargetsAgent(config, logger=Mock())
