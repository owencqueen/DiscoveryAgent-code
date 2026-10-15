"""The launcher must accept every config in configs/paper/."""

import importlib
from pathlib import Path

import pytest
import yaml

from DiscoveryAgent.utils.arg_parsers import create_run_parser
from DiscoveryAgent.utils.config_loader import build_arg_mapping_from_parser

REPO = Path(__file__).resolve().parents[1]
PAPER_CONFIGS = sorted((REPO / "configs" / "paper").glob("*.yaml"))


def test_launcher_imports():
    launcher = importlib.import_module("scripts.run")
    assert launcher.HYPOTHESIS_GENERATION_SYSTEM_PROMPT.lstrip().startswith(
        "You are a scientific expert"
    )


def test_all_six_campaigns_ship_a_config():
    assert {p.stem for p in PAPER_CONFIGS} == {
        "starembed_auto", "dms_auto", "dms_steer", "dwarf_select", "acs_select", "gw_cd4_select",
    }


@pytest.mark.parametrize("path", PAPER_CONFIGS, ids=lambda p: p.stem)
def test_paper_config_keys_are_launcher_flags(path):
    config = yaml.safe_load(path.read_text())
    known = build_arg_mapping_from_parser(create_run_parser())
    assert sorted(set(config) - set(known)) == []
    assert (REPO / config["codebase_structure_prompt_path"]).is_file()


@pytest.mark.parametrize("path", PAPER_CONFIGS, ids=lambda p: p.stem)
def test_paper_configs_keep_paper_workflow(path):
    config = yaml.safe_load(path.read_text())
    for flag in ("git_tracking", "review_design", "review_search", "sub_agent_architecture",
                 "allow_director_to_stop", "enable_compaction", "generate_background_report"):
        assert config[flag] is True, flag
    assert config["single_search_max_tokens"] == 1200
