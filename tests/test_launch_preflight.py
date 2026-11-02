"""Launch validation must not start work or expose credentials."""

import json
from unittest.mock import patch

import pytest

from DiscoveryAgent.utils.arg_parsers import create_run_parser
from scripts import run
from scripts.plan import campaign_plan


@pytest.mark.parametrize("anthropic,perplexity,missing", [
    ("", "", "ANTHROPIC_API_KEY"),
    ("test-anthropic", "", "PERPLEXITY_API_KEY"),
    ("replace_with_your_key", "test-perplexity", "ANTHROPIC_API_KEY"),
])
def test_missing_credentials_fail_before_workspace_creation(
    tmp_path, monkeypatch, anthropic, perplexity, missing,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic)
    monkeypatch.setenv("PERPLEXITY_API_KEY", perplexity)
    monkeypatch.setattr(run, "DATA_DIR", str(tmp_path))
    args = create_run_parser().parse_args([])
    with patch.object(run, "init_sandbox_dir") as initialize:
        with pytest.raises(ValueError, match=missing) as error:
            run.main(args)
    initialize.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    assert "test-anthropic" not in str(error.value)
    assert "test-perplexity" not in str(error.value)


def test_perplexity_optional_when_deep_research_is_disabled(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.setattr(run, "DATA_DIR", "/unused")
    args = create_run_parser().parse_args(["--omit_deepresearch"])
    run.validate_run_environment(args)


@pytest.mark.parametrize("campaign,has_task", [
    ("starembed_auto", False), ("dms_auto", False), ("dms_steer", False),
    ("dwarf_select", True), ("acs_select", True), ("gw_cd4_select", True),
])
def test_campaign_plan_uses_only_supplied_inputs(tmp_path, monkeypatch, campaign, has_task):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret-never-in-plan")
    with patch("socket.socket", side_effect=AssertionError("Network forbidden")):
        plan = campaign_plan(campaign, str(tmp_path / "dataset"))
    assert bool(plan["task_description"]) == has_task
    assert plan["dataset_description_template"]
    assert plan["workspace_specification"]
    create_run_parser().parse_args(plan["command"][3:])
    assert "test-secret-never-in-plan" not in json.dumps(plan)
    assert list(tmp_path.iterdir()) == []
    if campaign in {"dms_auto", "starembed_auto"}:
        assert plan["configuration"]["human_selection"] is False
        assert plan["configuration"]["no_auto_hypothesis"] is False
