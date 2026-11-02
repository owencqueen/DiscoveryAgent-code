"""Inspect paper campaign inputs and launch commands without starting a run."""

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_FLAGS = {
    "dwarf_select": ["--background_path"],
    "gw_cd4_select": ["--opentargets_data_path", "--opentargets_mcp_package_path"],
}


def campaign_plan(campaign, working_directory):
    config_path = ROOT / "configs" / "paper" / f"{campaign}.yaml"
    if not config_path.is_file() or campaign not in {
        p.stem for p in (ROOT / "configs" / "paper").glob("*.yaml")
    }:
        raise ValueError(f"Unknown campaign: {campaign}")
    inputs = ROOT / "inputs" / campaign
    task = inputs / "TASK_DESCRIPTION.md"
    config = yaml.safe_load(config_path.read_text())
    command = [
        sys.executable, "-m", "scripts.run", "--config-yaml", str(config_path),
        "--working_directory", str(Path(working_directory).resolve()),
        "--readme-path", str(inputs / "README.txt"),
    ]
    if task.is_file():
        command.extend(["--task_description_file", str(task)])
    return {
        "campaign": campaign,
        "run_from": str(ROOT),
        "command": command,
        "additional_resource_flags": RESOURCE_FLAGS.get(campaign, []),
        "configuration": config,
        "dataset_description_template": (inputs / "README.txt").read_text(),
        "task_description": task.read_text() if task.is_file() else None,
        "workspace_specification": (ROOT / config["codebase_structure_prompt_path"]).read_text(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=sorted(
        p.stem for p in (ROOT / "configs" / "paper").glob("*.yaml")))
    parser.add_argument("--working-directory", required=True)
    args = parser.parse_args()
    print(json.dumps(campaign_plan(args.campaign, args.working_directory), indent=2))


if __name__ == "__main__":
    main()
