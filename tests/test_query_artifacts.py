import json
import os
from pathlib import Path

import pytest

os.environ["CLAUDE_CODE_CLI_PATH"] = "/usr/bin/true"

from DiscoveryAgent.director_workflow import DirectorWorkflow
from DiscoveryAgent.checkpoint_manager import CheckpointManager
from DiscoveryAgent.utils import progress_recorder
from DiscoveryAgent.utils.query_artifacts import (
    QueryArtifactCollisionError,
    artifact_filename,
    persist_query_input,
    persist_query_report,
    query_directory,
    query_directory_name,
)


class _Logger:
    def log_action(self, *_args, **_kwargs):
        pass


class _CheckpointManager:
    def __init__(self, round_dir: Path):
        self.round_dir = round_dir

    def get_checkpoint_dir(self, query_num):
        return query_directory(self.round_dir, query_num)


class _Agent:
    def __init__(self, callback):
        self.callback = callback

    def execute(self, query):
        return self.callback(query)

    def cleanup(self):
        pass


def _workflow(round_dir: Path, agent: _Agent) -> DirectorWorkflow:
    workflow = DirectorWorkflow.__new__(DirectorWorkflow)
    workflow.query_num = 1
    workflow.round_num = 5
    workflow.log_dir = str(round_dir.parent.parent)
    workflow.checkpoint_manager = _CheckpointManager(round_dir)
    workflow.agents = {"execution_agent": agent}
    workflow.logger = _Logger()
    return workflow


@pytest.mark.parametrize(
    ("query_num", "directory", "input_name", "report_name"),
    [
        (9, "query_09", "query_input_09.md", "query_report_09.md"),
        (10, "query_10", "query_input_10.md", "query_report_10.md"),
        (100, "query_100", "query_input_100.md", "query_report_100.md"),
    ],
)
def test_two_digit_minimum_width_names(query_num, directory, input_name, report_name):
    assert query_directory_name(query_num) == directory
    assert artifact_filename(query_num, "input") == input_name
    assert artifact_filename(query_num, "report") == report_name


def test_workflow_persists_exact_input_before_dispatch_and_pairs_successful_report(tmp_path):
    round_dir = tmp_path / "checkpoints" / "round_5"
    exact = "# Dataset\n\nUse `/input/a.csv`.\n\n- Verify the estimate."
    report = "# Result\n\nThe estimate is 0.42."

    def execute(query):
        assert query == exact
        assert (round_dir / "query_01" / "query_input_01.md").read_text() == exact
        assert not (round_dir / "query_01" / "query_report_01.md").exists()
        return report

    output, sent = _workflow(round_dir, _Agent(execute)).execute_agent(
        f"<execution_agent>{exact}</execution_agent>"
    )

    query_dir = round_dir / "query_01"
    assert sent == exact
    assert output == report
    assert (query_dir / "query_input_01.md").read_text() == exact
    assert (query_dir / "query_report_01.md").read_text() == report
    assert not (query_dir / "query_01.md").exists()


def test_failure_or_refusal_preserves_input_without_inventing_report(tmp_path):
    round_dir = tmp_path / "round_2"

    def refuse(_query):
        assert (round_dir / "query_01" / "query_input_01.md").is_file()
        raise RuntimeError("refused")

    with pytest.raises(RuntimeError, match="refused"):
        _workflow(round_dir, _Agent(refuse)).execute_agent(
            "<execution_agent>Do not lose this request.</execution_agent>"
        )
    assert (round_dir / "query_01" / "query_input_01.md").read_text() == "Do not lose this request."
    assert not (round_dir / "query_01" / "query_report_01.md").exists()


def test_empty_agent_response_does_not_create_a_report_artifact(tmp_path):
    round_dir = tmp_path / "round_2"
    assert persist_query_report(round_dir, 1, None) is None
    assert persist_query_report(round_dir, 1, "\n\t") is None
    assert not (round_dir / "query_01").exists()


def test_retry_is_idempotent_but_conflicting_input_is_explicit(tmp_path):
    round_dir = tmp_path / "round_3"
    first = persist_query_input(round_dir, 1, "same exact input")
    stat = first.stat()
    assert persist_query_input(round_dir, 1, "same exact input") == first
    assert first.stat().st_ino == stat.st_ino
    with pytest.raises(QueryArtifactCollisionError, match="different content"):
        persist_query_input(round_dir, 1, "different input")
    assert first.read_text() == "same exact input"


def test_existing_legacy_folder_is_reused_without_migration(tmp_path):
    round_dir = tmp_path / "round_4"
    legacy = round_dir / "query_001"
    legacy.mkdir(parents=True)
    legacy_report = legacy / "query_001.md"
    legacy_report.write_text("historical")

    input_path = persist_query_input(round_dir, 1, "resumed input")
    report_path = persist_query_report(round_dir, 1, "resumed report")

    assert input_path == legacy / "query_input_01.md"
    assert report_path == legacy / "query_report_01.md"
    assert legacy_report.read_text() == "historical"
    assert not (round_dir / "query_01").exists()


def test_checkpoint_metadata_joins_the_canonical_folder_without_duplicate_markdown(tmp_path):
    manager = CheckpointManager(str(tmp_path), object(), round_num=5)
    manager._create_checkpoint_data = lambda query_num: {"query_num": query_num}

    checkpoint = Path(manager.save_checkpoint(1))

    assert checkpoint == tmp_path / "checkpoints" / "round_5" / "query_01" / "checkpoint.json"
    assert checkpoint.is_file()
    assert not (checkpoint.parent / "query_001.md").exists()
    assert (checkpoint.parent.parent / "latest").resolve() == checkpoint.parent.resolve()


def test_query_directory_rejects_symlink_escape(tmp_path):
    round_dir = tmp_path / "round_6"
    outside = tmp_path / "outside"
    round_dir.mkdir()
    outside.mkdir()
    (round_dir / "query_01").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes|symlink"):
        persist_query_input(round_dir, 1, "private")
    assert not (outside / "query_input_01.md").exists()


def test_ambiguous_new_and_legacy_directories_fail_explicitly(tmp_path):
    round_dir = tmp_path / "round_7"
    (round_dir / "query_01").mkdir(parents=True)
    (round_dir / "query_001").mkdir()
    with pytest.raises(QueryArtifactCollisionError, match="multiple query directories"):
        persist_query_input(round_dir, 1, "do not choose silently")


def test_query_started_persists_exact_input_without_changing_bounded_objective(tmp_path):
    exact = "Compare the following cohorts:\n\n- treatment\n- control\n\n" + ("x" * 1400)

    progress_recorder.write_query_started(tmp_path, 7, objective=exact)

    payload = json.loads(
        (tmp_path / "query_07" / "query_status.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 2
    assert payload["query_input"] == exact.strip()
    assert payload["objective"].endswith("... [truncated]")
    assert len(payload["objective"]) < len(payload["query_input"])


def test_later_status_updates_preserve_exact_query_input(tmp_path):
    exact = "Measure effect sizes.\nPreserve this second line exactly."
    progress_recorder.write_query_started(tmp_path, 3, objective=exact)

    progress_recorder.write_query_step(tmp_path, 3, "analyzing_results")
    progress_recorder.write_query_completed(tmp_path, 3, summary="Completed analysis.")

    payload = json.loads(
        (tmp_path / "query_03" / "query_status.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 2
    assert payload["query_input"] == exact
    assert payload["status"] == "completed"


def test_missing_input_is_represented_as_null_without_persisting_other_context(tmp_path):
    progress_recorder.write_query_started(tmp_path, 1)

    payload = json.loads(
        (tmp_path / "query_01" / "query_status.json").read_text(encoding="utf-8")
    )
    assert payload["query_input"] is None
    assert payload["objective"] == progress_recorder.DEFAULT_OBJECTIVE
