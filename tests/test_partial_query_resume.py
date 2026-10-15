import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ["CLAUDE_CODE_CLI_PATH"] = "/usr/bin/true"

from DiscoveryAgent.director_workflow import DirectorWorkflow
from DiscoveryAgent.tool_agents.execution_agent_cc import (
    ExecutionAgentEmptyResponseError,
    ExecutionAgentMalformedReportError,
    ExecutionAgentRefusalError,
)
from DiscoveryAgent.utils import progress_recorder
from DiscoveryAgent.utils.query_artifacts import (
    artifact_filename,
    persist_query_input,
    persist_query_report,
    query_directory,
)


class _Logger:
    def __init__(self):
        self.actions = []
        self.errors = []

    def log_action(self, action, details=None):
        self.actions.append((action, details))

    def log_error(self, message, code=None):
        self.errors.append((message, code))

    def log_response(self, *_args, **_kwargs):
        pass


class _Director:
    last_refusal = None

    def __init__(self):
        self.calls = 0

    def estimate_context_tokens(self):
        return 0

    def generate_query(self, **_kwargs):
        self.calls += 1
        raise AssertionError("resume must not regenerate a persisted unfinished query")


class _Agent:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def execute(self, query):
        self.calls.append(query)
        return self.callback(query)

    def cleanup(self):
        pass


class _Checkpoints:
    def __init__(self, round_dir: Path, workflow):
        self.round_dir = round_dir
        self.workflow = workflow

    def get_checkpoint_dir(self, query_num):
        return query_directory(self.round_dir, query_num)

    def save_checkpoint(self, query_num):
        path = self.get_checkpoint_dir(query_num) / "checkpoint.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "query_num": query_num,
            "results": self.workflow.results,
        }), encoding="utf-8")
        return str(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _workflow(round_dir: Path, callback) -> tuple[DirectorWorkflow, _Agent, _Director]:
    workflow = DirectorWorkflow.__new__(DirectorWorkflow)
    workflow.query_num = 4
    workflow.round_num = 1
    workflow.dataset_description = "fixture dataset"
    workflow.task_description = None
    workflow.max_queries = 5
    workflow.logger = _Logger()
    workflow.hypothesis_config = SimpleNamespace(
        human_selection_mode=False,
        selected_hypothesis_file=None,
    )
    director = _Director()
    workflow.director_agent = director
    workflow.hypothesis_gen_agent = SimpleNamespace(last_refusal=None)
    workflow.multiturn_hypothesis = True
    workflow.report = "four-query report"
    workflow.cumulative_report = "four-query cumulative report"
    workflow.background_report = ""
    workflow.max_tool_invocation_retries = 1
    workflow.allow_director_to_stop = False
    workflow.results = [
        {"query_num": number, "result": f"result-{number}"}
        for number in range(1, 5)
    ]
    workflow.git_vc = None
    workflow.review_optimization_steps = None
    workflow.agents = {"execution_agent": _Agent(callback)}
    workflow.log_dir = str(round_dir.parent.parent)
    workflow._save_final_report = lambda report=None: round_dir / "final_report.md"
    workflow.checkpoint_manager = _Checkpoints(round_dir, workflow)
    return workflow, workflow.agents["execution_agent"], director


def _partial_input(round_dir: Path, content: str = "exact persisted query five") -> Path:
    path = persist_query_input(round_dir, 5, content)
    progress_recorder.write_query_started(
        round_dir, 5, objective=content, agent_tag="execution_agent",
    )
    progress_recorder.write_query_failed(round_dir, 5, "execution_agent_final_response_missing")
    return path


def test_resume_retries_exact_partial_input_without_regeneration_or_duplicate_artifacts(tmp_path):
    round_dir = tmp_path / "checkpoints" / "round_1"
    for number in range(1, 5):
        checkpoint = query_directory(round_dir, number) / "checkpoint.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(f"completed-{number}", encoding="utf-8")
    completed_before = {
        number: _digest(query_directory(round_dir, number) / "checkpoint.json")
        for number in range(1, 5)
    }
    input_path = _partial_input(round_dir)
    input_stat = input_path.stat()
    log_path = input_path.parent / artifact_filename(5, "execution_log")
    log_path.write_text("successful tool output\n", encoding="utf-8")

    workflow, agent, director = _workflow(round_dir, lambda _query: "one recovered report")
    result = workflow.run(exploratory_mode=True)

    assert director.calls == 0
    assert agent.calls == ["exact persisted query five"]
    assert input_path.stat().st_ino == input_stat.st_ino
    assert input_path.read_text() == "exact persisted query five"
    assert log_path.read_text() == "successful tool output\n"
    assert (input_path.parent / artifact_filename(5, "report")).read_text() == "one recovered report"
    assert (input_path.parent / "checkpoint.json").is_file()
    assert [item["query_num"] for item in result["results"]] == [1, 2, 3, 4, 5]
    assert all(
        _digest(query_directory(round_dir, number) / "checkpoint.json") == completed_before[number]
        for number in range(1, 5)
    )
    status = json.loads((input_path.parent / "query_status.json").read_text())
    assert status["status"] == "completed"
    assert status["query_num"] == 5


def test_resume_replays_existing_report_into_checkpoint_without_agent_or_rewrite(tmp_path):
    round_dir = tmp_path / "checkpoints" / "round_1"
    input_path = _partial_input(round_dir)
    report_path = persist_query_report(round_dir, 5, "durable report before exit")
    report_stat = report_path.stat()
    workflow, agent, director = _workflow(
        round_dir, lambda _query: (_ for _ in ()).throw(AssertionError("must not execute")),
    )

    result = workflow.run(exploratory_mode=True)

    assert director.calls == 0
    assert agent.calls == []
    assert report_path.stat().st_ino == report_stat.st_ino
    assert report_path.read_text() == "durable report before exit"
    assert (input_path.parent / "checkpoint.json").is_file()
    assert result["results"][-1]["query_parsed"] == "exact persisted query five"


@pytest.mark.parametrize(
    "failure",
    [
        ExecutionAgentRefusalError(),
        ExecutionAgentEmptyResponseError(),
        ExecutionAgentMalformedReportError(),
    ],
)
def test_failed_partial_retry_keeps_input_and_has_no_report(tmp_path, failure):
    round_dir = tmp_path / "checkpoints" / "round_1"
    input_path = _partial_input(round_dir)
    before = _digest(input_path)
    workflow, _agent, director = _workflow(
        round_dir, lambda _query: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(type(failure), match=failure.category):
        workflow.run(exploratory_mode=True)

    assert director.calls == 0
    assert _digest(input_path) == before
    assert not (input_path.parent / artifact_filename(5, "report")).exists()
    status = json.loads((input_path.parent / "query_status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == failure.category


def test_partial_input_without_a_recoverable_agent_identity_fails_before_director_call(tmp_path):
    round_dir = tmp_path / "checkpoints" / "round_1"
    persist_query_input(round_dir, 5, "ambiguous persisted input")
    workflow, agent, director = _workflow(round_dir, lambda _query: "must not run")

    with pytest.raises(RuntimeError, match="helper identity cannot be recovered"):
        workflow.run(exploratory_mode=True)

    assert director.calls == 0
    assert agent.calls == []
