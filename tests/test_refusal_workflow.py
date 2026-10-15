import json
import os
from types import SimpleNamespace

from DiscoveryAgent.director_workflow import DirectorWorkflow
from DiscoveryAgent.utils.llm_diagnostics import REFUSAL_ADVICE


class CaptureLogger:
    def __init__(self):
        self.actions = []
        self.errors = []
        self.responses = []

    def log_action(self, action, details=None):
        self.actions.append((action, details))

    def log_error(self, message, code=None):
        self.errors.append((message, code))

    def log_response(self, response, source=None):
        self.responses.append((response, source))


class RefusingDirector:
    def __init__(self, outputs, refusal="[REFUSAL] Model declined to respond"):
        self.outputs = iter(outputs)
        self.last_refusal = refusal
        self.feedback = []

    def estimate_context_tokens(self):
        return 0

    def generate_query(self, **kwargs):
        self.feedback.append(kwargs.get("feedback"))
        return next(self.outputs)

    def generate_final_report(self, report):
        return ""


class Checkpoints:
    def __init__(self, round_dir):
        self.round_dir = round_dir

    def get_checkpoint_dir(self, query_num):
        return self.round_dir / f"query_{query_num:03d}"


def read_events(round_dir):
    path = round_dir / "round_activity.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def director_workflow(tmp_path, director, *, max_queries=1):
    workflow = DirectorWorkflow.__new__(DirectorWorkflow)
    workflow.dataset_description = "dataset"
    workflow.task_description = None
    workflow.max_queries = max_queries
    workflow.logger = CaptureLogger()
    workflow.hypothesis_config = SimpleNamespace(
        human_selection_mode=False,
        selected_hypothesis_file=None,
    )
    workflow.director_agent = director
    workflow.hypothesis_gen_agent = SimpleNamespace(last_refusal=None)
    workflow.multiturn_hypothesis = True
    workflow.report = ""
    workflow.cumulative_report = "existing report"
    workflow.background_report = ""
    workflow.max_tool_invocation_retries = 1
    workflow.allow_director_to_stop = False
    workflow.results = []
    workflow.git_vc = None
    workflow.review_optimization_steps = None
    workflow.checkpoint_manager = Checkpoints(tmp_path)
    workflow._save_final_report = lambda report=None: tmp_path / "final_report.md"
    return workflow


def test_director_refusal_retries_without_malformed_query_feedback_and_records_events(tmp_path):
    refusal = "[REFUSAL] Model declined to respond (category: bio)"
    director = RefusingDirector(["", ""], refusal=refusal)
    workflow = director_workflow(tmp_path, director)

    result = workflow.run(exploratory_mode=True)

    assert result["results"] == []
    assert director.feedback == [None, None]
    assert not any(action == "tool_invocation_error" for action, _ in workflow.logger.actions)
    assert any(action == "model_refusal" for action, _ in workflow.logger.actions)
    assert any(
        "Refused by the API on 1 consecutive attempts" in message and REFUSAL_ADVICE in message
        for message, _ in workflow.logger.errors
    )

    events = read_events(tmp_path)
    assert [event["event_type"] for event in events] == [
        "model_refusal",
        "model_refusal",
        "early_stopping",
    ]
    assert refusal in events[0]["message"]
    assert "Refused by the API" in events[-1]["message"]


def test_malformed_non_refusal_still_uses_shorter_query_feedback(tmp_path):
    director = RefusingDirector(["", "<execution>valid</execution>"], refusal=None)
    workflow = director_workflow(tmp_path, director)
    workflow.execute_agent = lambda query: ("result", "valid")
    workflow.report_update = lambda *_args: None
    workflow.checkpoint_manager.save_checkpoint = lambda _query: None

    result = workflow.run(exploratory_mode=True)

    assert len(result["results"]) == 1
    assert director.feedback[0] is None
    assert "SHORTER and more concise query" in director.feedback[1]
    assert any(action == "tool_invocation_error" for action, _ in workflow.logger.actions)
    assert not any(action == "model_refusal" for action, _ in workflow.logger.actions)


def test_no_query_records_an_early_stop_reason(tmp_path):
    director = RefusingDirector([None], refusal=None)
    workflow = director_workflow(tmp_path, director)

    workflow.run(exploratory_mode=True)

    events = read_events(tmp_path)
    assert events == [
        {
            "timestamp": events[0]["timestamp"],
            "event_type": "early_stopping",
            "message": "Query 1: director returned no query",
        }
    ]


def test_final_report_refusal_records_activity_and_falls_back(tmp_path):
    refusal = "[REFUSAL] Model declined to respond (category: policy)"
    director = RefusingDirector([], refusal=refusal)
    workflow = director_workflow(tmp_path, director, max_queries=0)

    result = workflow.run(final_report_step=True, exploratory_mode=True)

    assert result["report"] == "existing report"
    events = read_events(tmp_path)
    assert events[-1]["event_type"] == "model_refusal"
    assert events[-1]["message"] == f"Final report: {refusal}"
    assert any(
        code is not None and refusal in code
        for _message, code in workflow.logger.errors
    )
