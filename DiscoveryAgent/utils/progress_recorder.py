"""
Lightweight, additive progress-recording utility for the workflow/orchestration layer.

This module records *state* about query/round progress (query_status.json,
round_activity.jsonl) for the frontend Round Monitor to consume. It makes no
scientific decisions, calls no LLM, and does not alter query planning or
execution logic.

Architecture principle: agents do science, the workflow/controller records
state, the frontend displays state.

Every public function here is designed to never raise -- a failure to write a
progress artifact must never break a research run.
"""

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .query_artifacts import artifact_filename, query_directory

PathLike = Union[str, Path]

SCHEMA_VERSION = 2

ALLOWED_STATUSES = {"running", "completed", "failed"}
ALLOWED_STEPS = {
    "planning",
    "executing_code",
    "running_literature_check",
    "analyzing_results",
    "writing_query_report",
    "completed",
    "failed",
}

DEFAULT_OBJECTIVE = "Objective not yet available."
RUNNING_SUMMARY = "Query is running; results have not been summarized yet."
NO_SUMMARY_FALLBACK = "Query completed; detailed results were saved to query artifacts."
FAILED_SUMMARY = "Query failed before results were summarized."

_MAX_SUMMARY_CHARS = 500
_MAX_OBJECTIVE_CHARS = 1000
_MAX_ERROR_CHARS = 500


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _query_name(query_num: int) -> str:
    return f"query_{query_num:02d}"


def _query_dir(round_dir: PathLike, query_num: int) -> Path:
    return query_directory(round_dir, query_num)


def _status_path(round_dir: PathLike, query_num: int) -> Path:
    return _query_dir(round_dir, query_num) / "query_status.json"


def _truncate(text: Optional[str], max_chars: int) -> Optional[str]:
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    os.replace(tmp_path, path)


def _load_existing(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _detect_artifacts(round_dir: PathLike, query_num: int) -> List[str]:
    """Best-effort, deterministic listing of files already present in query_NNN/.

    Lists canonical query artifacts and checkpoint.json first (if present), then any other
    files already in the directory, excluding query_status.json itself.
    """
    try:
        qdir = _query_dir(round_dir, query_num)
        if not qdir.exists():
            return []
        names: List[str] = []
        for candidate in (
            artifact_filename(query_num, "input"),
            artifact_filename(query_num, "report"),
            f"{qdir.name}.md",
            "checkpoint.json",
        ):
            if (qdir / candidate).exists():
                names.append(candidate)
        for entry in sorted(qdir.iterdir()):
            if entry.is_file() and entry.name not in names and entry.name != "query_status.json":
                names.append(entry.name)
        return names
    except Exception:
        return []


def _append_round_event(
    round_dir: PathLike,
    event_type: str,
    message: Optional[str] = None,
    query_name: Optional[str] = None,
) -> None:
    try:
        path = Path(round_dir) / "round_activity.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry: Dict[str, Any] = {"timestamp": _now_iso(), "event_type": event_type}
        if query_name is not None:
            entry["query_name"] = query_name
        if message is not None:
            entry["message"] = _truncate(message, _MAX_SUMMARY_CHARS)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def write_round_event(
    round_dir: PathLike,
    event_type: str,
    message: Optional[str] = None,
    query_name: Optional[str] = None,
) -> None:
    """Append a single event line to round_dir/round_activity.jsonl. Never raises."""
    _append_round_event(round_dir, event_type, message=message, query_name=query_name)


def _write_status(round_dir: PathLike, query_num: int, **updates: Any) -> None:
    """Merge `updates` into the existing query_status.json (if any) and write atomically.

    Fields not present in `updates` (or explicitly None, except `error`/`completed_at`
    which may be intentionally cleared) are preserved from the existing file.
    """
    path = _status_path(round_dir, query_num)
    existing = _load_existing(path) or {}

    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "query_num": query_num,
        "query_name": _query_dir(round_dir, query_num).name,
        "status": existing.get("status", "running"),
        "current_step": existing.get("current_step", "planning"),
        "objective": existing.get("objective", DEFAULT_OBJECTIVE),
        "query_input": existing.get("query_input"),
        "agent_tag": existing.get("agent_tag"),
        "summary": existing.get("summary", RUNNING_SUMMARY),
        "started_at": existing.get("started_at"),
        "updated_at": None,
        "completed_at": existing.get("completed_at"),
        "artifacts_written": existing.get("artifacts_written", []),
        "error": existing.get("error"),
    }

    for key, value in updates.items():
        if value is not None or key in ("error", "completed_at"):
            record[key] = value

    if record.get("started_at") is None:
        record["started_at"] = _now_iso()
    record["updated_at"] = _now_iso()

    if record["status"] not in ALLOWED_STATUSES:
        record["status"] = "running"
    if record["current_step"] not in ALLOWED_STEPS:
        record["current_step"] = "planning"

    _atomic_write_json(path, record)


def write_query_started(
    round_dir: PathLike,
    query_num: int,
    objective: Optional[str] = None,
    agent_tag: Optional[str] = None,
) -> None:
    """Create/update query_status.json marking a query as started (status=running).

    `objective` should be the actual query/task text being pursued (e.g. the parsed
    instruction sent to a helper agent for this query), not dataset README/context
    boilerplate. Schema 2 retains that exact instruction separately as
    `query_input`; the concise objective remains bounded for existing status UI.
    Falls back to DEFAULT_OBJECTIVE if not yet available.
    """
    try:
        query_input = str(objective).strip() if objective is not None else None
        query_input = query_input or None
        objective_value = _truncate(query_input, _MAX_OBJECTIVE_CHARS) or DEFAULT_OBJECTIVE
        _write_status(
            round_dir,
            query_num,
            status="running",
            current_step="planning",
            objective=objective_value,
            query_input=query_input,
            agent_tag=str(agent_tag).strip() if agent_tag else None,
            summary=RUNNING_SUMMARY,
            completed_at=None,
            error=None,
        )
        _append_round_event(round_dir, "query_started", query_name=_query_dir(round_dir, query_num).name)
    except Exception:
        pass


def write_query_step(
    round_dir: PathLike,
    query_num: int,
    current_step: str,
    summary: Optional[str] = None,
    artifacts_written: Optional[List[str]] = None,
) -> None:
    """Update current_step (and optionally summary/artifacts_written) for a running query."""
    try:
        step = current_step if current_step in ALLOWED_STEPS else "planning"
        updates: Dict[str, Any] = {"current_step": step}
        truncated_summary = _truncate(summary, _MAX_SUMMARY_CHARS)
        if truncated_summary:
            updates["summary"] = truncated_summary
        if artifacts_written is not None:
            updates["artifacts_written"] = list(artifacts_written)
        _write_status(round_dir, query_num, **updates)
        _append_round_event(round_dir, "query_step", message=step, query_name=_query_dir(round_dir, query_num).name)
    except Exception:
        pass


def write_query_completed(
    round_dir: PathLike,
    query_num: int,
    summary: Optional[str] = None,
    artifacts_written: Optional[List[str]] = None,
) -> None:
    """Mark a query as completed (status=completed, current_step=completed).

    `summary` should be an existing query-level result/summary text already produced
    by the workflow (e.g. the helper agent's own output), truncated here -- never
    invented. Falls back to NO_SUMMARY_FALLBACK if none is available. If
    `artifacts_written` is not supplied, it is auto-detected from the query directory.
    """
    try:
        final_summary = _truncate(summary, _MAX_SUMMARY_CHARS) or NO_SUMMARY_FALLBACK
        detected_artifacts = (
            list(artifacts_written) if artifacts_written is not None else _detect_artifacts(round_dir, query_num)
        )
        _write_status(
            round_dir,
            query_num,
            status="completed",
            current_step="completed",
            summary=final_summary,
            completed_at=_now_iso(),
            artifacts_written=detected_artifacts,
            error=None,
        )
        _append_round_event(round_dir, "query_completed", query_name=_query_dir(round_dir, query_num).name)
    except Exception:
        pass


def write_query_failed(round_dir: PathLike, query_num: int, error: str) -> None:
    """Mark a query as failed (status=failed, current_step=failed).

    Never raises and never masks the original exception -- callers must still
    re-raise (or otherwise propagate) the original error after calling this.
    """
    try:
        _write_status(
            round_dir,
            query_num,
            status="failed",
            current_step="failed",
            summary=FAILED_SUMMARY,
            completed_at=_now_iso(),
            error=_truncate(str(error), _MAX_ERROR_CHARS),
        )
        _append_round_event(
            round_dir,
            "query_failed",
            message=str(error),
            query_name=_query_dir(round_dir, query_num).name,
        )
    except Exception:
        pass
