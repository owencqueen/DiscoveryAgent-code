"""Canonical, collision-safe persistence for per-query Markdown artifacts."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]
QUERY_ARTIFACT_MAX_BYTES = 4 * 1024 * 1024


class QueryArtifactCollisionError(RuntimeError):
    """Raised when a query artifact already contains different content."""


def query_suffix(query_num: int) -> str:
    if not isinstance(query_num, int) or isinstance(query_num, bool) or query_num < 0:
        raise ValueError("query_num must be a non-negative integer")
    return f"{query_num:02d}"


def query_directory_name(query_num: int) -> str:
    return f"query_{query_suffix(query_num)}"


def legacy_query_directory_name(query_num: int) -> str:
    return f"query_{query_num:03d}"


def query_directory(round_dir: PathLike, query_num: int) -> Path:
    """Return the canonical query directory, reusing an existing legacy folder.

    Existing three-digit folders are never renamed. For new work the two-digit
    minimum-width name is used. If a candidate exists, its resolved location
    must remain contained by the supplied round directory.
    """

    root = Path(round_dir).expanduser().absolute()
    canonical = root / query_directory_name(query_num)
    legacy = root / legacy_query_directory_name(query_num)
    candidates = (canonical,) if canonical == legacy else (canonical, legacy)
    existing = [
        candidate for candidate in candidates
        if candidate.exists() or candidate.is_symlink()
    ]
    if len(existing) > 1:
        raise QueryArtifactCollisionError(
            f"multiple query directories exist for query {query_num}: "
            + ", ".join(str(candidate) for candidate in existing)
        )
    selected = existing[0] if existing else canonical

    root_resolved = root.resolve(strict=False)
    selected_resolved = selected.resolve(strict=False)
    try:
        selected_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("query artifact directory escapes its round directory") from exc
    if selected.is_symlink():
        raise ValueError("query artifact directory must not be a symlink")
    return selected


def artifact_filename(query_num: int, role: str) -> str:
    if role == "execution_log":
        return f"execution_log_{query_suffix(query_num)}.txt"
    if role not in {"input", "report"}:
        raise ValueError(f"unsupported query artifact role: {role}")
    return f"query_{role}_{query_suffix(query_num)}.md"


def execution_log_filename(query_num: int) -> str:
    """Return the public, per-query ExecutionAgent log filename."""
    return artifact_filename(query_num, "execution_log")


def _atomic_create_idempotent(path: Path, content: str) -> Path:
    data = content.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("query artifact parent must be a real directory")

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise QueryArtifactCollisionError(f"query artifact target is not a regular file: {path}")
        if path.read_bytes() == data:
            return path
        raise QueryArtifactCollisionError(f"query artifact already exists with different content: {path}")

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise QueryArtifactCollisionError(
                    f"query artifact was concurrently created with different content: {path}"
                )
        return path
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def persist_query_artifact(round_dir: PathLike, query_num: int, role: str, content: str) -> Path:
    if not isinstance(content, str):
        content = str(content)
    directory = query_directory(round_dir, query_num)
    return _atomic_create_idempotent(directory / artifact_filename(query_num, role), content)


def persist_query_input(round_dir: PathLike, query_num: int, content: str) -> Path:
    return persist_query_artifact(round_dir, query_num, "input", content)


def persist_query_report(round_dir: PathLike, query_num: int, content: object) -> Optional[Path]:
    """Persist a produced report, or return ``None`` for an empty response."""
    if not isinstance(content, str) or not content.strip():
        return None
    return persist_query_artifact(round_dir, query_num, "report", content)


def read_query_artifact(
    round_dir: PathLike,
    query_num: int,
    role: str,
    *,
    max_bytes: int = QUERY_ARTIFACT_MAX_BYTES,
) -> Optional[str]:
    """Read one canonical query artifact without following an artifact symlink.

    Missing artifacts return ``None``. Invalid, oversized, or non-UTF-8 files are
    rejected explicitly so resume never guesses from a corrupt partial write.
    """
    if role not in {"input", "report"}:
        raise ValueError(f"unsupported readable query artifact role: {role}")
    directory = query_directory(round_dir, query_num)
    path = directory / artifact_filename(query_num, role)
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"query {role} artifact must be a regular file")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"query {role} artifact is empty")
    if size > max_bytes:
        raise ValueError(f"query {role} artifact exceeds the resume read limit")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"query {role} artifact is not valid UTF-8") from exc
