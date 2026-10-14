"""
Agent-readable on-disk artifacts for tool-call outputs, under the agent's own
filesystem at ``<first allowed root>/.grasp/tasks/<call_id>.<ext>``:

- ``<call_id>.log`` — a backgrounded task's *streamed* output, appended as it
  runs so it survives a crash and the agent can ``Read`` / ``Grep`` a running
  task.
- ``<call_id>.result`` — a tool call's *full result* (or error), written once by
  :func:`spill_if_large` when it exceeds the inline cap: a foreground call's
  result, or a backgrounded task's outcome when it finishes. The streamed log
  holds text; this holds the structured result (e.g. a ``BashResult`` with split
  stdout / stderr / exit code) — what the inline excerpt points at.

A backgrounded task's ``TaskRecord`` indexes its log via ``log_path``.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grasp_agents.file_backend.base import FileBackend

logger = logging.getLogger(__name__)

# Directory (under the backend's first allowed root) holding per-call artifacts.
TASK_LOG_DIR = (".grasp", "tasks")


def _task_file_name(name: str, suffix: str) -> str:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name) or "task"
    return f"{safe}{suffix}"


def task_log_name(name: str) -> str:
    """Basename of a task's streamed-output log (``<sanitised name>.log``)."""
    return _task_file_name(name, ".log")


async def _open_task_file(
    file_backend: FileBackend, *, name: str, suffix: str
) -> str | None:
    """Resolve (creating the parent dir) ``<grasp tasks>/<name><suffix>``."""
    if not file_backend.allowed_roots:
        return None
    log_dir = file_backend.allowed_roots[0].joinpath(*TASK_LOG_DIR)
    try:
        await file_backend.mkdir(log_dir)
        resolved = await file_backend.validate_path(
            log_dir / _task_file_name(name, suffix), must_exist=False, access="write"
        )
    except Exception:
        logger.warning("Could not open task file for %r (%s)", name, suffix)
        return None
    return str(resolved)


async def open_task_log(file_backend: FileBackend, *, name: str) -> str | None:
    """
    Resolve the streamed-output log file for the task ``name``.

    Returns the resolved path, or ``None`` if the backend has no roots or
    rejects the path. Best-effort — never raises.
    """
    return await _open_task_file(file_backend, name=name, suffix=".log")


async def append_task_log(file_backend: FileBackend, path: str, data: bytes) -> None:
    """Append ``data`` to the log at ``path``. Best-effort — never raises."""
    with contextlib.suppress(Exception):
        await file_backend.append_bytes(Path(path), data, mode=0o644)


async def write_result_file(
    file_backend: FileBackend, *, name: str, text: str
) -> str | None:
    """
    Write a tool call's full result to ``<name>.result`` and return its path
    (``None`` if it could not be written). Overwrites — written once. Used for a
    backgrounded task's terminal result and for a foreground spill (the
    ``call_id`` namespaces the two). Best-effort: never raises.
    """
    path = await _open_task_file(file_backend, name=name, suffix=".result")
    if path is None:
        return None
    try:
        await file_backend.write_bytes(
            Path(path), text.encode(), mode=0o644, overwrite=True
        )
    except Exception:
        logger.warning("Could not write result file for %r", name)
        return None

    return path


def excerpt_for_inline(
    text: str,
    cap: int | None,
    *,
    file: str | None = None,
    file_kind: str = "full output",
) -> tuple[str, bool]:
    """
    Fit a result into the limited inline space (a transcript message or a
    completion note): ``(text, truncated)``.

    Returns ``text`` unchanged when no cap applies or it already fits. Otherwise
    keeps a head + tail of ``cap`` chars total with a middle marker; when a
    ``file`` is known (a spilled result or a task's ``.grasp`` log), the marker
    points there — worded as ``file_kind`` so the model knows whether it is the
    full text or only the streamed output — so it can ``Read`` / ``Grep`` the
    rest on demand rather than bloating the transcript with it.
    """
    if cap is None or len(text) <= cap:
        return text, False
    head = cap // 2
    tail = cap - head
    omitted = len(text) - cap
    pointer = f" — {file_kind} in {file}" if file else ""
    marker = f"\n... [{omitted} chars omitted{pointer}] ...\n"

    return text[:head] + marker + text[-tail:], True


async def spill_if_large(
    file_backend: FileBackend | None,
    *,
    name: str,
    text: str,
    cap: int | None,
    file_kind: str = "full output",
) -> tuple[str, str | None]:
    """
    Bound how much of a tool result is inlined: when ``text`` exceeds ``cap``,
    spill the full text to a ``.result`` file and return a head+tail excerpt that
    points at it (``Read`` / ``Grep`` recovers the rest), with that file's path.

    Returns ``(text, None)`` when it fits, when ``cap`` is ``None``, or when the
    spill is impossible (no backend) or fails — truncating with no recovery path
    would hide output unrecoverably.
    """
    if cap is None or len(text) <= cap or file_backend is None:
        return text, None
    result_path = await write_result_file(file_backend, name=name, text=text)
    if result_path is None:
        return text, None
    excerpt, _ = excerpt_for_inline(text, cap, file=result_path, file_kind=file_kind)
    return excerpt, result_path
