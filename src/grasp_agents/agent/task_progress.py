"""
Agent-readable on-disk artifacts for tool-call outputs, under the agent's own
filesystem at ``<first allowed root>/.grasp/tasks/<call_id>.<ext>``:

- ``<call_id>.log`` — a backgrounded task's *streamed* output, mirrored
  incrementally as it runs so it survives a crash and the agent can ``Read`` /
  ``Grep`` a running task. Appended to at the manager's turn-boundary flush.
- ``<call_id>.result`` — a tool call's *full result*, written once when it
  exceeds the inline cap: a backgrounded task's terminal result on completion,
  or a foreground call's result spilled by :func:`spill_if_large`. The streamed
  log holds text; this holds the structured result (e.g. a ``BashResult`` with
  split stdout / stderr / exit code) — what the inline excerpt points at.

A backgrounded task's ``TaskRecord`` indexes its log via ``output_path``.
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
    text: str, cap: int | None, *, output_file: str | None = None
) -> tuple[str, bool]:
    """
    Fit a result into the limited inline space (a transcript message or a
    completion note): ``(text, truncated)``.

    Returns ``text`` unchanged when no cap applies or it already fits. Otherwise
    keeps a head + tail of ``cap`` chars total with a middle marker; when an
    ``output_file`` is known (a spilled result or a task's ``.grasp`` log), the
    marker points there so the model can ``Read`` / ``Grep`` the full output on
    demand rather than bloating the transcript with it.
    """
    if cap is None or len(text) <= cap:
        return text, False
    head = cap // 2
    tail = cap - head
    omitted = len(text) - cap
    pointer = f" — full output in {output_file}" if output_file else ""
    marker = f"\n... [{omitted} chars omitted{pointer}] ...\n"
    return text[:head] + marker + text[-tail:], True


async def spill_if_large(
    file_backend: FileBackend | None, *, name: str, text: str, cap: int | None
) -> str:
    """
    Bound how much of a tool result is inlined: when ``text`` exceeds ``cap``,
    spill the full text to a ``.result`` file and return a head+tail excerpt that
    points at it (``Read`` / ``Grep`` recovers the rest).

    Returns ``text`` unchanged when it fits, when ``cap`` is ``None``, or when
    there is no backend to spill to — truncating with no recovery path would
    hide output unrecoverably.
    """
    if cap is None or len(text) <= cap:
        return text
    output_file = (
        await write_result_file(file_backend, name=name, text=text)
        if file_backend is not None
        else None
    )
    if output_file is None:
        return text
    excerpt, _ = excerpt_for_inline(text, cap, output_file=output_file)
    return excerpt
