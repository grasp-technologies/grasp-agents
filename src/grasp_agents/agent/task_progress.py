"""
Agent-readable on-disk artifacts for backgrounded tasks, under the agent's own
filesystem at ``<first allowed root>/.grasp/tasks/<call_id>.<ext>``:

- ``<call_id>.log`` — the task's *streamed* output, mirrored incrementally as it
  runs so it survives a crash and the agent can ``Read`` / ``Grep`` a running
  task. Appended to at the manager's turn-boundary flush.
- ``<call_id>.result`` — the task's *full terminal result*, written once on
  completion only when it exceeds the inline cap. The streamed log holds text;
  this holds the structured result (e.g. a ``BashResult`` with split
  stdout/stderr/exit code) — what the truncated completion note points at.

The ``TaskRecord`` only indexes the log via ``output_path``.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grasp_agents.file_backend.base import FileBackend

logger = logging.getLogger(__name__)

# Directory (under the backend's first allowed root) holding per-task artifacts.
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
    Write the task's full terminal result to ``<name>.result`` and return its
    path (``None`` if it could not be written). Overwrites — written once on
    completion. Best-effort: never raises.
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
