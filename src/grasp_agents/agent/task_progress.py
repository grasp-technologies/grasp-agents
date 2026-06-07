"""
Agent-readable progress logs for backgrounded tasks.

A backgrounded task's streamed output is mirrored to a file under the agent's
own filesystem — ``<first allowed root>/.grasp/tasks/<call_id>.log`` — so it
survives a crash and the agent can ``Read`` / ``Grep`` it. The
:class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager` calls these
helpers at its turn-boundary flush. The file is the single source of truth for a
task's output; the ``TaskRecord`` only indexes it via ``output_path``.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.file_backend.base import FileBackend

logger = logging.getLogger(__name__)

# Directory (under the backend's first allowed root) holding per-task logs.
TASK_LOG_DIR = (".grasp", "tasks")


async def open_task_log(file_backend: FileBackend, *, name: str) -> str | None:
    """
    Resolve (creating the parent dir) the log file for the task ``name``.

    Returns the resolved path, or ``None`` if the backend has no roots or
    rejects the path. Best-effort — never raises.
    """
    if not file_backend.allowed_roots:
        return None
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name) or "task"
    log_dir = file_backend.allowed_roots[0].joinpath(*TASK_LOG_DIR)
    try:
        await file_backend.mkdir(log_dir)
        resolved = await file_backend.validate_path(
            log_dir / f"{safe}.log", must_exist=False, access="write"
        )
    except Exception:
        logger.warning("Could not open task log file for %r", name)
        return None
    return str(resolved)


async def write_task_log(file_backend: FileBackend, path: str, text: str) -> None:
    """Overwrite the log at ``path`` with ``text``. Best-effort — never raises."""
    with contextlib.suppress(Exception):
        await file_backend.write_bytes(
            Path(path), text.encode(), mode=0o644, overwrite=True
        )
