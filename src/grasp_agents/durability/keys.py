"""
Centralized checkpoint-store key construction.

All store keys produced by the framework pass through helpers in this
module, so the layout is visible in one place.

Layout:

- Processor checkpoints (``Processor._checkpoint_store_key``):
  ``"{kind}/{session_key}[/{session_path}]"`` — see
  :class:`~.checkpoints.CheckpointKind` for the ``kind`` enum. The path
  segment is set by container processors when adopting subprocs and
  (since B2 step C) by :class:`BackgroundTaskManager` when spawning
  background subagents under the parent session.

- Task records (background tool executions):
  ``"task/{session_key}/{task_id}"``. Built via :func:`task_key` /
  :func:`task_session_prefix`; never constructed by hand.
"""

from __future__ import annotations

TASK_KEY_PREFIX = "task"


def task_key(session_key: str, task_id: str) -> str:
    """Checkpoint-store key for a single :class:`~.task_record.TaskRecord`."""
    return f"{TASK_KEY_PREFIX}/{session_key}/{task_id}"


def task_session_prefix(session_key: str) -> str:
    """
    Prefix for :meth:`CheckpointStore.list_keys` to scan all task records
    under a session — e.g. for resume scan or ``prune_delivered``.
    """
    return f"{TASK_KEY_PREFIX}/{session_key}/"


def background_child_session_subpath(task_id: str) -> list[str]:
    """
    Session-path segment under which a background subagent's checkpoints
    live, relative to the parent agent's session key.

    Keeping the child under the parent session — rather than minting a
    separate ``session_key`` — means session-scoped services (approval
    store, file-edit store, usage tracker) stay shared between parent
    and child without any extra plumbing.
    """
    return [TASK_KEY_PREFIX, task_id]
