"""
Checkpoint-store key construction.

Layout: ``"<session_key>/<kind>/<path>"``. The ``kind`` segment names the
record type (``agent`` / ``workflow`` / ``parallel`` / ``runner`` /
``task``); each kind tree is type-homogeneous. Tool calls contribute a
``tc_<call_id>`` segment; parallel replicas a combined ``<subproc>_<idx>``.
Always use the helpers — never compose keys inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .checkpoints import CheckpointKind

if TYPE_CHECKING:
    from collections.abc import Sequence

TOOL_CALL_PREFIX = "tc_"


def make_store_key(
    session_key: str,
    kind: CheckpointKind | str,
    path: Sequence[str] | None = None,
) -> str:
    """Compose ``"<session_key>/<kind>[/<path>]"``."""
    base = f"{session_key}/{kind}"
    if not path:
        return base
    return f"{base}/{'/'.join(path)}"


def make_tool_call_path(
    parent_path: Sequence[str] | None, tool_call_id: str
) -> list[str] | None:
    """Append a ``tc_<call_id>`` segment; ``None`` propagates ``None``."""
    if parent_path is None:
        return None
    return [*parent_path, f"{TOOL_CALL_PREFIX}{tool_call_id}"]


def session_prefix(session_key: str) -> str:
    """Prefix for :meth:`CheckpointStore.list_keys` to scan one session."""
    return f"{session_key}/"


def task_prefix(session_key: str) -> str:
    """Prefix for :meth:`CheckpointStore.list_keys` to scan task records."""
    return f"{session_key}/{CheckpointKind.TASK}/"
