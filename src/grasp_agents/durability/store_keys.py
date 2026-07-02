"""
Checkpoint-store key construction.

Layout: ``"<session_key>/<kind>/<path>"``. The ``kind`` segment names the
record type (``agent`` / ``workflow`` / ``parallel`` / ``runner`` / ``team`` /
``task`` / ``session``); each kind tree is type-homogeneous. The ``session``
record is a singleton per session key (no path). Tool calls contribute a
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


def key_leaf(key: str) -> str:
    """The final segment of a store key — the record id under its prefix."""
    return key.rsplit("/", 1)[-1]


def is_under(key: str, prefix: str) -> bool:
    """
    Whether ``key`` lies under ``prefix``.

    A plain string-prefix test: store keys are a logical ``/``-joined namespace, not
    filesystem paths, so there is no path normalization (``//`` / ``.`` / ``..`` /
    trailing-slash) — matching must be exact.
    """
    return key.startswith(prefix)


def is_direct_child(key: str, prefix: str) -> bool:
    """
    Whether ``key`` is an *immediate* child of ``prefix`` — under it, with no further
    ``/`` beyond it. Selects the top-level records of a kind, excluding nested ones
    (e.g. a task record but not its per-tool-call children).
    """
    return key.startswith(prefix) and "/" not in key[len(prefix) :]
