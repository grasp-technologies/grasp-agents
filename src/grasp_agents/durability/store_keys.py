"""
Checkpoint-store key construction.

Layout: ``"<session_key>/<kind>/<full_path>"`` — with ``"…/lifecycle"``
appended for background-task records. Tool calls contribute a
``tc_<call_id>`` segment; parallel replicas a combined ``<subproc>_<idx>``.
Always use the helpers — never compose keys inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .checkpoints import CheckpointKind

TOOL_CALL_PREFIX = "tc_"
LIFECYCLE_LEAF = "lifecycle"


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


def make_lifecycle_key(
    session_key: str,
    kind: CheckpointKind | str,
    path: Sequence[str],
) -> str:
    """Lifecycle-metadata key — sibling to the processor's checkpoint at ``path``."""
    return make_store_key(session_key, kind, [*path, LIFECYCLE_LEAF])


def make_tool_call_path(
    parent_session_path: Sequence[str] | None, tool_call_id: str
) -> list[str] | None:
    """Append a ``tc_<call_id>`` segment; ``None`` propagates ``None``."""
    if parent_session_path is None:
        return None
    return [*parent_session_path, f"{TOOL_CALL_PREFIX}{tool_call_id}"]


def session_prefix(session_key: str) -> str:
    """Prefix for :meth:`CheckpointStore.list_keys` to scan one session."""
    return f"{session_key}/"


def is_lifecycle_key(key: str) -> bool:
    return key.rsplit("/", 1)[-1] == LIFECYCLE_LEAF


def lifecycle_state_key(lifecycle_key: str) -> str:
    """Inverse of :func:`make_lifecycle_key` — strip the trailing ``/lifecycle``."""
    if not is_lifecycle_key(lifecycle_key):
        raise ValueError(f"Not a lifecycle key: {lifecycle_key!r}")
    return lifecycle_key[: -(len(LIFECYCLE_LEAF) + 1)]
