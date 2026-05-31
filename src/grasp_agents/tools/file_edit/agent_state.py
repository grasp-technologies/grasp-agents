"""
ContextVar-propagated reference to the currently-active agent's
:class:`FileEditSessionState`.

Each :class:`AgentLoop` owns its own :class:`FileEditSessionState`.
``execute_stream`` sets this ContextVar at run-start (and resets it in a
``finally``) so file-edit tools, file-search tools, and
:class:`MemoryProvider` can find the active state without threading it
through every backend method.

The default is ``None`` so standalone tool use outside any agent works:
tools treat ``None`` as "no read-before-write enforcement, no recording"
(power-user escape hatch). Child asyncio tasks created by
``asyncio.gather`` / ``stream_concurrent`` inherit the parent's
ContextVar value, so the state propagates naturally into parallel tool
dispatch.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_state import FileEditSessionState


_current_file_edit_state: ContextVar[FileEditSessionState | None] = ContextVar(
    "_grasp_current_file_edit_state", default=None
)


def get_current_file_edit_state() -> FileEditSessionState | None:
    """Return the active agent's state, or ``None`` outside any agent."""
    return _current_file_edit_state.get()


def set_current_file_edit_state(
    state: FileEditSessionState | None,
) -> Token[FileEditSessionState | None]:
    """Bind ``state`` as the active agent's state and return a reset token."""
    return _current_file_edit_state.set(state)


def reset_current_file_edit_state(
    token: Token[FileEditSessionState | None],
) -> None:
    """Restore whatever was active before the matching :func:`set` call."""
    _current_file_edit_state.reset(token)


__all__ = [
    "get_current_file_edit_state",
    "reset_current_file_edit_state",
    "set_current_file_edit_state",
]
