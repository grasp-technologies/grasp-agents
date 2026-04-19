"""
``FileEditStore`` — session-keyed storage of read-before-write state.

The store owns the mutable state the file-edit tools need across calls:
per-session ``FileEditSessionState`` (read records + dotfile overrides).
Tools reach into the store at call time, keyed by ``ctx.session_key``,
so separate sessions stay isolated inside one process and consecutive
calls in the same session share state without explicit plumbing.

One store instance can serve many sessions. Reset is per-session and
explicit — no TTL, no implicit eviction tied to request boundaries.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from .session_state import FileEditSessionState


@runtime_checkable
class FileEditStore(Protocol):
    """
    Contract for session-keyed file-edit state.

    Implementations may be in-memory, disk-backed, or remote. The tools
    depend only on this protocol.
    """

    async def get_session_state(self, session_key: str) -> FileEditSessionState:
        """
        Return the :class:`FileEditSessionState` for ``session_key``.

        Implementations should lazily create a fresh state on first
        access — callers do not need to register a session before use.
        """
        ...

    async def reset_session(self, session_key: str) -> None:
        """
        Drop ``session_key``'s state entirely.

        After reset, :meth:`get_session_state` for the same key returns
        a fresh, empty :class:`FileEditSessionState`.
        """
        ...


class InMemoryFileEditStore:
    """
    Default :class:`FileEditStore` backed by an in-process dict.

    State survives across tool calls and across agent turns within the
    same process. Not persisted — a process restart clears everything.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, FileEditSessionState] = {}
        self._lock = asyncio.Lock()

    async def get_session_state(self, session_key: str) -> FileEditSessionState:
        async with self._lock:
            state = self._sessions.get(session_key)
            if state is None:
                state = FileEditSessionState()
                self._sessions[session_key] = state
            return state

    async def reset_session(self, session_key: str) -> None:
        async with self._lock:
            self._sessions.pop(session_key, None)
