"""
Persistent-shell-session reset detection.

When the holder reopens a session that closed (e.g. killed by a timeout), it
loses cwd / env / shell variables. The holder flags that so ``BashSession`` can
tell the model its shell state was reset, instead of silently continuing in a
fresh shell.
"""

from __future__ import annotations

import pytest

from grasp_agents.tools.bash_session import BashSessionHolder


class _FakeSession:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeBackend:
    name = "fake"

    def __init__(self) -> None:
        self.opened: list[_FakeSession] = []

    async def open_session(self) -> _FakeSession:
        s = _FakeSession()
        self.opened.append(s)
        return s


@pytest.mark.asyncio
async def test_first_open_is_not_a_reset() -> None:
    holder = BashSessionHolder()
    backend = _FakeBackend()
    s = await holder.get(backend)
    assert s is backend.opened[0]
    assert holder.take_reset() is False


@pytest.mark.asyncio
async def test_reopen_after_close_flags_reset() -> None:
    holder = BashSessionHolder()
    backend = _FakeBackend()
    s1 = await holder.get(backend)
    s1.closed = True
    s2 = await holder.get(backend)
    assert s2 is not s1
    assert holder.take_reset() is True
    assert holder.take_reset() is False  # cleared after taking


@pytest.mark.asyncio
async def test_live_session_reused_without_reset() -> None:
    holder = BashSessionHolder()
    backend = _FakeBackend()
    s1 = await holder.get(backend)
    s2 = await holder.get(backend)  # still alive → same session, no reset
    assert s2 is s1
    assert holder.take_reset() is False
    assert len(backend.opened) == 1
