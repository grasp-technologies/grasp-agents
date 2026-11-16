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
        self.close_calls = 0

    async def close(self) -> None:
        self.closed = True
        self.close_calls += 1


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


@pytest.mark.asyncio
async def test_reopen_closes_the_session_it_replaces() -> None:
    # A session can count as closed without close() having run (its E2B
    # sandbox was replaced), which still holds its output stream open.
    holder = BashSessionHolder()
    backend = _FakeBackend()
    s1 = await holder.get(backend)
    s1.closed = True
    await holder.get(backend)
    assert s1.close_calls == 1


@pytest.mark.asyncio
async def test_holder_close_closes_a_session_that_already_counts_as_closed() -> None:
    holder = BashSessionHolder()
    backend = _FakeBackend()
    s1 = await holder.get(backend)
    s1.closed = True
    await holder.close()
    assert s1.close_calls == 1
