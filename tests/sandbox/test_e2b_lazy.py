"""
``e2b_environment(lazy=True)``: the sandbox starts on first use.

Uses the E2B test doubles from ``test_e2b.py`` and patches ``AsyncSandbox``
in the environment module the same way those tests do.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("e2b")

from grasp_agents.sandbox.e2b import environment as e2b_env
from grasp_agents.sandbox.e2b.environment import (
    FRESH_SNAPSHOT_REF,
    e2b_environment,
)
from tests.sandbox.test_e2b import _WS, _FakeFiles, _FakeSandbox

pytestmark = pytest.mark.asyncio


class _Creator:
    """Stands in for ``AsyncSandbox``; counts and records creations."""

    def __init__(self, *sandboxes: _FakeSandbox) -> None:
        self.__name__ = "FakeAsyncSandbox"
        self._queue = list(sandboxes)
        self.created: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeSandbox:
        self.created.append(kwargs)
        await asyncio.sleep(0)
        return self._queue.pop(0) if self._queue else _FakeSandbox()


@pytest.fixture
def creator(monkeypatch: pytest.MonkeyPatch) -> _Creator:
    fake = _Creator()
    monkeypatch.setattr(e2b_env, "AsyncSandbox", fake)
    return fake


async def test_entering_does_not_start_a_sandbox(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        assert not env.started
    assert creator.created == []


async def test_first_file_call_starts_it_once(creator: _Creator) -> None:
    sandbox = _FakeSandbox()
    creator._queue.append(sandbox)

    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{_WS}/a"))
        await env.file_backend.mkdir(Path(f"{_WS}/b"))
        assert env.started

    assert len(creator.created) == 1
    assert sandbox.files.made_dirs[0] == _WS
    assert sandbox.killed


async def test_concurrent_first_calls_share_one_sandbox(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        await asyncio.gather(
            *(env.file_backend.mkdir(Path(f"{_WS}/d{i}")) for i in range(5))
        )

    assert len(creator.created) == 1


async def test_first_exec_call_starts_it(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        await env.exec_backend.execute("echo hi")
        assert env.started

    assert len(creator.created) == 1


async def test_calls_after_exit_fail_without_starting(creator: _Creator) -> None:
    env = e2b_environment(allowed_roots=[_WS], lazy=True)
    async with env:
        pass

    with pytest.raises(RuntimeError, match="not entered"):
        await env.file_backend.mkdir(Path(f"{_WS}/late"))
    assert creator.created == []


async def test_pause_on_exit_pauses_a_started_sandbox(creator: _Creator) -> None:
    sandbox = _FakeSandbox()
    creator._queue.append(sandbox)

    async with e2b_environment(
        allowed_roots=[_WS], lazy=True, pause_on_exit=True
    ) as env:
        await env.file_backend.mkdir(Path(f"{_WS}/x"))

    assert sandbox.paused
    assert not sandbox.killed


async def test_snapshot_before_start_is_the_fresh_ref(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        assert await env.snapshot() == FRESH_SNAPSHOT_REF
        assert not env.started
    assert creator.created == []


async def test_restoring_fresh_drops_the_sandbox_until_next_use(
    creator: _Creator,
) -> None:
    first, second = _FakeSandbox(sandbox_id="first"), _FakeSandbox(sandbox_id="second")
    creator._queue.extend([first, second])

    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{_WS}/x"))
        await env.restore(FRESH_SNAPSHOT_REF)
        assert first.killed
        assert not env.started
        await env.file_backend.mkdir(Path(f"{_WS}/y"))

    assert len(creator.created) == 2
    assert f"{_WS}/y" in second.files.made_dirs


async def test_restoring_fresh_without_lazy_starts_a_new_sandbox(
    creator: _Creator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS]) as env:
        await env.restore(FRESH_SNAPSHOT_REF)
        assert env.started

    assert len(creator.created) == 2
    assert creator.created[1].get("template") is None


async def test_pause_before_start_is_refused(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        with pytest.raises(RuntimeError, match="never started"):
            await env.pause()


class _FailingFiles(_FakeFiles):
    async def make_dir(self, path: str) -> Any:
        raise OSError("disk full")


async def test_failed_setup_kills_the_sandbox_and_retries_later(
    creator: _Creator,
) -> None:
    broken = _FakeSandbox(files=_FailingFiles())
    creator._queue.extend([broken, _FakeSandbox()])

    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        with pytest.raises(OSError, match="disk full"):
            await env.file_backend.mkdir(Path(f"{_WS}/x"))
        assert broken.killed
        assert not env.started
        await env.file_backend.mkdir(Path(f"{_WS}/x"))
        assert env.started

    assert len(creator.created) == 2
