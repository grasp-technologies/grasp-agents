"""
``e2b_environment(lazy=True)``: the sandbox starts on first use.

Uses the E2B test doubles from ``test_e2b.py`` and patches ``AsyncSandbox``
in the environment module the same way those tests do.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("e2b")

from grasp_agents.sandbox.e2b import environment as e2b_env
from grasp_agents.sandbox.e2b.environment import (
    FRESH_SNAPSHOT_REF,
    e2b_environment,
)
from tests.sandbox.test_e2b import _WS, _FakeFiles, _FakeSandbox

if TYPE_CHECKING:
    from e2b import AsyncSandbox

pytestmark = pytest.mark.asyncio


class _Creator:
    """Stands in for ``AsyncSandbox``; counts and records creations."""

    def __init__(self, *sandboxes: _FakeSandbox) -> None:
        self.__name__ = "FakeAsyncSandbox"
        self._queue = list(sandboxes)
        self.created: list[dict[str, Any]] = []
        self.made: list[_FakeSandbox] = []
        self.connected: list[_FakeSandbox] = []

    async def create(self, **kwargs: Any) -> _FakeSandbox:
        self.created.append(kwargs)
        sandbox = (
            self._queue.pop(0)
            if self._queue
            else _FakeSandbox(sandbox_id=f"sbx_{len(self.created)}")
        )
        self.made.append(sandbox)
        await asyncio.sleep(0)
        return sandbox

    async def connect(self, sandbox_id: str) -> _FakeSandbox:
        sandbox = _FakeSandbox(sandbox_id=sandbox_id)
        self.connected.append(sandbox)
        return sandbox


class _SlowCreator(_Creator):
    """
    Holds each create without a template (a lazy start) until ``gate`` opens.
    The sandbox exists from the moment the create is sent, as on E2B, so a
    caller cancelled mid-create leaves it running unless something kills it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.creating = asyncio.Event()

    async def create(self, **kwargs: Any) -> _FakeSandbox:
        sandbox = await super().create(**kwargs)
        if kwargs.get("template") is None:
            self.creating.set()
            await self.gate.wait()
        return sandbox


@pytest.fixture
def creator(monkeypatch: pytest.MonkeyPatch) -> _Creator:
    fake = _Creator()
    monkeypatch.setattr(e2b_env, "AsyncSandbox", fake)
    return fake


@pytest.fixture
def slow_creator(monkeypatch: pytest.MonkeyPatch) -> _SlowCreator:
    fake = _SlowCreator()
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


# --- lifecycle changes while a start is in flight ---------------------------


async def test_exit_waits_for_a_start_and_kills_its_sandbox(
    slow_creator: _SlowCreator,
) -> None:
    env = e2b_environment(allowed_roots=[_WS], lazy=True)
    async with env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        await slow_creator.creating.wait()
        # Finish the create only once the exit is under way.
        asyncio.get_running_loop().call_soon(slow_creator.gate.set)

    with pytest.raises(RuntimeError, match="while it was starting"):
        await first_use
    (started,) = slow_creator.made
    assert started.killed
    assert not env.started


async def test_restore_waits_for_a_start_in_flight(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        await slow_creator.creating.wait()
        restoring = asyncio.create_task(env.restore("snap-1"))
        await asyncio.sleep(0)
        slow_creator.gate.set()
        await asyncio.gather(first_use, restoring)
        await env.file_backend.mkdir(Path(f"{_WS}/b"))

    started, restored = slow_creator.made
    assert started.killed
    assert f"{_WS}/b" in restored.files.made_dirs


async def test_pause_waits_for_a_start_in_flight(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        await slow_creator.creating.wait()
        pausing = asyncio.create_task(env.pause())
        await asyncio.sleep(0)
        slow_creator.gate.set()
        await first_use
        (started,) = slow_creator.made
        assert await pausing == started.sandbox_id
        assert started.paused


async def test_resume_waits_for_a_start_in_flight(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        await slow_creator.creating.wait()
        resuming = asyncio.create_task(env.resume("sbx_paused"))
        await asyncio.sleep(0)
        slow_creator.gate.set()
        await asyncio.gather(first_use, resuming)
        await env.file_backend.mkdir(Path(f"{_WS}/b"))

    (resumed,) = slow_creator.connected
    assert f"{_WS}/b" in resumed.files.made_dirs


async def test_start_finding_the_handle_filled_kills_its_sandbox(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        await slow_creator.creating.wait()
        env._holder.sandbox = cast("AsyncSandbox", _FakeSandbox(sandbox_id="other"))
        slow_creator.gate.set()
        with pytest.raises(RuntimeError, match="while it was starting"):
            await first_use

    (started,) = slow_creator.made
    assert started.killed


async def test_cancelled_first_call_kills_the_sandbox_it_started(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        await slow_creator.creating.wait()
        first_use.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_use
        slow_creator.gate.set()
        # pause() waits for the start, which no caller awaits any more.
        with pytest.raises(RuntimeError, match="never started"):
            await env.pause()

        (started,) = slow_creator.made
        assert started.killed
        assert not env.started


async def test_cancelling_one_first_call_leaves_the_others_their_sandbox(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[_WS], lazy=True) as env:
        cancelled = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/a")))
        waiting = asyncio.create_task(env.file_backend.mkdir(Path(f"{_WS}/b")))
        await slow_creator.creating.wait()
        cancelled.cancel()
        slow_creator.gate.set()
        await waiting

        (started,) = slow_creator.made
        assert not started.killed
        assert f"{_WS}/b" in started.files.made_dirs
