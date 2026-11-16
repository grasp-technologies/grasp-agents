"""
``e2b_environment(lazy=True)``: the sandbox starts on first use.

Uses the E2B test doubles from ``_e2b_fakes.py`` and patches ``AsyncSandbox``
in the environment module the same way ``test_e2b.py`` does.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("e2b")

from grasp_agents.sandbox import E2BSandboxExpiredError
from grasp_agents.sandbox.e2b import _handle as e2b_handle
from grasp_agents.sandbox.e2b import environment as e2b_env
from grasp_agents.sandbox.e2b.config import E2BEnvironmentConfig
from grasp_agents.sandbox.e2b.environment import (
    FRESH_SNAPSHOT_REF,
    E2BEnvironment,
    e2b_environment,
)
from grasp_agents.sandbox.exec_backend import ExecResult, TerminationReason
from grasp_agents.sandbox.kernel import CellResult
from grasp_agents.sandbox.policy import SandboxPolicy
from grasp_agents.tools.bash_session import BashSessionHolder
from grasp_agents.tools.notebook_exec import KernelHolder
from tests.sandbox._e2b_fakes import (
    WS,
    CodeSandbox,
    FakeFiles,
    FakeSandbox,
    ShellCommands,
)

if TYPE_CHECKING:
    from e2b import AsyncSandbox

    from grasp_agents.sandbox.exec_backend import ExecSession
    from grasp_agents.sandbox.kernel import KernelSession

pytestmark = pytest.mark.asyncio


class _Creator:
    """Stands in for ``AsyncSandbox``; counts and records creations."""

    def __init__(self, *sandboxes: FakeSandbox) -> None:
        self.__name__ = "FakeAsyncSandbox"
        self._queue = list(sandboxes)
        self.created: list[dict[str, Any]] = []
        self.made: list[FakeSandbox] = []
        self.connected: list[FakeSandbox] = []
        self.reconnects: dict[str, FakeSandbox] = {}

    async def create(self, **kwargs: Any) -> FakeSandbox:
        self.created.append(kwargs)
        sandbox = (
            self._queue.pop(0)
            if self._queue
            else FakeSandbox(sandbox_id=f"sbx_{len(self.created)}")
        )
        self.made.append(sandbox)
        await asyncio.sleep(0)
        return sandbox

    async def connect(self, sandbox_id: str, **opts: Any) -> FakeSandbox:
        del opts
        sandbox = self.reconnects.get(sandbox_id) or FakeSandbox(sandbox_id=sandbox_id)
        self.connected.append(sandbox)
        return sandbox


class _SlowCreator(_Creator):
    """
    Holds each create from a ``gated`` template (by default, none: a lazy
    start) and, with ``gate_connects``, each connect until ``gate`` opens. The
    sandbox exists from the moment the request is sent, as on E2B, so a caller
    cancelled midway leaves it running unless something kills it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.creating = asyncio.Event()
        self.gated: set[str | None] = {None}
        self.gate_connects = False

    async def create(self, **kwargs: Any) -> FakeSandbox:
        sandbox = await super().create(**kwargs)
        if kwargs.get("template") in self.gated:
            self.creating.set()
            await self.gate.wait()
        return sandbox

    async def connect(self, sandbox_id: str, **opts: Any) -> FakeSandbox:
        sandbox = await super().connect(sandbox_id, **opts)
        if self.gate_connects:
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
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        assert not env.started
    assert creator.created == []


async def test_first_file_call_starts_it_once(creator: _Creator) -> None:
    sandbox = FakeSandbox()
    creator._queue.append(sandbox)

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{WS}/a"))
        await env.file_backend.mkdir(Path(f"{WS}/b"))
        assert env.started

    assert len(creator.created) == 1
    assert sandbox.files.made_dirs[0] == WS
    assert sandbox.killed


async def test_concurrent_first_calls_share_one_sandbox(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await asyncio.gather(
            *(env.file_backend.mkdir(Path(f"{WS}/d{i}")) for i in range(5))
        )

    assert len(creator.created) == 1


async def test_first_exec_call_starts_it(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.exec_backend.execute("echo hi")
        assert env.started

    assert len(creator.created) == 1


async def test_calls_after_exit_fail_without_starting(creator: _Creator) -> None:
    env = e2b_environment(allowed_roots=[WS], lazy=True)
    async with env:
        pass

    with pytest.raises(RuntimeError, match="not entered"):
        await env.file_backend.mkdir(Path(f"{WS}/late"))
    assert creator.created == []


async def test_pause_on_exit_pauses_a_started_sandbox(creator: _Creator) -> None:
    sandbox = FakeSandbox()
    creator._queue.append(sandbox)

    async with e2b_environment(
        allowed_roots=[WS], lazy=True, pause_on_exit=True
    ) as env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))

    assert sandbox.paused
    assert not sandbox.killed


async def test_snapshot_before_start_is_the_fresh_ref(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        assert await env.snapshot() == FRESH_SNAPSHOT_REF
        assert not env.started
    assert creator.created == []


async def test_restoring_fresh_drops_the_sandbox_until_next_use(
    creator: _Creator,
) -> None:
    first, second = FakeSandbox(sandbox_id="first"), FakeSandbox(sandbox_id="second")
    creator._queue.extend([first, second])

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        await env.restore(FRESH_SNAPSHOT_REF)
        assert first.killed
        assert not env.started
        await env.file_backend.mkdir(Path(f"{WS}/y"))

    assert len(creator.created) == 2
    assert f"{WS}/y" in second.files.made_dirs


async def test_restoring_fresh_without_lazy_starts_a_new_sandbox(
    creator: _Creator,
) -> None:
    async with e2b_environment(allowed_roots=[WS]) as env:
        await env.restore(FRESH_SNAPSHOT_REF)
        assert env.started

    assert len(creator.created) == 2
    assert creator.created[1].get("template") is None


async def test_pause_before_start_is_refused(creator: _Creator) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        with pytest.raises(RuntimeError, match="never started"):
            await env.pause()


class _FailingFiles(FakeFiles):
    async def make_dir(self, path: str) -> Any:
        raise OSError("disk full")


async def test_failed_setup_fails_fast_until_reentered(creator: _Creator) -> None:
    broken = FakeSandbox(files=_FailingFiles())
    creator._queue.append(broken)
    env = e2b_environment(allowed_roots=[WS], lazy=True)

    async with env:
        with pytest.raises(OSError, match="disk full"):
            await env.file_backend.mkdir(Path(f"{WS}/x"))
        assert broken.killed
        assert not env.started
        # No new (billed) sandbox per call while the setup keeps failing.
        with pytest.raises(RuntimeError, match="failed to start") as failed:
            await env.file_backend.mkdir(Path(f"{WS}/x"))
        assert isinstance(failed.value.__cause__, OSError)
        assert len(creator.created) == 1

    async with env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        assert env.started

    assert len(creator.created) == 2


async def test_restoring_fresh_retries_a_failed_start(creator: _Creator) -> None:
    creator._queue.append(FakeSandbox(files=_FailingFiles()))

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        with pytest.raises(OSError, match="disk full"):
            await env.file_backend.mkdir(Path(f"{WS}/x"))
        await env.restore(FRESH_SNAPSHOT_REF)
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        assert env.started

    assert len(creator.created) == 2


async def test_restore_and_resume_after_exit_are_refused(creator: _Creator) -> None:
    env = e2b_environment(allowed_roots=[WS])
    async with env:
        pass

    # Nothing would kill a sandbox they created or connected after the exit.
    with pytest.raises(RuntimeError, match="not entered"):
        await env.restore("snap-1")
    with pytest.raises(RuntimeError, match="not entered"):
        await env.restore(FRESH_SNAPSHOT_REF)
    with pytest.raises(RuntimeError, match="not entered"):
        await env.resume("sbx_paused")
    assert len(creator.created) == 1
    assert creator.connected == []


class _KillFailsSandbox(FakeSandbox):
    async def kill(self) -> Any:
        raise RuntimeError("kill failed")


async def test_restore_logs_a_failed_kill_of_the_previous_sandbox(
    creator: _Creator, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=e2b_handle.__name__)
    creator._queue.append(_KillFailsSandbox(sandbox_id="first"))

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        await env.restore("snap-1")

    assert [r.args for r in caplog.records] == [("first",)]


async def test_restoring_fresh_over_a_wrapped_sandbox_is_refused(
    creator: _Creator,
) -> None:
    wrapped = FakeSandbox(sandbox_id="wrapped")
    env = E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", wrapped),
        policy=SandboxPolicy(allowed_roots=(Path(WS),)),
    )

    async with env:
        # It has no create params to start one from, and would not own it.
        with pytest.raises(RuntimeError, match="from_sandbox"):
            await env.restore(FRESH_SNAPSHOT_REF)
        assert env.started

    assert creator.created == []
    assert not wrapped.killed


async def test_config_lazy_defers_the_start(creator: _Creator) -> None:
    env = await E2BEnvironmentConfig(lazy=True).build()

    async with env:
        assert not env.started
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        assert env.started

    assert len(creator.created) == 1


async def test_config_passes_the_label(creator: _Creator) -> None:
    env = await E2BEnvironmentConfig(label="prod").build()

    async with env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))

    assert creator.created[0]["metadata"] == {"grasp-agents": "prod"}


async def test_snapshots_get_unique_names_that_carry_the_label_and_time(
    creator: _Creator,
) -> None:
    sandbox = FakeSandbox(sandbox_id="first")
    creator._queue.append(sandbox)

    async with e2b_environment(allowed_roots=[WS], lazy=True, label="prod") as env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        before = dt.datetime.now(dt.UTC).replace(microsecond=0)
        await env.snapshot()
        await env.snapshot()

    first, second = sandbox.snapshot_names
    assert first != second
    for name in (first, second):
        assert name is not None
        found = re.fullmatch(r"grasp-agents-prod-(\d{8}t\d{6})-[0-9a-f]{8}", name)
        assert found is not None, name
        taken = dt.datetime.strptime(f"{found.group(1)}+0000", "%Y%m%dt%H%M%S%z")
        assert abs(taken - before) < dt.timedelta(minutes=1)


# --- lifecycle changes while a start is in flight ---------------------------


async def test_exit_waits_for_a_start_and_kills_its_sandbox(
    slow_creator: _SlowCreator,
) -> None:
    env = e2b_environment(allowed_roots=[WS], lazy=True)
    async with env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
        await slow_creator.creating.wait()
        # Finish the create only once the exit is under way.
        asyncio.get_running_loop().call_soon(slow_creator.gate.set)

    # Exit returned only after the start finished and killed its sandbox.
    (started,) = slow_creator.made
    assert started.killed
    assert not env.started
    with pytest.raises(RuntimeError, match="while it was starting"):
        await first_use


async def test_restore_waits_for_a_start_in_flight(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
        await slow_creator.creating.wait()
        restoring = asyncio.create_task(env.restore("snap-1"))
        await asyncio.sleep(0)
        slow_creator.gate.set()
        await asyncio.gather(first_use, restoring)
        await env.file_backend.mkdir(Path(f"{WS}/b"))

    started, restored = slow_creator.made
    assert started.killed
    assert f"{WS}/b" in restored.files.made_dirs


async def test_pause_waits_for_a_start_in_flight(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
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
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
        await slow_creator.creating.wait()
        resuming = asyncio.create_task(env.resume("sbx_paused"))
        await asyncio.sleep(0)
        slow_creator.gate.set()
        await asyncio.gather(first_use, resuming)
        await env.file_backend.mkdir(Path(f"{WS}/b"))

    (resumed,) = slow_creator.connected
    assert f"{WS}/b" in resumed.files.made_dirs


async def test_cancelled_restore_kills_the_sandbox_it_was_creating(
    slow_creator: _SlowCreator,
) -> None:
    slow_creator.gated = {"snap-1"}
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{WS}/a"))
        restoring = asyncio.create_task(env.restore("snap-1"))
        await slow_creator.creating.wait()
        restoring.cancel()
        with pytest.raises(asyncio.CancelledError):
            await restoring
        slow_creator.gate.set()
        async with env._holder.lock:  # held until the create finishes
            pass

        started, created = slow_creator.made
        assert created.killed
        assert not started.killed
        await env.file_backend.mkdir(Path(f"{WS}/b"))
        assert f"{WS}/b" in started.files.made_dirs


async def test_cancelled_resume_lets_go_of_the_sandbox_it_was_connecting(
    slow_creator: _SlowCreator,
) -> None:
    slow_creator.gate_connects = True
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        resuming = asyncio.create_task(env.resume("sbx_paused"))
        await slow_creator.creating.wait()
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resuming
        slow_creator.gate.set()
        async with env._holder.lock:  # held until the connect finishes
            pass

        (resumed,) = slow_creator.connected
        assert resumed.killed
        assert not env.started


async def test_cancelled_resume_of_the_live_sandbox_keeps_it(
    slow_creator: _SlowCreator,
) -> None:
    slow_creator.gated = set()
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{WS}/a"))
        (started,) = slow_creator.made
        slow_creator.reconnects[started.sandbox_id] = FakeSandbox(
            sandbox_id=started.sandbox_id, files=started.files
        )
        slow_creator.gate_connects = True
        resuming = asyncio.create_task(env.resume())
        await slow_creator.creating.wait()
        resuming.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resuming
        slow_creator.gate.set()
        async with env._holder.lock:
            pass

        (reconnected,) = slow_creator.connected
        assert not reconnected.killed
        assert not started.killed
        await env.file_backend.mkdir(Path(f"{WS}/b"))
        assert f"{WS}/b" in started.files.made_dirs


async def test_resuming_another_sandbox_kills_the_one_it_replaces(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    creator._queue.append(first)
    other = FakeSandbox(sandbox_id="other", commands=ShellCommands())
    creator.reconnects["other"] = other
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        session = await holder.get(env.exec_backend)
        await _run(session, "true")
        await env.resume("other")
        assert first.killed
        assert session.closed

        await _run(await holder.get(env.exec_backend), "true")
        assert holder.take_reset()

    assert other.killed  # by the exit


async def test_resuming_another_sandbox_pauses_the_one_it_replaces_with_pause_on_exit(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first")
    creator._queue.append(first)

    async with e2b_environment(
        allowed_roots=[WS], lazy=True, pause_on_exit=True
    ) as env:
        await env.file_backend.mkdir(Path(f"{WS}/x"))
        await env.resume("other")
        assert first.paused
        assert not first.killed


async def test_resuming_another_sandbox_leaves_one_it_does_not_own(
    creator: _Creator,
) -> None:
    wrapped = FakeSandbox(sandbox_id="wrapped")
    env = E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", wrapped),
        policy=SandboxPolicy(allowed_roots=(Path(WS),)),
    )

    async with env:
        await env.resume("other")

    assert not wrapped.killed
    assert not wrapped.paused


async def test_start_finding_the_handle_filled_kills_its_sandbox(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        first_use = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
        await slow_creator.creating.wait()
        env._holder.sandbox = cast("AsyncSandbox", FakeSandbox(sandbox_id="other"))
        slow_creator.gate.set()
        with pytest.raises(RuntimeError, match="while it was starting"):
            await first_use

    (started,) = slow_creator.made
    assert started.killed


async def test_a_start_outlives_its_cancelled_callers(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        first = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
        await slow_creator.creating.wait()
        first.cancel()
        second = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/b")))
        await asyncio.sleep(0)  # it joins the start in flight
        second.cancel()
        for call in (first, second):
            with pytest.raises(asyncio.CancelledError):
                await call
        slow_creator.gate.set()
        async with env._holder.lock:  # held until the start finishes
            pass

        # Kept for the next call, not killed for want of a caller.
        assert env.started
        await env.file_backend.mkdir(Path(f"{WS}/c"))

    (started,) = slow_creator.made
    assert f"{WS}/c" in started.files.made_dirs
    assert started.killed  # by the exit


async def test_cancelling_one_first_call_leaves_the_others_their_sandbox(
    slow_creator: _SlowCreator,
) -> None:
    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        cancelled = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/a")))
        waiting = asyncio.create_task(env.file_backend.mkdir(Path(f"{WS}/b")))
        await slow_creator.creating.wait()
        cancelled.cancel()
        slow_creator.gate.set()
        await waiting

        (started,) = slow_creator.made
        assert not started.killed
        assert f"{WS}/b" in started.files.made_dirs


# --- sessions and kernels across a sandbox swap ------------------------------


async def _run(session: ExecSession, command: str, **kwargs: float) -> ExecResult:
    result: ExecResult | None = None
    async for item in session.run(command, **kwargs):
        if isinstance(item, ExecResult):
            result = item
    assert result is not None
    return result


async def _execute(kernel: KernelSession, code: str) -> CellResult:
    result: CellResult | None = None
    async for item in kernel.execute(code):
        if isinstance(item, CellResult):
            result = item
    assert result is not None
    return result


async def test_session_reopens_on_the_next_sandbox_after_a_fresh_restore(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    second = FakeSandbox(sandbox_id="second", commands=ShellCommands())
    creator._queue.extend([first, second])
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        session = await holder.get(env.exec_backend)
        await _run(session, "cd /tmp")
        await env.restore(FRESH_SNAPSHOT_REF)
        assert session.closed

        reopened = await holder.get(env.exec_backend)
        assert holder.take_reset()  # BashSession tells the model its shell reset
        assert (await _run(reopened, "pwd")).returncode == 0

    assert len(first.commands.stdin_writes) == 1
    assert len(second.commands.stdin_writes) == 1
    # A deadline on the shell's stream would cancel whichever task opened it.
    assert first.commands.shell_timeouts == [0]
    assert second.commands.shell_timeouts == [0]


async def test_stale_session_never_touches_the_next_sandbox(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    second = FakeSandbox(sandbox_id="second", commands=ShellCommands())
    creator._queue.extend([first, second])

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        stale = await env.exec_backend.open_session()
        await _run(stale, "cd /tmp")
        await env.restore(FRESH_SNAPSHOT_REF)
        # Another loop's shell starts first on the next sandbox, so it gets
        # the pid the stale shell had on the old one.
        other = await env.exec_backend.open_session()
        await _run(other, "true")

        with pytest.raises(RuntimeError, match="closed"):
            await _run(stale, "rm -rf build", timeout=0.1)
        await stale.close()

    assert [pid for pid, _ in second.commands.stdin_writes] == [1]
    assert second.commands.killed == []


async def test_closing_a_session_kills_its_shell_and_disconnects(
    creator: _Creator,
) -> None:
    sandbox = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    creator._queue.append(sandbox)

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        session = await env.exec_backend.open_session()
        await _run(session, "true")
        await session.close()

    assert sandbox.commands.killed == [1]
    assert sandbox.commands.shells[1].disconnected


async def test_a_replaced_session_is_disconnected_without_a_kill(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    restored = FakeSandbox(sandbox_id="restored", commands=ShellCommands())
    creator._queue.extend([first, restored])
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await _run(await holder.get(env.exec_backend), "true")
        await env.restore("snap-1")
        await _run(await holder.get(env.exec_backend), "true")
        # The shell has no stream deadline: only a disconnect ends its stream.
        assert first.commands.shells[1].disconnected

        await holder.close()
        assert restored.commands.shells[1].disconnected

    assert first.commands.killed == []
    assert restored.commands.killed == [1]


async def test_session_reattaches_its_output_after_a_snapshot(
    creator: _Creator,
) -> None:
    sandbox = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    creator._queue.append(sandbox)
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await _run(await holder.get(env.exec_backend), "cd /tmp")
        await env.snapshot()

        result = await _run(await holder.get(env.exec_backend), "pwd", timeout=0.5)
        assert result.reason is TerminationReason.EXIT
        assert not holder.take_reset()  # the same shell: its state is kept

    assert sandbox.commands.connects == [(1, 0)]
    assert [pid for pid, _ in sandbox.commands.stdin_writes] == [1, 1]


async def test_session_reattaches_its_output_after_pause_and_resume(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    creator._queue.append(first)
    # resume() reconnects through a new client to the same sandbox and shells.
    creator.reconnects["first"] = FakeSandbox(
        sandbox_id="first", commands=first.commands
    )
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await _run(await holder.get(env.exec_backend), "cd /tmp")
        await env.pause()
        await env.resume()

        result = await _run(await holder.get(env.exec_backend), "pwd", timeout=0.5)
        assert result.reason is TerminationReason.EXIT
        assert not holder.take_reset()

    assert first.commands.connects == [(1, 0)]


async def test_session_replaces_a_shell_whose_output_it_cannot_reattach(
    creator: _Creator,
) -> None:
    commands = ShellCommands(refuse_connect=True)
    creator._queue.append(FakeSandbox(sandbox_id="first", commands=commands))
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await _run(await holder.get(env.exec_backend), "cd /tmp")
        await env.snapshot()

        result = await _run(await holder.get(env.exec_backend), "pwd", timeout=0.5)
        assert result.reason is TerminationReason.EXIT
        assert holder.take_reset()  # BashSession tells the model its shell reset

    assert commands.killed[0] == 1
    assert commands.shells[1].disconnected
    assert [pid for pid, _ in commands.stdin_writes] == [1, 2]


async def test_kernel_reopens_on_the_next_sandbox_after_a_fresh_restore(
    creator: _Creator,
) -> None:
    first, second = CodeSandbox(sandbox_id="first"), CodeSandbox(sandbox_id="second")
    creator._queue.extend([first, second])
    holder = KernelHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        kernel = await holder.get(env.exec_backend)
        await _execute(kernel, "x = 1")
        await env.restore(FRESH_SNAPSHOT_REF)
        assert kernel.closed

        reopened = await holder.get(env.exec_backend)
        assert (await _execute(reopened, "x = 2")).status == "ok"
        assert holder.take_reset()  # RunCell tells the model its variables are gone

    assert second.ran == [("second-ctx0", "x = 2")]


async def test_kernel_reattaches_after_a_snapshot_rollback_without_a_reset(
    creator: _Creator,
) -> None:
    first = CodeSandbox(sandbox_id="first")
    # The snapshot keeps the kernel's memory, so its context lives on there.
    restored = CodeSandbox(sandbox_id="restored", contexts=("first-ctx0",))
    creator._queue.extend([first, restored])
    holder = KernelHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        kernel = await holder.get(env.exec_backend)
        await _execute(kernel, "x = 1")
        await env.restore("snap-1")
        holder.rebind(kernel.context_id)  # what rollback_to_step does

        reattached = await holder.get(env.exec_backend)
        assert (await _execute(reattached, "x")).status == "ok"
        assert not holder.take_reset()

    assert restored.ran == [("first-ctx0", "x")]


async def test_rollback_past_a_kernel_opens_a_fresh_one(creator: _Creator) -> None:
    first = CodeSandbox(sandbox_id="first")
    restored = CodeSandbox(sandbox_id="restored")  # the snapshot predates the kernel
    creator._queue.extend([first, restored])
    holder = KernelHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        kernel = await holder.get(env.exec_backend)
        await _execute(kernel, "x = 1")
        await env.restore("snap-0")
        holder.rebind(None)  # what rollback_to_step does for this boundary
        # The next checkpoint records this id with the restored snapshot.
        assert holder.context_id is None

        reopened = await holder.get(env.exec_backend)
        assert (await _execute(reopened, "x = 2")).status == "ok"
        # The rolled-back transcript never used the kernel: nothing was lost.
        assert not holder.take_reset()

    assert restored.ran == [("restored-ctx0", "x = 2")]


async def test_seeded_kernel_opens_a_fresh_context_the_sandbox_lacks(
    creator: _Creator,
) -> None:
    first = CodeSandbox(sandbox_id="first", contexts=("ctx-x",))
    second = CodeSandbox(sandbox_id="second")
    creator._queue.extend([first, second])
    holder = KernelHolder(context_id="ctx-x")  # what a cold resume seeds
    env = e2b_environment(allowed_roots=[WS], lazy=True)

    async with env:
        kernel = await holder.get(env.exec_backend)
        assert (await _execute(kernel, "x = 1")).status == "ok"
        assert not holder.take_reset()

    # Re-entered without a rebind: the new sandbox never had the seeded context.
    async with env:
        reopened = await holder.get(env.exec_backend)
        assert (await _execute(reopened, "x")).status == "ok"
        assert holder.take_reset()  # RunCell tells the model its variables are gone
        assert (await _execute(reopened, "x")).status == "ok"
        assert not holder.take_reset()

    assert first.ran == [("ctx-x", "x = 1")]
    assert second.ran == [("second-ctx0", "x"), ("second-ctx0", "x")]


async def test_kernel_survives_a_reconnect_to_the_same_sandbox(
    creator: _Creator,
) -> None:
    first = CodeSandbox(sandbox_id="first")
    # resume() reconnects through a new client object for the same sandbox.
    reconnected = CodeSandbox(sandbox_id="first")
    reconnected.contexts = first.contexts
    creator._queue.append(first)
    creator.reconnects["first"] = reconnected
    holder = KernelHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        kernel = await holder.get(env.exec_backend)
        await _execute(kernel, "x = 1")
        await env.pause()
        await env.resume()

        assert await holder.get(env.exec_backend) is kernel
        assert (await _execute(kernel, "x")).status == "ok"
        assert not first.killed

    assert reconnected.ran == [("first-ctx0", "x")]


# --- a sandbox E2B removed ---------------------------------------------------


@pytest.mark.parametrize("lazy", [True, False])
async def test_a_sandbox_e2b_removed_is_replaced_on_the_next_call(
    creator: _Creator, lazy: bool
) -> None:
    first, second = FakeSandbox(sandbox_id="first"), FakeSandbox(sandbox_id="second")
    creator._queue.extend([first, second])

    async with e2b_environment(allowed_roots=[WS], lazy=lazy) as env:
        await env.file_backend.mkdir(Path(f"{WS}/a"))
        first.expire()  # its keep-alive timeout, say

        # The call that finds out fails plainly; it is not retried in secret.
        with pytest.raises(E2BSandboxExpiredError, match="first expired"):
            await env.file_backend.mkdir(Path(f"{WS}/b"))
        assert not env.started

        await env.file_backend.mkdir(Path(f"{WS}/c"))
        assert env.started

    assert len(creator.created) == 2
    assert f"{WS}/b" not in second.files.made_dirs
    assert f"{WS}/c" in second.files.made_dirs
    assert second.killed


async def test_a_command_in_a_removed_sandbox_fails_plainly(creator: _Creator) -> None:
    first = FakeSandbox(sandbox_id="first")
    creator._queue.append(first)

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.exec_backend.execute("true")
        first.expire()
        with pytest.raises(E2BSandboxExpiredError):
            await env.exec_backend.execute("true")
        assert (await env.exec_backend.execute("true")).returncode == 0

    assert len(creator.created) == 2


async def test_a_shell_in_a_removed_sandbox_reopens_with_a_reset(
    creator: _Creator,
) -> None:
    first = FakeSandbox(sandbox_id="first", commands=ShellCommands())
    second = FakeSandbox(sandbox_id="second", commands=ShellCommands())
    creator._queue.extend([first, second])
    holder = BashSessionHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        session = await holder.get(env.exec_backend)
        await _run(session, "cd /tmp")
        first.expire()
        with pytest.raises(E2BSandboxExpiredError):
            await _run(session, "pwd")
        assert session.closed

        assert (await _run(await holder.get(env.exec_backend), "pwd")).returncode == 0
        assert holder.take_reset()

    assert [pid for pid, _ in second.commands.stdin_writes] == [1]


async def test_a_kernel_in_a_removed_sandbox_reopens_with_a_reset(
    creator: _Creator,
) -> None:
    first, second = CodeSandbox(sandbox_id="first"), CodeSandbox(sandbox_id="second")
    creator._queue.extend([first, second])
    holder = KernelHolder()

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        kernel = await holder.get(env.exec_backend)
        await _execute(kernel, "x = 1")
        first.expire()
        with pytest.raises(E2BSandboxExpiredError):
            await _execute(kernel, "x")
        assert kernel.closed

        reopened = await holder.get(env.exec_backend)
        assert (await _execute(reopened, "x = 2")).status == "ok"
        assert holder.take_reset()

    assert second.ran == [("second-ctx0", "x = 2")]


async def test_a_removed_sandbox_the_environment_cannot_replace_stays_an_error(
    creator: _Creator,
) -> None:
    wrapped = FakeSandbox(sandbox_id="wrapped")
    env = E2BEnvironment.from_sandbox(
        cast("AsyncSandbox", wrapped),
        policy=SandboxPolicy(allowed_roots=(Path(WS),)),
    )

    async with env:
        wrapped.expire()
        with pytest.raises(E2BSandboxExpiredError, match="cannot start another"):
            await env.file_backend.mkdir(Path(f"{WS}/a"))
        with pytest.raises(E2BSandboxExpiredError, match="cannot start another"):
            await env.file_backend.mkdir(Path(f"{WS}/b"))

    assert creator.created == []


async def test_a_failure_the_sandbox_survived_keeps_it(creator: _Creator) -> None:
    sandbox = FakeSandbox(sandbox_id="first")
    creator._queue.append(sandbox)

    async with e2b_environment(allowed_roots=[WS], lazy=True) as env:
        await env.file_backend.mkdir(Path(f"{WS}/a"))
        with pytest.raises(KeyError):  # the fake's missing file
            await env.file_backend.read_text(Path(f"{WS}/missing.txt"))
        assert env.started

    assert len(creator.created) == 1
