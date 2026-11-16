"""
Live E2B checks of the sandbox lifecycle that the fakes in ``_e2b_fakes.py``
cannot prove: what E2B itself does on a pause, a snapshot, a restore, a resume
and a kill.

Needs ``e2b`` installed, ``E2B_API_KEY`` and network; run unsandboxed:
``uv run pytest -m integration tests/sandbox/test_e2b_live_lifecycle.py``.

Every sandbox a test creates carries a metadata tag of its own (key
:data:`TAG_KEY`), and its environments a label of their own, which names
their snapshots. The test kills every sandbox with that tag, and deletes every
snapshot named with that label or recorded by the test, in a ``finally``.
Sandboxes also get a short keep-alive (:data:`KEEP_ALIVE`), so one a failed
cleanup leaves behind dies soon, unless paused; find leftovers by listing
sandboxes with that metadata key and snapshots named ``grasp-agents-t...``.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.sandbox import (
    FRESH_SNAPSHOT_REF,
    E2BSandboxExpiredError,
    e2b_environment,
)
from grasp_agents.sandbox.exec_backend import TerminationReason
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import BaseTool
from grasp_agents.tools.bash_common import BashInput, BashResult
from grasp_agents.tools.bash_session import BashSession, BashSessionHolder
from grasp_agents.tools.code_interpreter import RunPython, RunPythonInput
from grasp_agents.types.content import InputText
from tests._helpers import _make_agent_ctx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from grasp_agents.sandbox.e2b.environment import E2BEnvironment

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

_HAS_E2B = importlib.util.find_spec("e2b") is not None
# bool, not the key itself — a raw key in a module global can surface in logs.
_HAS_E2B_KEY = bool(os.getenv("E2B_API_KEY"))
_live = pytest.mark.skipif(
    not (_HAS_E2B and _HAS_E2B_KEY), reason="needs e2b installed + E2B_API_KEY"
)

WS = "/home/user/workspace"
TAG_KEY = "grasp-agents-live-test"
KEEP_ALIVE = 180  # seconds; a leftover sandbox dies on its own after this
PAUSE_KEEP_ALIVE = 20  # seconds; short, so a test can wait past it


class LiveRun:
    """The sandboxes and snapshots one live test creates, and their cleanup."""

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex
        self.label = f"t{self.tag[:12]}"
        self.snapshots: list[str] = []

    def env(
        self, *, sandbox_timeout: int = KEEP_ALIVE, **kwargs: Any
    ) -> E2BEnvironment:
        return e2b_environment(
            allowed_roots=[WS],
            metadata={TAG_KEY: self.tag},
            sandbox_timeout=sandbox_timeout,
            label=self.label,
            **kwargs,
        )

    async def create_sandbox(self) -> Any:
        """A sandbox outside any environment, tagged like the environment's."""
        from e2b import AsyncSandbox

        return await AsyncSandbox.create(
            metadata={TAG_KEY: self.tag}, timeout=KEEP_ALIVE
        )

    async def tagged(self) -> set[str]:
        """Ids of this test's sandboxes E2B still has, running or paused."""
        from e2b import (
            AsyncSandbox,
            SandboxQuery,
        )

        pages = AsyncSandbox.list(query=SandboxQuery(metadata={TAG_KEY: self.tag}))
        ids: set[str] = set()
        while pages.has_next:
            ids.update(info.sandbox_id for info in await pages.next_items())
        return ids

    async def named(self) -> set[str]:
        """Ids of the snapshots this test's environments named."""
        from e2b import AsyncSandbox

        prefix = f"grasp-agents-{self.label}-"
        pages = AsyncSandbox.list_snapshots()
        ids: set[str] = set()
        while pages.has_next:
            for snapshot in await pages.next_items():
                names = [name.rsplit("/", 1)[-1] for name in snapshot.names]
                if any(name.startswith(prefix) for name in names):
                    ids.add(snapshot.snapshot_id)
        return ids

    async def cleanup(self) -> None:
        from e2b import AsyncSandbox

        for sandbox_id in await self.tagged():
            with contextlib.suppress(Exception):
                await AsyncSandbox.kill(sandbox_id)
        for snapshot_id in {*self.snapshots, *await self.named()}:
            with contextlib.suppress(Exception):
                await AsyncSandbox.delete_snapshot(snapshot_id)


async def until(check: Callable[[], Awaitable[bool]], *, timeout: float = 30.0) -> None:
    """Poll ``check`` until it holds: E2B's listing lags behind creates and kills."""
    deadline = time.monotonic() + timeout
    while not await check():
        if time.monotonic() > deadline:
            raise AssertionError(f"still false after {timeout:g}s")
        await asyncio.sleep(1.0)


def live_id(env: E2BEnvironment) -> str:
    sandbox = env._holder.sandbox
    assert sandbox is not None
    return str(sandbox.sandbox_id)


async def paused_by_e2b(sandbox_id: str) -> None:
    """Wait until E2B paused the sandbox at the end of its keep-alive."""
    from e2b import AsyncSandbox, SandboxState

    async def paused() -> bool:
        info = await AsyncSandbox.get_info(sandbox_id)
        return info.state == SandboxState.PAUSED

    await until(paused, timeout=PAUSE_KEEP_ALIVE + 30)


async def keep_alive_left(sandbox_id: str) -> float:
    from e2b import AsyncSandbox

    info = await AsyncSandbox.get_info(sandbox_id)
    return info.end_at.timestamp() - time.time()


class EchoLive(BaseTool[BashInput, str, Any]):
    """A tool that does nothing, so a run has a tool round to checkpoint."""

    def __init__(self) -> None:
        super().__init__(name="echo_live", description="Echoes")

    async def _run(
        self,
        inp: BashInput,
        *,
        ctx: Any = None,
        exec_id: str | None = None,
        progress_callback: Any = None,
        path: Any = None,
        agent_ctx: Any = None,
    ) -> str:
        return inp.command


class Agent:
    """The per-agent state the shell and kernel tools read, over ``env``."""

    def __init__(self, env: E2BEnvironment) -> None:
        self.ctx = SessionContext[None](environment=env)
        self.agent_ctx = _make_agent_ctx()
        self.agent_ctx.session_holder = BashSessionHolder()

    async def shell(self, command: str, *, timeout: float = 30) -> BashResult:
        return await BashSession()._run(
            BashInput(command=command, timeout=timeout),
            ctx=self.ctx,
            agent_ctx=self.agent_ctx,
        )

    async def python(self, code: str) -> str:
        parts = await RunPython()._run(
            RunPythonInput(code=code), ctx=self.ctx, agent_ctx=self.agent_ctx
        )
        return "".join(part.text for part in parts if isinstance(part, InputText))


# --- lazy start --------------------------------------------------------------


@_live
async def test_a_lazy_start_creates_one_sandbox_and_exit_kills_it() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            assert await run.tagged() == set()

            await asyncio.gather(
                *(env.file_backend.mkdir(Path(f"{WS}/d{i}")) for i in range(4)),
                env.exec_backend.execute("true"),
            )
            assert env.started

            async def only_the_live_one() -> bool:
                return await run.tagged() == {live_id(env)}

            await until(only_the_live_one)

        async def none_left() -> bool:
            return await run.tagged() == set()

        await until(none_left)
    finally:
        await run.cleanup()


@_live
async def test_restoring_the_fresh_ref_starts_over_on_the_next_call() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            note = Path(f"{WS}/note.txt")
            await env.file_backend.write_bytes(note, b"before", mode=0o644)
            first = live_id(env)

            await env.restore(FRESH_SNAPSHOT_REF)
            assert not env.started

            async def first_is_gone() -> bool:
                return first not in await run.tagged()

            await until(first_is_gone)
            assert not await env.file_backend.exists(note)
            assert live_id(env) != first
    finally:
        await run.cleanup()


# --- commands ----------------------------------------------------------------


@_live
async def test_a_command_past_its_timeout_does_not_cancel_the_caller() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            await env.exec_backend.execute("true")  # start outside the timing
            t0 = time.monotonic()
            result = await env.exec_backend.execute("echo started; sleep 60", timeout=3)

            assert result.reason is TerminationReason.OVERALL_TIMEOUT
            assert "started" in result.stdout
            assert time.monotonic() - t0 < 20
            task = asyncio.current_task()
            assert task is not None
            assert task.cancelling() == 0
            await asyncio.sleep(0.5)  # a stray cancel would land here
            assert (await env.exec_backend.execute("true")).returncode == 0
    finally:
        await run.cleanup()


@_live
@pytest.mark.xfail(
    strict=True,
    reason=(
        "E2B's kill reaches the shell it runs the command in (/bin/bash -l -c), "
        "not the processes that shell started; docs/sandbox.md, Limitations"
    ),
)
async def test_a_timeout_stops_the_processes_the_command_started() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            await env.exec_backend.execute("true")
            await env.exec_backend.execute("sleep 41.37; echo done", timeout=2)
            await asyncio.sleep(1)
            left = await env.exec_backend.execute(
                "ps -eo args | grep -c '^sleep 41.37' || true"
            )
            assert left.stdout.strip() == "0"
    finally:
        await run.cleanup()


# --- shells and kernels across a pause, a snapshot and a restore ------------


@_live
async def test_a_shell_answers_after_a_snapshot_and_a_pause() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            agent = Agent(env)
            await agent.shell("cd /tmp && export MARK=kept")
            run.snapshots.append(await env.snapshot())
            after_snapshot = await agent.shell("echo $MARK; pwd")
            await agent.shell("cd /tmp && export MARK=kept")
            await env.pause()
            await env.resume()
            after_resume = await agent.shell("echo $MARK; pwd")

        for result in (after_snapshot, after_resume):
            # The command ran: it reports its output, never a timeout.
            assert result.reason is TerminationReason.EXIT
            assert result.returncode == 0
            reattached = result.stdout.split() == ["kept", "/tmp"]
            reopened = "shell session was reset" in result.stderr
            assert reattached or reopened, result
    finally:
        await run.cleanup()


@_live
async def test_a_kernel_keeps_its_variables_across_pause_and_resume() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True, code_interpreter=True) as env:
            agent = Agent(env)
            assert "status=ok" in await agent.python("x = 41")
            await env.pause()
            await env.resume()

            out = await agent.python("print(x + 1)")
            assert "42" in out
            assert "kernel was restarted" not in out
    finally:
        await run.cleanup()


@_live
async def test_a_kernel_restored_past_opens_a_fresh_context_with_a_notice() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True, code_interpreter=True) as env:
            await env.exec_backend.execute("true")
            before_kernel = await env.snapshot()
            run.snapshots.append(before_kernel)
            agent = Agent(env)
            assert "status=ok" in await agent.python("x = 1")
            holder = agent.agent_ctx.ipy_kernel_holder
            assert holder is not None
            # Seeded as a cold resume would be, with a context the snapshot lacks.
            holder.rebind(holder.context_id)

            await env.restore(before_kernel)

            out = await agent.python("print('x' in globals())")
            assert "status=ok" in out
            assert "False" in out
            assert "kernel was restarted" in out
    finally:
        await run.cleanup()


# --- a sandbox E2B removed ---------------------------------------------------


@_live
async def test_a_sandbox_killed_from_outside_is_replaced_on_the_call_after() -> None:
    from e2b import AsyncSandbox

    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            ctx = SessionContext[None](environment=env)
            agent_ctx = _make_agent_ctx()
            agent_ctx.session_holder = BashSessionHolder()
            shell = BashSession()
            note = Path(f"{WS}/note.txt")
            await env.file_backend.write_bytes(note, b"before", mode=0o644)
            await shell._run(BashInput(command="cd /tmp"), ctx=ctx, agent_ctx=agent_ctx)
            first = live_id(env)

            await AsyncSandbox.kill(first)

            with pytest.raises(E2BSandboxExpiredError, match="expired"):
                await env.file_backend.read_text(note)
            assert not env.started

            assert not await env.file_backend.exists(note)
            assert live_id(env) != first
            result = await shell._run(
                BashInput(command="pwd"), ctx=ctx, agent_ctx=agent_ctx
            )
            assert result.returncode == 0
            assert "shell session was reset" in result.stderr
            assert result.stdout.strip() == WS
    finally:
        await run.cleanup()


# --- a sandbox E2B paused at its keep-alive ----------------------------------


@_live
async def test_calls_resume_a_sandbox_e2b_paused_and_find_its_state() -> None:
    run = LiveRun()
    try:
        async with run.env(
            lazy=True,
            code_interpreter=True,
            sandbox_timeout=PAUSE_KEEP_ALIVE,
            on_timeout="pause",
        ) as env:
            agent = Agent(env)
            note = Path(f"{WS}/note.txt")
            await env.file_backend.write_bytes(note, b"kept", mode=0o644)
            await agent.shell("cd /tmp && export MARK=kept")
            assert "status=ok" in await agent.python("x = 41")
            sandbox_id = live_id(env)

            # After each pause another kind of call is the first to find it.
            await paused_by_e2b(sandbox_id)
            assert (await env.file_backend.read_text(note))[0] == "kept"

            await paused_by_e2b(sandbox_id)
            shell = await agent.shell("echo $MARK; pwd")
            assert shell.reason is TerminationReason.EXIT
            assert shell.stdout.split() == ["kept", "/tmp"]
            assert "shell session was reset" not in shell.stderr

            await paused_by_e2b(sandbox_id)
            out = await agent.python("print(x + 1)")
            assert "42" in out
            assert "kernel was restarted" not in out

            await paused_by_e2b(sandbox_id)
            run.snapshots.append(await env.snapshot())
            assert (await env.exec_backend.execute("cat note.txt")).stdout == "kept"
            assert live_id(env) == sandbox_id
    finally:
        await run.cleanup()


@_live
async def test_a_sandbox_e2b_paused_with_auto_resume_keeps_its_keep_alive() -> None:
    run = LiveRun()
    try:
        async with run.env(
            lazy=True,
            code_interpreter=True,
            sandbox_timeout=PAUSE_KEEP_ALIVE,
            on_timeout="pause",
            auto_resume=True,
        ) as env:
            agent = Agent(env)
            await agent.shell("cd /tmp && export MARK=kept")
            assert "status=ok" in await agent.python("x = 41")
            sandbox_id = live_id(env)

            await paused_by_e2b(sandbox_id)
            shell = await agent.shell("echo $MARK; pwd")
            assert shell.reason is TerminationReason.EXIT
            assert shell.stdout.split() == ["kept", "/tmp"]
            assert "shell session was reset" not in shell.stderr
            # Resumed with sandbox_timeout, not E2B's own default for traffic.
            assert await keep_alive_left(sandbox_id) <= PAUSE_KEEP_ALIVE
            assert "42" in await agent.python("print(x + 1)")
    finally:
        await run.cleanup()


# --- network -------------------------------------------------------------------

PYPI = ("pypi.org", "files.pythonhosted.org")


async def reaches(env: E2BEnvironment, host: str) -> bool:
    """Whether a command in the sandbox gets any HTTP answer from ``host``."""
    result = await env.exec_backend.execute(
        f"curl -sS -o /dev/null -w '%{{http_code}}' --max-time 8 https://{host}/ "
        "|| true",
        timeout=30,
    )
    return result.stdout.strip() not in {"", "000"}


@_live
async def test_install_through_an_allowlist_then_cut_the_network() -> None:
    from grasp_agents.sandbox import NetworkPolicy

    run = LiveRun()
    try:
        async with run.env(
            network=NetworkPolicy.ALLOWLIST, allowed_domains=PYPI
        ) as env:
            install = await env.exec_backend.execute(
                "pip install --user --no-deps --no-cache-dir six==1.16.0 && "
                "python3 -c 'import six; print(six.__version__)'",
                timeout=180,
            )
            assert install.returncode == 0, install.stderr
            assert install.stdout.strip().endswith("1.16.0")
            assert await reaches(env, "pypi.org")
            assert not await reaches(env, "example.com")

            await env.update_network(NetworkPolicy.NONE)

            assert env.exec_backend.policy.network is NetworkPolicy.NONE
            assert not await reaches(env, "pypi.org")
            assert not await reaches(env, "example.com")
            # A sandbox restored after the change starts without network too.
            snapshot = await env.snapshot()
            run.snapshots.append(snapshot)
            await env.restore(snapshot)
            assert not await reaches(env, "pypi.org")
    finally:
        await run.cleanup()


# --- what environments leave on the account ---------------------------------


@_live
async def test_sandboxes_are_tagged_and_snapshots_named_and_deletable() -> None:
    from e2b import AsyncSandbox, SandboxException

    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            note = Path(f"{WS}/note.txt")
            await env.file_backend.write_bytes(note, b"kept", mode=0o644)
            info = await AsyncSandbox.get_info(live_id(env))
            assert info.metadata["grasp-agents"] == run.label

            snapshot = await env.snapshot()
            run.snapshots.append(snapshot)
            assert await run.named() == {snapshot}
            await env.restore(snapshot)
            assert (await env.file_backend.read_text(note))[0] == "kept"
            # The restored sandbox was created from it, so E2B keeps it.
            assert not await env.delete_snapshot(snapshot)

        assert await env.delete_snapshot(snapshot)
        assert await run.named() == set()
        with pytest.raises(SandboxException):
            await AsyncSandbox.create(template=snapshot, timeout=30)
    finally:
        await run.cleanup()


@_live
async def test_a_snapshot_outlives_the_one_its_sandbox_was_restored_from() -> None:
    # Release deletes a snapshot once no record holds it, even when a held
    # one was taken in a sandbox restored from it.
    from e2b import AsyncSandbox

    run = LiveRun()
    try:
        async with run.env() as env:
            base = Path(f"{WS}/base.txt")
            await env.file_backend.write_bytes(base, b"from r1", mode=0o644)
            first = await env.snapshot()
            run.snapshots.append(first)
            await env.restore(first)
            await env.file_backend.write_bytes(Path(f"{WS}/top.txt"), b"r2", mode=0o644)
            second = await env.snapshot()
            run.snapshots.append(second)
            # A fresh sandbox, so none is left using the first snapshot.
            await env.restore(FRESH_SNAPSHOT_REF)

            async def first_is_free() -> bool:
                return await env.delete_snapshot(first)

            await until(first_is_free)
            assert await run.named() == {second}

            await env.restore(second)
            assert (await env.file_backend.read_text(base))[0] == "from r1"
            assert (await env.file_backend.read_text(Path(f"{WS}/top.txt")))[0] == "r2"
            assert first not in {s.snapshot_id async for s in _snapshots(AsyncSandbox)}
    finally:
        await run.cleanup()


async def _snapshots(sandbox_cls: Any) -> Any:
    pages = sandbox_cls.list_snapshots()
    while pages.has_next:
        for snapshot in await pages.next_items():
            yield snapshot


@_live
async def test_a_turn_policy_session_keeps_only_the_snapshots_its_records_hold() -> (
    None
):
    from grasp_agents import LLMAgent
    from grasp_agents.durability import InMemoryCheckpointStore, read_fs_snapshot_refs
    from tests._helpers import MockLLM, _text_response, _tool_call_response

    run = LiveRun()
    store = InMemoryCheckpointStore()
    try:
        # Eager: a lazy environment that never starts snapshots nothing.
        async with run.env() as env:
            ctx = SessionContext[None](
                environment=env,
                checkpoint_store=store,
                session_key="live",
                fs_snapshot_policy="turn",
            )
            llm = MockLLM(
                responses_queue=[
                    _tool_call_response("echo_live", '{"text": "hi"}', "c1"),
                    _text_response("done"),
                ]
            )
            agent = LLMAgent[str, str, None](
                name="writer", llm=llm, ctx=ctx, tools=[EchoLive()]
            )
            await agent.run("go")

            held = await read_fs_snapshot_refs(store, "live")
            assert len(held) == 1

            async def only_the_held_one() -> bool:
                return await run.named() == held

            await until(only_the_held_one)
    finally:
        await run.cleanup()


@_live
async def test_a_sweep_removes_what_an_ended_process_left_but_kept_snapshots() -> None:
    from grasp_agents.sandbox import sweep_e2b

    run = LiveRun()
    try:
        async with run.env(pause_on_exit=True) as env:
            left_paused = live_id(env)
            kept = await env.snapshot()
            dropped = await env.snapshot()
            run.snapshots += [kept, dropped]

        async def both_listed() -> bool:
            return await run.named() == {kept, dropped}

        await until(both_listed)
        dry = await sweep_e2b(
            older_than=timedelta(0), keep_snapshots={kept}, label=run.label
        )
        assert [s.sandbox_id for s in dry.killed] == [left_paused]
        assert [s.snapshot_id for s in dry.deleted] == [dropped]
        assert [s.snapshot_id for s in dry.kept] == [kept]
        assert await run.tagged() == {left_paused}

        report = await sweep_e2b(
            older_than=timedelta(0),
            keep_snapshots={kept},
            label=run.label,
            dry_run=False,
        )
        assert [s.sandbox_id for s in report.killed] == [left_paused]
        assert [s.snapshot_id for s in report.deleted] == [dropped]

        async def only_the_kept_one() -> bool:
            return await run.tagged() == set() and await run.named() == {kept}

        await until(only_the_kept_one)
    finally:
        await run.cleanup()


# --- restore -----------------------------------------------------------------


@_live
async def test_cancelled_restore_leaves_no_sandbox_behind() -> None:
    run = LiveRun()
    try:
        async with run.env(lazy=True) as env:
            await env.exec_backend.execute("true")
            snapshot = await env.snapshot()
            run.snapshots.append(snapshot)

            restoring = asyncio.create_task(env.restore(snapshot))
            await asyncio.sleep(0.5)  # E2B has the create request by now
            restoring.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await restoring
            async with env._holder.lock:  # held until the create finishes
                pass

            # Whether the cancel landed during the create or after the swap,
            # one sandbox is left: the one the environment holds.
            held = live_id(env)

            async def only_the_held_one() -> bool:
                return await run.tagged() == {held}

            await until(only_the_held_one)
            assert (await env.exec_backend.execute("true")).returncode == 0
    finally:
        await run.cleanup()


# --- resume ------------------------------------------------------------------


@_live
async def test_resuming_another_sandbox_kills_the_previous_one() -> None:
    run = LiveRun()
    try:
        other = await run.create_sandbox()
        await other.files.make_dir(WS)  # as an environment like this one would
        async with run.env(lazy=True) as env:
            await env.exec_backend.execute("true")
            first = live_id(env)

            await env.resume(str(other.sandbox_id))

            assert (await env.exec_backend.execute("true")).returncode == 0
            assert live_id(env) == other.sandbox_id

            async def first_is_gone() -> bool:
                return first not in await run.tagged()

            await until(first_is_gone)
    finally:
        await run.cleanup()
