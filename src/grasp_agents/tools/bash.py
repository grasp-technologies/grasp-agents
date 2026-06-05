"""
``Bash`` — run a shell command via ``ctx.exec_backend`` — plus the
``BashOutput`` / ``KillBash`` companions for auto-backgrounded commands.

Opt-in and stateless between calls: the tools consume the
:class:`ExecBackend` wired onto :attr:`RunContext.exec_backend` (set it via
:func:`local_environment` or by constructing a backend directly). Without an
exec backend the tools refuse to run — an agent gets no shell access by
default.

Non-interactive contract: no TTY, and each call is a fresh ``/bin/sh -c`` — no
shell state (``cd``, environment, started processes) carries over. This is the
:class:`ExecBackend` contract, uniform across every backend (local subprocess,
Seatbelt/bwrap, Docker, remote); pass ``cd ... &&`` or ``cwd`` for a different
directory. For a persistent shell instead, use
:class:`~grasp_agents.tools.bash_session.BashSession`.

Long-running commands:

* **Heartbeat** — while a command runs, ``Bash`` reports progress through the
  tool progress callback (first beat at ``progress_at``, then every
  ``heartbeat_every`` seconds).
* **Auto-background** — when ``auto_background_at`` is set and a command
  outlives it, ``Bash`` returns early with ``status="backgrounded"`` and a
  ``bash_id``; the command keeps running. Poll it with ``BashOutput``, stop it
  with ``KillBash``; a completion note is injected when it finishes. Build the
  trio with :func:`bash_tools`.

A backgrounded command is tracked as a general, *ephemeral*, non-answer-blocking
task in the agent loop's
:class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager` (the same
manager that runs background subagent tools), carrying the live
:class:`~grasp_agents.tools.bash_common.BackgroundCommand` as its monitor. It is
deliberately **not durable** (no ``TaskRecord``): an OS process does not survive
the host, so a persisted record would lie on resume. The manager is bound per
:class:`AgentLoop` through a ContextVar, so sub-agents and parallel replicas
keep their own background commands; passing an explicit ``manager`` overrides
the ContextVar (standalone use outside an agent loop).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..types.tool import BaseTool, ToolProgressCallback
from .bash_common import (
    DEFAULT_BASH_TIMEOUT,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_PROGRESS_AT,
    LEADING_SLEEP,
    MAX_BASH_TIMEOUT,
    BackgroundCommand,
    BashInput,
    BashResult,
    preview,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..agent.background_tasks import BackgroundTaskManager
    from ..run_context import RunContext

DEFAULT_AUTO_BACKGROUND_AT = 120.0


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class _ManagerBound:
    """
    Mixin for bash tools that the agent loop wires its background task manager
    onto. The manager is otherwise resolved from the tool's own ``_manager``
    (set explicitly at construction, or :meth:`bind_manager`-ed by the loop).
    """

    _manager: BackgroundTaskManager[Any] | None

    @property
    def manager(self) -> BackgroundTaskManager[Any] | None:
        """The bound background task manager, if any."""
        return self._manager

    def bind_manager(self, manager: BackgroundTaskManager[Any]) -> None:
        """Wire ``manager`` unless one was already set explicitly."""
        if self._manager is None:
            self._manager = manager


class Bash(_ManagerBound, BaseTool[BashInput, BashResult, Any]):
    """
    Execute a shell command in the agent's environment via ``ctx.exec_backend``.

    Stateless wrapper around the bound :class:`ExecBackend`; the isolation that
    applies (none / seatbelt / bwrap / docker / e2b / ...) is whatever the
    wired backend provides and is reported back in :attr:`BashResult.backend`.
    Each call is a fresh process — shell state (``cd`` / env / variables) does
    not persist across calls; use
    :class:`~grasp_agents.tools.bash_session.BashSession` for a persistent
    session.

    Args:
        default_timeout: Used when the model passes no ``timeout``.
        max_timeout: Hard clamp on the per-call ``timeout``.
        progress_at: Seconds before the first heartbeat progress report.
        heartbeat_every: Interval between subsequent heartbeats.
        auto_background_at: When set, commands outliving this many seconds
            return early as ``status="backgrounded"`` with a ``bash_id``
            (prefer :func:`bash_tools`, which also wires the ``BashOutput`` /
            ``KillBash`` companions). ``None`` (default) never backgrounds.
            Backgrounding needs a background task manager in scope (the running
            loop's or an explicit one); without one the command runs in the
            foreground to completion or timeout.
        block_leading_sleep: Reject commands whose first statement is a bare
            ``sleep`` — they block the loop and produce nothing.
        manager: Explicit background task manager, overriding the per-agent-loop
            one. Default ``None``: backgrounded commands go to the running
            loop's manager. Pass the same manager to all three tools or to none.
        timeout: Standard per-tool timeout (outer asyncio ceiling).

    """

    name = "Bash"
    description = (
        "Run a shell command in the agent's environment and return its "
        "stdout, stderr, and exit code. Use this for running programs, build "
        "and test commands, and git operations. Prefer the dedicated file "
        "tools (Read / Write / Edit / Delete / Glob / Grep) over shell equivalents "
        "(cat / sed / find / grep) — they are safer and give better output. "
        "Commands run non-interactively (no prompts, no TTY) and share no "
        "state between calls; pass `cd ... &&` if you need a different "
        "directory in one call, or set `cwd`."
    )

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_BASH_TIMEOUT,
        max_timeout: float = MAX_BASH_TIMEOUT,
        progress_at: float = DEFAULT_PROGRESS_AT,
        heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
        auto_background_at: float | None = None,
        block_leading_sleep: bool = True,
        manager: BackgroundTaskManager[Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._progress_at = progress_at
        self._heartbeat_every = heartbeat_every
        self._auto_background_at = auto_background_at
        self._block_leading_sleep = block_leading_sleep
        self._manager = manager
        if auto_background_at is not None:
            self.description += (
                f" Commands running longer than {auto_background_at:g}s are "
                "moved to the background: you get a `bash_id` back, and a "
                "notification is injected when the command finishes — there "
                "is no need to poll for completion. Use BashOutput to "
                "inspect output in the meantime (e.g. to decide whether to "
                "KillBash), and KillBash to stop it."
            )

    async def _run(
        self,
        inp: BashInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        if ctx is None or ctx.exec_backend is None:
            raise ValueError(
                "Bash requires ctx.exec_backend. Wire an ExecBackend on "
                "RunContext (e.g. via local_environment(...)) before running "
                "the agent."
            )
        if self._block_leading_sleep and LEADING_SLEEP.match(inp.command):
            raise ValueError(
                "Blocked: a leading `sleep` stalls the agent loop and "
                "produces no output. Run the actual command (with a timeout) "
                "and poll its result instead."
            )

        requested = inp.timeout if inp.timeout is not None else self._default_timeout
        effective_timeout = min(requested, self._max_timeout)
        cwd = Path(inp.cwd) if inp.cwd is not None else None

        # Drain in a task and heartbeat while waiting. At the auto-background
        # deadline, hand the still-running command to the background task
        # manager (wired by the agent loop, or passed explicitly) and return
        # early.
        stream = ctx.exec_backend.stream(
            inp.command, cwd=cwd, timeout=effective_timeout
        )
        manager = self._manager
        # Background only with a deadline AND a manager to track it in.
        deadline = self._auto_background_at if manager is not None else None

        command = BackgroundCommand(
            bash_id="", command=inp.command, started_at=time.monotonic()
        )
        command.start(stream)
        drain = command.drain
        assert drain is not None

        next_beat = self._progress_at
        try:
            while not drain.done():
                elapsed = time.monotonic() - command.started_at
                if deadline is not None and elapsed >= deadline:
                    assert manager is not None  # implied by deadline is not None
                    try:
                        bash_id = manager.track(
                            drain,
                            label="Bash",
                            exec_id=exec_id or "",
                            blocks_final_answer=False,
                            monitor=command,
                            id_prefix="bash",
                        )
                    except RuntimeError:
                        await command.kill()  # cap full: don't leak the process
                        raise
                    command.bash_id = bash_id
                    return command.running_result()

                wait = max(0.0, next_beat - elapsed)
                if deadline is not None:
                    wait = min(wait, deadline - elapsed)
                # wait_for cancels its target on timeout; shield keeps the drain
                # task (and the running command) alive — we only peek for
                # completion here, then heartbeat. wait_for never returns before
                # the timeout, so the next beat boundary is always crossed.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(drain), timeout=wait)

                elapsed = time.monotonic() - command.started_at
                if not drain.done() and elapsed >= next_beat:
                    next_beat = elapsed + self._heartbeat_every
                    if progress_callback is not None:
                        await progress_callback(
                            elapsed,
                            effective_timeout,
                            f"Bash: `{preview(inp.command)}` running for "
                            f"{elapsed:.0f}s",
                        )

        except asyncio.CancelledError:
            # The tool was aborted (turn cancel / outer timeout) while the
            # command was still foreground. Cancelling the drain closes the
            # backend stream, so the supervisor kills the process group and we
            # don't leak a live process. Backgrounded commands have already
            # returned and keep running under the manager.
            await command.kill()
            raise

        return command.final_result()


class BashIdInput(BaseModel):
    """Input schema for ``BashOutput`` / ``KillBash``."""

    bash_id: str = Field(description="The id returned by a backgrounded Bash call.")


def _resolve_manager(
    manager: BackgroundTaskManager[Any] | None, tool_name: str
) -> BackgroundTaskManager[Any]:
    if manager is None:
        raise ValueError(
            f"{tool_name} found no background task manager: run under an "
            "agent loop (which wires one onto its bash tools), or construct "
            "the tool with an explicit manager."
        )
    return manager


def _bash_command(
    manager: BackgroundTaskManager[Any], bash_id: str
) -> BackgroundCommand:
    """The :class:`BackgroundCommand` behind ``bash_id``, or a clear error."""
    monitor = manager.get(bash_id).monitor
    if isinstance(monitor, BackgroundCommand):
        return monitor
    raise ValueError(f"{bash_id!r} is not a backgrounded Bash command.")


class BashOutput(_ManagerBound, BaseTool[BashIdInput, BashResult, Any]):
    """
    Poll a backgrounded ``Bash`` command for new output.

    Returns output produced since the previous poll. While the command is
    still running, ``status="backgrounded"`` and ``returncode`` is ``None``;
    once it finishes, the final result fields are filled in and the command
    is removed from the manager.
    """

    name = "BashOutput"
    description = (
        "Get new output from a backgrounded Bash command: stdout / stderr "
        "produced since the last check, plus the exit code once the command "
        "has finished. You are notified automatically when a background "
        "command finishes, so do not call this in a loop just to wait for "
        "completion — use it to inspect progress (e.g. to decide whether to "
        "KillBash) or to read the output after a completion notice."
    )

    def __init__(self, manager: BackgroundTaskManager[Any] | None = None) -> None:
        super().__init__()
        self._manager = manager

    async def _run(
        self,
        inp: BashIdInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        del ctx, exec_id, progress_callback
        manager = _resolve_manager(self._manager, self.name)
        command = _bash_command(manager, inp.bash_id)
        if not command.running:
            manager.remove(inp.bash_id)
            return command.final_result()
        return command.running_result()


class KillBash(_ManagerBound, BaseTool[BashIdInput, BashResult, Any]):
    """Terminate a backgrounded ``Bash`` command (process group)."""

    name = "KillBash"
    description = (
        "Kill a backgrounded Bash command by its bash_id. Returns any output "
        "produced since the last poll."
    )

    def __init__(self, manager: BackgroundTaskManager[Any] | None = None) -> None:
        super().__init__()
        self._manager = manager

    async def _run(
        self,
        inp: BashIdInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        del ctx, exec_id, progress_callback
        manager = _resolve_manager(self._manager, self.name)
        command = _bash_command(manager, inp.bash_id)
        await manager.cancel(inp.bash_id)
        manager.remove(inp.bash_id)
        return command.final_result()


def bind_bash_manager(
    tools: Iterable[BaseTool[Any, Any, Any]],
    manager: BackgroundTaskManager[Any],
) -> None:
    """
    Wire ``manager`` into the bash tools in ``tools`` that don't already have
    one. The agent loop calls this at setup so its ``Bash`` / ``BashOutput`` /
    ``KillBash`` tools share the loop's manager; an explicitly-constructed
    ``manager`` (standalone use) is left untouched. Because the agent owns its
    tools (deep-copied in ``LLMAgent.__init__``) and ``agent.copy()`` copies the
    tools and the manager together, a replica's tools point at its own manager.
    """
    for tool in tools:
        if isinstance(tool, _ManagerBound):
            tool.bind_manager(manager)


def bash_tools(
    *,
    default_timeout: float = DEFAULT_BASH_TIMEOUT,
    max_timeout: float = MAX_BASH_TIMEOUT,
    progress_at: float = DEFAULT_PROGRESS_AT,
    heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
    auto_background_at: float = DEFAULT_AUTO_BACKGROUND_AT,
    block_leading_sleep: bool = True,
    manager: BackgroundTaskManager[Any] | None = None,
    timeout: float | None = None,
) -> list[BaseTool[Any, Any, Any]]:
    """
    Build the ``[Bash, BashOutput, KillBash]`` trio with auto-backgrounding
    enabled.

    Each command is a fresh process, and long ones move to the agent loop's
    background task manager. The trio resolves the running agent loop's manager
    at call time, so every agent (including sub-agents and parallel replicas)
    keeps its own background commands. Pass an explicit ``manager`` only for
    standalone use outside a loop. For a stateful shell instead (``cd`` / env
    persist across calls), use the
    :class:`~grasp_agents.tools.bash_session.BashSession` tool.
    """
    return [
        Bash(
            default_timeout=default_timeout,
            max_timeout=max_timeout,
            progress_at=progress_at,
            heartbeat_every=heartbeat_every,
            auto_background_at=auto_background_at,
            block_leading_sleep=block_leading_sleep,
            manager=manager,
            timeout=timeout,
        ),
        BashOutput(manager),
        KillBash(manager),
    ]
