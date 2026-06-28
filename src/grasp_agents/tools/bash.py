"""
``Bash`` — run a shell command via ``ctx.exec_backend``.

Opt-in and stateless between calls: the tool consumes the :class:`ExecBackend`
wired onto :attr:`RunContext.exec_backend` (set it via :func:`local_environment`
or by constructing a backend directly). Without an exec backend the tool refuses
to run — an agent gets no shell access by default.

Non-interactive contract: no TTY, and each call is a fresh ``/bin/sh -c`` — no
shell state (``cd``, environment, started processes) carries over (the working
directory is the one exception: it is round-tripped across calls so ``cd``
sticks — see :class:`~grasp_agents.tools.bash_common.ShellState`). This is the
:class:`ExecBackend` contract, uniform across every backend (local subprocess,
Seatbelt/bwrap, Docker, remote); pass ``cd ... &&`` or ``cwd`` for a one-off
directory. For a fully persistent shell (env + variables too), use
:class:`~grasp_agents.tools.bash_session.BashSession`.

``Bash`` owns no backgrounding. It simply streams the backend's output
(:class:`~grasp_agents.tools.bash_common.ExecStreamEvent` per chunk, then a
terminal :class:`BashResult`); when a command outlives ``auto_background_at`` the
agent loop's
:class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager` keeps
consuming that same stream in the background, mirroring its output to a
``.grasp`` log and notifying the model with the result on completion (stop it
early with the generic ``KillTask`` tool). Build the pair with
:func:`bash_tools`.
"""

from __future__ import annotations

import contextlib
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grasp_agents.tools.base import BaseTool, ToolProgressCallback
from grasp_agents.types.events import ToolOutputEvent

from .bash_common import (
    DEFAULT_BASH_TIMEOUT,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_PROGRESS_AT,
    LEADING_SLEEP,
    MAX_BASH_TIMEOUT,
    BashInput,
    BashResult,
    ShellState,
    stream_command,
)
from .task_tools import KillTask

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.agent.background_tasks import BackgroundTaskManager
    from grasp_agents.run_context import RunContext
    from grasp_agents.sandbox.exec_backend import ExecChunk, ExecResult
    from grasp_agents.types.events import Event

DEFAULT_AUTO_BACKGROUND_AT = 120.0
# A backgrounded command's completion note inlines its result up to this many
# chars; a larger result is excerpted in the note, which points at the task's
# ``.grasp`` log holding the full output, keeping the transcript from bloating.
DEFAULT_MAX_INLINE_RESULT_CHARS = 8000


class Bash(BaseTool[BashInput, BashResult, Any]):
    """
    Execute a shell command in the agent's environment via ``ctx.exec_backend``.

    Stateless wrapper around the bound :class:`ExecBackend`; the isolation that
    applies (none / seatbelt / bwrap / docker / e2b / ...) is whatever the
    wired backend provides and is reported back in :attr:`BashResult.backend`.
    Each call is a fresh process — shell state (env / variables) does not persist
    across calls (``cd`` is round-tripped so it sticks); use
    :class:`~grasp_agents.tools.bash_session.BashSession` for a persistent
    session.

    Args:
        default_timeout: Used when the model passes no ``timeout``.
        max_timeout: Hard clamp on the per-call ``timeout``.
        progress_at: Seconds before the first heartbeat progress report.
        heartbeat_every: Interval between subsequent heartbeats.
        auto_background_at: When set, the agent loop moves the command to the
            background once it has run this many seconds (``0`` backgrounds
            immediately; ``None`` (default) never does). A backgrounded command
            keeps running; the loop is notified with its result on completion,
            and can stop it early with ``KillTask`` (prefer :func:`bash_tools`,
            which wires the companion). Outside an agent loop the command always
            runs to completion in the foreground.
        max_inline_result_chars: How much of a command's output is inlined into
            the agent's context. A larger result is excerpted, with a pointer to
            an on-disk ``.grasp`` file holding the full output for ``Read`` /
            ``Grep`` (a foreground call spills to a file; a backgrounded one
            points at its task log). ``None`` inlines the whole result.
        block_leading_sleep: Reject commands whose first statement is a bare
            ``sleep`` — they block the loop and produce nothing.
        timeout: Standard per-tool timeout (outer asyncio ceiling).

    """

    name = "Bash"
    description = (
        "Run a shell command in the agent's environment and return the result. "
        "Use it for running programs, build/test commands, and git operations.\n"
        "\n"
        "* Returns `stdout`, `stderr`, and `returncode` — the exit code of "
        "*this* command (0 = success), not of anything the command inspects — "
        "plus `reason` (exit / timeout / cancel), `timed_out`, and `truncated` "
        "(true when the shown output was clipped).\n"
        "* Prefer the dedicated file tools (Read / Write / Edit / Delete / "
        "Glob / Grep) over shell equivalents (cat / sed / find / grep): safer, "
        "with better output.\n"
        "* Runs non-interactively (no prompts, no TTY); calls share no state — "
        "pass `cd ... &&` or set `cwd` for a one-off directory."
    )
    untrusted_output = True

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_BASH_TIMEOUT,
        max_timeout: float = MAX_BASH_TIMEOUT,
        progress_at: float = DEFAULT_PROGRESS_AT,
        heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
        auto_background_at: float | None = None,
        max_inline_result_chars: int | None = DEFAULT_MAX_INLINE_RESULT_CHARS,
        block_leading_sleep: bool = True,
        blocks_final_answer: bool = True,
        timeout: float | None = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            auto_background_at=auto_background_at,
            # Default: a backgrounded command's result is waited for before the
            # final answer (the model can't "wait" — only the loop can, by
            # gating the answer on the result). Set False only for genuine
            # fire-and-forget work (e.g. starting a server) where "I launched it"
            # IS the answer; then the run finalizes without it.
            blocks_final_answer=blocks_final_answer,
            max_inline_result_chars=max_inline_result_chars,
            # Streams incremental output (ExecStreamEvent) → mirrored to a
            # ``.grasp`` log the agent can Read / Grep while it runs.
            has_progress_log=True,
        )
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._progress_at = progress_at
        self._heartbeat_every = heartbeat_every
        self._block_leading_sleep = block_leading_sleep
        if auto_background_at is not None:
            self.description += (
                "\n"
                f"* Commands running longer than {auto_background_at:g}s are "
                "moved to the background: you get a `task_id` back, and a "
                "notification with the command's output is injected when it "
                "finishes — there is no need to poll for completion. The "
                "notification points at an output log file you can Read / Grep "
                "if the output was truncated; use KillTask with the task_id to "
                "stop a command you no longer need."
            )

    def _prepare(
        self,
        inp: BashInput,
        ctx: RunContext[Any] | None,
        shell_state: ShellState | None,
    ) -> tuple[AsyncIterator[ExecChunk | ExecResult], float, Path | None]:
        """
        Validate the call and open the backend exec stream.

        With a ``shell_state`` (running under an agent loop), the command starts
        in the round-tripped working directory and is wrapped to record its
        final directory to a scratch state-file, returned here so the caller can
        read it back into ``shell_state`` (so ``cd`` sticks across calls).
        """
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

        # cwd: an explicit per-call override wins; otherwise resume the shell's
        # round-tripped working directory.
        if inp.cwd is not None:
            cwd = Path(inp.cwd)
        elif shell_state is not None and shell_state.cwd is not None:
            cwd = Path(shell_state.cwd)
        else:
            cwd = None

        command = inp.command
        state_file: Path | None = None
        roots = ctx.exec_backend.policy.allowed_roots
        if shell_state is not None and roots:
            state_file = shell_state.next_state_file(roots[0])
            # Record the final working directory without disturbing the command's
            # own stdout/stderr or its exit code.
            command = (
                f"{inp.command}\n"
                "__grasp_rc=$?\n"
                f"command pwd -P > {shlex.quote(str(state_file))} 2>/dev/null || true\n"
                "exit $__grasp_rc\n"
            )

        stream = ctx.exec_backend.stream(command, cwd=cwd, timeout=effective_timeout)
        return stream, effective_timeout, state_file

    def concurrency_conflict_keys(self, inp: BashInput) -> list[str] | None:
        # A shell command can write anywhere in the workspace — claim global
        # exclusivity so it never interleaves with concurrent writers.
        del inp
        return ["/"]

    async def _run(
        self,
        inp: BashInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> BashResult:
        result: BashResult | None = None
        async for event in self._run_stream(
            inp,
            ctx=ctx,
            exec_id=exec_id,
            progress_callback=progress_callback,
            path=path,
            agent_ctx=agent_ctx,
        ):
            if isinstance(event, ToolOutputEvent):
                result = event.data
        assert result is not None
        return result

    async def _run_stream(
        self,
        inp: BashInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del path
        shell_state = agent_ctx.shell_state if agent_ctx is not None else None
        stream, effective_timeout, state_file = self._prepare(inp, ctx, shell_state)
        async for event in stream_command(
            stream,
            command=inp.command,
            progress_callback=progress_callback,
            progress_at=self._progress_at,
            heartbeat_every=self._heartbeat_every,
            effective_timeout=effective_timeout,
            source=self.name,
            exec_id=exec_id,
        ):
            yield event
        # Round-trip the working directory so ``cd`` sticks for the next call
        # (best-effort; a command cancelled before the trailer leaves it as-is).
        if state_file is not None and shell_state is not None and ctx is not None:
            await self._persist_cwd(ctx, shell_state, state_file)

    @staticmethod
    async def _persist_cwd(
        ctx: RunContext[Any], shell_state: ShellState, state_file: Path
    ) -> None:
        """Read the command's recorded final directory into ``shell_state``."""
        backend = ctx.file_backend
        if backend is None:
            return
        try:
            text, _ = await backend.read_text(state_file)
        except Exception:  # best-effort; a missing file just keeps the old cwd
            return
        new_cwd = text.strip()
        if new_cwd:
            shell_state.cwd = new_cwd
        with contextlib.suppress(Exception):
            await backend.delete(state_file)


def bash_tools(
    *,
    default_timeout: float = DEFAULT_BASH_TIMEOUT,
    max_timeout: float = MAX_BASH_TIMEOUT,
    progress_at: float = DEFAULT_PROGRESS_AT,
    heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
    auto_background_at: float = DEFAULT_AUTO_BACKGROUND_AT,
    max_inline_result_chars: int | None = DEFAULT_MAX_INLINE_RESULT_CHARS,
    block_leading_sleep: bool = True,
    blocks_final_answer: bool = True,
    manager: BackgroundTaskManager[Any] | None = None,
    timeout: float | None = None,
) -> list[BaseTool[Any, Any, Any]]:
    """
    Build the ``[Bash, KillTask]`` pair with auto-backgrounding enabled.

    Each command is a fresh process. The agent loop moves a long one to its
    background task manager at ``auto_background_at``, notifying the model with
    the result on completion (its streamed output is mirrored to a ``.grasp``
    log for ``Read`` / ``Grep``); the generic ``KillTask`` companion resolves
    that same manager from the call's :class:`AgentContext`, so every agent
    (including sub-agents and parallel replicas) keeps its own background work.
    Pass an explicit ``manager`` only to kill from the companion outside a loop.
    For a stateful shell instead (``cd`` / env persist across calls), use the
    :class:`~grasp_agents.tools.bash_session.BashSession` tool.
    """
    return [
        Bash(
            default_timeout=default_timeout,
            max_timeout=max_timeout,
            progress_at=progress_at,
            heartbeat_every=heartbeat_every,
            auto_background_at=auto_background_at,
            max_inline_result_chars=max_inline_result_chars,
            block_leading_sleep=block_leading_sleep,
            blocks_final_answer=blocks_final_answer,
            timeout=timeout,
        ),
        KillTask(manager),
    ]
