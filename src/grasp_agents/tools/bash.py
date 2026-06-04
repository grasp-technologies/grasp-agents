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
directory. The backend owns the real timeout (it kills the process group on
expiry); the per-call ``timeout`` is clamped to ``max_timeout``. Output is
captured and size-capped by the backend; ``truncated`` flags when a cap was
hit.

Long-running commands:

* **Heartbeat** — while a command runs, ``Bash`` reports progress through the
  tool progress callback (first beat at ``progress_at``, then every
  ``heartbeat_every`` seconds).
* **Auto-background** — when ``auto_background_at`` is set and a command
  outlives it, ``Bash`` returns early with ``status="backgrounded"`` and a
  ``bash_id``; the process keeps running under the supervisor. Poll it with
  ``BashOutput``, stop it with ``KillBash``. Backgrounded commands still honor
  the overall timeout they were started with. Build the trio with
  :func:`bash_tools` so all three resolve the same :class:`BashProcessRegistry`.

The registry is deliberately **not durable** (no ``TaskRecord`` /
``BackgroundTaskManager``): an OS process does not survive the host process,
so a persisted record would lie on resume. Backgrounded commands die with the
host; sessions that must survive restarts belong in real background *tools*.

Registry scoping mirrors ``FileEditSessionState``: each :class:`AgentLoop`
owns one :class:`BashProcessRegistry` and provides it to the tools through a
ContextVar during runs, so sub-agents and parallel replicas get their own
sessions — one agent's completion notes and idle waits never leak into
another's transcript. Passing an explicit ``registry`` to the tools overrides
the ContextVar (standalone use outside an agent loop).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shlex
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..sandbox.exec_backend import (
    ExecChunk,
    ExecResult,
    ExecSession,
    SessionCapable,
    TerminationReason,
)
from ..types.tool import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..run_context import RunContext

DEFAULT_BASH_TIMEOUT = 120.0
MAX_BASH_TIMEOUT = 600.0
DEFAULT_PROGRESS_AT = 2.0
DEFAULT_HEARTBEAT_EVERY = 10.0
DEFAULT_AUTO_BACKGROUND_AT = 120.0
DEFAULT_MAX_SESSIONS = 16

# Command-preview length in heartbeats and completion notes.
_PREVIEW_MAXLEN = 60
# A command whose first statement is a plain `sleep` blocks the loop for its
# whole duration and produces nothing — reject it with guidance instead.
_LEADING_SLEEP = re.compile(r"^\s*sleep\s+\d")


def _preview(command: str, maxlen: int = _PREVIEW_MAXLEN) -> str:
    return command if len(command) <= maxlen else command[: maxlen - 3] + "..."


class BashInput(BaseModel):
    """Input schema for the ``Bash`` tool."""

    command: str = Field(
        description=(
            "The shell command to run, non-interactively, via `/bin/sh -c`. "
            "Quote paths with spaces. Chain steps with `&&`. Do not launch "
            "interactive programs (there is no TTY) or long-lived servers "
            "(they cannot be reached and are killed when the run ends)."
        )
    )
    cwd: str | None = Field(
        default=None,
        description=(
            "Working directory for the command. Must resolve under the "
            "environment's allowed roots. Defaults to the first allowed root."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description=(
            "Overall wall-clock timeout in seconds. Defaults to "
            f"{DEFAULT_BASH_TIMEOUT:g}s; clamped to {MAX_BASH_TIMEOUT:g}s."
        ),
        gt=0,
    )


class BashResult(BaseModel):
    """
    Output schema for ``Bash`` / ``BashOutput`` / ``KillBash``.

    ``stdout`` and ``stderr`` are kept separate and labeled so the model can
    tell them apart (interleave order across the two is not preserved — a
    single ordered stream needs a PTY-backed session).

    This is the LLM-facing transport schema, not the backend's
    :class:`ExecResult`: it must also represent the *running* state of a
    backgrounded command (``returncode is None``, ``reason="running"``), so it
    is a flat Pydantic model rather than a subclass of the frozen, always-
    finished ``ExecResult``. Finished results map across in
    :meth:`_BashSession.final_result`.
    """

    stdout: str
    stderr: str
    returncode: int | None
    reason: TerminationReason | Literal["running"]
    status: Literal["completed", "backgrounded"] = "completed"
    bash_id: str | None = None
    timed_out: bool = False
    truncated: bool = False
    runtime_ms: float = 0.0
    backend: str = ""


# ---------------------------------------------------------------------------
# Background process registry
# ---------------------------------------------------------------------------


@dataclass
class _BashSession:
    """A live (or finished) backgrounded command and its buffered output."""

    bash_id: str
    command: str
    started_at: float
    chunks: list[ExecChunk] = field(default_factory=list[ExecChunk])
    result: ExecResult | None = None
    cursor: int = 0  # chunks already delivered through BashOutput
    drain: asyncio.Task[None] | None = None
    announced: bool = False  # completion note already emitted via collect_notes

    def start(self, stream: AsyncIterator[ExecChunk | ExecResult]) -> None:
        """Begin draining the backend stream into this session's buffer."""
        self.drain = asyncio.create_task(self._drain(stream))

    async def _drain(self, stream: AsyncIterator[ExecChunk | ExecResult]) -> None:
        async for item in stream:
            if isinstance(item, ExecResult):
                self.result = item
            else:
                self.chunks.append(item)

    async def kill(self) -> None:
        """
        Stop draining: cancelling the task closes the backend stream, which
        makes the supervisor kill the process group (SIGTERM, then SIGKILL
        after the grace period).
        """
        if self.drain is not None and not self.drain.done():
            self.drain.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.drain

    @property
    def running(self) -> bool:
        return self.drain is not None and not self.drain.done()

    @property
    def runtime_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000.0

    def _render(self, since: int) -> tuple[str, str, int]:
        """Split stdout / stderr from ``chunks[since:]``; return the new cursor."""
        out: list[str] = []
        err: list[str] = []
        for chunk in self.chunks[since:]:
            (out if chunk.stream == "stdout" else err).append(chunk.data)
        return "".join(out), "".join(err), len(self.chunks)

    def running_result(self) -> BashResult:
        """Output since the last read for a still-running session; advances cursor."""
        stdout, stderr, self.cursor = self._render(self.cursor)
        return BashResult(
            stdout=stdout,
            stderr=stderr,
            returncode=None,
            reason="running",
            status="backgrounded",
            bash_id=self.bash_id or None,
            runtime_ms=self.runtime_ms,
        )

    def final_result(self) -> BashResult:
        """Final result, output from the cursor to the end of the buffer."""
        stdout, stderr, self.cursor = self._render(self.cursor)
        result = self.result
        if result is None:
            # Stream ended without a terminal result (killed mid-drain).
            return BashResult(
                stdout=stdout,
                stderr=stderr,
                returncode=-1,
                reason=TerminationReason.MANUAL_CANCEL,
                runtime_ms=self.runtime_ms,
            )
        return BashResult(
            stdout=stdout,
            stderr=stderr + result.stderr,  # spawn errors arrive on the terminal
            returncode=result.returncode,
            reason=result.reason,
            timed_out=result.timed_out,
            truncated=result.truncated,
            runtime_ms=result.runtime_ms,
            backend=result.backend,
        )


class BashProcessRegistry:
    """
    In-memory registry of auto-backgrounded ``Bash`` commands.

    Holds each backgrounded session (its drain task keeps consuming the backend
    stream — cancelling it makes the supervisor kill the process group) and its
    buffered output. Foreground commands never enter the registry; only those
    that outlive the auto-background deadline are :meth:`adopt`-ed. Contents die
    with the host process by design.

    One registry per :class:`AgentLoop` (see the module docstring): sessions,
    completion notes, and idle-wait futures are scoped to the agent that
    started them.
    """

    def __init__(self, *, max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        self._sessions: dict[str, _BashSession] = {}
        self._max_sessions = max_sessions
        self._counter = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> BashProcessRegistry:
        # Sessions are process-local runtime state (live asyncio tasks are
        # not copyable, and a copied agent is a *new* context that must not
        # see — or announce — another agent's sessions). Copies start empty;
        # the memo keeps one fresh registry per copied object graph, so a
        # tool trio sharing a registry still shares the copy.
        fresh = BashProcessRegistry(max_sessions=self._max_sessions)
        memo[id(self)] = fresh
        return fresh

    def adopt(self, session: _BashSession) -> str:
        """
        Register a still-running session that outlived the auto-background
        deadline, assigning it a stable ``bash_id``.

        Only backgrounded commands enter the registry, so the ``max_sessions``
        cap counts live background work, not transient foreground calls.
        """
        if len(self._sessions) >= self._max_sessions:
            raise RuntimeError(
                f"Too many background commands ({self._max_sessions}); "
                "kill or drain existing ones with KillBash / BashOutput first."
            )
        self._counter += 1
        session.bash_id = f"bash_{self._counter}"
        self._sessions[session.bash_id] = session
        return session.bash_id

    def get(self, bash_id: str) -> _BashSession:
        session = self._sessions.get(bash_id)
        if session is None:
            known = ", ".join(sorted(self._sessions)) or "none"
            raise ValueError(f"Unknown bash_id {bash_id!r} (known: {known}).")
        return session

    def remove(self, bash_id: str) -> None:
        self._sessions.pop(bash_id, None)

    async def kill(self, bash_id: str) -> _BashSession:
        """Stop the session's command (see :meth:`_BashSession.kill`)."""
        session = self.get(bash_id)
        await session.kill()
        return session

    def pending_drains(self) -> list[asyncio.Future[None]]:
        """
        Futures of sessions not yet announced.

        Exposed to the agent loop's idle wait so that a model with nothing
        else to do blocks until the next completion instead of spinning
        poll turns. Done-but-unannounced sessions are included — their
        already-done futures make the wait return immediately so the
        pending note gets delivered in the same drain cycle.
        """
        return [
            session.drain
            for session in self._sessions.values()
            if session.drain is not None and not session.announced
        ]

    def collect_notes(self) -> list[str]:
        """
        Turn-boundary completion notes, one per newly finished session.

        The agent loop that owns the registry polls this in PRE-ACT and
        injects each note as a user-role message, so the model hears about
        completions at the next turn instead of polling blind. Each session
        is announced at most once; its output still arrives through
        ``BashOutput``.
        """
        notes: list[str] = []
        for session in self._sessions.values():
            if session.announced or session.drain is None or not session.drain.done():
                continue
            session.announced = True
            rc = session.result.returncode if session.result is not None else None
            notes.append(
                f"Background command {session.bash_id} (`{_preview(session.command)}`) "
                f"finished (returncode={rc}).\nCall BashOutput with "
                f"bash_id={session.bash_id!r} to read its output."
            )
        return notes


_current_bash_registry: ContextVar[BashProcessRegistry | None] = ContextVar(
    "_grasp_current_bash_registry", default=None
)


def get_current_bash_registry() -> BashProcessRegistry | None:
    """The registry of the agent loop currently running, if any."""
    return _current_bash_registry.get()


def set_current_bash_registry(
    registry: BashProcessRegistry | None,
) -> Token[BashProcessRegistry | None]:
    """Bind ``registry`` for the current async context; returns a reset token."""
    return _current_bash_registry.set(registry)


def reset_current_bash_registry(
    token: Token[BashProcessRegistry | None],
) -> None:
    """Undo a :func:`set_current_bash_registry`."""
    _current_bash_registry.reset(token)


# ---------------------------------------------------------------------------
# Persistent-session holder
# ---------------------------------------------------------------------------


class BashSessionHolder:
    """
    Lazily opens and caches one persistent :class:`ExecSession` per agent loop.

    Scoped through a ContextVar like :class:`BashProcessRegistry`, so sub-agents
    and parallel replicas each get their own stateful shell. The session is
    opened on first use and reopened if a command closed it (e.g. a command that
    ignored the interrupt and forced a session-level kill).
    """

    def __init__(self) -> None:
        self._session: ExecSession | None = None
        self._lock = asyncio.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> BashSessionHolder:
        # A copied agent is a new context: its own (initially unopened) holder.
        fresh = BashSessionHolder()
        memo[id(self)] = fresh
        return fresh

    async def get(self, backend: SessionCapable) -> ExecSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = await backend.open_session()
            return self._session

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()


_current_session_holder: ContextVar[BashSessionHolder | None] = ContextVar(
    "_grasp_current_bash_session_holder", default=None
)


def get_current_session_holder() -> BashSessionHolder | None:
    """The persistent-session holder of the agent loop currently running, if any."""
    return _current_session_holder.get()


def set_current_session_holder(
    holder: BashSessionHolder | None,
) -> Token[BashSessionHolder | None]:
    """Bind ``holder`` for the current async context; returns a reset token."""
    return _current_session_holder.set(holder)


def reset_current_session_holder(
    token: Token[BashSessionHolder | None],
) -> None:
    """Undo a :func:`set_current_session_holder`."""
    _current_session_holder.reset(token)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class Bash(BaseTool[BashInput, BashResult, Any]):
    """
    Execute a shell command in the agent's environment via ``ctx.exec_backend``.

    Stateless wrapper around the bound :class:`ExecBackend`; the isolation that
    applies (none / seatbelt / bwrap / docker / e2b / ...) is whatever the
    wired backend provides and is reported back in :attr:`BashResult.backend`.

    Args:
        default_timeout: Used when the model passes no ``timeout``.
        max_timeout: Hard clamp on the per-call ``timeout``.
        progress_at: Seconds before the first heartbeat progress report.
        heartbeat_every: Interval between subsequent heartbeats.
        auto_background_at: When set, commands outliving this many seconds
            return early as ``status="backgrounded"`` with a ``bash_id``
            (prefer :func:`bash_tools`, which also wires the ``BashOutput`` /
            ``KillBash`` companions). ``None`` (default) never backgrounds.
            Backgrounding needs a registry in scope (the running loop's or an
            explicit one); without one the command runs in the foreground to
            completion or timeout.
        block_leading_sleep: Reject commands whose first statement is a bare
            ``sleep`` — they block the loop and produce nothing.
        persistent: Default ``True``: when the backend is
            :class:`~grasp_agents.sandbox.exec_backend.SessionCapable` and a
            session holder is in scope (the agent loop wires one), run commands
            in one long-lived shell per loop, so ``cd`` / env / variables
            persist across calls. The session is serial and foreground-only
            (no auto-background). Falls back to a fresh process per call when no
            session is available (standalone use, or a non-session backend).
        registry: Explicit background-session registry for the one-shot
            (non-persistent) path, overriding the per-agent-loop one. Default
            ``None``: backgrounded commands go to the running loop's registry.
            Pass the same registry to all three tools or to none.
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
        persistent: bool = True,
        registry: BashProcessRegistry | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._progress_at = progress_at
        self._heartbeat_every = heartbeat_every
        self._auto_background_at = auto_background_at
        self._block_leading_sleep = block_leading_sleep
        self._persistent = persistent
        self._registry = registry
        if auto_background_at is not None:
            self.description += (
                f" Commands running longer than {auto_background_at:g}s are "
                "moved to the background: you get a `bash_id` back, and a "
                "notification is injected when the command finishes — there "
                "is no need to poll for completion. Use BashOutput to "
                "inspect output in the meantime (e.g. to decide whether to "
                "KillBash), and KillBash to stop it."
            )

    @property
    def registry(self) -> BashProcessRegistry | None:
        """The explicit session registry, if one was provided at construction."""
        return self._registry

    async def _run(
        self,
        inp: BashInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        del exec_id

        if ctx is None or ctx.exec_backend is None:
            raise ValueError(
                "Bash requires ctx.exec_backend. Wire an ExecBackend on "
                "RunContext (e.g. via local_environment(...)) before running "
                "the agent."
            )
        if self._block_leading_sleep and _LEADING_SLEEP.match(inp.command):
            raise ValueError(
                "Blocked: a leading `sleep` stalls the agent loop and "
                "produces no output. Run the actual command (with a timeout) "
                "and poll its result instead."
            )

        requested = inp.timeout if inp.timeout is not None else self._default_timeout
        effective_timeout = min(requested, self._max_timeout)
        cwd = Path(inp.cwd) if inp.cwd is not None else None

        # Persistent shell (stateful) takes precedence when enabled and a
        # session holder is in scope: one long-lived shell per agent loop, so
        # cd / env / variables carry across calls. It is serial and
        # foreground-only — no auto-background.
        exec_session: ExecSession | None = None
        if self._persistent and isinstance(ctx.exec_backend, SessionCapable):
            holder = get_current_session_holder()
            if holder is not None:
                exec_session = await holder.get(ctx.exec_backend)

        if exec_session is not None:
            command = inp.command
            if cwd is not None:
                # One-off cwd: a subshell, so the session's persistent cwd is
                # left untouched.
                command = f"( cd -- {shlex.quote(str(cwd))} && {inp.command} )"
            stream = exec_session.run(command, timeout=effective_timeout)
            registry = None
            deadline = None
        else:
            # One supervision path: drain in a task and heartbeat while waiting.
            # At the auto-background deadline, hand the still-running session to
            # the registry (if one is in scope) and return early. Resolution:
            # explicit ctor registry > the running agent loop's.
            stream = ctx.exec_backend.stream(
                inp.command, cwd=cwd, timeout=effective_timeout
            )
            registry = self._registry or get_current_bash_registry()
            # Background only with a deadline AND a registry to park it in.
            deadline = self._auto_background_at if registry is not None else None

        session = _BashSession(
            bash_id="", command=inp.command, started_at=time.monotonic()
        )
        session.start(stream)
        drain = session.drain
        assert drain is not None

        next_beat = self._progress_at
        try:
            while not drain.done():
                elapsed = time.monotonic() - session.started_at
                if deadline is not None and elapsed >= deadline:
                    assert registry is not None  # implied by deadline is not None
                    try:
                        registry.adopt(session)
                    except RuntimeError:
                        await session.kill()  # cap full: don't leak the process
                        raise
                    return session.running_result()

                wait = max(0.0, next_beat - elapsed)
                if deadline is not None:
                    wait = min(wait, deadline - elapsed)
                # wait_for cancels its target on timeout; shield keeps the drain
                # task (and the running command) alive — we only peek for
                # completion here, then heartbeat. wait_for never returns before
                # the timeout, so the next beat boundary is always crossed.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(drain), timeout=wait)

                elapsed = time.monotonic() - session.started_at
                if not drain.done() and elapsed >= next_beat:
                    next_beat = elapsed + self._heartbeat_every
                    if progress_callback is not None:
                        await progress_callback(
                            elapsed,
                            effective_timeout,
                            f"Bash: `{_preview(inp.command)}` running for "
                            f"{elapsed:.0f}s",
                        )

        except asyncio.CancelledError:
            # The tool was aborted (turn cancel / outer timeout) while the
            # command was still foreground. Cancelling the drain closes the
            # backend stream, so the supervisor kills the process group and we
            # don't leak a live process. Backgrounded commands have already
            # returned and keep running under the registry.
            await session.kill()
            raise

        return session.final_result()


class BashIdInput(BaseModel):
    """Input schema for ``BashOutput`` / ``KillBash``."""

    bash_id: str = Field(description="The id returned by a backgrounded Bash call.")


def _resolve_registry(
    explicit: BashProcessRegistry | None, tool_name: str
) -> BashProcessRegistry:
    registry = explicit or get_current_bash_registry()
    if registry is None:
        raise ValueError(
            f"{tool_name} found no bash session registry: run under an "
            "agent loop, or construct the bash tools with one explicit "
            "shared registry."
        )
    return registry


class BashOutput(BaseTool[BashIdInput, BashResult, Any]):
    """
    Poll a backgrounded ``Bash`` command for new output.

    Returns output produced since the previous poll. While the command is
    still running, ``status="backgrounded"`` and ``returncode`` is ``None``;
    once it finishes, the final result fields are filled in and the session
    is removed from the registry.
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

    def __init__(self, registry: BashProcessRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry

    async def _run(
        self,
        inp: BashIdInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        del ctx, exec_id, progress_callback
        registry = _resolve_registry(self._registry, self.name)
        session = registry.get(inp.bash_id)
        if not session.running:
            registry.remove(inp.bash_id)
            return session.final_result()
        return session.running_result()


class KillBash(BaseTool[BashIdInput, BashResult, Any]):
    """Terminate a backgrounded ``Bash`` command (process group)."""

    name = "KillBash"
    description = (
        "Kill a backgrounded Bash command by its bash_id. Returns any output "
        "produced since the last poll."
    )

    def __init__(self, registry: BashProcessRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry

    async def _run(
        self,
        inp: BashIdInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> BashResult:
        del ctx, exec_id, progress_callback
        registry = _resolve_registry(self._registry, self.name)
        session = await registry.kill(inp.bash_id)
        registry.remove(inp.bash_id)
        return session.final_result()


def bash_tools(
    *,
    default_timeout: float = DEFAULT_BASH_TIMEOUT,
    max_timeout: float = MAX_BASH_TIMEOUT,
    progress_at: float = DEFAULT_PROGRESS_AT,
    heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
    auto_background_at: float = DEFAULT_AUTO_BACKGROUND_AT,
    block_leading_sleep: bool = True,
    registry: BashProcessRegistry | None = None,
    timeout: float | None = None,
) -> list[BaseTool[Any, Any, Any]]:
    """
    Build the ephemeral ``[Bash, BashOutput, KillBash]`` trio with
    auto-backgrounding enabled.

    This trio is the one-shot / background model (``persistent=False``): each
    command is a fresh process, and long ones move to the background registry.
    It resolves the running agent loop's registry at call time, so every agent
    (including sub-agents and parallel replicas) keeps its own background
    sessions. Pass an explicit ``registry`` only for standalone use outside a
    loop. For the stateful shell instead (``cd`` / env persist), use a plain
    ``Bash()`` (``persistent=True`` by default), which needs no companions.
    """
    return [
        Bash(
            default_timeout=default_timeout,
            max_timeout=max_timeout,
            progress_at=progress_at,
            heartbeat_every=heartbeat_every,
            auto_background_at=auto_background_at,
            block_leading_sleep=block_leading_sleep,
            persistent=False,
            registry=registry,
            timeout=timeout,
        ),
        BashOutput(registry),
        KillBash(registry),
    ]
