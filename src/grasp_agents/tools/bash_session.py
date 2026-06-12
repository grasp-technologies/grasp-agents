"""
``BashSession`` — run a command in a *persistent* shell session via
``ctx.exec_backend``, so ``cd``, environment changes, and shell variables carry
across calls (one long-lived ``/bin/sh`` per agent loop).

This is the stateful counterpart to :class:`~grasp_agents.tools.bash.Bash`
(which runs a fresh process per command). It requires a
:class:`~grasp_agents.sandbox.exec_backend.SessionCapable` backend (local /
Seatbelt / srt / E2B all qualify) and is **serial and foreground-only** — one
command at a time, no auto-backgrounding (a session's foreground command is not
cleanly detachable; background long-running work with ``Bash`` instead).

Each :class:`AgentLoop` owns one :class:`BashSessionHolder` on its
:class:`AgentContext`; the stateless ``BashSession`` tool resolves it from the
call context (so sub-agents and parallel replicas each get their own shell).
Outside an agent loop (no holder on the context) each call opens a throwaway
session, so state does not persist across calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
from typing import TYPE_CHECKING, Any

from ..sandbox.exec_backend import SessionCapable
from ..types.tool import BaseTool, ToolProgressCallback
from .bash_common import (
    DEFAULT_BASH_TIMEOUT,
    DEFAULT_HEARTBEAT_EVERY,
    DEFAULT_PROGRESS_AT,
    LEADING_SLEEP,
    MAX_BASH_TIMEOUT,
    BashInput,
    BashResult,
    run_foreground,
)

if TYPE_CHECKING:
    from ..agent.agent_context import AgentContext
    from ..run_context import RunContext
    from ..sandbox.exec_backend import ExecSession


class BashSessionHolder:
    """
    Lazily opens and caches one persistent :class:`ExecSession` per agent loop.

    Owned by the agent loop and wired onto its ``BashSession`` tool; because the
    agent owns its tools (deep-copied at construction), sub-agents and parallel
    replicas each get their own stateful shell. The session is opened on first
    use and reopened if a command closed it (e.g. one that ignored the interrupt
    and forced a session-level kill) or if it reached its lifetime cap (E2B's
    per-session timeout). A reopen loses shell state, so it is flagged for the
    ``BashSession`` tool to surface to the model (see :meth:`take_reset`).
    """

    def __init__(self) -> None:
        self._session: ExecSession | None = None
        self._lock = asyncio.Lock()
        self._was_reset = False

    def __deepcopy__(self, memo: dict[int, Any]) -> BashSessionHolder:
        # A live shell isn't copyable, and a copied agent is a new context: it
        # gets its own (initially unopened) holder. The memo seeding makes the
        # loop's holder and its BashSession tool's holder the same fresh object.
        fresh = BashSessionHolder()
        memo[id(self)] = fresh
        return fresh

    async def get(self, backend: SessionCapable) -> ExecSession:
        async with self._lock:
            current = self._session
            if current is None:
                current = await backend.open_session()
            elif current.closed or getattr(current, "expired", False):
                # Prior shell died or hit its lifetime cap — a fresh shell loses
                # cwd / env / shell variables, so flag it for BashSession to
                # report. Close a still-open-but-expired one first.
                if not current.closed:
                    with contextlib.suppress(Exception):
                        await current.close()
                current = await backend.open_session()
                self._was_reset = True
            self._session = current
            return current

    def take_reset(self) -> bool:
        """Return and clear the 'session was reset since last command' flag."""
        was = self._was_reset
        self._was_reset = False
        return was

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()


class BashSession(BaseTool[BashInput, BashResult, Any]):
    """
    Run a command in a persistent shell session (state carries across calls).

    Args:
        default_timeout: Used when the model passes no ``timeout``.
        max_timeout: Hard clamp on the per-call ``timeout``.
        progress_at: Seconds before the first heartbeat progress report.
        heartbeat_every: Interval between subsequent heartbeats.
        block_leading_sleep: Reject commands whose first statement is a bare
            ``sleep`` — they block the loop and produce nothing.
        timeout: Standard per-tool timeout (outer asyncio ceiling).

    Under an agent loop the persistent session comes from the loop's
    :class:`AgentContext`; used standalone (no ``AgentContext``) each call
    opens a throwaway session, so state does not persist across calls.

    """

    name = "BashSession"
    description = (
        "Run a shell command in a persistent shell session and return the "
        "result. Unlike Bash, state carries across calls: `cd`, exported "
        "environment variables, and shell variables set in one call are still "
        "in effect in the next.\n"
        "\n"
        "* Returns `stdout`, `stderr`, and `returncode` (this command's own "
        "exit code, 0 = success), plus `reason`, `timed_out`, and `truncated`.\n"
        "* Use it when a sequence of commands must share a working directory "
        "or environment.\n"
        "* Runs non-interactively (no prompts, no TTY), one command at a time. "
        "Prefer the dedicated file tools (Read / Write / Edit / Delete / Glob "
        "/ Grep) over shell equivalents.\n"
        "* Set `cwd` to run in a different directory for this call only, "
        "without changing the session's directory."
    )
    untrusted_output = True

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_BASH_TIMEOUT,
        max_timeout: float = MAX_BASH_TIMEOUT,
        progress_at: float = DEFAULT_PROGRESS_AT,
        heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
        block_leading_sleep: bool = True,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._default_timeout = default_timeout
        self._max_timeout = max_timeout
        self._progress_at = progress_at
        self._heartbeat_every = heartbeat_every
        self._block_leading_sleep = block_leading_sleep

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
        del exec_id, path

        backend = ctx.exec_backend if ctx is not None else None
        if backend is None:
            raise ValueError(
                "BashSession requires ctx.exec_backend. Wire an ExecBackend on "
                "RunContext (e.g. via local_environment(...)) before running "
                "the agent."
            )
        session_backend = backend if isinstance(backend, SessionCapable) else None
        if session_backend is None:
            raise ValueError(
                f"BashSession requires a session-capable exec backend; "
                f"{backend.name!r} cannot hold a persistent shell. Use Bash "
                "instead."
            )
        if self._block_leading_sleep and LEADING_SLEEP.match(inp.command):
            raise ValueError(
                "Blocked: a leading `sleep` stalls the agent loop and produces "
                "no output. Run the actual command (with a timeout) instead."
            )

        requested = inp.timeout if inp.timeout is not None else self._default_timeout
        effective_timeout = min(requested, self._max_timeout)

        command = inp.command
        if inp.cwd is not None:
            # One-off cwd: a subshell, so the session's persistent cwd is
            # left untouched.
            command = f"( cd -- {shlex.quote(inp.cwd)} && {inp.command} )"

        # Use the agent loop's per-loop session when running under a loop;
        # otherwise open a throwaway session for this call (no persistence).
        holder = agent_ctx.session_holder if agent_ctx is not None else None
        own_session = holder is None
        session = (
            await session_backend.open_session()
            if holder is None
            else await holder.get(session_backend)
        )
        try:
            result = await run_foreground(
                session.run(command, timeout=effective_timeout),
                command=inp.command,
                progress_callback=progress_callback,
                progress_at=self._progress_at,
                heartbeat_every=self._heartbeat_every,
                effective_timeout=effective_timeout,
            )
            if holder is not None and holder.take_reset():
                notice = (
                    "[shell session expired and was reset: working directory, "
                    "exported environment, and shell variables set in earlier "
                    "calls were lost]\n"
                )
                result = result.model_copy(update={"stderr": notice + result.stderr})
            return result
        finally:
            if own_session and not session.closed:
                with contextlib.suppress(Exception):
                    await session.close()


__all__ = [
    "BashSession",
    "BashSessionHolder",
]
