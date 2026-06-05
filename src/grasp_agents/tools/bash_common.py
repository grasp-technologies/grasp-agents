"""
Shared pieces for the shell tools: the LLM-facing :class:`BashInput` /
:class:`BashResult` schemas, the timeout / heartbeat defaults, and
:func:`run_foreground` — drain a foreground exec stream into a
:class:`BashResult` while emitting heartbeat progress.

Used by both :mod:`grasp_agents.tools.bash` (fresh process per command) and
:mod:`grasp_agents.tools.bash_session` (one persistent shell session).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from ..sandbox.exec_backend import ExecChunk, ExecResult, TerminationReason

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..types.tool import ToolProgressCallback

DEFAULT_BASH_TIMEOUT = 120.0
MAX_BASH_TIMEOUT = 600.0
DEFAULT_PROGRESS_AT = 2.0
DEFAULT_HEARTBEAT_EVERY = 10.0

# Command-preview length in heartbeats and completion notes.
_PREVIEW_MAXLEN = 60
# A command whose first statement is a plain `sleep` blocks the loop for its
# whole duration and produces nothing — reject it with guidance instead.
LEADING_SLEEP = re.compile(r"^\s*sleep\s+\d")


def preview(command: str, maxlen: int = _PREVIEW_MAXLEN) -> str:
    return command if len(command) <= maxlen else command[: maxlen - 3] + "..."


class BashInput(BaseModel):
    """Input schema for the shell tools."""

    command: str = Field(
        description=(
            "The shell command to run, non-interactively, via `/bin/sh`. "
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
    Output schema for the shell tools (``Bash`` / ``Shell`` / ``BashOutput`` /
    ``KillBash``).

    ``stdout`` and ``stderr`` are kept separate and labeled so the model can
    tell them apart. This is the LLM-facing transport schema, not the backend's
    :class:`ExecResult`: it must also represent the *running* state of a
    backgrounded command (``returncode is None``, ``reason="running"``), so it
    is a flat model rather than a subclass of the frozen, always-finished
    ``ExecResult``. The ``status`` / ``bash_id`` fields are set only by the
    backgrounding ``Bash`` tool; the persistent ``Shell`` leaves them at their
    foreground defaults.
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


def foreground_result(
    stdout: list[str],
    stderr: list[str],
    result: ExecResult | None,
    runtime_ms: float,
) -> BashResult:
    """Build a finished :class:`BashResult` from drained chunks + the terminal."""
    if result is None:
        # Stream ended without a terminal result (killed mid-drain).
        return BashResult(
            stdout="".join(stdout),
            stderr="".join(stderr),
            returncode=-1,
            reason=TerminationReason.MANUAL_CANCEL,
            runtime_ms=runtime_ms,
        )
    return BashResult(
        stdout="".join(stdout),
        stderr="".join(stderr) + result.stderr,  # spawn errors arrive on the terminal
        returncode=result.returncode,
        reason=result.reason,
        timed_out=result.timed_out,
        truncated=result.truncated,
        runtime_ms=result.runtime_ms,
        backend=result.backend,
    )


async def run_foreground(
    stream: AsyncIterator[ExecChunk | ExecResult],
    *,
    command: str,
    progress_callback: ToolProgressCallback | None,
    progress_at: float,
    heartbeat_every: float,
    effective_timeout: float,
) -> BashResult:
    """
    Drain a foreground exec ``stream`` to completion, accumulating stdout /
    stderr and emitting a heartbeat at ``progress_at`` then every
    ``heartbeat_every`` seconds while it runs.

    On cancellation (turn abort / outer timeout) the consuming task is
    cancelled, which closes the stream so the backend (or session) kills the
    running command; the ``CancelledError`` then propagates.
    """
    started = time.monotonic()
    out: list[str] = []
    err: list[str] = []
    result: ExecResult | None = None

    async def _consume() -> None:
        nonlocal result
        async for item in stream:
            if isinstance(item, ExecResult):
                result = item
            elif item.stream == "stdout":
                out.append(item.data)
            else:
                err.append(item.data)

    task = asyncio.create_task(_consume())
    next_beat = progress_at
    try:
        while not task.done():
            elapsed = time.monotonic() - started
            wait = max(0.0, next_beat - elapsed)
            # wait_for cancels its target on timeout; shield keeps the consume
            # task (and the running command) alive — we only peek for
            # completion here, then heartbeat.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(task), timeout=wait)
            elapsed = time.monotonic() - started
            if not task.done() and elapsed >= next_beat:
                next_beat = elapsed + heartbeat_every
                if progress_callback is not None:
                    await progress_callback(
                        elapsed,
                        effective_timeout,
                        f"`{preview(command)}` running for {elapsed:.0f}s",
                    )
        await task  # surface any exception from the consumer
    except asyncio.CancelledError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        raise

    runtime_ms = (time.monotonic() - started) * 1000.0
    return foreground_result(out, err, result, runtime_ms)


@dataclass
class BackgroundCommand:
    """
    A live (or finished) backgrounded command and its buffered output.

    Created by :class:`~grasp_agents.tools.bash.Bash` when a command outlives the
    auto-background deadline, then owned by the
    :class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager`: its drain
    task keeps consuming the backend stream (cancelling it makes the supervisor
    kill the process group), and ``BashOutput`` / ``KillBash`` read / stop it by
    ``bash_id``.
    """

    bash_id: str
    command: str
    started_at: float
    chunks: list[ExecChunk] = field(default_factory=list[ExecChunk])
    result: ExecResult | None = None
    cursor: int = 0  # chunks already delivered through BashOutput
    drain: asyncio.Task[None] | None = None
    announced: bool = False  # completion note already emitted

    def start(self, stream: AsyncIterator[ExecChunk | ExecResult]) -> None:
        """Begin draining the backend stream into this command's buffer."""
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
        """Output since the last read for a still-running command; advances cursor."""
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

    def summary(self) -> str:
        """
        One-line completion summary for the background task manager's
        turn-boundary note (satisfies the manager's ``BackgroundMonitor``).
        """
        rc = self.result.returncode if self.result is not None else None
        return (
            f"finished (returncode={rc}); call BashOutput with "
            f"bash_id={self.bash_id!r} to read its output"
        )


__all__ = [
    "DEFAULT_BASH_TIMEOUT",
    "DEFAULT_HEARTBEAT_EVERY",
    "DEFAULT_PROGRESS_AT",
    "LEADING_SLEEP",
    "MAX_BASH_TIMEOUT",
    "BackgroundCommand",
    "BashInput",
    "BashResult",
    "foreground_result",
    "preview",
    "run_foreground",
]
