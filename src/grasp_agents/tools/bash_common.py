"""
Shared pieces for the shell tools: the LLM-facing :class:`BashInput` /
:class:`BashResult` schemas, the timeout / heartbeat defaults, the
:class:`ExecStreamEvent` incremental-output event, and :func:`stream_command` —
drain an exec stream into per-chunk events plus a terminal result.

Used by both :mod:`grasp_agents.tools.bash` (fresh process per command) and
:mod:`grasp_agents.tools.bash_session` (one persistent shell session). Neither
tool implements backgrounding: a long command is sidelined by the agent loop's
:class:`~grasp_agents.agent.background_tasks.BackgroundTaskManager`, which simply
keeps consuming the same stream and buffers the events it yields.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from ..sandbox.exec_backend import ExecChunk, ExecResult, TerminationReason
from ..types.events import Event, ToolOutputEvent, ToolStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

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


@dataclass(frozen=True)
class ExecStreamChunk:
    """One stdout/stderr fragment from a running command; ``str()`` is its text."""

    channel: Literal["stdout", "stderr"]
    text: str

    def __str__(self) -> str:
        return self.text


class ExecStreamEvent(ToolStreamEvent, frozen=True):
    """
    The shell tools' incremental-output event: a
    :class:`~grasp_agents.types.events.ToolStreamEvent` whose ``data`` is an
    :class:`ExecStreamChunk` (text + the stdout/stderr ``channel`` it arrived on).

    Generic consumers render it via ``str(data)`` (the chunk's text); a
    structure-aware consumer (e.g. a console that styles stderr)
    ``isinstance``-checks this subclass and reads ``data.channel`` / ``data.text``.
    """

    type: Literal["tool.exec_stream"] = "tool.exec_stream"  # pyright: ignore[reportIncompatibleVariableOverride]
    data: ExecStreamChunk


@dataclass
class ShellState:
    """
    Per-agent state the fresh :class:`~grasp_agents.tools.bash.Bash` tool
    round-trips across calls.

    Each ``Bash`` command runs in a *fresh* process, so shell state would
    normally reset every call. To make ``cd`` stick (the common expectation —
    ``Bash`` is the primary shell tool), the working directory is captured after
    each command (``pwd -P`` to a scratch state-file) and replayed as the next
    command's ``cwd``. ``token`` namespaces the state-file per agent loop;
    ``seq`` makes it unique per call (so concurrent calls in one turn don't
    clobber each other's file — the resulting ``cwd`` is last-writer-wins, which
    is all that's well-defined for two concurrent ``cd`` commands anyway).

    Lives on the :class:`~grasp_agents.agent.agent_context.AgentContext`; richer
    continuity (env mutations, shell variables) belongs to ``BashSession``.
    """

    cwd: str | None = None
    token: str = field(default_factory=lambda: uuid4().hex[:8])
    seq: int = 0

    def next_state_file(self, root: Path) -> Path:
        """Allocate this call's unique scratch path for the cwd round-trip."""
        self.seq += 1
        return root / f".grasp-cwd-{self.token}-{self.seq}"


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
    Output schema for the shell tools (``Bash`` / ``BashSession``).

    ``stdout`` and ``stderr`` are kept separate and labeled so the model can
    tell them apart. This is the LLM-facing transport schema, not the backend's
    :class:`ExecResult`: it is a flat model (not a subclass of the frozen
    ``ExecResult``) so it can also stand in for a command that was cancelled
    mid-drain. Always a *terminal* result — a backgrounded command's live,
    incremental output is streamed as ``ExecStreamEvent``s (bubbled to the
    parent stream and mirrored to its ``.grasp`` log) instead.
    """

    stdout: str
    stderr: str
    returncode: int | None
    reason: TerminationReason
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


async def stream_command(
    stream: AsyncIterator[ExecChunk | ExecResult],
    *,
    command: str,
    progress_callback: ToolProgressCallback | None = None,
    progress_at: float = DEFAULT_PROGRESS_AT,
    heartbeat_every: float = DEFAULT_HEARTBEAT_EVERY,
    effective_timeout: float = DEFAULT_BASH_TIMEOUT,
    source: str = "Bash",
    exec_id: str | None = None,
) -> AsyncIterator[Event[object]]:
    """
    Drain an exec ``stream``, yielding an :class:`ExecStreamEvent` per output
    chunk and a terminal :class:`ToolOutputEvent` carrying the assembled
    :class:`BashResult`. A heartbeat fires through ``progress_callback`` at
    ``progress_at`` then every ``heartbeat_every`` seconds while the command
    is quiet — independent of the chunk loop, so it never cancels a pipe read.

    On cancellation (turn abort / ``KillTask``) the backend stream is closed,
    which makes the supervisor kill the process group; the ``CancelledError``
    then propagates (and the terminal event is *not* emitted).
    """
    started = time.monotonic()
    out: list[str] = []
    err: list[str] = []
    result: ExecResult | None = None

    async def _heartbeat() -> None:
        if progress_callback is None:
            return
        delay = progress_at
        while True:
            await asyncio.sleep(delay)
            elapsed = time.monotonic() - started
            await progress_callback(
                elapsed,
                effective_timeout,
                f"`{preview(command)}` running for {elapsed:.0f}s",
            )
            delay = heartbeat_every

    beat = asyncio.create_task(_heartbeat())
    try:
        async for item in stream:
            if isinstance(item, ExecResult):
                result = item
            else:
                (out if item.stream == "stdout" else err).append(item.data)
                yield ExecStreamEvent(
                    data=ExecStreamChunk(channel=item.stream, text=item.data),
                    source=source,
                    exec_id=exec_id,
                )
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
        # If we were cancelled while suspended at our own ``yield`` (not inside
        # a pipe read), the backend stream is still open — close it so the
        # supervisor kills the process group.
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()

    runtime_ms = (time.monotonic() - started) * 1000.0
    yield ToolOutputEvent(
        data=foreground_result(out, err, result, runtime_ms),
        source=source,
        exec_id=exec_id,
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
    Drain a foreground exec ``stream`` to completion, returning the assembled
    :class:`BashResult`. A thin wrapper over :func:`stream_command` for callers
    that want only the terminal result (e.g. the non-streaming ``BashSession``);
    the per-chunk stream events are discarded.
    """
    result: BashResult | None = None
    async for event in stream_command(
        stream,
        command=command,
        progress_callback=progress_callback,
        progress_at=progress_at,
        heartbeat_every=heartbeat_every,
        effective_timeout=effective_timeout,
    ):
        if isinstance(event, ToolOutputEvent):
            result = event.data
    assert result is not None
    return result


__all__ = [
    "DEFAULT_BASH_TIMEOUT",
    "DEFAULT_HEARTBEAT_EVERY",
    "DEFAULT_PROGRESS_AT",
    "LEADING_SLEEP",
    "MAX_BASH_TIMEOUT",
    "BashInput",
    "BashResult",
    "ExecStreamChunk",
    "ExecStreamEvent",
    "ShellState",
    "foreground_result",
    "preview",
    "run_foreground",
    "stream_command",
]
