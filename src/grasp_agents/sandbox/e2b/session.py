"""
``E2BExecSession`` — the remote analogue of
:class:`~grasp_agents.sandbox.local_session.LocalExecSession`: one long-lived
``/bin/sh`` running on an E2B sandbox, so ``cd`` / environment / shell variables
persist across :meth:`run` calls.

The shell is started via ``commands.run("/bin/sh", background=True, stdin=True,
...)`` and fed commands through ``commands.send_stdin``; output arrives on the
separate ``on_stdout`` / ``on_stderr`` callbacks, so the two streams stay split.
A unique per-command marker on each stream delimits the end of output and
carries the exit code (see :mod:`.session_protocol`), exactly as in the local
session. Serial and process-local.

Unlike the local session there is no per-command interrupt over the ``commands``
API (no SIGINT to a process group), so a timeout kills the shell and closes the
session — a fresh one is opened on the next call.

The shell lives in the sandbox the session was opened on. Once a restore or an
exit replaces that sandbox, the session counts as closed and never addresses
the new one: its shell's pid there could belong to another process, such as
another agent's shell.

A pause, a snapshot (E2B pauses the sandbox while taking it) or a resume drops
the shell's output stream while the shell keeps running, and so does a pause
E2B makes on its own at the sandbox's keep-alive. The next command first
resumes the sandbox if E2B paused it, then reattaches to the shell's output;
if that fails (the shell exited, say), the session kills the shell, starts a
new one in its place and reports the reset through
:meth:`E2BExecSession.take_reset`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from grasp_agents.sandbox.exec_backend import (
    ExecChunk,
    ExecResult,
    ExecSession,
    TerminationReason,
)
from grasp_agents.sandbox.session_protocol import frame_command, parse_exit_code

from ._handle import E2BSandboxExpiredError, consume_outcome

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from e2b import AsyncCommandHandle, AsyncSandbox

    from ._handle import SandboxHandle

logger = logging.getLogger(__name__)


class _StreamScanner:
    """
    Per-stream marker scanner for :class:`E2BExecSession`.

    Accumulates callback-delivered output, holding back the last ``len(marker)``
    chars so a marker split across deliveries is never emitted, and caps the
    emitted total. ``found`` flips once the end-of-command marker is seen; the
    stdout scanner also captures the exit code.
    """

    def __init__(self, stream: Literal["stdout", "stderr"], marker: str, cap: int):
        self.stream: Literal["stdout", "stderr"] = stream
        self._marker = marker
        self._cap = cap
        self._buf = ""
        self._emitted = 0
        self.found = False
        self.rc: int | None = None
        self.truncated = False

    def feed(self, data: str) -> list[ExecChunk]:
        self._buf += data
        marker_len = len(self._marker)
        idx = self._buf.find(self._marker)
        if idx != -1:
            head = self._buf[:idx]
            self.rc = parse_exit_code(self._buf[idx + marker_len :])
            self.found = True
            self._buf = ""
            return self._emit(head)
        if len(self._buf) > marker_len:
            head = self._buf[:-marker_len]
            self._buf = self._buf[-marker_len:]
            return self._emit(head)
        return []

    def _emit(self, text: str) -> list[ExecChunk]:
        if not text or self._emitted >= self._cap:
            self.truncated = self.truncated or bool(text)
            return []
        allow = text[: self._cap - self._emitted]
        self._emitted += len(allow)
        self.truncated = self.truncated or len(allow) < len(text)
        return [ExecChunk(stream=self.stream, data=allow)]


class E2BExecSession(ExecSession):
    """A long-lived shell on an E2B sandbox. See the module docstring."""

    def __init__(
        self,
        holder: SandboxHandle,
        sandbox_id: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        backend: str,
        max_output_chars: int,
    ) -> None:
        self._holder = holder
        # The sandbox the session was opened on.
        self._sandbox_id = sandbox_id
        self._cwd = cwd
        self._env = env
        self._backend = backend
        self._max_output_chars = max_output_chars
        self._handle: AsyncCommandHandle | None = None
        # Ends when the shell's output stream does.
        self._output: asyncio.Task[Any] | None = None
        self._pid: int | None = None
        self._stream_epoch = 0
        self._queue: asyncio.Queue[tuple[Literal["stdout", "stderr"], str]] = (
            asyncio.Queue()
        )
        self._lock = asyncio.Lock()
        self._closed = False
        self._was_reset = False

    @property
    def backend(self) -> str:
        return self._backend

    def _sandbox(self) -> AsyncSandbox | None:
        """The sandbox the session was opened on while it is the live one."""
        return self._holder.live(self._sandbox_id)

    @property
    def closed(self) -> bool:
        return self._closed or self._sandbox() is None

    def take_reset(self) -> bool:
        was = self._was_reset
        self._was_reset = False
        return was

    def _follow(self, handle: AsyncCommandHandle) -> None:
        self._handle = handle
        self._output = asyncio.ensure_future(handle.wait())
        self._output.add_done_callback(consume_outcome)

    def _output_lost(self) -> bool:
        """
        True once the shell's output stream was dropped, whether the
        environment announced it (a newer stream epoch) or the stream ended
        on its own: E2B ends it when it pauses the sandbox at its keep-alive,
        and so does the shell exiting.
        """
        ended = self._output is not None and self._output.done()
        return ended or self._stream_epoch != self._holder.stream_epoch

    def _on_stdout(self, data: str) -> None:
        self._queue.put_nowait(("stdout", data))

    def _on_stderr(self, data: str) -> None:
        self._queue.put_nowait(("stderr", data))

    async def _ensure_started(self, sb: AsyncSandbox) -> None:
        if self._handle is not None:
            return
        self._stream_epoch = self._holder.stream_epoch
        start = partial(
            sb.commands.run,
            "/bin/sh",
            background=True,
            cwd=self._cwd,
            envs=self._env,
            # No connection deadline: the shell lives until closed, and an E2B
            # timeout would cancel whichever task happened to open it.
            timeout=0,
            stdin=True,
            on_stdout=self._on_stdout,
            on_stderr=self._on_stderr,
        )
        handle = await self._holder.retrying(sb, start)
        self._follow(handle)
        self._pid = handle.pid

    async def _reattach(self, sb: AsyncSandbox, pid: int) -> None:
        """
        Reconnect to the shell's output after its stream was dropped, or
        replace the shell (losing its state) if that fails.
        """
        stale, self._handle = self._handle, None
        if stale is not None:
            with contextlib.suppress(Exception):
                await stale.disconnect()
        epoch = self._holder.stream_epoch

        async def connect() -> AsyncCommandHandle:
            nonlocal epoch
            epoch = self._holder.stream_epoch
            return await sb.commands.connect(
                pid, timeout=0, on_stdout=self._on_stdout, on_stderr=self._on_stderr
            )

        try:
            # A sandbox E2B paused without the environment knowing refuses
            # the connect until it is resumed, which drops streams again.
            handle = await self._holder.retrying(sb, connect)
        except E2BSandboxExpiredError:
            raise
        except Exception:
            logger.warning(
                "Could not reattach to the output of E2B shell %s on sandbox %s; "
                "starting a new shell (its state is lost).",
                pid,
                sb.sandbox_id,
                exc_info=True,
            )
            await self._kill()
            self._was_reset = True
            await self._ensure_started(sb)
            return
        self._follow(handle)
        self._stream_epoch = epoch

    async def run(
        self, command: str, *, timeout: float | None = None
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        # The lock spans the yields so one command owns the shell at a time;
        # the consumer closes this stream when it stops early, releasing it.
        # ruff: disable[yield-in-context-manager-in-async-generator]
        async with self._lock:
            sb = self._sandbox()
            if self._closed or sb is None:
                raise RuntimeError("shell session is closed; open a new one")
            start = time.monotonic()
            await self._holder.ready(sb)
            await self._ensure_started(sb)
            assert self._pid is not None
            if self._output_lost():
                await self._reattach(sb, self._pid)

            # Fresh queue per command — the shell is idle between commands, so
            # the callbacks deliver only this command's output.
            self._queue = asyncio.Queue()
            marker, payload = frame_command(command)
            try:
                await sb.commands.send_stdin(self._pid, payload)
            except Exception as exc:
                await self.close()
                await self._holder.recover(sb, exc)
                yield ExecChunk(stream="stderr", data=str(exc))
                yield self._result(-1, start, reason=TerminationReason.SIGNAL)
                return

            scanners = {
                "stdout": _StreamScanner("stdout", marker, self._max_output_chars),
                "stderr": _StreamScanner("stderr", marker, self._max_output_chars),
            }
            completed = False
            try:
                while not (scanners["stdout"].found and scanners["stderr"].found):
                    wait = (
                        max(0.0, timeout - (time.monotonic() - start))
                        if timeout is not None
                        else None
                    )
                    try:
                        stream, data = await asyncio.wait_for(self._queue.get(), wait)
                    except TimeoutError as exc:
                        await self.close()
                        completed = True
                        # A sandbox E2B removed mid-command never sends the
                        # end markers either.
                        await self._holder.recover(sb, exc)
                        yield self._result(
                            -1,
                            start,
                            reason=TerminationReason.OVERALL_TIMEOUT,
                            truncated=True,
                        )
                        return
                    for chunk in scanners[stream].feed(data):
                        yield chunk
                rc = scanners["stdout"].rc
                truncated = scanners["stdout"].truncated or scanners["stderr"].truncated
                completed = True
                yield self._result(
                    rc if rc is not None else -1,
                    start,
                    reason=TerminationReason.EXIT,
                    truncated=truncated,
                )
            finally:
                if not completed and not self._closed:
                    # Abandoned mid-command (external cancel): kill the shell so
                    # the next run starts clean.
                    await self.close()
        # ruff: enable[yield-in-context-manager-in-async-generator]

    def _result(
        self,
        returncode: int,
        start: float,
        *,
        reason: TerminationReason,
        truncated: bool = False,
    ) -> ExecResult:
        return ExecResult(
            stdout="",
            stderr="",
            returncode=returncode,
            reason=reason,
            runtime_ms=(time.monotonic() - start) * 1000.0,
            backend=self._backend,
            truncated=truncated,
        )

    async def _kill(self) -> None:
        sb = self._sandbox()
        if self._pid is None or sb is None:
            # The shell went with its sandbox; on the live one the pid may
            # belong to another process.
            return
        with contextlib.suppress(Exception):
            await sb.commands.kill(self._pid)

    async def close(self) -> None:
        # Always disconnect: the shell's stream has no deadline, so it would
        # otherwise stay open until its sandbox dies.
        self._closed = True
        handle, self._handle = self._handle, None
        await self._kill()
        if handle is not None:
            with contextlib.suppress(Exception):
                await handle.disconnect()


__all__ = ["E2BExecSession"]
