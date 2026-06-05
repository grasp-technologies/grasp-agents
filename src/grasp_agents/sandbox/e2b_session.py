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
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Literal

from .exec_backend import ExecChunk, ExecResult, ExecSession, TerminationReason
from .session_protocol import frame_command, parse_exit_code

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from e2b import AsyncCommandHandle, AsyncSandbox

# Max lifetime of a persistent session shell on the sandbox.
_SESSION_SHELL_TIMEOUT = 3600.0


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
        get_sandbox: Callable[[], AsyncSandbox],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        backend: str,
        max_output_chars: int,
    ) -> None:
        self._get_sandbox = get_sandbox
        self._cwd = cwd
        self._env = env
        self._backend = backend
        self._max_output_chars = max_output_chars
        self._handle: AsyncCommandHandle | None = None
        self._pid: int | None = None
        self._queue: asyncio.Queue[tuple[Literal["stdout", "stderr"], str]] = (
            asyncio.Queue()
        )
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def closed(self) -> bool:
        return self._closed

    async def _ensure_started(self) -> None:
        if self._handle is not None:
            return
        sb = self._get_sandbox()

        def on_stdout(data: str) -> None:
            self._queue.put_nowait(("stdout", data))

        def on_stderr(data: str) -> None:
            self._queue.put_nowait(("stderr", data))

        self._handle = await sb.commands.run(
            "/bin/sh",
            background=True,
            cwd=self._cwd,
            envs=self._env,
            timeout=_SESSION_SHELL_TIMEOUT,
            stdin=True,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        self._pid = self._handle.pid

    async def run(
        self, command: str, *, timeout: float | None = None
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        async with self._lock:
            if self._closed:
                raise RuntimeError("shell session is closed; open a new one")
            start = time.monotonic()
            await self._ensure_started()
            sb = self._get_sandbox()
            assert self._pid is not None

            # Fresh queue per command — the shell is idle between commands, so
            # the callbacks deliver only this command's output.
            self._queue = asyncio.Queue()
            marker, payload = frame_command(command)
            try:
                await sb.commands.send_stdin(self._pid, payload)
            except Exception as exc:
                self._closed = True
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
                    except TimeoutError:
                        await self._kill()
                        self._closed = True
                        completed = True
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
                    await self._kill()
                    self._closed = True

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
        if self._pid is None:
            return
        with contextlib.suppress(Exception):
            sb = self._get_sandbox()
            await sb.commands.kill(self._pid)

    async def close(self) -> None:
        self._closed = True
        await self._kill()
        self._handle = None


__all__ = ["E2BExecSession"]
