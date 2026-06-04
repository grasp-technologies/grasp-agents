"""
``LocalExecSession`` — the local-subprocess implementation of the
``ExecSession`` protocol: one long-lived ``/bin/sh`` (optionally wrapped by a
confinement) that keeps ``cd``, environment, and shell variables across
:meth:`run` calls. Used by every ``LocalExecBackend`` (local / Seatbelt / srt),
which differ only in the argv that launches the shell.

Each command is written to the shell's stdin and bracketed by a unique
per-command marker on *each* of stdout and stderr, so a command's output is
delimited without a TTY and the two streams stay separate (the same
``stdout`` / ``stderr`` shape the one-shot backends produce). The marker line on
stdout also carries the command's exit code.

Constraints, by design:

* **Serial** — :meth:`run` calls are serialized; the shell runs one command at
  a time. Real fan-out belongs in separate sessions (e.g. one per agent loop /
  parallel replica), not concurrent commands in one shell.
* **Process-local** — the shell dies with the host; nothing is persisted. A
  resumed run opens a fresh session.
* **Non-interactive** — no TTY, so a program that block-buffers when its stdout
  is a pipe streams its output only when it flushes (its full output still
  arrives before the command's marker).
* **Timeout closes the session** — there is no per-command interrupt without job
  control, so a timed-out command kills the whole shell; :attr:`closed` then
  becomes True and a new session must be opened.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import os
import signal
import time
from typing import TYPE_CHECKING, Literal

from .exec_backend import ExecChunk, ExecResult, ExecSession, TerminationReason
from .session_protocol import frame_command, parse_exit_code

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from .supervisor import SupervisorLimits

_READ_CHUNK = 65_536


class LocalExecSession(ExecSession):
    """A long-lived shell process. See the module docstring for the contract."""

    def __init__(
        self,
        *,
        argv: tuple[str, ...],
        cwd: Path,
        env: Mapping[str, str],
        backend: str,
        limits: SupervisorLimits,
    ) -> None:
        self._argv = argv
        self._cwd = cwd
        self._env = dict(env)
        self._backend = backend
        self._limits = limits
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def closed(self) -> bool:
        return self._closed

    async def _ensure_started(self) -> None:
        if self._proc is not None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
            start_new_session=True,
        )
        # Catch SIGINT with a no-op handler so a per-command interrupt
        # (:meth:`_interrupt`) terminates the running command — which resets
        # INT to the default action on exec — without taking the shell down.
        if self._proc.stdin is not None:
            self._proc.stdin.write(b"trap ':' INT\n")
            await self._proc.stdin.drain()

    async def run(
        self, command: str, *, timeout: float | None = None
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        # Serialize: the shell is a single command interpreter. The lock is held
        # for the whole generator, so concurrent run() calls queue rather than
        # interleaving their writes into one stdin.
        async with self._lock:
            if self._closed:
                raise RuntimeError("shell session is closed; open a new one")
            start = time.monotonic()
            await self._ensure_started()
            proc = self._proc
            assert proc is not None
            assert proc.stdin is not None

            marker, payload = frame_command(command)
            try:
                proc.stdin.write(payload.encode())
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                self._closed = True
                yield self._result(-1, start, reason=TerminationReason.SIGNAL)
                return

            queue: asyncio.Queue[tuple[Literal["stdout", "stderr"], str | None]] = (
                asyncio.Queue()
            )
            readers = (
                asyncio.create_task(self._read(proc.stdout, "stdout", marker, queue)),
                asyncio.create_task(self._read(proc.stderr, "stderr", marker, queue)),
            )
            emitted = {"stdout": 0, "stderr": 0}
            truncated = False
            cap = self._limits.max_output_chars
            completed = False
            try:
                # On timeout, SIGINT the command and grant a grace window for its
                # marker to arrive: the trapped shell survives, so the session is
                # recovered. If grace also overruns, the command is ignoring
                # SIGINT — kill and close the whole session.
                interrupted = False
                deadline = start + timeout if timeout is not None else None
                done = 0
                while done < 2:
                    wait = (
                        max(0.0, deadline - time.monotonic())
                        if deadline is not None
                        else None
                    )
                    try:
                        stream, data = await asyncio.wait_for(queue.get(), wait)
                    except TimeoutError:
                        if interrupted:
                            await self._cancel(readers)
                            await self._terminate()
                            self._closed = True
                            completed = True
                            yield self._result(
                                -1,
                                start,
                                reason=TerminationReason.OVERALL_TIMEOUT,
                                truncated=truncated,
                            )
                            return
                        interrupted = True
                        await self._interrupt()
                        deadline = time.monotonic() + self._limits.kill_grace_period
                        continue
                    if data is None:
                        done += 1
                        continue
                    used = emitted[stream]
                    if used >= cap:
                        truncated = True
                        continue
                    allow = data[: cap - used]
                    emitted[stream] = used + len(allow)
                    truncated = truncated or len(allow) < len(data)
                    yield ExecChunk(stream=stream, data=allow)

                found_out, rc = await readers[0]
                found_err, _ = await readers[1]
                code = rc if rc is not None else -1
                if not (found_out and found_err):
                    # EOF before the marker: the shell exited mid-command.
                    self._closed = True
                    reason = TerminationReason.SIGNAL
                else:
                    reason = (
                        TerminationReason.OVERALL_TIMEOUT
                        if interrupted
                        else TerminationReason.EXIT
                    )
                completed = True
                yield self._result(code, start, reason=reason, truncated=truncated)
            finally:
                await self._cancel(readers)
                if not completed and not self._closed:
                    # Abandoned mid-command (external cancel / turn abort): the
                    # shell is mid-command, so close to avoid corrupting the
                    # next run.
                    await self._terminate()
                    self._closed = True

    async def _read(
        self,
        reader: asyncio.StreamReader | None,
        stream: Literal["stdout", "stderr"],
        marker: str,
        queue: asyncio.Queue[tuple[Literal["stdout", "stderr"], str | None]],
    ) -> tuple[bool, int | None]:
        """
        Push ``(stream, data)`` chunks until ``marker``; return
        ``(marker_found, exit_code)``. Holds back the last ``len(marker)`` chars
        so a marker split across reads is never emitted as output.
        """
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        buf = ""
        marker_len = len(marker)
        try:
            while True:
                raw = await reader.read(_READ_CHUNK) if reader is not None else b""
                if not raw:
                    buf += decoder.decode(b"", final=True)
                    if buf:
                        await queue.put((stream, buf))
                    return False, None
                buf += decoder.decode(raw)
                idx = buf.find(marker)
                if idx != -1:
                    if idx:
                        await queue.put((stream, buf[:idx]))
                    return True, parse_exit_code(buf[idx + marker_len :])
                if len(buf) > marker_len:
                    await queue.put((stream, buf[:-marker_len]))
                    buf = buf[-marker_len:]
        finally:
            await queue.put((stream, None))

    def _result(
        self,
        returncode: int,
        start: float,
        *,
        reason: TerminationReason,
        truncated: bool = False,
    ) -> ExecResult:
        # Output already streamed as chunks; the terminal result carries only
        # the code/reason/runtime (same convention as ProcessSupervisor).
        return ExecResult(
            stdout="",
            stderr="",
            returncode=returncode,
            reason=reason,
            runtime_ms=(time.monotonic() - start) * 1000.0,
            backend=self._backend,
            truncated=truncated,
        )

    @staticmethod
    async def _cancel(tasks: tuple[asyncio.Task[object], ...]) -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _pgid(self) -> int | None:
        if self._proc is None:
            return None
        try:
            return os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return None

    async def _interrupt(self) -> None:
        """
        SIGINT the process group: the trapped shell survives (see
        :meth:`_ensure_started`); the running command — INT reset to the default
        action on exec — is terminated, so the session can keep going.
        """
        pgid = self._pgid()
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGINT)

    async def _terminate(self) -> None:
        """SIGTERM the shell's process group, then SIGKILL after the grace period."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        pgid = self._pgid()
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), self._limits.kill_grace_period)
        except TimeoutError:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)
            await proc.wait()

    async def close(self) -> None:
        """Close stdin (EOF → the shell exits); force-kill if it lingers."""
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        if proc.stdin is not None and not proc.stdin.is_closing():
            with contextlib.suppress(OSError):
                proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), self._limits.kill_grace_period)
        except TimeoutError:
            await self._terminate()


__all__ = ["LocalExecSession"]
