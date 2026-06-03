"""
``ProcessSupervisor`` — the one place process lifecycle lives.

An :class:`~grasp_agents.sandbox.exec_backend.ExecBackend` builds an
:class:`ExecSpec` (argv + cwd + env — no spawning) and hands it to a shared
supervisor that owns the spawn, the dual timeout timers (overall wall-clock +
idle no-output), output caps, process-group kill, and streaming. Splitting
*build the spec* from *run the spec* keeps every backend (local / Seatbelt /
bwrap / Docker / remote) thin: they differ only in the argv they assemble,
never in how a process is supervised.

POSIX only: process-group containment uses ``start_new_session`` (``setsid``)
at spawn so the whole tree can be killed with :func:`os.killpg`. Windows is an
explicit gap (see ``docs/roadmap/15-sandbox-and-terminal.md``).
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import math
import os
import signal
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .exec_backend import ExecChunk, ExecResult, TerminationReason

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping
    from pathlib import Path


@dataclass(frozen=True)
class ExecSpec:
    """
    A ready-to-spawn command: argv + working dir + environment.

    Built by an ``ExecBackend`` (which assembles the argv — bare
    ``/bin/sh -c`` for the unconfined local backend, a ``sandbox-exec`` /
    ``bwrap`` / ``docker exec`` wrapper for the confined ones) and consumed by
    the :class:`ProcessSupervisor`. Building a spec never spawns anything.
    """

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    backend: str


@dataclass(frozen=True)
class SupervisorLimits:
    """
    Ceilings the supervisor enforces on every run.

    ``overall_timeout`` is the wall-clock ceiling; ``idle_timeout`` (when set)
    re-arms on every output chunk and fires when a command goes quiet for too
    long. ``max_output_chars`` caps *each* stream independently — past it the
    stream is dropped and the result is flagged ``truncated``. ``None`` on
    either timeout disables it.

    Resource limits (POSIX ``setrlimit``, applied to the spawned process tree;
    ``None`` disables each):

    - ``cpu_timeout`` — CPU-seconds ceiling (``RLIMIT_CPU``); catches busy loops
      that a wall-clock timeout would let burn a core. Rounded up to whole
      seconds.
    - ``max_memory_mb`` — virtual address space (``RLIMIT_AS``). Blunt — some
      runtimes reserve large virtual memory and may fail; use a hosted/Docker
      backend for a hard physical cap.
    - ``max_file_size_mb`` — largest file the process may write (``RLIMIT_FSIZE``).

    (``RLIMIT_NPROC`` is intentionally not exposed — it is per-UID, so a low
    value would throttle the user's *other* processes; use cgroups/Docker for
    a per-tree process cap.)
    """

    overall_timeout: float | None = 600.0
    idle_timeout: float | None = None
    max_output_chars: int = 1_000_000
    kill_grace_period: float = 5.0
    read_chunk_size: int = 65_536
    cpu_timeout: float | None = None
    max_memory_mb: int | None = None
    max_file_size_mb: int | None = None


class ProcessSupervisor:
    """
    Spawns, supervises, and reaps a single subprocess per :meth:`run`.

    One instance is shared across every command of an exec backend; it holds
    only the default :class:`SupervisorLimits`, so it is stateless between
    runs and safe to share.
    """

    def __init__(self, limits: SupervisorLimits | None = None) -> None:
        self._limits = limits or SupervisorLimits()

    @property
    def limits(self) -> SupervisorLimits:
        return self._limits

    async def run(
        self,
        spec: ExecSpec,
        *,
        stdin: bytes | None = None,
        limits: SupervisorLimits | None = None,
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        """
        Spawn ``spec`` and stream its output.

        Yields a sequence of :class:`ExecChunk` items as output arrives,
        followed by exactly one terminal :class:`ExecResult`. The terminal
        result carries the return code, :class:`TerminationReason`, runtime,
        and ``truncated`` flag, but **empty** ``stdout`` / ``stderr`` — the
        bytes were already delivered as chunks. Callers that want the full
        captured output assemble it from the chunks (see
        :meth:`LocalExecBackend.execute`).
        """
        eff = limits or self._limits
        start = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                *spec.argv,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=spec.cwd,
                env=dict(spec.env),
                start_new_session=True,
                preexec_fn=self._make_preexec(eff),
            )
        except (OSError, ValueError) as exc:
            yield ExecResult(
                stdout="",
                stderr=f"spawn failed: {exc}",
                returncode=-1,
                reason=TerminationReason.SPAWN_ERROR,
                runtime_ms=self._ms(start),
                backend=spec.backend,
            )
            return

        if stdin is not None and proc.stdin is not None:
            with contextlib.suppress(OSError):
                proc.stdin.write(stdin)
                await proc.stdin.drain()
                proc.stdin.close()

        queue: asyncio.Queue[ExecChunk | None] = asyncio.Queue()
        readers = [
            asyncio.create_task(self._pump(proc.stdout, "stdout", queue, eff)),
            asyncio.create_task(self._pump(proc.stderr, "stderr", queue, eff)),
        ]
        emitted = {"stdout": 0, "stderr": 0}
        truncated = False
        reason: TerminationReason | None = None

        try:
            eof_count = 0
            while eof_count < len(readers):
                wait_t = self._next_timeout(start, eff)
                try:
                    item = await self._get(queue, wait_t)
                except TimeoutError:
                    reason = (
                        TerminationReason.OVERALL_TIMEOUT
                        if self._overall_expired(start, eff)
                        else TerminationReason.NO_OUTPUT_TIMEOUT
                    )
                    break
                if item is None:
                    eof_count += 1
                    continue
                chunk, truncated = self._cap(item, emitted, eff, truncated=truncated)
                if chunk is not None:
                    yield chunk

            if reason is None:
                rc = await self._await_exit(proc, start, eff)
                if rc is None:
                    reason = TerminationReason.OVERALL_TIMEOUT
                    rc = await self._kill(proc, eff)
                else:
                    reason = (
                        TerminationReason.SIGNAL if rc < 0 else TerminationReason.EXIT
                    )
            else:
                rc = await self._kill(proc, eff)

            yield ExecResult(
                stdout="",
                stderr="",
                returncode=rc,
                reason=reason,
                runtime_ms=self._ms(start),
                backend=spec.backend,
                truncated=truncated,
            )
        finally:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            await self._kill(proc, eff)

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _ms(start: float) -> float:
        return (time.monotonic() - start) * 1000.0

    @staticmethod
    def _make_preexec(limits: SupervisorLimits) -> Callable[[], None] | None:
        """
        Build a ``preexec_fn`` that applies the resource ``setrlimit`` ceilings
        in the child before ``exec``. Returns ``None`` (no preexec) when no
        resource limit is set, so the module never imports ``resource`` on
        platforms that lack it unless a limit is actually requested.
        """
        if (
            limits.cpu_timeout is None
            and limits.max_memory_mb is None
            and limits.max_file_size_mb is None
        ):
            return None

        import resource  # noqa: PLC0415  (POSIX-only; imported only when used)

        rlimits: list[tuple[int, int]] = []
        if limits.cpu_timeout is not None:
            secs = math.ceil(limits.cpu_timeout)
            rlimits.append((resource.RLIMIT_CPU, secs))
        if limits.max_memory_mb is not None:
            rlimits.append((resource.RLIMIT_AS, limits.max_memory_mb * 1024 * 1024))
        if limits.max_file_size_mb is not None:
            rlimits.append(
                (resource.RLIMIT_FSIZE, limits.max_file_size_mb * 1024 * 1024)
            )

        def _apply() -> None:
            for res_id, value in rlimits:
                resource.setrlimit(res_id, (value, value))

        return _apply

    async def _pump(
        self,
        reader: asyncio.StreamReader | None,
        stream: str,
        queue: asyncio.Queue[ExecChunk | None],
        limits: SupervisorLimits,
    ) -> None:
        """Read one pipe to EOF, decoding incrementally, and push chunks."""
        if reader is None:
            await queue.put(None)
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            raw = await reader.read(limits.read_chunk_size)
            if not raw:
                tail = decoder.decode(b"", final=True)
                if tail:
                    await queue.put(ExecChunk(stream=stream, data=tail))  # type: ignore[arg-type]
                await queue.put(None)
                return
            text = decoder.decode(raw)
            if text:
                await queue.put(ExecChunk(stream=stream, data=text))  # type: ignore[arg-type]

    @staticmethod
    async def _get(
        queue: asyncio.Queue[ExecChunk | None], wait_t: float | None
    ) -> ExecChunk | None:
        if wait_t is None:
            return await queue.get()
        return await asyncio.wait_for(queue.get(), wait_t)

    @staticmethod
    def _next_timeout(start: float, limits: SupervisorLimits) -> float | None:
        candidates: list[float] = []
        if limits.overall_timeout is not None:
            elapsed = time.monotonic() - start
            candidates.append(max(0.0, limits.overall_timeout - elapsed))
        if limits.idle_timeout is not None:
            candidates.append(limits.idle_timeout)
        return min(candidates) if candidates else None

    @staticmethod
    def _overall_expired(start: float, limits: SupervisorLimits) -> bool:
        return (
            limits.overall_timeout is not None
            and (time.monotonic() - start) >= limits.overall_timeout
        )

    @staticmethod
    def _cap(
        chunk: ExecChunk,
        emitted: dict[str, int],
        limits: SupervisorLimits,
        *,
        truncated: bool,
    ) -> tuple[ExecChunk | None, bool]:
        cap = limits.max_output_chars
        used = emitted[chunk.stream]
        if used >= cap:
            return None, True
        remaining = cap - used
        if len(chunk.data) <= remaining:
            emitted[chunk.stream] = used + len(chunk.data)
            return chunk, truncated
        emitted[chunk.stream] = cap
        return ExecChunk(stream=chunk.stream, data=chunk.data[:remaining]), True

    @staticmethod
    async def _await_exit(
        proc: asyncio.subprocess.Process, start: float, limits: SupervisorLimits
    ) -> int | None:
        """Wait for a clean exit, bounded by the remaining overall budget."""
        if limits.overall_timeout is None:
            return await proc.wait()
        remaining = max(0.0, limits.overall_timeout - (time.monotonic() - start))
        try:
            return await asyncio.wait_for(proc.wait(), remaining)
        except TimeoutError:
            return None

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process, limits: SupervisorLimits) -> int:
        """SIGTERM the process group, then SIGKILL after the grace period."""
        if proc.returncode is not None:
            return proc.returncode
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return proc.returncode if proc.returncode is not None else -1
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), limits.kill_grace_period)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            await proc.wait()
        return proc.returncode if proc.returncode is not None else -1


__all__ = ["ExecSpec", "ProcessSupervisor", "SupervisorLimits"]
