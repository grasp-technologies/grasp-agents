"""
``LocalKernel`` — a local Jupyter kernel behind the :class:`KernelSession`
protocol, driven over ZMQ with :mod:`jupyter_client`.

It mirrors :class:`~grasp_agents.sandbox.local_session.LocalExecSession`: the
backend hands it the argv that launches the kernel (already wrapped by the
backend's confinement, exactly like ``_session_argv`` wraps ``/bin/sh``), and
the session owns the subprocess + its lifecycle. The kernel is launched
*manually* (``python -m ipykernel_launcher -f <connection_file>``) rather than
through ``KernelManager`` so the confinement wrapper applies to the kernel
process the same way it applies to a shell.

Transport is TCP loopback. The kernel's ZMQ channels need loopback, so the
confinement profile must permit it. Verified matrix (macOS):

* ``confinement="none"`` — works (the local-first default).
* ``confinement="srt"`` — works (the recommended confined env): the policy's
  egress allowlist + write confinement apply, and ``build_srt_settings`` sets
  ``allowLocalBinding`` so the kernel can bind its loopback ZMQ ports (loopback
  only — external egress stays allowlist-gated). srt manages its own writable
  temp dir for the sandboxed process, so no extra ``allowWrite`` entry is needed.
* ``confinement="seatbelt"`` with ``network=ALL`` — works (``network=ALL`` lets
  the kernel bind its loopback sockets; no egress allowlist, so prefer srt).

Output is bounded (stream text + image count) so a runaway cell cannot blow up
the context.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from typing import TYPE_CHECKING, Any

from ..kernel import CellOutput, CellResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence
    from pathlib import Path

DEFAULT_KERNEL_STARTUP_TIMEOUT = 60.0
DEFAULT_KILL_GRACE = 5.0
# Bound the captured output of a single cell. Full outputs still land in the
# notebook; these caps protect the in-memory collection + the model context.
DEFAULT_MAX_STREAM_CHARS = 200_000
DEFAULT_MAX_IMAGES = 50


class KernelStartError(RuntimeError):
    """The kernel process failed to launch or never became ready."""


class LocalKernel:
    """A local ``jupyter_client`` kernel. See the module docstring for the contract."""

    def __init__(
        self,
        *,
        launch_argv: Callable[[str], Sequence[str]],
        cwd: Path,
        env: Mapping[str, str],
        backend: str = "local",
        startup_timeout: float = DEFAULT_KERNEL_STARTUP_TIMEOUT,
        kill_grace: float = DEFAULT_KILL_GRACE,
        max_stream_chars: int = DEFAULT_MAX_STREAM_CHARS,
        max_images: int = DEFAULT_MAX_IMAGES,
    ) -> None:
        self._launch_argv = launch_argv
        self._cwd = cwd
        self._env = dict(env)
        self._backend = backend
        self._startup_timeout = startup_timeout
        self._kill_grace = kill_grace
        self._max_stream_chars = max_stream_chars
        self._max_images = max_images

        self._proc: asyncio.subprocess.Process | None = None
        self._client: Any = None
        self._connection_file: str | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def context_id(self) -> str | None:
        # A local kernel is a child process; its state cannot outlive it, so
        # there is no durable context to persist or re-attach to.
        return None

    # --- lifecycle ---------------------------------------------------------

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._client is not None:
            return
        from jupyter_client.asynchronous.client import (  # noqa: PLC0415
            AsyncKernelClient,
        )
        from jupyter_client.connect import write_connection_file  # noqa: PLC0415

        connection_file, _ = write_connection_file(ip="127.0.0.1", transport="tcp")
        self._connection_file = connection_file
        argv = tuple(self._launch_argv(connection_file))

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=self._cwd,
            env=self._env,
            start_new_session=True,
        )

        client = AsyncKernelClient(connection_file=connection_file)
        client.load_connection_file()
        client.start_channels()
        try:
            await client.wait_for_ready(timeout=self._startup_timeout)
        except Exception as exc:
            with contextlib.suppress(Exception):
                client.stop_channels()
            # Capture liveness before _terminate() nulls self._proc.
            died = self._proc.returncode is not None
            await self._terminate()
            self._cleanup_connection_file()
            self._closed = True
            detail = (
                "the kernel process exited during startup"
                if died
                else f"the kernel was not ready within {self._startup_timeout}s"
            )
            raise KernelStartError(
                f"Failed to start the local Jupyter kernel ({detail}). "
                "Ensure ipykernel is installed (grasp-agents[notebook-exec]); "
                "under a confined backend, loopback networking must be permitted."
            ) from exc
        self._client = client

    async def execute(
        self, code: str, *, timeout: float | None = None
    ) -> AsyncIterator[CellOutput | CellResult]:
        async with self._lock:
            if self._closed:
                raise RuntimeError("kernel session is closed; open a new one")
            start = time.monotonic()
            await self._ensure_started()
            client = self._client
            assert client is not None

            msg_id = client.execute(
                code,
                silent=False,
                store_history=True,
                user_expressions={},
                allow_stdin=False,
                stop_on_error=False,
            )

            stream_chars = 0
            n_images = 0
            truncated = False
            interrupted = False
            execution_count: int | None = None
            deadline = start + timeout if timeout is not None else None

            while True:
                wait = (
                    max(0.0, deadline - time.monotonic())
                    if deadline is not None
                    else None
                )
                try:
                    msg = await asyncio.wait_for(client.get_iopub_msg(), wait)
                except TimeoutError:
                    if interrupted:
                        await self._terminate()
                        self._cleanup_connection_file()
                        self._closed = True
                        yield CellResult(
                            status="abort",
                            execution_count=execution_count,
                            runtime_ms=(time.monotonic() - start) * 1000.0,
                            timed_out=True,
                        )
                        return
                    interrupted = True
                    await self.interrupt()
                    deadline = time.monotonic() + self._kill_grace
                    continue

                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                msg_type = msg.get("msg_type")
                content = msg.get("content", {})

                if msg_type == "status":
                    if content.get("execution_state") == "idle":
                        break
                    continue
                if msg_type == "execute_input":
                    execution_count = content.get("execution_count", execution_count)
                    continue
                if msg_type == "stream":
                    text = str(content.get("text", ""))
                    remaining = self._max_stream_chars - stream_chars
                    if remaining <= 0:
                        truncated = True
                        continue
                    if len(text) > remaining:
                        text = text[:remaining]
                        truncated = True
                    stream_chars += len(text)
                    yield CellOutput(
                        output_type="stream",
                        name=str(content.get("name", "stdout")),
                        text=text,
                    )
                elif msg_type in {"execute_result", "display_data"}:
                    data = dict(content.get("data", {}))
                    if "image/png" in data:
                        if n_images >= self._max_images:
                            data.pop("image/png", None)
                            truncated = True
                        else:
                            n_images += 1
                    yield CellOutput(
                        output_type=msg_type,  # type: ignore[arg-type]
                        data=data,
                        execution_count=content.get("execution_count"),
                    )
                elif msg_type == "error":
                    yield CellOutput(
                        output_type="error",
                        ename=str(content.get("ename", "")),
                        evalue=str(content.get("evalue", "")),
                        traceback=tuple(str(t) for t in content.get("traceback", [])),
                    )

            status = await self._collect_shell_status(client, msg_id)
            if truncated:
                yield CellOutput(
                    output_type="stream",
                    name="stderr",
                    text="\n[output truncated]",
                )
            yield CellResult(
                status=status[0],
                execution_count=status[1] if status[1] is not None else execution_count,
                runtime_ms=(time.monotonic() - start) * 1000.0,
            )

    @staticmethod
    async def _collect_shell_status(
        client: Any, msg_id: str
    ) -> tuple[Any, int | None]:
        """Drain the matching ``execute_reply`` for the final status + count."""
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                reply = await asyncio.wait_for(client.get_shell_msg(), timeout=1.0)
            except TimeoutError:
                break
            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            content = reply.get("content", {})
            return content.get("status", "ok"), content.get("execution_count")
        return "ok", None

    async def interrupt(self) -> None:
        pgid = self._pgid()
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGINT)

    async def restart(self) -> None:
        async with self._lock:
            await self._reset()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            await self._reset()

    # --- internals ---------------------------------------------------------

    async def _reset(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            with contextlib.suppress(Exception):
                client.stop_channels()
        await self._terminate()
        self._cleanup_connection_file()

    def _pgid(self) -> int | None:
        if self._proc is None:
            return None
        try:
            return os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return None

    async def _terminate(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        pgid = None
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pass
        if pgid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), self._kill_grace)
        except TimeoutError:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pgid, signal.SIGKILL)
            await proc.wait()

    def _cleanup_connection_file(self) -> None:
        path = self._connection_file
        self._connection_file = None
        if path is not None:
            with contextlib.suppress(OSError):
                from pathlib import Path  # noqa: PLC0415

                Path(path).unlink()


__all__ = [
    "DEFAULT_KERNEL_STARTUP_TIMEOUT",
    "KernelStartError",
    "LocalKernel",
]
