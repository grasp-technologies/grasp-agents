"""
``E2BExecBackend`` — a :class:`ExecBackend` over an E2B sandbox's ``commands``
API, plus ``open_session`` (persistent shell) and ``open_kernel`` (Jupyter
kernel via code-interpreter).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from e2b import AsyncCommandHandle, CommandExitException

from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.sandbox.exec_backend import (
    ExecBackend,
    ExecChunk,
    ExecResult,
    SessionCapable,
    TerminationReason,
)
from grasp_agents.sandbox.kernel import KernelCapable

from ._handle import (
    DEFAULT_EXEC_TIMEOUT,
    MAX_OUTPUT_CHARS,
    SandboxHandle,
    is_timeout,
    normalize_posix,
    wire,
    wrap_stdin,
)
from .kernel import E2BKernel
from .session import E2BExecSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from grasp_agents.sandbox.policy import SandboxPolicy


class E2BExecBackend(ExecBackend, SessionCapable, KernelCapable):
    """
    :class:`ExecBackend` over an E2B sandbox's ``commands`` API.

    Bridges E2B's callback-driven streaming onto our ``ExecChunk`` generator,
    returns non-zero exits as data (catching ``CommandExitException`` instead
    of letting it raise), and maps ``timeout=None`` to a bounded default
    (E2B would otherwise impose its own 60 s ceiling). Co-located with the
    paired :class:`E2BFileBackend` via the shared :class:`SandboxHandle`.
    """

    def __init__(
        self,
        holder: SandboxHandle,
        *,
        policy: SandboxPolicy,
        name: str = "e2b",
        default_timeout: float = DEFAULT_EXEC_TIMEOUT,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self._holder = holder
        self._policy = policy
        self._name = name
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def execute(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        out: list[str] = []
        err: list[str] = []
        terminal: ExecResult | None = None
        async for item in self.stream(
            command, cwd=cwd, timeout=timeout, stdin=stdin, env=env
        ):
            if isinstance(item, ExecChunk):
                (out if item.stream == "stdout" else err).append(item.data)
            else:
                terminal = item
        if terminal is None:  # pragma: no cover - stream always yields one
            raise RuntimeError("E2B stream produced no terminal ExecResult")
        return replace(terminal, stdout="".join(out), stderr="".join(err))

    async def stream(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        sb = self._holder.require()

        wrapped = wrap_stdin(command, stdin)
        eff_timeout = timeout if timeout is not None else self._default_timeout
        cwd_str = self._resolve_cwd(cwd)
        envs = self._merged_env(env)

        queue: asyncio.Queue[ExecChunk | None] = asyncio.Queue()
        emitted = 0
        truncated = False

        def _push(stream: Literal["stdout", "stderr"], data: str) -> None:
            nonlocal emitted, truncated
            if truncated:
                return
            budget = self._max_output_chars - emitted
            if budget <= 0:
                truncated = True
                return
            if len(data) > budget:
                data = data[:budget]
                truncated = True
            emitted += len(data)
            queue.put_nowait(ExecChunk(stream=stream, data=data))

        def on_stdout(data: str) -> None:
            _push("stdout", data)

        def on_stderr(data: str) -> None:
            _push("stderr", data)

        start = time.monotonic()
        try:
            handle = await sb.commands.run(
                wrapped,
                background=True,
                cwd=cwd_str,
                envs=envs,
                timeout=eff_timeout,
                stdin=False,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
        except BaseException as exc:
            reason = (
                TerminationReason.OVERALL_TIMEOUT
                if is_timeout(exc)
                else TerminationReason.SPAWN_ERROR
            )
            yield ExecChunk(stream="stderr", data=str(exc))
            yield ExecResult(
                stdout="",
                stderr="",
                returncode=-1,
                reason=reason,
                runtime_ms=(time.monotonic() - start) * 1000,
                backend=self._name,
            )
            return

        outcome: dict[str, BaseException] = {}

        async def _drain() -> None:
            try:
                await handle.wait()
            except BaseException as exc:
                outcome["exc"] = exc
            finally:
                queue.put_nowait(None)

        waiter = asyncio.create_task(_drain())
        completed = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    completed = True
                    break
                yield item
        finally:
            # Early exit (the consumer cancelled / closed us — e.g. KillTask):
            # kill the remote command so it doesn't keep running in the sandbox,
            # then cancel the drain (its ``handle.wait()`` would otherwise block
            # forever on a long-running command). On normal completion the drain
            # already finished, so just reap it.
            if not completed and not waiter.done():
                with contextlib.suppress(Exception):
                    await handle.kill()
                waiter.cancel()
            with contextlib.suppress(BaseException):
                await waiter

        yield self._terminal(
            handle,
            outcome.get("exc"),
            runtime_ms=(time.monotonic() - start) * 1000,
            truncated=truncated,
        )

    def _terminal(
        self,
        handle: AsyncCommandHandle,
        exc: BaseException | None,
        *,
        runtime_ms: float,
        truncated: bool,
    ) -> ExecResult:
        if exc is not None and is_timeout(exc):
            return ExecResult(
                stdout="",
                stderr="",
                returncode=-1,
                reason=TerminationReason.OVERALL_TIMEOUT,
                runtime_ms=runtime_ms,
                backend=self._name,
                truncated=truncated,
            )
        if exc is not None and isinstance(exc, CommandExitException):
            rc = int(getattr(exc, "exit_code", 1) or 1)
            return ExecResult(
                stdout="",
                stderr="",
                returncode=rc,
                reason=TerminationReason.EXIT,
                runtime_ms=runtime_ms,
                backend=self._name,
                truncated=truncated,
            )
        if exc is not None:
            raise exc  # genuine sandbox/transport failure — surface it
        rc = int(handle.exit_code or 0)
        return ExecResult(
            stdout="",
            stderr="",
            returncode=rc,
            reason=TerminationReason.EXIT,
            runtime_ms=runtime_ms,
            backend=self._name,
            truncated=truncated,
        )

    def _resolve_cwd(self, cwd: Path | None) -> str:
        roots = self._policy.allowed_roots
        if not roots:
            raise PathAccessError(
                "SandboxPolicy has no allowed_roots; cannot choose a working "
                "directory for the command."
            )
        if cwd is None:
            return wire(roots[0])
        # Normalize before containment: a literal ".." would pass the prefix
        # check here and escape the roots when the VM resolves it.
        candidate = normalize_posix(cwd)
        for root in roots:
            root_posix = normalize_posix(root)
            if candidate == root_posix or root_posix in candidate.parents:
                return str(candidate)
        raise PathAccessError(f"cwd {cwd} is outside the policy's allowed_roots")

    def _merged_env(self, env: Mapping[str, str] | None) -> dict[str, str] | None:
        merged = dict(self._policy.env)
        if env:
            merged.update(env)
        return merged or None

    async def open_session(
        self, *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> E2BExecSession:
        """Open a persistent shell on the sandbox (see :class:`E2BExecSession`)."""
        return E2BExecSession(
            self._holder.require,
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
            max_output_chars=self._max_output_chars,
        )

    async def open_kernel(
        self,
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        context_id: str | None = None,
    ) -> E2BKernel:
        """
        Open a Jupyter kernel on the sandbox (see :class:`E2BKernel`).

        Requires a **code-interpreter** sandbox (``e2b_environment(...,
        code_interpreter=True)``) — the base E2B template is shell + files only.

        Pass ``context_id`` (a value previously read from
        :attr:`E2BKernel.context_id`) to **re-attach** to an existing code
        context instead of creating a fresh one — e.g. when resuming a session
        whose sandbox was paused/snapshotted, so the kernel's variables persist.
        """
        sandbox = self._holder.require()
        if not hasattr(sandbox, "run_code"):
            raise ValueError(
                "RunCell on E2B needs a code-interpreter sandbox. Build the "
                "environment with e2b_environment(..., code_interpreter=True) "
                "(needs the e2b-code-interpreter package, in the 'e2b' extra)."
            )
        return E2BKernel(
            self._holder.require,
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
            default_timeout=self._default_timeout,
            context_id=context_id,
        )
