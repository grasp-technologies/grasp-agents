"""
``E2BKernel`` — a Jupyter kernel on an E2B code-interpreter sandbox (a
``KernelSession``), driving ``run_code`` over a dedicated code context and
mapping its rich ``Execution`` outputs onto our :class:`CellOutput`/
:class:`CellResult` seam.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any, cast

from e2b import TimeoutException

from grasp_agents.sandbox.kernel import CellOutput, CellResult, KernelSession

from ._handle import DEFAULT_EXEC_TIMEOUT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from e2b import AsyncSandbox
    from e2b_code_interpreter import AsyncSandbox as CodeInterpreterSandbox
    from e2b_code_interpreter.models import Context, Execution


_RESULT_MIME_ATTRS: tuple[tuple[str, str], ...] = (
    ("text", "text/plain"),
    ("html", "text/html"),
    ("markdown", "text/markdown"),
    ("svg", "image/svg+xml"),
    ("png", "image/png"),
    ("jpeg", "image/jpeg"),
    ("pdf", "application/pdf"),
    ("latex", "text/latex"),
    ("json", "application/json"),
    ("javascript", "application/javascript"),
)


def _execution_to_cell_outputs(execution: Execution) -> list[CellOutput]:
    """Map an e2b ``Execution`` to our nbformat-shaped :class:`CellOutput`s."""
    outs: list[CellOutput] = []
    logs = execution.logs
    stdout = "".join(logs.stdout) if logs and logs.stdout else ""
    if stdout:
        outs.append(CellOutput(output_type="stream", name="stdout", text=stdout))
    stderr = "".join(logs.stderr) if logs and logs.stderr else ""
    if stderr:
        outs.append(CellOutput(output_type="stream", name="stderr", text=stderr))
    for result in execution.results:
        data: dict[str, Any] = {}
        for attr, mime in _RESULT_MIME_ATTRS:
            value = getattr(result, attr, None)
            if value is not None:
                data[mime] = value
        if not data:
            continue
        if result.is_main_result:
            outs.append(
                CellOutput(
                    output_type="execute_result",
                    data=data,
                    execution_count=execution.execution_count,
                )
            )
        else:
            outs.append(CellOutput(output_type="display_data", data=data))
    error = execution.error
    if error is not None:
        traceback = error.traceback or ""
        outs.append(
            CellOutput(
                output_type="error",
                ename=error.name,
                evalue=error.value,
                traceback=tuple(traceback.splitlines()),
            )
        )
    return outs


class E2BKernel(KernelSession):
    """
    A Jupyter kernel on an E2B code-interpreter sandbox (a ``KernelSession``).

    Wraps ``sandbox.run_code`` over a dedicated **code context**, so sub-agents
    / parallel replicas get isolated kernel state on the shared sandbox (the
    kernel analogue of each loop getting its own :class:`E2BExecSession` shell).
    State persists across :meth:`execute`; :meth:`restart` resets the context.

    The context is **not** torn down on :meth:`close` — it lives with the
    sandbox (killed on teardown, preserved across E2B pause / snapshot). Persist
    :attr:`context_id` and pass it back to ``open_kernel(context_id=...)`` to
    **re-attach** to the same live kernel — this is what lets a session resumed
    from a checkpoint keep its in-memory variables instead of starting fresh.
    The notebook artifact lives on the same sandbox FS via the paired
    :class:`E2BFileBackend`.
    """

    def __init__(
        self,
        require_sandbox: Callable[[], AsyncSandbox],
        *,
        cwd: str,
        env: dict[str, str] | None,
        backend: str = "e2b",
        default_timeout: float = DEFAULT_EXEC_TIMEOUT,
        context_id: str | None = None,
    ) -> None:
        self._require = require_sandbox
        self._cwd = cwd
        self._env = env
        self._backend = backend
        self._default_timeout = default_timeout
        self._context_id = context_id
        self._context: Context | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def context_id(self) -> str | None:
        """
        Id of the live code context, or ``None`` before the first
        :meth:`execute`. Persist it (e.g. in your checkpoint, alongside the
        paused/snapshotted sandbox id) and pass it back to
        ``open_kernel(context_id=...)`` to re-attach to this kernel on resume.
        """
        if self._context is not None:
            return self._context.id
        return self._context_id

    def _sandbox(self) -> CodeInterpreterSandbox:
        """The live sandbox, narrowed to the code-interpreter type (has run_code)."""
        sandbox = self._require()
        if not hasattr(sandbox, "run_code"):
            raise RuntimeError(
                "E2B sandbox is not a code-interpreter sandbox (no run_code); "
                "build the environment with code_interpreter=True."
            )
        return cast("CodeInterpreterSandbox", sandbox)

    async def _bind_context(self, sandbox: CodeInterpreterSandbox) -> Context:
        """Re-attach to a persisted context id (resume), else create a fresh one."""
        if self._context_id is not None:
            from e2b_code_interpreter.models import Context  # noqa: PLC0415

            return Context(
                context_id=self._context_id, language="python", cwd=self._cwd
            )
        return await sandbox.create_code_context(cwd=self._cwd)

    async def execute(
        self, code: str, *, timeout: float | None = None
    ) -> AsyncIterator[CellOutput | CellResult]:
        if self._closed:
            raise RuntimeError("kernel session is closed; open a new one")
        eff_timeout = timeout if timeout is not None else self._default_timeout
        start = time.monotonic()
        timed_out = False
        execution: Execution | None = None
        async with self._lock:
            sandbox = self._sandbox()
            if self._context is None:
                self._context = await self._bind_context(sandbox)
            try:
                execution = await sandbox.run_code(
                    code,
                    context=self._context,
                    envs=self._env or None,
                    timeout=eff_timeout,
                )
            except TimeoutException:
                timed_out = True

        if execution is None:
            yield CellResult(
                status="error",
                runtime_ms=(time.monotonic() - start) * 1000.0,
                timed_out=timed_out,
            )
            return
        for output in _execution_to_cell_outputs(execution):
            yield output
        yield CellResult(
            status="error" if execution.error is not None else "ok",
            execution_count=execution.execution_count,
            runtime_ms=(time.monotonic() - start) * 1000.0,
        )

    async def interrupt(self) -> None:
        # E2B interrupts via run_code's own timeout (handled in execute); there
        # is no separate per-cell interrupt API, so this is a no-op.
        return

    async def restart(self) -> None:
        async with self._lock:
            context = self._context
            if context is not None:
                with contextlib.suppress(Exception):
                    await self._sandbox().restart_code_context(context)

    async def close(self) -> None:
        # The code context is intentionally left alive: it belongs to the
        # sandbox (which reaps it on teardown and preserves it across
        # pause/snapshot), so a run resumed from a checkpoint can re-attach to
        # it via ``context_id``. Use :meth:`restart` to reset kernel state.
        async with self._lock:
            self._closed = True
            self._context = None
