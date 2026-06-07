"""
``RunCell`` — execute a notebook code cell against a live kernel via
``ctx.exec_backend`` (a :class:`~grasp_agents.sandbox.kernel.KernelCapable`
backend), capture its rich outputs, write them back into the ``.ipynb``, and
return them to the model as content parts (text + plots as viewable images:
png / jpeg / gif / webp).

It composes the two halves of the notebook tooling: the cell is read/written
through the same ``FileBackend`` as :mod:`.file_edit.notebook` (sharing its
read-before-write + mtime invariants), and run through the same environment as
``Bash`` (so the kernel inherits the co-located filesystem + confinement).

Each :class:`AgentLoop` owns one :class:`KernelHolder` (mirroring
``BashSessionHolder``): a single persistent kernel per loop, opened lazily,
fresh on deep-copy (sub-agent / replica isolation), closed at run end. State
carries across ``RunCell`` calls (imports, variables) like cells in a live
notebook. Used outside a loop, each call opens a throwaway kernel.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..sandbox.kernel import CellOutput, CellResult, KernelCapable
from ..types.content import InputImage, InputText
from ..types.tool import BaseTool, ToolProgressCallback
from .cell_output import (
    DEFAULT_MAX_IMAGES,
    DEFAULT_OUTPUT_TEXT_CHARS,
    render_outputs_as_parts,
    sanitize_output_data,
)
from .file_edit.notebook import (
    cell_source,
    find_cell_index,
    make_output,
    open_notebook_for_write,
    write_notebook,
)

if TYPE_CHECKING:
    from ..agent.agent_context import AgentContext
    from ..run_context import RunContext
    from ..sandbox.kernel import KernelSession

DEFAULT_CELL_TIMEOUT = 120.0


class KernelHolder:
    """
    Lazily opens and caches one persistent :class:`KernelSession` per agent loop.

    Mirrors :class:`~grasp_agents.tools.bash_session.BashSessionHolder`: owned by
    the loop, deep-copied fresh so sub-agents / parallel replicas each get their
    own kernel. Reopened if a previous kernel was closed (e.g. a timeout that
    killed it).

    Seed ``context_id`` to **re-attach** to an existing kernel on resume: read
    :attr:`context_id` before pausing/snapshotting the sandbox, persist it in
    your checkpoint, and pass it back here when rebuilding the loop. Backends
    that cannot persist kernel state (local) ignore it; sub-agents/replicas get
    fresh kernels (the seed is not deep-copied).
    """

    def __init__(self, context_id: str | None = None) -> None:
        self._kernel: KernelSession | None = None
        self._context_id = context_id
        self._lock = asyncio.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> KernelHolder:
        fresh = KernelHolder()
        memo[id(self)] = fresh
        return fresh

    async def get(self, backend: KernelCapable) -> KernelSession:
        async with self._lock:
            if self._kernel is None or self._kernel.closed:
                self._kernel = await backend.open_kernel(context_id=self._context_id)
            return self._kernel

    @property
    def context_id(self) -> str | None:
        """
        The live kernel's durable context id (persist it for resume), or the
        seeded id if no kernel is open yet (``None`` for non-persistable kernels).
        """
        kernel = self._kernel
        if kernel is not None and kernel.context_id is not None:
            return kernel.context_id
        return self._context_id

    def rebind(self, context_id: str | None) -> None:
        """
        Seed the context id to re-attach to on the next (re)open — used by resume
        to point a freshly-built loop's kernel at its persisted context. Takes
        effect when the kernel is next opened; does not disturb an open kernel.
        """
        self._context_id = context_id

    async def close(self) -> None:
        kernel = self._kernel
        self._kernel = None
        if kernel is not None and not kernel.closed:
            await kernel.close()


class RunCellInput(BaseModel):
    """Input schema for the ``RunCell`` tool."""

    notebook_path: str = Field(
        description="Path to the .ipynb notebook containing the cell to run."
    )
    cell_id: str = Field(
        description=(
            "ID of the code cell to execute (from NotebookRead). The cell's "
            "source is run in the notebook's kernel; its outputs are written "
            "back into the cell."
        )
    )
    timeout: float | None = Field(
        default=None,
        description=(
            f"Max seconds to wait for the cell (default {DEFAULT_CELL_TIMEOUT:g}). "
            "On timeout the cell is interrupted."
        ),
    )


def cell_output_to_nbformat(output: CellOutput, *, sanitize: bool = True) -> Any:
    """
    Convert a :class:`CellOutput` to a schema-valid nbformat output node.

    When ``sanitize`` (default), strips browser-executable payloads from rich
    outputs (``application/javascript``; ``<script>`` / ``on*=`` / ``javascript:``
    in ``text/html``) so the persisted ``.ipynb`` can't run agent-authored JS in
    a reviewer's frontend — see :func:`sanitize_output_data`.
    """
    if output.output_type == "stream":
        return make_output(
            "stream", name=output.name or "stdout", text=output.text or ""
        )
    if output.output_type in {"execute_result", "display_data"}:
        data = dict(output.data or {})
        if sanitize:
            data, _ = sanitize_output_data(data)
        if output.output_type == "execute_result":
            return make_output(
                "execute_result",
                data=data,
                metadata={},
                execution_count=output.execution_count,
            )
        return make_output("display_data", data=data, metadata={})
    return make_output(
        "error",
        ename=output.ename or "",
        evalue=output.evalue or "",
        traceback=list(output.traceback),
    )


class RunCell(BaseTool[RunCellInput, list[InputText | InputImage], Any]):
    """
    Execute a notebook code cell in a live kernel and return its outputs.

    Stateless: the kernel comes from the call's :class:`AgentContext`
    (``kernel_holder``); the notebook is read/written via
    :attr:`RunContext.file_backend`.
    """

    name = "RunCell"
    description = (
        "Execute a code cell of a Jupyter notebook (.ipynb) in its live kernel "
        "and return the outputs (stdout, results, errors, and plots as images). "
        "Address the cell by its `id` from NotebookRead. Kernel state — imports "
        "and variables — persists across RunCell calls, like running cells in a "
        "notebook. The cell's outputs and execution count are written back into "
        "the notebook. Long-running training belongs in a background script (via "
        "Bash), not a cell, which blocks the kernel until it finishes."
    )

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_CELL_TIMEOUT,
        max_result_chars: int = DEFAULT_OUTPUT_TEXT_CHARS,
        max_images: int = DEFAULT_MAX_IMAGES,
        sanitize_outputs: bool = True,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._default_timeout = default_timeout
        self._max_result_chars = max_result_chars
        self._max_images = max_images
        self._sanitize_outputs = sanitize_outputs

    async def _run(
        self,
        inp: RunCellInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> list[InputText | InputImage]:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "RunCell requires ctx.file_backend. Wire a FileBackend on "
                "RunContext before running the agent."
            )
        backend = ctx.file_backend
        exec_backend = ctx.exec_backend
        if exec_backend is None:
            raise ValueError(
                "RunCell requires ctx.exec_backend. Wire an ExecBackend on "
                "RunContext (e.g. via local_environment(...)) before running."
            )
        kernel_backend = (
            exec_backend if isinstance(exec_backend, KernelCapable) else None
        )
        if kernel_backend is None:
            raise ValueError(
                f"RunCell requires a kernel-capable exec backend; "
                f"{exec_backend.name!r} cannot host a Jupyter kernel."
            )

        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        resolved, nb, mode = await open_notebook_for_write(
            backend, state, inp.notebook_path, overrides=overrides
        )

        index = find_cell_index(nb, inp.cell_id)
        if index is None:
            raise ValueError(f"No cell with id {inp.cell_id!r} in {resolved}.")
        cell = nb.cells[index]
        if cell.get("cell_type") != "code":
            raise ValueError(
                f"Cell {inp.cell_id!r} is a {cell.get('cell_type')!r} cell; "
                "only code cells can be run."
            )
        source = cell_source(cell)

        requested = inp.timeout if inp.timeout is not None else self._default_timeout

        holder = agent_ctx.kernel_holder if agent_ctx is not None else None
        own_kernel = holder is None
        kernel = (
            await kernel_backend.open_kernel()
            if holder is None
            else await holder.get(kernel_backend)
        )
        try:
            outputs: list[CellOutput] = []
            result: CellResult | None = None
            async for item in kernel.execute(source, timeout=requested):
                if isinstance(item, CellResult):
                    result = item
                else:
                    outputs.append(item)
        finally:
            if own_kernel and not kernel.closed:
                await kernel.close()

        if result is None:  # pragma: no cover - execute always yields a result
            raise RuntimeError("kernel produced no terminal result")

        cell["execution_count"] = result.execution_count
        cell["outputs"] = [
            cell_output_to_nbformat(o, sanitize=self._sanitize_outputs) for o in outputs
        ]
        await write_notebook(backend, resolved, nb, mode=mode, state=state)

        header = f"[execution_count={result.execution_count} status={result.status}]"
        if result.timed_out:
            header += " (timed out — cell interrupted)"
        return render_outputs_as_parts(
            cell["outputs"],
            header=header,
            max_text_chars=self._max_result_chars,
            max_images=self._max_images,
        )


__all__ = [
    "DEFAULT_CELL_TIMEOUT",
    "KernelHolder",
    "RunCell",
    "RunCellInput",
    "cell_output_to_nbformat",
]
