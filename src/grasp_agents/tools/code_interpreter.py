"""
``RunPython`` — a code-interpreter tool: run Python in a live kernel via
``ctx.exec_backend`` (a :class:`~grasp_agents.sandbox.kernel.KernelCapable`
backend) and return its output.

It surfaces results on two distinct channels, kept separate on purpose:

* **Display channel (to be *seen*)** — stdout, the last expression's value,
  tracebacks, and anything the code *displays* (``plt.show()``,
  ``IPython.display``) come back as text + inline images. This is how the model
  views a figure.
* **Artifact channel (to be *handed over*)** — paths the model lists in
  ``artifacts`` are reported by path / type / size (never inlined). This is how
  the model points at files it wrote (CSVs, checkpoints, saved figures) without
  dumping their bytes into the context.

Keeping them separate means a saved figure isn't also re-inlined (and a
checkpoint is never base64'd into the prompt); to *see* an image the model
displays it, to *hand over* a file it lists it.

``RunPython`` runs in its **own** persistent kernel (one per agent loop, via
``AgentContext.code_kernel_holder``) — a stateful REPL where variables/imports
persist across calls, so the model loads data once and explores it over several
calls. It is deliberately **not** the notebook kernel used by ``RunCell``: the
two are independent Python sessions. Used outside a loop, each call opens a
throwaway kernel.
"""

from __future__ import annotations

import mimetypes
import stat as stat_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..sandbox.kernel import CellOutput, CellResult, KernelCapable
from ..types.content import InputImage, InputText
from ..types.tool import BaseTool, ToolProgressCallback
from .cell_output import (
    DEFAULT_MAX_IMAGES,
    DEFAULT_OUTPUT_TEXT_CHARS,
    render_outputs_as_parts,
)
from .file_backend.paths import PathAccessError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..agent.agent_context import AgentContext
    from ..run_context import RunContext
    from .file_backend.base import FileBackend, FileStat

DEFAULT_CODE_TIMEOUT = 120.0
# Cap on files surfaced from ``artifacts`` in one call (bounds a directory
# expansion + the per-file stats); the rest are noted as omitted.
DEFAULT_MAX_ARTIFACT_FILES = 50

# POSIX file-type mask (S_IFMT). Present in ``st_mode`` on the local backend;
# absent on backends that report permission-only modes (e.g. E2B / MCP).
_S_IFMT = 0o170000


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{n} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


async def _is_directory(backend: FileBackend, resolved: Path, st: FileStat) -> bool:
    """
    Whether ``resolved`` is a directory, robust across backends.

    Prefers the POSIX type bits in ``st_mode`` (the local backend reports them);
    a backend that returns permission-only modes is probed with a listing.
    """
    if st.mode & _S_IFMT:
        return stat_module.S_ISDIR(st.mode)
    try:
        return bool(await backend.list_dir(resolved))
    except Exception:
        return False


def _file_reference_line(path: Path, st: FileStat) -> str:
    """One ``- path — mime, size`` artifact reference line."""
    mime, _ = mimetypes.guess_type(str(path))
    return f"- {path} — {mime or 'application/octet-stream'}, {_human_size(st.size)}"


def _outputs_to_render_dicts(outputs: Sequence[CellOutput]) -> list[dict[str, Any]]:
    """
    Convert :class:`CellOutput`s to the plain nbformat-shaped dicts
    :func:`render_outputs_as_parts` consumes — without taking an ``nbformat``
    dependency (these outputs are returned to the model, never persisted to a
    ``.ipynb``, so no schema validation or output sanitization is needed).
    """
    dicts: list[dict[str, Any]] = []
    for o in outputs:
        if o.output_type == "stream":
            dicts.append(
                {
                    "output_type": "stream",
                    "name": o.name or "stdout",
                    "text": o.text or "",
                }
            )
        elif o.output_type in {"execute_result", "display_data"}:
            dicts.append({"output_type": o.output_type, "data": dict(o.data or {})})
        else:  # error
            dicts.append(
                {
                    "output_type": "error",
                    "ename": o.ename or "",
                    "evalue": o.evalue or "",
                    "traceback": list(o.traceback),
                }
            )
    return dicts


class RunPythonInput(BaseModel):
    """Input schema for the ``RunPython`` tool."""

    code: str = Field(
        description=(
            "Python source to execute in the kernel. The last expression's "
            "value is returned (like a notebook cell)."
        )
    )
    artifacts: list[str] | None = Field(
        default=None,
        description=(
            "Files the code wrote to hand back to me (each a path to a file or "
            "a directory; relative to the working directory, or absolute). I "
            "report each by path, type, and size — not its contents. This does "
            "NOT display images: to show me a figure, display it inline in the "
            "code instead (see the tool description)."
        ),
    )
    timeout: float | None = Field(
        default=None,
        description=(
            f"Max seconds to wait for the code (default {DEFAULT_CODE_TIMEOUT:g}). "
            "On timeout the code is interrupted."
        ),
    )


class RunPython(BaseTool[RunPythonInput, list[InputText | InputImage], Any]):
    """
    Run Python in a live kernel and return its output (text + displayed images)
    plus references to any files named in ``artifacts``.

    Stateless: the kernel comes from the call's :class:`AgentContext`
    (``code_kernel_holder`` — its own, not the notebook's); files named in
    ``artifacts`` are read via :attr:`RunContext.file_backend`.
    """

    name = "RunPython"
    description = (
        "Run Python in a persistent, stateful kernel and return its output: "
        "stdout, the value of the last expression, tracebacks, and any images "
        "the code displays. Variables and imports persist across calls, so you "
        "can build up state like a REPL — load data once, then explore it over "
        "several calls. Prefer this over running `python -c` through Bash.\n"
        "There are two ways to surface results:\n"
        "1. To view an image, DISPLAY it inline — e.g. matplotlib `plt.show()`, "
        "or `from IPython.display import Image; display(Image('fig.png'))` to "
        "show an image already on disk. Displayed images are returned as image "
        "outputs. Saving a figure with `savefig` alone does NOT return it as an "
        "image — display it to view it.\n"
        "2. To surface a FILE the code wrote (a CSV, a checkpoint, a saved "
        "figure) without viewing it, list its path in `artifacts`; it is "
        "reported by path, type, and size — not its contents.\n"
        "Code runs synchronously and blocks the kernel until it finishes, so "
        "put long-running work (training loops, large downloads) in a "
        "background script via Bash and poll it — don't run it here."
    )

    def __init__(
        self,
        *,
        default_timeout: float = DEFAULT_CODE_TIMEOUT,
        max_result_chars: int = DEFAULT_OUTPUT_TEXT_CHARS,
        max_images: int = DEFAULT_MAX_IMAGES,
        max_artifact_files: int = DEFAULT_MAX_ARTIFACT_FILES,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._default_timeout = default_timeout
        self._max_result_chars = max_result_chars
        self._max_images = max_images
        self._max_artifact_files = max_artifact_files

    async def _run(
        self,
        inp: RunPythonInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> list[InputText | InputImage]:
        del exec_id, progress_callback, path

        if ctx is None or ctx.exec_backend is None:
            raise ValueError(
                "RunPython requires ctx.exec_backend. Wire an ExecBackend on "
                "RunContext (e.g. via local_environment(...)) before running."
            )
        exec_backend = ctx.exec_backend
        if not isinstance(exec_backend, KernelCapable):
            raise ValueError(
                f"RunPython requires a kernel-capable exec backend; "
                f"{exec_backend.name!r} cannot host a Jupyter kernel."
            )
        kernel_backend = exec_backend

        requested = inp.timeout if inp.timeout is not None else self._default_timeout

        holder = agent_ctx.code_kernel_holder if agent_ctx is not None else None
        own_kernel = holder is None
        kernel = (
            await kernel_backend.open_kernel()
            if holder is None
            else await holder.get(kernel_backend)
        )
        try:
            outputs: list[CellOutput] = []
            result: CellResult | None = None
            async for item in kernel.execute(inp.code, timeout=requested):
                if isinstance(item, CellResult):
                    result = item
                else:
                    outputs.append(item)
        finally:
            if own_kernel and not kernel.closed:
                await kernel.close()

        if result is None:  # pragma: no cover - execute always yields a result
            raise RuntimeError("kernel produced no terminal result")

        header = f"[execution_count={result.execution_count} status={result.status}]"
        if result.timed_out:
            header += " (timed out — code interrupted)"
        if holder is not None and holder.take_reset():
            header += (
                " (kernel was restarted — variables and imports from earlier "
                "calls were lost)"
            )
        parts = render_outputs_as_parts(
            _outputs_to_render_dicts(outputs),
            header=header,
            max_text_chars=self._max_result_chars,
            max_images=self._max_images,
        )
        if not inp.artifacts:
            return parts

        lines = await self._collect_artifacts(ctx, agent_ctx, inp.artifacts)
        if not lines:
            return parts
        artifact_text = InputText(text="\n\nFiles produced:\n" + "\n".join(lines))
        # Text (output + the file list) first, then the displayed images.
        return [parts[0], artifact_text, *parts[1:]]

    async def _collect_artifacts(
        self,
        ctx: RunContext[Any],
        agent_ctx: AgentContext | None,
        raw_paths: Sequence[str],
    ) -> list[str]:
        """Resolve ``artifacts`` to ``- path — mime, size`` reference lines."""
        backend = ctx.file_backend
        if backend is None:
            return ["(cannot surface artifacts: no file_backend on RunContext)"]

        exec_backend = ctx.exec_backend
        roots: Sequence[Path] = (
            exec_backend.policy.allowed_roots if exec_backend is not None else ()
        )
        base = roots[0] if roots else None  # the kernel's working directory
        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )

        lines: list[str] = []
        n_files = 0
        for raw in raw_paths:
            if n_files >= self._max_artifact_files:
                lines.append(
                    f"… (artifact limit {self._max_artifact_files} reached; "
                    "remaining not shown)"
                )
                break
            p = Path(raw)
            if not p.is_absolute() and base is not None:
                p = base / p
            try:
                resolved = await backend.validate_path(
                    p, must_exist=True, dotfile_overrides=overrides
                )
            except PathAccessError as exc:
                lines.append(f"- {raw} — not accessible: {exc}")
                continue
            st = await backend.stat(resolved)
            if await _is_directory(backend, resolved, st):
                entries = [e for e in await backend.list_dir(resolved) if not e.is_dir]
                entries.sort(key=lambda e: e.mtime, reverse=True)
                if not entries:
                    lines.append(f"- {resolved}/ — (empty directory)")
                for entry in entries:
                    if n_files >= self._max_artifact_files:
                        lines.append(
                            f"… (artifact limit {self._max_artifact_files} reached)"
                        )
                        break
                    lines.append(
                        _file_reference_line(entry.path, await backend.stat(entry.path))
                    )
                    n_files += 1
            else:
                lines.append(_file_reference_line(resolved, st))
                n_files += 1
        return lines


__all__ = [
    "DEFAULT_CODE_TIMEOUT",
    "DEFAULT_MAX_ARTIFACT_FILES",
    "RunPython",
    "RunPythonInput",
]
