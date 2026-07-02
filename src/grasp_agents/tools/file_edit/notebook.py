"""
``NotebookRead`` / ``NotebookEdit`` — cell-structured access to ``.ipynb``
notebooks via ``ctx.file_backend``.

A notebook is JSON, but edits are *cell*-structured, not raw-text fuzzy
match — running ``Edit``'s replacement chain over notebook JSON would
corrupt it. These tools parse with :mod:`nbformat`, address cells by a
stable ``cell_id`` (indices shift on insert/delete; IDs do not), and
re-serialize so unrelated cells + notebook metadata survive verbatim.

They inherit the file-edit invariants over :class:`FileBackend`:

1. ``backend.validate_path`` — sandbox + sensitive-path policy.
2. **Read-before-write + mtime staleness refusal** for ``NotebookEdit``,
   identical to :class:`EditTool` (the model must have ``NotebookRead`` /
   ``Read`` the notebook this session and it must not have changed since).
3. Atomic write of the re-serialized notebook; permission bits preserved.
4. **Code-cell replace clears ``execution_count`` and ``outputs``** — stale
   output displayed next to new source is a correctness footgun.

Cell IDs: notebooks at nbformat >= 4.5 carry stable per-cell IDs. Older
notebooks have none; we mint them *deterministically* from
``(index, source)`` so the IDs :class:`NotebookReadTool` shows match the
ones :class:`NotebookEditTool` computes when it re-reads the unchanged
file — without ``NotebookRead`` having to write. The first ``NotebookEdit``
persists native IDs (re-serialized at nbformat minor 5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.tools.base import BaseTool, ToolProgressCallback
from grasp_agents.tools.cell_output import (
    DEFAULT_MAX_IMAGES,
    coerce_text,
    image_parts_from_data,
)
from grasp_agents.types.content import InputImage, InputText

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.session_context import SessionContext

    from .redact import SecretRedactor

# Per-cell cap on captured output text in the structured view / redirect.
# Full outputs live in the .ipynb; the model sees a preview.
DEFAULT_OUTPUT_PREVIEW_CHARS = 2_000
# Whole-file ceiling for parsing a notebook into memory. Generous —
# notebooks with embedded figures are large, but the returned view is
# always previewed down regardless of file size.
DEFAULT_MAX_NOTEBOOK_BYTES = 25_000_000


CellType = Literal["code", "markdown", "raw"]
EditMode = Literal["replace", "insert", "delete"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NotebookCellView(BaseModel):
    """A single cell as the model sees it — never the raw JSON."""

    id: str
    cell_type: CellType
    source: str
    execution_count: int | None = None
    has_output: bool = False
    output_preview: str | None = None


class NotebookReadInput(BaseModel):
    """Input schema for the ``NotebookRead`` tool."""

    path: str = Field(
        description=(
            "Path to the .ipynb notebook to read. Must exist and resolve "
            "under one of the backend's allowed roots."
        )
    )
    cell_id: str | None = Field(
        default=None,
        description=("If given, return only this cell; otherwise return all cells."),
    )
    include_images: bool | None = Field(
        default=None,
        description=(
            "Whether to return stored cell outputs as viewable images "
            "(plots/figures) in addition to the text view. Defaults to True "
            "when `cell_id` is set (you're inspecting one cell) and False for a "
            "whole-notebook read (keeps the overview small). Set explicitly to "
            "override."
        ),
    )


class NotebookReadResult(BaseModel):
    """Output schema for the ``NotebookRead`` tool."""

    path: str
    cells: list[NotebookCellView]
    total_cells: int


class NotebookEditInput(BaseModel):
    """Input schema for the ``NotebookEdit`` tool."""

    notebook_path: str = Field(
        description=(
            "Path to the .ipynb notebook to edit. Must exist and resolve "
            "under one of the backend's allowed roots."
        )
    )
    cell_id: str | None = Field(
        default=None,
        description=(
            "Target cell ID. Required for `replace` and `delete`. For "
            "`insert`, the new cell is added immediately after this cell; "
            "omit it to insert at the very start of the notebook."
        ),
    )
    new_source: str = Field(
        default="",
        description=(
            "New source for the cell. Required for `replace` and `insert`; "
            "ignored for `delete`."
        ),
    )
    cell_type: Literal["code", "markdown"] | None = Field(
        default=None,
        description=(
            "Cell type. Required for `insert`. Ignored for `replace` "
            "(a cell's type is not changed) and `delete`."
        ),
    )
    edit_mode: EditMode = Field(
        default="replace",
        description=(
            "`replace` (default): overwrite the target cell's source. "
            "`insert`: add a new cell. `delete`: remove the target cell."
        ),
    )


class NotebookEditResult(BaseModel):
    """Output schema for the ``NotebookEdit`` tool (compact — no full JSON)."""

    path: str
    edit_mode: EditMode
    cell_id: str | None
    cell_type: CellType | None
    total_cells: int


# ---------------------------------------------------------------------------
# nbformat helpers (lazy import — nbformat is an optional dependency)
# ---------------------------------------------------------------------------


def _nbformat() -> Any:
    try:
        import nbformat  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ValueError(
            "Notebook editing tools require the 'nbformat' package. "
            "Install it with `pip install grasp-agents[notebook-edit]`."
        ) from exc
    return nbformat


def _load_notebook(text: str) -> Any:
    nbf = _nbformat()
    try:
        return nbf.reads(text, as_version=4)
    except Exception as exc:
        raise ValueError(f"Not a valid Jupyter notebook (.ipynb): {exc}") from exc


def _cell_source(cell: Any) -> str:
    """Normalize a cell's ``source`` (nbformat keeps it as str or list)."""
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(str(line) for line in src)  # type: ignore[reportUnknownArgumentType]
    return src if isinstance(src, str) else ""


def _deterministic_cell_id(index: int, source: str, taken: set[str]) -> str:
    digest = hashlib.sha1(  # noqa: S324 - id derivation, not security
        f"{index}\x00{source}".encode()
    ).hexdigest()[:8]
    cell_id = digest
    suffix = 0
    while cell_id in taken:
        suffix += 1
        cell_id = f"{digest[:6]}{suffix:02d}"
    taken.add(cell_id)
    return cell_id


def _ensure_cell_ids(nb: Any) -> bool:
    """
    Mint stable IDs for any cell missing one. Deterministic in
    ``(index, source)`` so a read and a later edit of the *unchanged* file
    agree on IDs without the read having to persist anything. Returns
    whether the notebook was changed (so the edit path knows to bump the
    minor version when it re-serializes).
    """
    cells: list[Any] = list(nb.cells)
    taken: set[str] = {c["id"] for c in cells if c.get("id")}
    changed = False
    for index, cell in enumerate(cells):
        if not cell.get("id"):
            cell["id"] = _deterministic_cell_id(index, _cell_source(cell), taken)
            changed = True
    if changed and int(getattr(nb, "nbformat_minor", 0)) < 5:
        nb.nbformat_minor = 5
    return changed


def _new_cell(cell_type: str, source: str) -> Any:
    nbf = _nbformat()
    if cell_type == "code":
        return nbf.v4.new_code_cell(source)
    if cell_type == "markdown":
        return nbf.v4.new_markdown_cell(source)
    raise ValueError(f"Unsupported cell_type {cell_type!r}; use 'code' or 'markdown'.")


def make_output(output_type: str, **fields: Any) -> Any:
    """Build a schema-valid nbformat output node (``nbformat.v4.new_output``)."""
    return _nbformat().v4.new_output(output_type, **fields)


def cell_source(cell: Any) -> str:
    """Public accessor for a cell's source as a single string."""
    return _cell_source(cell)


def find_cell_index(nb: Any, cell_id: str) -> int | None:
    for index, cell in enumerate(nb.cells):
        if cell.get("id") == cell_id:
            return index
    return None


def _available_ids(nb: Any, limit: int = 20) -> str:
    ids = [str(c.get("id")) for c in nb.cells][:limit]
    more = "" if len(nb.cells) <= limit else f", … (+{len(nb.cells) - limit} more)"
    return ", ".join(ids) + more


def _output_preview(cell: Any, max_chars: int) -> tuple[bool, str | None]:
    """Summarize a code cell's outputs into a short, model-friendly preview."""
    if cell.get("cell_type") != "code":
        return False, None
    outputs: list[Any] = list(cell.get("outputs", []))
    if not outputs:
        return False, None

    parts: list[str] = []
    for out in outputs:
        output_type = out.get("output_type")
        if output_type == "stream":
            parts.append(coerce_text(out.get("text", "")))
        elif output_type == "error":
            err_name = out.get("ename", "Error")
            err_value = out.get("evalue", "")
            parts.append(f"{err_name}: {err_value}")
        elif output_type in {"execute_result", "display_data"}:
            data: dict[str, Any] = out.get("data", {})
            if "text/plain" in data:
                parts.append(coerce_text(data["text/plain"]))
            mimes = [m for m in data if m != "text/plain"]
            for mime in mimes:
                parts.append(f"[{mime} output]")
    preview = "\n".join(p for p in parts if p)
    if len(preview) > max_chars:
        preview = preview[:max_chars] + f"\n… [output truncated at {max_chars} chars]"
    return True, preview or None


def _build_cell_view(
    cell: Any, *, output_preview_chars: int = DEFAULT_OUTPUT_PREVIEW_CHARS
) -> NotebookCellView:
    cell_type = cell.get("cell_type", "raw")
    if cell_type not in {"code", "markdown", "raw"}:
        cell_type = "raw"
    has_output, preview = _output_preview(cell, output_preview_chars)
    return NotebookCellView(
        id=str(cell.get("id", "")),
        cell_type=cell_type,
        source=_cell_source(cell),
        execution_count=cell.get("execution_count"),
        has_output=has_output,
        output_preview=preview,
    )


def build_cell_views(
    nb: Any, *, output_preview_chars: int = DEFAULT_OUTPUT_PREVIEW_CHARS
) -> list[NotebookCellView]:
    """Build the structured, ID-addressable view of every cell."""
    return [
        _build_cell_view(cell, output_preview_chars=output_preview_chars)
        for cell in nb.cells
    ]


def render_cell_views(path: Path, views: list[NotebookCellView]) -> str:
    """Render the cell view as compact text for the generic ``Read`` redirect."""
    lines: list[str] = [f"Notebook: {path} ({len(views)} cell(s))", ""]
    for index, view in enumerate(views):
        header = f"[cell {index}] id={view.id} type={view.cell_type}"
        if view.execution_count is not None:
            header += f" exec_count={view.execution_count}"
        lines.extend([header, view.source or "(empty)"])
        if view.has_output:
            lines.extend(["--- output ---", view.output_preview or "(non-text output)"])
        lines.append("")
    return "\n".join(lines)


def collect_cell_images(
    cell: Any, *, max_images: int = DEFAULT_MAX_IMAGES
) -> list[InputImage]:
    """Extract viewable images from a cell's stored outputs (capped)."""
    images: list[InputImage] = []
    for out in cell.get("outputs", []):
        if len(images) >= max_images:
            break
        if out.get("output_type") in {"execute_result", "display_data"}:
            images += image_parts_from_data(
                out.get("data", {}), max_images - len(images)
            )
    return images


async def read_notebook(
    backend: Any, resolved: Path, *, max_bytes: int = DEFAULT_MAX_NOTEBOOK_BYTES
) -> tuple[Any, float]:
    """
    Read + parse a notebook, returning ``(notebook_node, mtime)``.

    Mints in-memory cell IDs (no write) so the returned view is
    ID-addressable. The ``mtime`` is the file's current mtime — the caller
    records it for read-before-write.
    """
    size = (await backend.stat(resolved)).size
    if size > max_bytes:
        raise ValueError(
            f"Notebook is {size:,} bytes, exceeding the maximum readable "
            f"size ({max_bytes:,})."
        )
    text, mtime = await backend.read_text(resolved)
    nb = _load_notebook(text)
    _ensure_cell_ids(nb)
    return nb, mtime


async def read_notebook_as_text(backend: Any, resolved: Path) -> tuple[str, float]:
    """Rendered cell-view text + mtime — used by the generic ``Read`` redirect."""
    nb, mtime = await read_notebook(backend, resolved)
    views = build_cell_views(nb)
    return render_cell_views(resolved, views), mtime


async def open_notebook_for_write(
    backend: Any, state: Any, notebook_path: str, *, overrides: set[Path] | None
) -> tuple[Path, Any, int]:
    """
    Validate + read a notebook for a mutating operation, returning
    ``(resolved, notebook_node, mode)``.

    Enforces the file-edit write invariants shared by ``NotebookEdit`` and
    ``RunCell``: path policy (write access), read-before-write, mtime staleness
    refusal, and permission-bit capture. Parses the notebook and mints stable
    cell ids. Raises ``ValueError`` on any violation.
    """
    try:
        resolved = await backend.validate_path(
            Path(notebook_path),
            must_exist=True,
            access="write",
            dotfile_overrides=overrides,
        )
    except PathAccessError as exc:
        raise ValueError(str(exc)) from exc

    if state is not None:
        record = state.get_read_record(resolved)
        if record is None:
            raise ValueError(
                f"Must Read {resolved} before modifying it. Read-before-write "
                "enforcement prevents acting on a notebook the model hasn't seen."
            )
        current_stat = await backend.stat(resolved)
        if current_stat.mtime != record.mtime:
            raise ValueError(
                f"{resolved} was modified since you last read it "
                f"(recorded mtime {record.mtime!r}, "
                f"current {current_stat.mtime!r}). Re-Read before modifying."
            )
    else:
        current_stat = await backend.stat(resolved)

    mode = current_stat.mode & 0o7777
    data, _ = await backend.read_bytes(resolved)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Cannot modify {resolved}: file is not valid UTF-8 ({exc.reason})."
        ) from exc

    nb = _load_notebook(text)
    _ensure_cell_ids(nb)
    return resolved, nb, mode


async def write_notebook(
    backend: Any, resolved: Path, nb: Any, *, mode: int, state: Any
) -> None:
    """Re-serialize ``nb`` and write it atomically, refreshing the read record."""
    new_bytes = _nbformat().writes(nb).encode("utf-8")
    new_mtime = await backend.write_bytes(
        resolved, new_bytes, mode=mode, overwrite=True
    )
    if state is not None:
        state.record_write(resolved, new_mtime)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class NotebookReadTool(
    BaseTool[
        NotebookReadInput, "NotebookReadResult | list[InputText | InputImage]", Any
    ]
):
    """
    Read a Jupyter notebook as a cell-addressable view (optionally with images).

    Returns a structured :class:`NotebookReadResult` for a text overview, or — when
    images are requested — content parts (the text view + the cells' stored
    figures as viewable images). Stateless: backend + read-state bookkeeping live
    on :attr:`SessionContext.file_backend` and the call's :class:`AgentContext`.
    """

    name = "NotebookRead"
    description = (
        "Read a Jupyter notebook (.ipynb) as a list of cells. Each cell has a "
        "stable `id`, a `cell_type` (code/markdown/raw), its `source`, and — "
        "for code cells that have been run — an execution count and a preview "
        "of its outputs. Use the cell `id` (not a position) to target a cell in "
        "a subsequent NotebookEdit/RunCell. Pass `cell_id` to inspect one cell, "
        "which also returns its stored output images (plots/figures) so you can "
        "see them; pass `include_images=True` to view figures across the whole "
        "notebook. Reading a notebook here (or with Read) is what lets you edit "
        "it afterward."
    )
    untrusted_output = True

    def __init__(
        self,
        *,
        output_preview_chars: int = DEFAULT_OUTPUT_PREVIEW_CHARS,
        max_notebook_bytes: int = DEFAULT_MAX_NOTEBOOK_BYTES,
        max_images: int = DEFAULT_MAX_IMAGES,
        timeout: float | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        from .redact import DefaultSecretRedactor  # noqa: PLC0415

        super().__init__(timeout=timeout)
        self._output_preview_chars = output_preview_chars
        self._max_notebook_bytes = max_notebook_bytes
        self._max_images = max_images
        # Same final pass as ``Read`` — notebook sources and stored outputs
        # can hold secrets just like any file content.
        self._redactor: SecretRedactor = redactor or DefaultSecretRedactor()

    async def _run(
        self,
        inp: NotebookReadInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> NotebookReadResult | list[InputText | InputImage]:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "NotebookRead requires ctx.file_backend. Wire a FileBackend on "
                "SessionContext before running the agent."
            )

        backend = ctx.file_backend
        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        try:
            resolved = await backend.validate_path(
                Path(inp.path),
                must_exist=True,
                dotfile_overrides=overrides,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        nb, mtime = await read_notebook(
            backend, resolved, max_bytes=self._max_notebook_bytes
        )
        if state is not None:
            state.record_read(resolved, mtime)

        cells: list[Any] = list(nb.cells)
        if inp.cell_id is not None:
            cells = [c for c in cells if c.get("id") == inp.cell_id]
            if not cells:
                raise ValueError(
                    f"No cell with id {inp.cell_id!r} in {resolved}. "
                    f"Available ids: {_available_ids(nb)}."
                )
        views = [
            self._redact_view(
                _build_cell_view(c, output_preview_chars=self._output_preview_chars)
            )
            for c in cells
        ]

        include_images = (
            inp.include_images
            if inp.include_images is not None
            else inp.cell_id is not None
        )
        if not include_images:
            return NotebookReadResult(
                path=str(resolved), cells=views, total_cells=len(nb.cells)
            )

        images: list[InputImage] = []
        for cell in cells:
            if len(images) >= self._max_images:
                break
            images += collect_cell_images(
                cell, max_images=self._max_images - len(images)
            )
        return [InputText(text=render_cell_views(resolved, views)), *images]

    def _redact_view(self, view: NotebookCellView) -> NotebookCellView:
        view.source = self._redactor(view.source)
        if view.output_preview is not None:
            view.output_preview = self._redactor(view.output_preview)
        return view


class NotebookEditTool(BaseTool[NotebookEditInput, NotebookEditResult, Any]):
    """
    Replace / insert / delete a single cell in a Jupyter notebook.

    Stateless: backend + read-state bookkeeping live on
    :attr:`SessionContext.file_backend` and the call's :class:`AgentContext`.
    """

    name = "NotebookEdit"
    description = (
        "Edit a Jupyter notebook (.ipynb) one cell at a time, addressed by "
        "the stable cell `id` from NotebookRead.\n"
        "\n"
        "* `edit_mode`:\n"
        "  - `replace` (default): overwrite the target cell's source — for a "
        "code cell this also clears its execution count and stored outputs;\n"
        "  - `insert`: add a new cell after `cell_id` (or at the start if "
        "`cell_id` is omitted); requires `cell_type`;\n"
        "  - `delete`: remove the target cell.\n"
        "* Requires a prior read of the notebook this session, unchanged on "
        "disk since. Unrelated cells and notebook metadata are preserved.\n"
        "* Returns the applied `edit_mode`, the affected `cell_id` / "
        "`cell_type`, and `total_cells` after the edit."
    )

    def __init__(self, *, timeout: float | None = None) -> None:
        super().__init__(timeout=timeout)

    def concurrency_conflict_keys(self, inp: NotebookEditInput) -> list[str] | None:
        return [inp.notebook_path]

    async def _run(
        self,
        inp: NotebookEditInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> NotebookEditResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "NotebookEdit requires ctx.file_backend. Wire a FileBackend on "
                "SessionContext before running the agent."
            )

        backend = ctx.file_backend
        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )
        resolved, nb, mode = await open_notebook_for_write(
            backend, state, inp.notebook_path, overrides=overrides
        )
        result_cell_id, result_cell_type = self._apply_edit(nb, inp, resolved)
        await write_notebook(backend, resolved, nb, mode=mode, state=state)

        return NotebookEditResult(
            path=str(resolved),
            edit_mode=inp.edit_mode,
            cell_id=result_cell_id,
            cell_type=result_cell_type,
            total_cells=len(nb.cells),
        )

    @staticmethod
    def _apply_edit(
        nb: Any, inp: NotebookEditInput, resolved: Path
    ) -> tuple[str | None, CellType | None]:
        """Mutate ``nb`` in place; return the affected cell's id + type."""
        if inp.edit_mode == "insert":
            if inp.cell_type is None:
                raise ValueError("cell_type is required for an insert.")
            new_cell = _new_cell(inp.cell_type, inp.new_source)
            if inp.cell_id is None:
                nb.cells.insert(0, new_cell)
            else:
                index = find_cell_index(nb, inp.cell_id)
                if index is None:
                    raise ValueError(
                        f"No cell with id {inp.cell_id!r} in {resolved}. "
                        f"Available ids: {_available_ids(nb)}."
                    )
                nb.cells.insert(index + 1, new_cell)
            return str(new_cell.get("id")), inp.cell_type

        if inp.cell_id is None:
            raise ValueError(f"cell_id is required for a {inp.edit_mode}.")
        index = find_cell_index(nb, inp.cell_id)
        if index is None:
            raise ValueError(
                f"No cell with id {inp.cell_id!r} in {resolved}. "
                f"Available ids: {_available_ids(nb)}."
            )
        cell = nb.cells[index]
        cell_type: CellType = cell.get("cell_type", "raw")

        if inp.edit_mode == "delete":
            nb.cells.pop(index)
            return inp.cell_id, cell_type

        # replace
        cell["source"] = inp.new_source
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        return inp.cell_id, cell_type


__all__ = [
    "DEFAULT_MAX_NOTEBOOK_BYTES",
    "DEFAULT_OUTPUT_PREVIEW_CHARS",
    "NotebookCellView",
    "NotebookEditInput",
    "NotebookEditResult",
    "NotebookEditTool",
    "NotebookReadInput",
    "NotebookReadResult",
    "NotebookReadTool",
    "build_cell_views",
    "cell_source",
    "collect_cell_images",
    "find_cell_index",
    "make_output",
    "open_notebook_for_write",
    "read_notebook",
    "read_notebook_as_text",
    "render_cell_views",
    "write_notebook",
]
