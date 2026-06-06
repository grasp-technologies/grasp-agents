"""
``Edit`` — targeted string replacement inside an existing file, via
``ctx.file_backend``.

Invariants (same family as :class:`WriteTool`, plus matching rules):

1. ``backend.validate_path`` — sandbox + sensitive-path policy + must
   exist (Edit doesn't create files; use ``Write`` for that).
2. **Read-before-edit**: target must have been ``Read`` this session.
3. **mtime staleness refusal**: current mtime must match what was recorded
   at the last ``Read``.
4. ``old_string`` non-empty and different from ``new_string``.
5. Fuzzy-match chain (``fuzzy_find``) must produce at least one match.
6. Multiple matches require ``replace_all=True``; otherwise refuse with
   count + strategy name so the model knows what to tighten.
7. Quote-convention preservation: if the *file's* matched slice uses
   curly quotes but ``new_string`` uses straight quotes, rewrite to
   curly before the replacement lands.
8. Atomic write (backend-specific).
9. Mode preservation — the file keeps its permission bits across the
   edit (local backend only).
10. Post-write mtime refresh by the backend so a following ``Edit``
    doesn't see its own write as external drift.

The fuzzy-match chain is **9 strategies deep** — see :mod:`fuzzy_match`
for the detailed contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .fuzzy_match import apply_replacements, fuzzy_find, preserve_quote_style
from .paths import PathAccessError, has_binary_extension

if TYPE_CHECKING:
    from ...agent.agent_context import AgentContext
    from ...run_context import RunContext


class EditInput(BaseModel):
    """Input schema for the ``Edit`` tool."""

    path: str = Field(
        description=(
            "Path to the file to edit. Must exist and must resolve under "
            "one of the backend's allowed roots."
        )
    )
    old_string: str = Field(
        description=(
            "Exact text to find. Must be present in the file (fuzzy "
            "matching is attempted if an exact match fails — whitespace, "
            "indentation, smart-quote, and similar drift is tolerated). "
            "Must be non-empty and different from `new_string`."
        ),
        min_length=1,
    )
    new_string: str = Field(description="Text to substitute in place of `old_string`.")
    replace_all: bool = Field(
        default=False,
        description=(
            "If True, replace every occurrence; if False (default), "
            "require the match to be unique — multiple matches are "
            "refused so the model is forced to add context."
        ),
    )


class EditResult(BaseModel):
    """Output schema for the ``Edit`` tool."""

    path: str
    edits_applied: int
    strategy: str  # which fuzzy-match strategy won
    bytes_written: int


class EditTool(BaseTool[EditInput, EditResult, Any]):
    """
    Find-and-replace a string inside an existing file via ``ctx.file_backend``.

    Stateless: backend, allowed_roots, and read-state bookkeeping all
    live on the :class:`FileBackend` wired onto :attr:`RunContext.file_backend`.
    """

    name = "Edit"
    description = (
        "Replace an exact text block inside an existing file. `old_string` "
        "must be non-empty — `Edit` is find-and-replace, not append. Treat "
        "`old_string` as a unique anchor: a line or block from the existing "
        "file at the place you want to edit. The replacement occurs where that "
        "anchor matches. To insert before/after the anchor, include the anchor "
        "and the new text in `new_string`. To append to a file, use the file's "
        "current final text as the anchor and include it followed by the "
        "appended text in `new_string`. You must have Read the file earlier in "
        "this session and it must not have changed on disk since. Fuzzy matching "
        "is attempted when the exact `old_string` is not found. Ambiguous matches "
        "are refused unless `replace_all` is True. To replace a file's whole "
        "content or to create a new file, use the `Write` tool instead."
    )

    def __init__(
        self,
        *,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)

    async def _run(
        self,
        inp: EditInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> EditResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Edit requires ctx.file_backend. Wire a FileBackend on "
                "RunContext before running the agent."
            )

        if Path(inp.path).suffix == ".ipynb":
            raise ValueError(
                "Refusing to Edit a Jupyter notebook (.ipynb) as raw text — it "
                "would corrupt the cell JSON. Use NotebookEdit (cell-addressed) "
                "to modify notebook cells."
            )

        if inp.old_string == inp.new_string:
            raise ValueError("old_string and new_string are identical; no-op refused.")

        if has_binary_extension(inp.path):
            raise ValueError(
                f"Cannot edit binary file {inp.path!r}. Edit is text-only."
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
                access="write",
                dotfile_overrides=overrides,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        if state is not None:
            record = state.get_read_record(resolved)
            if record is None:
                raise ValueError(
                    f"Must Read {resolved} before editing it. "
                    "Read-before-edit enforcement prevents editing files whose "
                    "current content the model hasn't seen."
                )

            current_stat = await backend.stat(resolved)
            if current_stat.mtime != record.mtime:
                raise ValueError(
                    f"{resolved} was modified since you last read it "
                    f"(recorded mtime {record.mtime!r}, "
                    f"current {current_stat.mtime!r}). "
                    "Re-Read before editing."
                )
        else:
            current_stat = await backend.stat(resolved)

        # Preserve existing permission bits across the edit. The backend
        # returns full ``st_mode``; mask off the type bits for chmod.
        mode = current_stat.mode & 0o7777

        original_bytes, _ = await backend.read_bytes(resolved)
        try:
            content = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Cannot edit {resolved}: file is not valid UTF-8 ({exc.reason})."
            ) from exc

        matches, strategy, error = fuzzy_find(content, inp.old_string)
        if error is not None:
            raise ValueError(error)
        if not matches:
            raise ValueError(
                f"Could not find a match for old_string in {resolved}. "
                "The 9-strategy chain (exact → line-trim → whitespace-norm "
                "→ indent-flex → escape-norm → boundary-trim → unicode-"
                "norm → block-anchor → context-aware) all failed. Re-Read "
                "the file and copy the exact block."
            )
        if len(matches) > 1 and not inp.replace_all:
            raise ValueError(
                f"Found {len(matches)} matches for old_string (strategy: "
                f"{strategy}) in {resolved}. Tighten old_string with more "
                "surrounding context, or pass replace_all=True."
            )

        # Quote preservation: if the file's matched slice has curly quotes
        # and new_string has straight quotes, rewrite straight→curly so the
        # edit doesn't flip the file's convention.
        first_start, first_end = matches[0]
        first_slice = content[first_start:first_end]
        adjusted_new = preserve_quote_style(first_slice, inp.new_string)

        new_content = apply_replacements(content, matches, adjusted_new)
        data = new_content.encode("utf-8")

        new_mtime = await backend.write_bytes(
            resolved,
            data,
            mode=mode,
            overwrite=True,
        )
        if state is not None:
            state.record_write(resolved, new_mtime)

        return EditResult(
            path=str(resolved),
            edits_applied=len(matches),
            strategy=strategy or "unknown",
            bytes_written=len(data),
        )
