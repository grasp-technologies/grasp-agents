"""
``Edit`` — targeted string replacement inside an existing file, via
the configured :class:`FileBackend`.

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
10. Post-write mtime refresh on the session state so a following
    ``Edit`` doesn't see its own write as external drift.

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
    from ...run_context import RunContext
    from .backend import FileBackend
    from .session_state import FileEditSessionState
    from .store import FileEditStore


class EditInput(BaseModel):
    """Input schema for the ``Edit`` tool."""

    path: str = Field(
        description=(
            "Path to the file to edit. Must exist and must resolve under "
            "one of the toolkit's allowed roots."
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
    Find-and-replace a string inside an existing file via the configured backend.

    Attach via :class:`FileEditToolkit`; do not instantiate directly unless
    you're constructing a custom toolkit.
    """

    name = "Edit"
    description = (
        "Replace an exact text block inside an existing file. You must "
        "have Read the file earlier in this session and it must not have "
        "changed on disk since. Matching tolerates whitespace / "
        "indentation / smart-quote drift via a 9-strategy chain; "
        "ambiguous matches are refused unless `replace_all` is True. Use "
        "`Write` to create a new file or replace a file's whole content."
    )

    def __init__(
        self,
        *,
        store: FileEditStore,
        allowed_roots: list[str] | list[Any],
        backend: FileBackend | None = None,
        default_session_key: str = "default",
        include_dotfiles: bool = True,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        from .local_backend import LocalFileBackend  # noqa: PLC0415

        self._store = store
        self._backend = backend or LocalFileBackend()
        self._default_session_key = default_session_key
        self._allowed_roots: list[Path] = [Path(r) for r in allowed_roots]
        self._include_dotfiles = include_dotfiles

    async def _resolve_state(self, ctx: RunContext[Any] | None) -> FileEditSessionState:
        """Pick the session state this call should read/write."""
        if ctx is not None and ctx.file_edit_store is not None:
            return await ctx.file_edit_store.get_session_state(ctx.session_key)
        return await self._store.get_session_state(self._default_session_key)

    async def _run(
        self,
        inp: EditInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> EditResult:
        del exec_id, progress_callback

        if inp.old_string == inp.new_string:
            raise ValueError("old_string and new_string are identical; no-op refused.")

        if has_binary_extension(inp.path):
            raise ValueError(
                f"Cannot edit binary file {inp.path!r}. Edit is text-only."
            )

        state = await self._resolve_state(ctx)

        try:
            resolved = await self._backend.validate_path(
                Path(inp.path),
                self._allowed_roots,
                must_exist=True,
                dotfile_overrides=state.dotfile_overrides,
                include_dotfiles=self._include_dotfiles,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        record = state.get_read_record(resolved)
        if record is None:
            raise ValueError(
                f"Must Read {resolved} before editing it. "
                "Read-before-edit enforcement prevents editing files whose "
                "current content the model hasn't seen."
            )

        current_stat = await self._backend.stat(resolved)
        if current_stat.mtime != record.mtime:
            raise ValueError(
                f"{resolved} was modified since you last read it "
                f"(recorded mtime {record.mtime!r}, current {current_stat.mtime!r}). "
                "Re-Read before editing."
            )

        # Preserve existing permission bits across the edit. The backend
        # returns full ``st_mode``; mask off the type bits for chmod.
        mode = current_stat.mode & 0o7777

        original_bytes, _ = await self._backend.read_bytes(resolved)
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

        new_mtime = await self._backend.write_bytes(
            resolved, data, mode=mode, overwrite=True
        )

        state.record_write(resolved, new_mtime)

        return EditResult(
            path=str(resolved),
            edits_applied=len(matches),
            strategy=strategy or "unknown",
            bytes_written=len(data),
        )
