"""
``Edit`` — targeted string replacement inside an existing file.

Invariants (same family as :class:`WriteTool`, plus matching rules):

1. Path must resolve under the toolkit's ``allowed_roots`` and must exist
   (Edit doesn't create files; use ``Write`` for that).
2. Sensitive-path deny list (system baseline + optional user dotfiles).
3. **Read-before-edit**: target must have been ``Read`` this session.
4. **mtime staleness refusal**: current mtime must match what was recorded
   at the last ``Read``.
5. ``old_string`` non-empty and different from ``new_string``.
6. Fuzzy-match chain (``fuzzy_find``) must produce at least one match.
7. Multiple matches require ``replace_all=True``; otherwise refuse with
   count + strategy name so the model knows what to tighten.
8. Quote-convention preservation: if the *file's* matched slice uses curly
   quotes but ``new_string`` uses straight quotes, rewrite to curly before
   the replacement lands. Prevents the edit from silently changing the
   file's quote convention.
9. Atomic write via ``tempfile.mkstemp`` + ``os.replace``.
10. Mode preservation — the file keeps its permission bits across the edit.
11. Post-write mtime refresh on the session state so a following ``Edit``
    doesn't see its own write as external drift.

The fuzzy-match chain is **9 strategies deep** — see :mod:`fuzzy_match` for
the detailed contract. The ``strategy`` field on :class:`EditResult`
surfaces which strategy won, which is useful when an edit succeeded via a
loose strategy (e.g. ``context_aware``) and the consumer wants to audit
the match before trusting it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .atomic_write import atomic_write_bytes
from .fuzzy_match import apply_replacements, fuzzy_find, preserve_quote_style
from .paths import (
    PathAccessError,
    check_sensitive_path,
    has_binary_extension,
    is_blocked_device,
    resolve_safe,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ...run_context import RunContext
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
    Find-and-replace a string inside an existing file.

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
        allowed_roots: list[Path],
        default_session_key: str = "default",
        include_dotfiles: bool = True,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._store = store
        self._default_session_key = default_session_key
        self._allowed_roots = allowed_roots
        self._include_dotfiles = include_dotfiles

    async def _resolve_state(self, ctx: RunContext[Any] | None) -> FileEditSessionState:
        """
        Pick the session state this call should read/write.

        Prefers the store + session key on ``ctx``; falls back to the
        tool's own store with ``default_session_key``.
        """
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
        session_id: str | None = None,
    ) -> EditResult:
        del exec_id, progress_callback, session_id

        if inp.old_string == inp.new_string:
            raise ValueError("old_string and new_string are identical; no-op refused.")

        state = await self._resolve_state(ctx)

        # Device-path guard — reject /dev/stdin etc. even if they happen to
        # land inside an allowed root.
        if is_blocked_device(inp.path):
            raise ValueError(
                f"Cannot edit device path {inp.path!r}: blocks or produces "
                "infinite output."
            )

        if has_binary_extension(inp.path):
            raise ValueError(
                f"Cannot edit binary file {inp.path!r}. Edit is text-only."
            )

        # Edit requires existing file — must_exist=True.
        try:
            resolved = resolve_safe(inp.path, self._allowed_roots, must_exist=True)
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        # Sensitive-path check (system baseline + optional dotfiles).
        err = check_sensitive_path(
            resolved,
            include_dotfiles=self._include_dotfiles,
            session_overrides=state.dotfile_overrides,
        )
        if err is not None:
            raise ValueError(err)

        # Read-before-edit enforcement.
        record = state.get_read_record(resolved)
        if record is None:
            raise ValueError(
                f"Must Read {resolved} before editing it. "
                "Read-before-edit enforcement prevents editing files whose "
                "current content the model hasn't seen."
            )

        # Staleness refusal — stat + compare.
        current_mtime = await asyncio.to_thread(lambda: resolved.stat().st_mtime)
        if current_mtime != record.mtime:
            raise ValueError(
                f"{resolved} was modified since you last read it "
                f"(recorded mtime {record.mtime!r}, current {current_mtime!r}). "
                "Re-Read before editing."
            )

        # Preserve existing mode on write.
        mode = resolved.stat().st_mode & 0o7777

        # Read current content off the event loop. Invalid bytes are
        # replaced rather than crashing — the model-facing view and the
        # read-state snapshot already tolerate this, and refusing here
        # would be surprising.
        original_bytes = await asyncio.to_thread(resolved.read_bytes)
        try:
            content = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Cannot edit {resolved}: file is not valid UTF-8 ({exc.reason})."
            ) from exc

        # Run the fuzzy chain.
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
        # edit doesn't flip the file's convention. For replace_all we use
        # the first match's slice as the reference — per-slice preservation
        # would require per-match replacement text, which complicates the
        # engine without clear payoff (identical occurrences usually share
        # quote style).
        first_start, first_end = matches[0]
        first_slice = content[first_start:first_end]
        adjusted_new = preserve_quote_style(first_slice, inp.new_string)

        new_content = apply_replacements(content, matches, adjusted_new)
        data = new_content.encode("utf-8")

        await asyncio.to_thread(
            atomic_write_bytes, resolved, data, mode=mode, overwrite=True
        )

        # Refresh read_file_state so a follow-up Edit in the same session
        # doesn't trip the staleness check on our own write.
        new_mtime = await asyncio.to_thread(lambda: resolved.stat().st_mtime)
        state.record_write(resolved, new_mtime)

        # ``strategy`` is guaranteed non-None by the match-list success
        # above. Use a conservative fallback for the return schema rather
        # than asserting — keeps the tool robust against future refactors
        # of ``fuzzy_find``'s return contract.
        return EditResult(
            path=str(resolved),
            edits_applied=len(matches),
            strategy=strategy or "unknown",
            bytes_written=len(data),
        )
