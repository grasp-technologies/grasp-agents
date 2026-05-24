"""
``Delete`` — remove a file via the configured :class:`FileBackend`.

Same safety family as :class:`WriteTool` and :class:`EditTool`:

1. ``backend.validate_path`` — sandbox + sensitive-path policy. Local-FS
   backends layer credential-dotfile blocks on top of the system-path
   baseline; MCP backends trust their server's containment policy.
2. **Read-before-delete**: the target must have been ``Read`` earlier
   in this session — prevents blindly deleting files the model hasn't
   seen.
3. **mtime staleness refusal**: the file's current ``mtime`` must match
   the one recorded at the last ``Read``. Re-read if it has changed.
4. Refuses to delete directories — :class:`DeleteTool` is file-only;
   removing a directory tree should be an explicit user-level action
   outside the agent loop.

Successful deletion clears the corresponding ``read_file_state`` entry
so a later ``Write`` to the same path creates a fresh file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .paths import PathAccessError

if TYPE_CHECKING:
    from ...run_context import RunContext
    from .backend import FileBackend
    from .session_state import FileEditSessionState
    from .store import FileEditStore


class DeleteInput(BaseModel):
    """Input schema for the ``Delete`` tool."""

    path: str = Field(
        description=(
            "Path to the file to delete. Must exist and must resolve "
            "under one of the toolkit's allowed roots."
        )
    )


class DeleteResult(BaseModel):
    """Output schema for the ``Delete`` tool."""

    path: str
    deleted: bool


class DeleteTool(BaseTool[DeleteInput, DeleteResult, Any]):
    """
    Delete a file via the configured backend.

    Attach via :class:`FileEditToolkit`; do not instantiate directly
    unless you're constructing a custom toolkit.
    """

    name = "Delete"
    description = (
        "Delete a single file. You must have Read the file earlier in "
        "this session and it must not have changed on disk since — "
        "otherwise the delete is refused. Refuses to delete "
        "directories. Atomic: a crash leaves either the file present or "
        "the file removed, never a partial state."
    )

    def __init__(
        self,
        *,
        store: FileEditStore,
        allowed_roots: list[Path] | list[str],
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
        inp: DeleteInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> DeleteResult:
        del exec_id, progress_callback

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

        current_stat = await self._backend.stat(resolved)
        # Reject deletes on directories — :class:`DeleteTool` is for
        # files. Walking and recursively deleting trees should be an
        # explicit user-level action.
        import stat as stat_module  # noqa: PLC0415

        if current_stat.mode and stat_module.S_IFMT(current_stat.mode):
            if stat_module.S_ISDIR(current_stat.mode):
                raise ValueError(
                    f"Delete refuses directories: {resolved}. Remove "
                    "individual files or perform the directory removal "
                    "outside the agent loop."
                )

        record = state.get_read_record(resolved)
        if record is None:
            raise ValueError(
                f"Must Read {resolved} before deleting it. "
                "Read-before-delete enforcement prevents removing files "
                "whose current content the model hasn't seen."
            )

        if current_stat.mtime != record.mtime:
            raise ValueError(
                f"{resolved} was modified since you last read it "
                f"(recorded mtime {record.mtime!r}, "
                f"current {current_stat.mtime!r}). Re-Read before deleting."
            )

        await self._backend.delete(resolved)

        # Drop the read record so a later Write to the same path creates
        # a fresh file (no leftover staleness check from the dead file).
        state.read_file_state.pop(resolved, None)

        return DeleteResult(path=str(resolved), deleted=True)
