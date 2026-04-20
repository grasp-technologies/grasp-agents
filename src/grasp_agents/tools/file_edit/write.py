"""
``Write`` — create or overwrite a file under the toolkit's roots.

Invariants enforced on every call:

1. ``allowed_roots`` + sensitive-path deny list (system baseline +
   dotfile additions). No overwriting ``/etc/passwd`` or ``~/.ssh/id_rsa``.
2. **Read-before-write** on existing files. The model must have ``Read``
   the target within this session; otherwise the write is refused.
   Fresh files (parent exists, target doesn't) skip this check.
3. **mtime staleness refusal** on existing files. If the file's current
   ``mtime`` differs from the one recorded at the last ``Read``, the
   write is refused with guidance to re-read. Prevents the
   "model clobbers concurrent edit" bug.
4. Atomic tmpfile + ``os.replace`` so partial / torn files never appear.
5. Preserve the existing file's permission bits when overwriting.

On success the session state's ``read_file_state`` is refreshed with
the post-write mtime — consecutive edits by the same agent don't
re-trip the staleness check on their own prior writes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .atomic_write import atomic_write_bytes
from .paths import PathAccessError, check_sensitive_path, resolve_safe

if TYPE_CHECKING:
    from pathlib import Path

    from ...run_context import RunContext
    from .session_state import FileEditSessionState
    from .store import FileEditStore

# Permissions applied to newly-created files. Existing files preserve
# their current mode (e.g. an executable script stays executable).
DEFAULT_NEW_FILE_MODE = 0o644


class WriteInput(BaseModel):
    """Input schema for the ``Write`` tool."""

    path: str = Field(
        description=(
            "Path to the file to write. Created if it doesn't exist, "
            "overwritten atomically if it does."
        )
    )
    content: str = Field(description="Full file content to write.")


class WriteResult(BaseModel):
    """Output schema for the ``Write`` tool."""

    path: str
    bytes_written: int
    created: bool  # True if the file did not previously exist


class WriteTool(BaseTool[WriteInput, WriteResult, Any]):
    """
    Create or overwrite a file atomically.

    Attach via :class:`FileEditToolkit`; do not instantiate directly
    unless you're constructing a custom toolkit.
    """

    name = "Write"
    description = (
        "Create or overwrite a file. For existing files you must have "
        "Read them earlier in this session and the file must not have "
        "changed on disk since — otherwise the write is refused. Parent "
        "directory must exist. Writes are atomic: a crash leaves either "
        "the old content or the new content, never partial bytes. Use "
        "`Edit` for targeted string replacement in large files."
    )

    def __init__(
        self,
        *,
        store: FileEditStore,
        allowed_roots: list[Path],
        default_session_key: str = "default",
        include_dotfiles: bool = True,
        new_file_mode: int = DEFAULT_NEW_FILE_MODE,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._store = store
        self._default_session_key = default_session_key
        self._allowed_roots = allowed_roots
        self._include_dotfiles = include_dotfiles
        self._new_file_mode = new_file_mode

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
        inp: WriteInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> WriteResult:
        del exec_id, progress_callback

        state = await self._resolve_state(ctx)

        # Resolve with must_exist=False — Write may create new files.
        try:
            resolved = resolve_safe(inp.path, self._allowed_roots, must_exist=False)
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

        target_exists = resolved.exists()

        if target_exists:
            # Read-before-write enforcement.
            record = state.get_read_record(resolved)
            if record is None:
                raise ValueError(
                    f"Must Read {resolved} before writing to it. "
                    "Read-before-write enforcement prevents clobbering "
                    "files whose current content the model hasn't seen."
                )

            # Staleness refusal — file was modified since the last Read.
            current_mtime = await asyncio.to_thread(lambda: resolved.stat().st_mtime)
            if current_mtime != record.mtime:
                raise ValueError(
                    f"{resolved} was modified since you last read it "
                    f"(recorded mtime {record.mtime!r}, "
                    f"current {current_mtime!r}). Re-Read before writing."
                )

            # Preserve existing mode when overwriting.
            mode = resolved.stat().st_mode & 0o7777
        else:
            mode = self._new_file_mode

        # Parent directory must exist — refuse to silently create missing
        # intermediate dirs. If the model wants that, it can Write to the
        # parent first.
        if not resolved.parent.exists():
            raise ValueError(
                f"Parent directory does not exist: {resolved.parent}. "
                "Create it explicitly first."
            )

        data = inp.content.encode("utf-8")
        await asyncio.to_thread(
            atomic_write_bytes, resolved, data, mode=mode, overwrite=True
        )

        # Refresh read_file_state with the new mtime so the next Edit
        # doesn't see its own write as an external modification.
        new_mtime = await asyncio.to_thread(lambda: resolved.stat().st_mtime)
        state.record_write(resolved, new_mtime)

        return WriteResult(
            path=str(resolved),
            bytes_written=len(data),
            created=not target_exists,
        )
