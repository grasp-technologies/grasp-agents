"""
``Write`` — create or overwrite a file via ``ctx.file_backend``.

Invariants enforced on every call:

1. ``backend.validate_path`` — sandbox + sensitive-path policy. Local-FS
   backends layer credential-dotfile blocks on top of the system-path
   baseline; MCP backends trust their server's containment policy.
2. **Read-before-write** on existing files. The model must have ``Read``
   the target within this session; otherwise the write is refused.
   Fresh files (parent exists, target doesn't) skip this check.
3. **mtime staleness refusal** on existing files. If the file's current
   ``mtime`` differs from the one recorded at the last ``Read``, the
   write is refused with guidance to re-read.
4. Atomic write (backend-specific — local uses tmpfile + ``os.replace``;
   MCP backends defer to the server).
5. Preserve the existing file's permission bits when overwriting (local
   backend only — MCP servers may not honor the ``mode`` argument).

On success the backend refreshes its read record with the post-write
mtime so consecutive edits by the same agent don't re-trip the
staleness check on their own prior writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from .paths import PathAccessError

if TYPE_CHECKING:
    from ...agent.agent_context import AgentContext
    from ...run_context import RunContext

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
    Create or overwrite a file atomically via ``ctx.file_backend``.

    Stateless: backend, allowed_roots, and read-state bookkeeping all
    live on the :class:`FileBackend` wired onto :attr:`RunContext.file_backend`.
    """

    name = "Write"
    description = (
        "Create a new file or replace an existing file's entire content. "
        "Prefer `Edit` for any targeted change to an existing file "
        "(adding a line, updating a field, fixing a typo) — `Write` "
        "REPLACES the file, so anything not in your `content` is lost. "
        "For existing files you must have `Read` them earlier in this "
        "session and the file must not have changed on disk since — "
        "otherwise the write is refused. Parent directory must exist. "
        "Writes are atomic: a crash leaves either the old content or "
        "the new content, never partial bytes."
    )

    def __init__(
        self,
        *,
        new_file_mode: int = DEFAULT_NEW_FILE_MODE,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._new_file_mode = new_file_mode

    async def _run(
        self,
        inp: WriteInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> WriteResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Write requires ctx.file_backend. Wire a FileBackend on "
                "RunContext before running the agent."
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
                must_exist=False,
                access="write",
                dotfile_overrides=overrides,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        target_exists = await backend.exists(resolved)

        if target_exists:
            # Read-before-write only enforced inside an agent — standalone
            # tool use (state is None) is a power-user escape hatch.
            if state is not None:
                record = state.get_read_record(resolved)
                if record is None:
                    raise ValueError(
                        f"Must Read {resolved} before writing to it. "
                        "Read-before-write enforcement prevents clobbering "
                        "files whose current content the model hasn't seen."
                    )

                current_stat = await backend.stat(resolved)
                if current_stat.mtime != record.mtime:
                    raise ValueError(
                        f"{resolved} was modified since you last read it "
                        f"(recorded mtime {record.mtime!r}, "
                        f"current {current_stat.mtime!r}). Re-Read before writing."
                    )

                # Preserve existing mode when overwriting. Mask off the
                # file-type bits — the backend returns full ``st_mode``
                # so callers can isdir/isfile-check.
                mode = current_stat.mode & 0o7777
            else:
                current_stat = await backend.stat(resolved)
                mode = current_stat.mode & 0o7777
        else:
            mode = self._new_file_mode

        # Parent directory must exist — refuse to silently create missing
        # intermediate dirs.
        if not await backend.parent_exists(resolved):
            raise ValueError(
                f"Parent directory for {resolved} does not exist. "
                "Create it explicitly first."
            )

        data = inp.content.encode("utf-8")
        new_mtime = await backend.write_bytes(
            resolved,
            data,
            mode=mode,
            overwrite=True,
        )
        if state is not None:
            state.record_write(resolved, new_mtime)

        return WriteResult(
            path=str(resolved),
            bytes_written=len(data),
            created=not target_exists,
        )
