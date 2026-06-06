"""
``Delete`` — remove a file via ``ctx.file_backend``.

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

Successful deletion clears the corresponding read record in the
backend's session state so a later ``Write`` to the same path creates a
fresh file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from ..file_backend.paths import PathAccessError

if TYPE_CHECKING:
    from ...agent.agent_context import AgentContext
    from ...run_context import RunContext


class DeleteInput(BaseModel):
    """Input schema for the ``Delete`` tool."""

    path: str = Field(
        description=(
            "Path to the file to delete. Must exist and must resolve "
            "under one of the backend's allowed roots."
        )
    )


class DeleteResult(BaseModel):
    """Output schema for the ``Delete`` tool."""

    path: str
    deleted: bool


class DeleteTool(BaseTool[DeleteInput, DeleteResult, Any]):
    """
    Delete a file via ``ctx.file_backend``.

    Stateless: backend, allowed_roots, and read-state bookkeeping all
    live on the :class:`FileBackend` wired onto :attr:`RunContext.file_backend`.
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
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)

    async def _run(
        self,
        inp: DeleteInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> DeleteResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Delete requires ctx.file_backend. Wire a FileBackend on "
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
                must_exist=True,
                access="write",
                dotfile_overrides=overrides,
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        current_stat = await backend.stat(resolved)
        # Reject deletes on directories — :class:`DeleteTool` is for
        # files. Walking and recursively deleting trees should be an
        # explicit user-level action.
        import stat as stat_module  # noqa: PLC0415

        if (
            current_stat.mode
            and stat_module.S_IFMT(current_stat.mode)
            and stat_module.S_ISDIR(current_stat.mode)
        ):
            raise ValueError(
                f"Delete refuses directories: {resolved}. Remove "
                "individual files or perform the directory removal "
                "outside the agent loop."
            )

        if state is not None:
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

        await backend.delete(resolved)
        if state is not None:
            state.read_file_state.pop(resolved, None)

        return DeleteResult(path=str(resolved), deleted=True)
