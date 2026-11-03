"""
``Copy`` — copy a file or a directory tree via ``ctx.file_backend``.

Bytes move backend to backend inside the tool, so file content never
passes through the model. Across the mounts of a
:class:`~grasp_agents.file_backend.CompositeFileBackend` this is how files
enter a sandbox mount and how results come back.

Rules:

1. ``validate_path`` on the source (read) and on every destination file
   (write), so read-only mounts and sandbox policies apply.
2. A file copied onto an existing directory lands inside it under its own
   name. A directory's contents land under ``destination``, which is
   created if missing (``a/b.txt`` → ``destination/a/b.txt``).
3. Existing destination files are refused unless ``overwrite`` is set. The
   whole plan is checked before anything is written.
4. When overwriting a file the agent read this session, the write is
   conditional on the recorded mtime, as for ``Write``.
5. File-count and byte caps are checked before anything is written.
6. Every written file is recorded in the session state, so a following
   ``Edit`` needs no extra ``Read``.
"""

from __future__ import annotations

import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from grasp_agents.file_backend.base import StaleFileError
from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.tools.base import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.file_backend.base import FileBackend
    from grasp_agents.file_backend.paths import AccessMode
    from grasp_agents.session_context import SessionContext

DEFAULT_MAX_COPY_FILES = 1_000
DEFAULT_MAX_COPY_BYTES = 50 * 1024 * 1024
_DEFAULT_MODE = 0o644
_MAX_LISTED_CONFLICTS = 5


class CopyInput(BaseModel):
    """Input schema for the ``Copy`` tool."""

    source: str = Field(description="File or directory to copy. Must exist.")
    destination: str = Field(
        description=(
            "Target path. For a file: the new file's path, or an existing "
            "directory to copy into. For a directory: the directory that "
            "receives its contents (created if missing)."
        )
    )
    overwrite: bool = Field(
        default=False,
        description=(
            "Replace destination files that already exist. Without it, any "
            "existing destination file makes the whole copy refuse."
        ),
    )


class CopyResult(BaseModel):
    """Output schema for the ``Copy`` tool."""

    source: str
    destination: str
    files_copied: int
    bytes_copied: int


class _PathCheck(Protocol):
    async def __call__(
        self, raw: Path, *, must_exist: bool, access: AccessMode
    ) -> Path: ...


@dataclass(frozen=True)
class _Step:
    source: Path
    destination: Path
    size: int
    mode: int


class CopyTool(BaseTool[CopyInput, CopyResult, Any]):
    """Copy files without routing their content through the model."""

    name = "Copy"
    description = (
        "Copy a file or a whole directory to another path, including between "
        "mounts (for example into the sandbox and back). File content never "
        "passes through the conversation.\n"
        "\n"
        "* A file copied onto an existing directory lands inside it. A "
        "directory's contents land under `destination`, which is created if "
        "missing.\n"
        "* Existing destination files are refused unless `overwrite` is true; "
        "nothing is written when the copy is refused.\n"
        "* Copied files count as read, so you can `Edit` them right away.\n"
        "* Returns `files_copied` and `bytes_copied`."
    )

    def __init__(
        self,
        *,
        max_files: int = DEFAULT_MAX_COPY_FILES,
        max_bytes: int = DEFAULT_MAX_COPY_BYTES,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._max_files = max_files
        self._max_bytes = max_bytes

    def concurrency_conflict_keys(self, inp: CopyInput) -> list[str] | None:
        return [inp.destination]

    async def _run(
        self,
        inp: CopyInput,
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> CopyResult:
        del exec_id, progress_callback, path

        if ctx is None or ctx.file_backend is None:
            raise ValueError(
                "Copy requires ctx.file_backend. Wire a FileBackend on "
                "SessionContext before running the agent."
            )
        backend = ctx.file_backend
        state = agent_ctx.file_edit_state if agent_ctx is not None else None
        overrides = (
            set(state.dotfile_overrides)
            if state is not None and state.dotfile_overrides
            else None
        )

        async def check(raw: Path, *, must_exist: bool, access: AccessMode) -> Path:
            try:
                return await backend.validate_path(
                    raw,
                    must_exist=must_exist,
                    access=access,
                    dotfile_overrides=overrides,
                )
            except PathAccessError as exc:
                raise ValueError(str(exc)) from exc

        source = await check(Path(inp.source), must_exist=True, access="read")
        destination = await check(
            Path(inp.destination), must_exist=False, access="write"
        )
        if destination == source or source in destination.parents:
            raise ValueError(
                f"Cannot copy {source} onto or into itself ({destination})."
            )

        source_stat = await backend.stat(source)
        if _is_dir(source_stat.mode):
            target_root = destination
            steps = await self._plan_tree(backend, source, destination, check)
        else:
            target_root = destination
            if await backend.exists(destination) and _is_dir(
                (await backend.stat(destination)).mode
            ):
                target_root = await check(
                    destination / source.name, must_exist=False, access="write"
                )
            if not await backend.parent_exists(target_root):
                raise ValueError(
                    f"Parent directory for {target_root} does not exist. "
                    "Create it explicitly first."
                )
            steps = [
                _Step(
                    source, target_root, source_stat.size, _mode_bits(source_stat.mode)
                )
            ]

        total_bytes = sum(s.size for s in steps)
        if len(steps) > self._max_files:
            raise ValueError(
                f"Copy would write {len(steps)} files, over the limit of "
                f"{self._max_files}. Copy a smaller directory."
            )
        if total_bytes > self._max_bytes:
            raise ValueError(
                f"Copy would write {total_bytes:,} bytes, over the limit of "
                f"{self._max_bytes:,} bytes. Copy a smaller directory."
            )

        existing: list[Path] = []
        for step in steps:
            if not await backend.exists(step.destination):
                continue
            if _is_dir((await backend.stat(step.destination)).mode):
                raise ValueError(
                    f"{step.destination} is a directory; it cannot be overwritten."
                )
            existing.append(step.destination)
        if existing and not inp.overwrite:
            listed = ", ".join(str(p) for p in existing[:_MAX_LISTED_CONFLICTS])
            more = len(existing) - _MAX_LISTED_CONFLICTS
            suffix = f" and {more} more" if more > 0 else ""
            raise ValueError(
                f"{len(existing)} destination file(s) already exist: {listed}{suffix}. "
                "Pass overwrite=true to replace them."
            )

        copied = 0
        made_dirs: set[Path] = set()
        for step in steps:
            parent = step.destination.parent
            if parent not in made_dirs and parent != target_root.parent:
                await backend.mkdir(parent)
                made_dirs.add(parent)
            record = (
                state.get_read_record(step.destination) if state is not None else None
            )
            expected = (
                record.mtime
                if record is not None and step.destination in existing
                else None
            )
            data, _ = await backend.read_bytes(step.source)
            try:
                new_mtime = await backend.write_bytes(
                    step.destination,
                    data,
                    mode=step.mode,
                    overwrite=True,
                    expected_mtime=expected,
                )
            except StaleFileError as exc:
                raise ValueError(
                    f"{step.destination} was modified since you last read it "
                    f"({exc}). {copied} of {len(steps)} files were copied before "
                    "this one; Re-Read it before overwriting."
                ) from exc
            if state is not None:
                state.record_write(step.destination, new_mtime)
            copied += 1

        return CopyResult(
            source=str(source),
            destination=str(target_root),
            files_copied=copied,
            bytes_copied=total_bytes,
        )

    async def _plan_tree(
        self,
        backend: FileBackend,
        source: Path,
        destination: Path,
        check: _PathCheck,
    ) -> list[_Step]:
        steps: list[_Step] = []
        for entry in await backend.list_dir(source, recursive=True):
            if entry.is_dir:
                continue
            rel = entry.path.relative_to(source)
            target = await check(destination / rel, must_exist=False, access="write")
            st = await backend.stat(entry.path)
            steps.append(_Step(entry.path, target, st.size, _mode_bits(st.mode)))
        steps.sort(key=lambda s: s.destination.parts)
        return steps


def _is_dir(mode: int) -> bool:
    return bool(stat_module.S_IFMT(mode)) and stat_module.S_ISDIR(mode)


def _mode_bits(mode: int) -> int:
    return (mode & 0o7777) or _DEFAULT_MODE


__all__ = ["CopyInput", "CopyResult", "CopyTool"]
