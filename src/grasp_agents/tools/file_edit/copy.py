"""
``Copy`` — copy a file or a directory tree via ``ctx.file_backend``.

Bytes move backend to backend inside the tool, so file content never
passes through the model. Across the mounts of a
:class:`~grasp_agents.file_backend.CompositeFileBackend` this is how files
enter a sandbox mount and how results come back.

Rules:

1. ``validate_path`` on the source (read) and on every destination file
   (write), so read-only mounts and sandbox policies apply. Inside a
   directory every file is validated for reading too. A file the policy
   refuses, a symbolic link (its resolved path is not the listed one) and
   anything that is not a regular file are skipped, never read, and listed
   in ``skipped``.
2. A file copied onto an existing directory lands inside it under its own
   name. A directory's contents land under ``destination``, which is
   created if missing (``a/b.txt`` → ``destination/a/b.txt``).
3. The whole plan is checked before anything is written: the file count
   right after listing, the byte cap, destination files that already exist,
   and destination directories that already exist as files.
4. Existing destination files are refused unless ``overwrite`` is set.
   Inside an agent, overwriting a file also needs a read record of it,
   unchanged since, as ``Write`` does; the write is conditional on that
   record's mtime.
5. Every source file is read before anything is written, and the byte cap
   counts the bytes actually read, not the sizes the backend reported.
6. Every written file is recorded in the session state, so a following
   ``Edit`` needs no extra ``Read``.
"""

from __future__ import annotations

import stat as stat_module
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from grasp_agents.file_backend.paths import PathAccessError
from grasp_agents.tools.base import BaseTool, ToolProgressCallback

from .session_state import refuse_if_stale

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.file_backend.base import FileBackend
    from grasp_agents.file_backend.paths import AccessMode
    from grasp_agents.session_context import SessionContext

    from .session_state import FileEditSessionState

DEFAULT_MAX_COPY_FILES = 1_000
DEFAULT_MAX_COPY_BYTES = 50 * 1024 * 1024
_DEFAULT_MODE = 0o644
_MAX_LISTED_CONFLICTS = 5
_MAX_LISTED_SKIPS = 20


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
    """
    Output schema for the ``Copy`` tool.

    ``skipped`` lists the first skipped source files as ``"<path>: <reason>"``;
    ``files_skipped`` counts all of them.
    """

    source: str
    destination: str
    files_copied: int
    bytes_copied: int
    files_skipped: int = 0
    skipped: list[str] = Field(default_factory=list[str])


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
        "* In a directory, symbolic links, special files and files you may not "
        "read are skipped and listed in `skipped`.\n"
        "* Existing destination files are refused unless `overwrite` is true. "
        "As with `Write`, overwriting a file also needs a `Read` of it earlier "
        "this session, unchanged since. Nothing is written when the copy is "
        "refused.\n"
        "* Copied files count as read, so you can `Edit` them right away.\n"
        "* Returns `files_copied`, `bytes_copied` and what was skipped."
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

        async def validate(raw: Path, *, must_exist: bool, access: AccessMode) -> Path:
            return await backend.validate_path(
                raw, must_exist=must_exist, access=access, dotfile_overrides=overrides
            )

        async def check(raw: Path, *, must_exist: bool, access: AccessMode) -> Path:
            try:
                return await validate(raw, must_exist=must_exist, access=access)
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

        skipped: list[str] = []
        source_stat = await backend.stat(source)
        if _is_dir(source_stat.mode):
            target_root = destination
            steps = await self._plan_tree(
                backend, source, destination, validate, check, skipped
            )
        else:
            if not _may_be_regular(source_stat.mode):
                raise ValueError(f"{source} is not a regular file or a directory.")
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

        reported_bytes = sum(s.size for s in steps)
        if reported_bytes > self._max_bytes:
            raise ValueError(
                f"Copy would write {reported_bytes:,} bytes, over the limit of "
                f"{self._max_bytes:,} bytes. Copy a smaller directory."
            )
        await _refuse_files_in_the_way(backend, steps)
        expected = await _check_overwrites(
            backend, steps, state, overwrite=inp.overwrite
        )
        contents = await self._read_sources(backend, steps)

        copied = 0
        made_dirs: set[Path] = set()
        for step, data in zip(steps, contents, strict=True):
            parent = step.destination.parent
            if parent not in made_dirs and parent != target_root.parent:
                await backend.mkdir(parent)
                made_dirs.add(parent)
            with refuse_if_stale(
                step.destination,
                action="overwriting it",
                note=f"{copied} of {len(steps)} files were copied before this one. ",
            ):
                new_mtime = await backend.write_bytes(
                    step.destination,
                    data,
                    mode=step.mode,
                    overwrite=True,
                    expected_mtime=expected.get(step.destination),
                )
            if state is not None:
                state.record_write(step.destination, new_mtime)
            copied += 1

        return CopyResult(
            source=str(source),
            destination=str(target_root),
            files_copied=copied,
            bytes_copied=sum(len(data) for data in contents),
            files_skipped=len(skipped),
            skipped=skipped[:_MAX_LISTED_SKIPS],
        )

    async def _plan_tree(
        self,
        backend: FileBackend,
        source: Path,
        destination: Path,
        validate: _PathCheck,
        check: _PathCheck,
        skipped: list[str],
    ) -> list[_Step]:
        listed = [
            e for e in await backend.list_dir(source, recursive=True) if not e.is_dir
        ]
        if len(listed) > self._max_files:
            raise ValueError(
                f"{source} holds {len(listed):,} files, over the limit of "
                f"{self._max_files:,}. Copy a smaller directory."
            )
        steps: list[_Step] = []
        for entry in sorted(listed, key=lambda e: e.path.parts):
            try:
                resolved = await validate(entry.path, must_exist=True, access="read")
            except PathAccessError as exc:
                skipped.append(f"{entry.path}: {exc}")
                continue
            if resolved != entry.path or source not in resolved.parents:
                skipped.append(f"{entry.path}: a symbolic link, not followed")
                continue
            st = await backend.stat(resolved)
            if not _may_be_regular(st.mode):
                skipped.append(f"{entry.path}: not a regular file")
                continue
            target = await check(
                destination / resolved.relative_to(source),
                must_exist=False,
                access="write",
            )
            steps.append(_Step(resolved, target, st.size, _mode_bits(st.mode)))
        return steps

    async def _read_sources(
        self, backend: FileBackend, steps: Sequence[_Step]
    ) -> list[bytes]:
        contents: list[bytes] = []
        total = 0
        for step in steps:
            data, _ = await backend.read_bytes(step.source)
            total += len(data)
            if total > self._max_bytes:
                raise ValueError(
                    f"The source files hold over {self._max_bytes:,} bytes, the "
                    "limit (the sizes the backend reported were smaller). Nothing "
                    "was copied. Copy a smaller directory."
                )
            contents.append(data)
        return contents


async def _refuse_files_in_the_way(
    backend: FileBackend, steps: Sequence[_Step]
) -> None:
    """Refuse when a directory a destination needs already exists as a file."""
    seen: set[Path] = set()
    for step in steps:
        directory = step.destination.parent
        while directory not in seen:
            seen.add(directory)
            if await backend.exists(directory):
                if _is_file((await backend.stat(directory)).mode):
                    raise ValueError(
                        f"{directory} is a file, so {step.destination} cannot be "
                        "created. Nothing was copied."
                    )
                break
            if directory.parent == directory:
                break
            directory = directory.parent


async def _check_overwrites(
    backend: FileBackend,
    steps: Sequence[_Step],
    state: FileEditSessionState | None,
    *,
    overwrite: bool,
) -> dict[Path, float]:
    """
    Refuse destinations that cannot be overwritten; return the expected
    mtime of each existing destination file (none outside an agent).
    """
    current: dict[Path, float] = {}
    for step in steps:
        if not await backend.exists(step.destination):
            continue
        st = await backend.stat(step.destination)
        if _is_dir(st.mode):
            raise ValueError(
                f"{step.destination} is a directory; it cannot be overwritten."
            )
        current[step.destination] = st.mtime
    if not current:
        return {}
    if not overwrite:
        raise ValueError(
            f"{len(current)} destination file(s) already exist: "
            f"{_list_paths(list(current))}. Pass overwrite=true to replace them."
        )
    if state is None:
        return {}

    expected: dict[Path, float] = {}
    unread: list[Path] = []
    changed: list[Path] = []
    for dest, mtime in current.items():
        record = state.get_read_record(dest)
        if record is None:
            unread.append(dest)
        elif record.mtime != mtime:
            changed.append(dest)
        else:
            expected[dest] = record.mtime
    if unread:
        raise ValueError(
            f"Must Read {_list_paths(unread)} before overwriting. As with Write, "
            "read-before-write keeps you from replacing content you haven't seen."
        )
    if changed:
        raise ValueError(
            f"{_list_paths(changed)} {'was' if len(changed) == 1 else 'were'} "
            "modified since you last read it. Re-Read before overwriting."
        )
    return expected


def _list_paths(paths: Sequence[Path]) -> str:
    listed = ", ".join(str(p) for p in paths[:_MAX_LISTED_CONFLICTS])
    more = len(paths) - _MAX_LISTED_CONFLICTS
    return f"{listed} and {more} more" if more > 0 else listed


def _is_dir(mode: int) -> bool:
    return bool(stat_module.S_IFMT(mode)) and stat_module.S_ISDIR(mode)


def _may_be_regular(mode: int) -> bool:
    """False only when the mode's type bits name something else."""
    return not stat_module.S_IFMT(mode) or stat_module.S_ISREG(mode)


def _is_file(mode: int) -> bool:
    """True only when the mode's type bits name something other than a directory."""
    return bool(stat_module.S_IFMT(mode)) and not stat_module.S_ISDIR(mode)


def _mode_bits(mode: int) -> int:
    return (mode & 0o7777) or _DEFAULT_MODE


__all__ = ["CopyInput", "CopyResult", "CopyTool"]
