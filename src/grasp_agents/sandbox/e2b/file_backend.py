"""
``E2BFileBackend`` — a :class:`FileBackend` over an E2B sandbox's ``files`` API.

Lives with the e2b exec environment (not in the file-tool layer like
``LocalFileBackend``) because E2B file I/O is inseparable from the live sandbox
handle: it cannot exist without an entered :class:`E2BEnvironment`.
"""

from __future__ import annotations

import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from e2b import AsyncSandbox, CommandExitException

from ...tools.file_backend.base import (
    FileBackend,
    FileEntry,
    FileStat,
    GrepRawResult,
)
from ...tools.file_backend.local import glob_filter_entries
from ...tools.file_backend.paths import PathAccessError, check_access_path
from ._handle import (
    RECURSIVE_DEPTH,
    SandboxHandle,
    is_dir,
    mtime,
    normalize_posix,
    wire,
)

if TYPE_CHECKING:
    from ...tools.file_backend.base import GrepOutputMode
    from ...tools.file_backend.paths import AccessMode
    from ..policy import SandboxPolicy


class E2BFileBackend(FileBackend):
    """
    :class:`FileBackend` over an E2B sandbox's ``files`` API.

    Consumes the *same* :class:`SandboxPolicy` as the paired
    :class:`E2BExecBackend` (one shared policy, both planes). ``allowed_roots``
    are POSIX paths inside the sandbox; validation is string containment (the
    remote FS cannot be host-resolved) plus the policy's FS carve-outs
    (``deny_read`` / ``allow_read`` / ``deny_write``) enforced on the tool plane
    via :func:`check_access_path`, exactly as :class:`LocalFileBackend` does. The
    host credential-dotfile denylist does **not** apply (those are host paths;
    the remote sandbox has its own). Read-before-write bookkeeping lives on the
    agent, as for every backend.
    """

    name = "e2b"

    def __init__(self, holder: SandboxHandle, *, policy: SandboxPolicy) -> None:
        self._holder = holder
        self._policy = policy
        self._allowed_roots: list[Path] = list(policy.allowed_roots)

    @property
    def allowed_roots(self) -> list[Path]:
        return list(self._allowed_roots)

    def add_allowed_root(self, root: Path) -> None:
        resolved = Path(root)
        if any(resolved == r or r in resolved.parents for r in self._allowed_roots):
            return
        self._allowed_roots.append(resolved)

    async def validate_path(
        self,
        path: Path,
        *,
        must_exist: bool,
        access: AccessMode = "read",
        dotfile_overrides: set[Path] | None = None,
    ) -> Path:
        del dotfile_overrides  # host credential denylist doesn't apply remotely

        if not self._allowed_roots:
            raise PathAccessError("No allowed_roots configured for E2B file backend.")

        # Normalize before containment: a literal ".." would pass the prefix
        # check here and escape the roots when the VM resolves it.
        candidate = normalize_posix(path)
        for root in self._allowed_roots:
            root_posix = normalize_posix(root)
            if candidate == root_posix or root_posix in candidate.parents:
                resolved = type(path)(str(candidate))
                break
        else:
            roots = ", ".join(str(r) for r in self._allowed_roots)
            raise PathAccessError(f"Path {path} is outside allowed roots [{roots}]")

        # Enforce the policy's FS carve-outs on the tool plane — the same shared
        # policy the exec plane consumes, as LocalFileBackend does.
        access_err = check_access_path(
            resolved,
            access=access,
            deny_read=self._policy.deny_read,
            allow_read=self._policy.allow_read,
            deny_write=self._policy.deny_write,
        )
        if access_err is not None:
            raise PathAccessError(access_err)

        if must_exist and not await self.exists(resolved):
            raise PathAccessError(f"Path does not exist: {resolved}")

        return resolved

    async def stat(self, path: Path) -> FileStat:
        info = await self._holder.require().files.get_info(wire(path))
        return FileStat(
            mtime=mtime(info),
            mode=int(getattr(info, "mode", 0o644) or 0o644),
            size=int(getattr(info, "size", 0) or 0),
        )

    async def exists(self, path: Path) -> bool:
        return bool(await self._holder.require().files.exists(wire(path)))

    async def parent_exists(self, path: Path) -> bool:
        parent = PurePosixPath(path).parent
        if str(parent) == wire(path):
            return True
        return bool(await self._holder.require().files.exists(str(parent)))

    async def read_text(self, path: Path) -> tuple[str, float]:
        sb = self._holder.require()
        content = await sb.files.read(wire(path), "text")
        info = await sb.files.get_info(wire(path))
        return str(content), mtime(info)

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        sb = self._holder.require()
        content = await sb.files.read(wire(path), "bytes")
        info = await sb.files.get_info(wire(path))
        return bytes(content), mtime(info)

    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
    ) -> float:
        del mode, overwrite  # E2B writes always overwrite; no mode on the API
        sb = self._holder.require()
        # e2b types write()'s data param as a bare ``IO`` (-> ``IO[Unknown]``).
        await sb.files.write(wire(path), data)  # pyright: ignore[reportUnknownMemberType]
        info = await sb.files.get_info(wire(path))
        return mtime(info)

    async def delete(self, path: Path) -> None:
        await self._holder.require().files.remove(wire(path))

    async def mkdir(self, path: Path) -> None:
        await self._holder.require().files.make_dir(wire(path))

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        depth = RECURSIVE_DEPTH if recursive else 1
        entries = await self._holder.require().files.list(wire(path), depth)
        out: list[FileEntry] = []
        for e in entries:
            p = type(path)(str(e.path))
            out.append(
                FileEntry(
                    name=str(getattr(e, "name", p.name)),
                    path=p,
                    is_dir=is_dir(e),
                    mtime=mtime(e),
                )
            )
        return out

    async def find_files(
        self,
        root: Path,
        pattern: str,
        *,
        include_hidden: bool = False,
        head_limit: int = 250,
    ) -> tuple[list[FileEntry], bool]:
        flat = await self.list_dir(root, recursive=True)
        return glob_filter_entries(
            flat,
            root,
            pattern,
            include_hidden=include_hidden,
            head_limit=head_limit,
        )

    async def grep(
        self,
        root: Path,
        pattern: str,
        *,
        glob: str | None = None,
        file_type: str | None = None,
        case_insensitive: bool = False,
        multiline: bool = False,
        output_mode: GrepOutputMode = "files_with_matches",
        show_line_numbers: bool = True,
        before_context: int | None = None,
        after_context: int | None = None,
        context: int | None = None,
    ) -> GrepRawResult:
        if multiline:
            raise NotImplementedError(
                "E2BFileBackend.grep does not support multiline mode (remote "
                "grep -Pz is fragile). Use single-line patterns, or Read the "
                "file and search in-process."
            )
        cmd = _build_grep_cmd(
            root,
            pattern,
            glob=glob,
            file_type=file_type,
            case_insensitive=case_insensitive,
            output_mode=output_mode,
            show_line_numbers=show_line_numbers,
            before_context=before_context,
            after_context=after_context,
            context=context,
        )
        stdout, code = await _run_capture(
            self._holder.require(), cmd, cwd=wire(self._allowed_roots[0])
        )
        if code >= 2:
            raise OSError(f"remote grep failed (exit {code}) for pattern {pattern!r}")
        return _parse_grep(stdout, output_mode)


async def _run_capture(
    sandbox: AsyncSandbox, command: str, *, cwd: str, timeout: float = 60.0
) -> tuple[str, int]:
    """
    Run ``command`` in the sandbox and return ``(stdout, exit_code)`` without
    raising on non-zero (so callers like grep can treat exit 1 as "no match").
    """
    try:
        result = await sandbox.commands.run(command, cwd=cwd, timeout=timeout)
    except CommandExitException as exc:
        return str(getattr(exc, "stdout", "") or ""), int(
            getattr(exc, "exit_code", 1) or 1
        )
    return str(getattr(result, "stdout", "") or ""), int(
        getattr(result, "exit_code", 0) or 0
    )


def _build_grep_cmd(
    root: Path,
    pattern: str,
    *,
    glob: str | None,
    file_type: str | None,
    case_insensitive: bool,
    output_mode: GrepOutputMode,
    show_line_numbers: bool,
    before_context: int | None,
    after_context: int | None,
    context: int | None,
) -> str:
    args = ["grep", "-r", "-E"]
    if case_insensitive:
        args.append("-i")
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    elif show_line_numbers:
        args.append("-n")
    if output_mode == "content":
        if context is not None:
            args.append(f"-C{context}")
        else:
            if before_context is not None:
                args.append(f"-B{before_context}")
            if after_context is not None:
                args.append(f"-A{after_context}")
    if glob is not None:
        args.append(f"--include={glob}")
    if file_type is not None:
        args.append(f"--include=*.{file_type}")
    args.extend(["-e", pattern, wire(root)])
    return " ".join(shlex.quote(a) for a in args)


def _parse_grep(stdout: str, output_mode: GrepOutputMode) -> GrepRawResult:
    lines = [ln for ln in stdout.splitlines() if ln]
    if output_mode == "files_with_matches":
        files = [Path(ln) for ln in lines]
        return GrepRawResult(files=files, num_files_matched=len(files))
    if output_mode == "count":
        counts: list[tuple[Path, int]] = []
        total = 0
        for ln in lines:
            path_str, _, num = ln.rpartition(":")
            if path_str and num.isdigit():
                n = int(num)
                counts.append((Path(path_str), n))
                total += n
        return GrepRawResult(
            counts=counts, num_matches=total, num_files_matched=len(counts)
        )
    matched = [ln for ln in lines if ln != "--"]
    return GrepRawResult(lines=matched, num_matches=len(matched))
