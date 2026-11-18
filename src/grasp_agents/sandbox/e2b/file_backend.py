"""
``E2BFileBackend`` — a :class:`FileBackend` over an E2B sandbox's ``files`` API.

Lives with the e2b exec environment (not in the file-tool layer like
``LocalFileBackend``) because E2B file I/O is inseparable from the live sandbox
handle: it cannot exist without an entered :class:`E2BEnvironment`.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import shlex
import stat as stat_module
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from e2b import AsyncSandbox, CommandExitException, NotFoundException

from grasp_agents.file_backend.base import (
    FileBackend,
    FileEntry,
    FileStat,
    GrepRawResult,
    StaleFileError,
)
from grasp_agents.file_backend.local import glob_filter_entries
from grasp_agents.file_backend.paths import PathAccessError, check_access_path

from ._handle import (
    RECURSIVE_DEPTH,
    SandboxHandle,
    is_dir,
    mtime,
    normalize_posix,
    wire,
)

if TYPE_CHECKING:
    from e2b import EntryInfo

    from grasp_agents.file_backend.base import GrepOutputMode
    from grasp_agents.file_backend.paths import AccessMode
    from grasp_agents.sandbox.policy import SandboxPolicy


# The globs behind grep's ``type`` filter: these types as ``rg --type-list``
# (ripgrep 14) defines them, so a type selects the same files here as on the
# local ripgrep backend.
# cspell:ignore jspx sarif jpynb mdwn mkdn pht tcsh tcshrc cshrc kshrc zlogin
# cspell:ignore zlogout zprofile zshenv dtx rng sch xjb irbrc
_FILE_TYPE_GLOBS: dict[str, tuple[str, ...]] = {
    "c": ("*.[chH]", "*.[chH].in", "*.cats"),
    "config": ("*.cfg", "*.conf", "*.config", "*.ini"),
    "cpp": (
        "*.[ChH]",
        "*.[ChH].in",
        "*.[ch]pp",
        "*.[ch]pp.in",
        "*.[ch]xx",
        "*.[ch]xx.in",
        "*.cc",
        "*.cc.in",
        "*.hh",
        "*.hh.in",
        "*.inl",
    ),
    "css": ("*.css", "*.scss"),
    "csv": ("*.csv",),
    "go": ("*.go",),
    "html": ("*.ejs", "*.htm", "*.html"),
    "java": ("*.java", "*.jsp", "*.jspx", "*.properties"),
    "js": ("*.cjs", "*.js", "*.jsx", "*.mjs", "*.vue"),
    "json": ("*.json", "*.sarif", "composer.lock"),
    "jsonl": ("*.jsonl",),
    "jupyter": ("*.ipynb", "*.jpynb"),
    "kotlin": ("*.kt", "*.kts"),
    "lua": ("*.lua",),
    "markdown": ("*.markdown", "*.md", "*.mdown", "*.mdwn", "*.mdx", "*.mkd", "*.mkdn"),
    "md": ("*.markdown", "*.md", "*.mdown", "*.mdwn", "*.mdx", "*.mkd", "*.mkdn"),
    "php": (
        "*.php",
        "*.php3",
        "*.php4",
        "*.php5",
        "*.php7",
        "*.php8",
        "*.pht",
        "*.phtml",
    ),
    "py": ("*.py", "*.pyi"),
    "r": ("*.R", "*.Rmd", "*.Rnw", "*.r"),
    "rst": ("*.rst",),
    "ruby": (
        "*.gemspec",
        "*.rb",
        "*.rbw",
        ".irbrc",
        "Gemfile",
        "Rakefile",
        "config.ru",
    ),
    "rust": ("*.rs",),
    "scala": ("*.sbt", "*.scala"),
    "sh": (
        "*.bash",
        "*.bashrc",
        "*.csh",
        "*.cshrc",
        "*.ksh",
        "*.kshrc",
        "*.sh",
        "*.tcsh",
        "*.zsh",
        ".bash_login",
        ".bash_logout",
        ".bash_profile",
        ".bashrc",
        ".cshrc",
        ".kshrc",
        ".login",
        ".logout",
        ".profile",
        ".tcshrc",
        ".zlogin",
        ".zlogout",
        ".zprofile",
        ".zshenv",
        ".zshrc",
        "bash_login",
        "bash_logout",
        "bash_profile",
        "bashrc",
        "profile",
        "zlogin",
        "zlogout",
        "zprofile",
        "zshenv",
        "zshrc",
    ),
    "sql": ("*.psql", "*.sql"),
    "swift": ("*.swift",),
    "tex": ("*.bib", "*.cls", "*.dtx", "*.ins", "*.ltx", "*.sty", "*.tex"),
    "toml": ("*.toml", "Cargo.lock"),
    "ts": ("*.cts", "*.mts", "*.ts", "*.tsx"),
    "txt": ("*.txt",),
    "xml": (
        "*.dtd",
        "*.rng",
        "*.sch",
        "*.xhtml",
        "*.xjb",
        "*.xml",
        "*.xml.dist",
        "*.xsd",
        "*.xsl",
        "*.xslt",
    ),
    "yaml": ("*.yaml", "*.yml"),
}


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
        async with self._holder.use() as sb:
            info = await sb.files.get_info(wire(path))
        return FileStat(
            mtime=mtime(info),
            mode=_file_mode(info),
            size=int(getattr(info, "size", 0) or 0),
        )

    async def exists(self, path: Path) -> bool:
        async with self._holder.use() as sb:
            return bool(await sb.files.exists(wire(path)))

    async def parent_exists(self, path: Path) -> bool:
        parent = PurePosixPath(path).parent
        if str(parent) == wire(path):
            return True
        async with self._holder.use() as sb:
            return bool(await sb.files.exists(str(parent)))

    # Both reads take the mtime before the content: a write landing in between
    # then makes the next conditional write refuse, instead of letting an
    # mtime newer than the content pass over that write.

    async def read_text(self, path: Path) -> tuple[str, float]:
        async with self._holder.use() as sb:
            info = await sb.files.get_info(wire(path))
            content = await sb.files.read(wire(path), "text")
        return str(content), mtime(info)

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        async with self._holder.use() as sb:
            info = await sb.files.get_info(wire(path))
            content = await sb.files.read(wire(path), "bytes")
        return bytes(content), mtime(info)

    async def write_bytes(
        self,
        path: Path,
        data: bytes,
        *,
        mode: int,
        overwrite: bool = True,
        expected_mtime: float | None = None,
    ) -> float:
        del mode, overwrite  # E2B writes always overwrite; no mode on the API
        async with self._holder.use() as sb:
            if expected_mtime is not None:
                await _check_expected_mtime(sb, path, expected_mtime)
            # e2b types write()'s data param as a bare ``IO`` (-> ``IO[Unknown]``).
            await sb.files.write(wire(path), data)  # pyright: ignore[reportUnknownMemberType]
            info = await sb.files.get_info(wire(path))
        return mtime(info)

    async def delete(self, path: Path, *, expected_mtime: float | None = None) -> None:
        async with self._holder.use() as sb:
            if expected_mtime is not None:
                await _check_expected_mtime(sb, path, expected_mtime)
            await sb.files.remove(wire(path))

    async def mkdir(self, path: Path) -> None:
        async with self._holder.use() as sb:
            await sb.files.make_dir(wire(path))

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        depth = RECURSIVE_DEPTH if recursive else 1
        async with self._holder.use() as sb:
            entries = await sb.files.list(wire(path), depth)
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
        return await asyncio.to_thread(
            glob_filter_entries,
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
        type_globs: tuple[str, ...] | None = None
        if file_type is not None:
            type_globs = _FILE_TYPE_GLOBS.get(file_type)
            if type_globs is None:
                raise ValueError(
                    f"file type {file_type!r} is not supported on E2B; filter "
                    "with glob instead"
                )
        # grep ORs its --include filters, so with a glob the type is applied
        # to the results instead.
        cmd = _build_grep_cmd(
            root,
            pattern,
            includes=(glob,) if glob is not None else type_globs or (),
            case_insensitive=case_insensitive,
            output_mode=output_mode,
            before_context=before_context,
            after_context=after_context,
            context=context,
        )
        async with self._holder.use() as sb:
            stdout, code = await _run_capture(sb, cmd, cwd=wire(self._allowed_roots[0]))
        if code >= 2:
            raise OSError(f"remote grep failed (exit {code}) for pattern {pattern!r}")
        return _parse_grep(
            stdout,
            output_mode,
            show_line_numbers=show_line_numbers,
            type_globs=type_globs,
        )


def _file_mode(info: EntryInfo) -> int:
    """
    Return the entry's mode with its file-type bits.

    The sandbox reports permission bits only, so without the type bits from
    the entry type every directory would look like a regular file.
    """
    mode = int(getattr(info, "mode", 0) or 0)
    if stat_module.S_IFMT(mode):
        return mode
    if is_dir(info):
        return stat_module.S_IFDIR | (mode or 0o755)
    return stat_module.S_IFREG | (mode or 0o644)


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
    includes: tuple[str, ...],
    case_insensitive: bool,
    output_mode: GrepOutputMode,
    before_context: int | None,
    after_context: int | None,
    context: int | None,
) -> str:
    # -H names the file even when ``root`` is one, and -Z ends each name with
    # a NUL, so names holding ':' or '-' still parse.
    args = ["grep", "-r", "-E", "-H", "-Z"]
    if case_insensitive:
        args.append("-i")
    if output_mode == "files_with_matches":
        args.append("-l")
    elif output_mode == "count":
        args.append("-c")
    else:
        # Numbered even when the caller hides line numbers: the separator
        # after the number tells a match (':') from a context line ('-').
        args.append("-n")
        if context is not None:
            args.append(f"-C{context}")
        else:
            if before_context is not None:
                args.append(f"-B{before_context}")
            if after_context is not None:
                args.append(f"-A{after_context}")
    args.extend(f"--include={glob}" for glob in includes)
    args.extend(["-e", pattern, wire(root)])
    return " ".join(shlex.quote(a) for a in args)


# ``line_no`` then ':' (a match) or '-' (a context line), then the line text.
_CONTENT_RECORD = re.compile(r"(\d+)([:-])(.*)", re.DOTALL)


def _parse_grep(
    stdout: str,
    output_mode: GrepOutputMode,
    *,
    show_line_numbers: bool,
    type_globs: tuple[str, ...] | None,
) -> GrepRawResult:
    """
    Parse ``grep -H -Z`` output into the shape the ripgrep backend returns.

    ``-l`` output is NUL-separated names; ``-c`` and ``-n`` output is one
    line per record, ``name NUL rest``. Lines without a NUL (the ``--``
    between context groups) are skipped, as are files of another type.
    """

    def wanted(name: str) -> bool:
        base = PurePosixPath(name).name
        return type_globs is None or any(
            fnmatch.fnmatchcase(base, type_glob) for type_glob in type_globs
        )

    if output_mode == "files_with_matches":
        files = [Path(name) for name in stdout.split("\0") if name and wanted(name)]
        return GrepRawResult(
            files=files, num_matches=len(files), num_files_matched=len(files)
        )

    records = [line.partition("\0") for line in stdout.split("\n")]
    if output_mode == "count":
        counts = [
            (Path(name), int(num))
            for name, sep, num in records
            if sep and num.isdigit() and int(num) > 0 and wanted(name)
        ]
        return GrepRawResult(
            counts=counts,
            num_matches=sum(n for _, n in counts),
            num_files_matched=len(counts),
        )

    lines: list[str] = []
    num_matches = 0
    matched_files: set[str] = set()
    for name, sep, rest in records:
        record = _CONTENT_RECORD.fullmatch(rest) if sep else None
        if record is None or not wanted(name):
            continue
        line_no, kind, text = record.groups()
        number = f"{line_no}{kind}" if show_line_numbers else ""
        lines.append(f"{name}{kind}{number}{text}")
        if kind == ":":
            num_matches += 1
            matched_files.add(name)
    return GrepRawResult(
        lines=lines, num_matches=num_matches, num_files_matched=len(matched_files)
    )


async def _check_expected_mtime(
    sb: AsyncSandbox, path: Path, expected_mtime: float
) -> None:
    """
    Best-effort precondition for a conditional write or delete.

    E2B has no conditional write, so a change landing between this check
    and the write goes unnoticed.
    """
    try:
        info = await sb.files.get_info(wire(path))
    except NotFoundException:
        raise StaleFileError(f"{path} no longer exists") from None
    current = mtime(info)
    if current != expected_mtime:
        raise StaleFileError(
            f"{path} changed (expected mtime {expected_mtime!r}, current {current!r})"
        )
