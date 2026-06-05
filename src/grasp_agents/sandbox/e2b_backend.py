"""
``E2BEnvironment`` — backend #3 of Phase D: a hosted, co-located filesystem +
exec pair backed by an E2B cloud sandbox (https://e2b.dev).

A thin adapter over the optional ``e2b`` SDK — it does **not** reimplement E2B's
transport. ``E2BFileBackend`` maps our :class:`FileBackend` calls onto
``sandbox.files`` and ``E2BExecBackend`` maps :class:`ExecBackend` onto
``sandbox.commands``; both address the *same* remote sandbox (co-location by
construction — they share one live handle). The async context-manager lifecycle
owns sandbox create + teardown, and :class:`SnapshotCapable` is wired to E2B's
native ``pause`` / ``connect``.

**Boundary doc (the three questions every backend answers):**

1. *What can this backend contain?* **Everything the command touches**, because
   the command runs on a remote E2B microVM, not the host. The whole sandbox
   filesystem is the workspace; ``allowed_roots`` are POSIX paths *inside* it.
   Network egress follows :attr:`SandboxPolicy.network` (``NONE`` →
   ``allow_internet_access=False``; ``ALL`` → on). A blast radius bounded by the
   provider, not by host trust — the strongest isolation of the three backends.
2. *What is outside the boundary / trusted to the provider?* E2B (the cloud
   account, the microVM image, the egress rules). The host process only holds an
   API token and an HTTP/RPC handle; nothing the command does can reach the host
   filesystem. The policy's FS carve-outs (``deny_read`` / ``allow_read`` /
   ``deny_write``) ARE enforced on the **tool plane** (the file tools, via
   ``check_access_path``) — the same two-plane model as the local/Seatbelt
   backends; they are not *additionally* enforced at the remote OS level, where
   the sandbox itself is the outer boundary. Per-domain network allowlists are
   not enforced (``ALLOWLIST`` / ``LOOPBACK`` raise ``NotImplementedError``;
   only ``NONE`` / ``ALL``).
3. *What did we configure and why?* ``find_files`` reuses ``files.list`` + our
   glob filter (no GNU-``find`` dependency, exact ``**`` semantics); ``grep``
   shells out to ``grep`` *inside* the sandbox through the co-located exec — the
   only way to search contents without downloading every file, and the payoff of
   E2B being a real co-located filesystem (unlike the MCP backend).
   :class:`SnapshotCapable` ``snapshot()`` creates a *persistent* E2B snapshot
   (``create_snapshot`` — survives deletion) and ``restore()`` spawns a fresh
   sandbox from it (a true rewind); separately, ``pause()`` / ``resume()`` are
   suspend/resume of the same sandbox (``pause`` / ``connect``).

Requires the optional ``e2b`` package (``pip install grasp-agents[e2b]``), which
is imported at module top (like the Anthropic / Gemini providers). This module
is loaded lazily by :mod:`grasp_agents.sandbox` (PEP 562), so the extra is only
needed when the E2B backend is actually used.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import shlex
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Self

from e2b import (
    AsyncCommandHandle,
    AsyncSandbox,
    CommandExitException,
    EntryInfo,
    TimeoutException,
)

from ..tools.file_edit.backend import (
    FileBackend,
    FileEntry,
    FileStat,
    GrepRawResult,
)
from ..tools.file_edit.local_backend import glob_filter_entries
from ..tools.file_edit.paths import PathAccessError, check_access_path
from .e2b_session import E2BExecSession
from .environment import ExecutionEnvironment, SnapshotCapable
from .exec_backend import (
    ExecBackend,
    ExecChunk,
    ExecResult,
    SessionCapable,
    TerminationReason,
)
from .policy import NetworkPolicy, SandboxPolicy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping, Sequence

    from ..tools.file_edit.backend import GrepOutputMode
    from ..tools.file_edit.paths import AccessMode


# Depth passed to ``files.list`` for a recursive listing. E2B has no "infinite"
# depth; this is deep enough for real trees and the Glob tool caps results.
_RECURSIVE_DEPTH = 64

_DEFAULT_WORKSPACE = "/home/user/workspace"
_DEFAULT_EXEC_TIMEOUT = 600.0  # matches the local supervisor's overall default
_MAX_OUTPUT_CHARS = 1_000_000  # per the local supervisor's per-stream cap


_TRANSPORT_TIMEOUT_NAMES = frozenset(
    {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException"}
)


def _is_timeout(exc: BaseException) -> bool:
    """
    True if ``exc`` (or its cause chain) is a command timeout.

    A timed-out command surfaces either as e2b's ``TimeoutException`` or as a
    raw transport ``ReadTimeout`` from the underlying httpx/httpcore stream, so
    both are recognized.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, TimeoutException):
            return True
        cls = type(cur)
        if cls.__name__ in _TRANSPORT_TIMEOUT_NAMES and cls.__module__.startswith(
            ("httpcore", "httpx")
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _mtime(entry: EntryInfo) -> float:
    """Epoch seconds from an e2b ``EntryInfo.modified_time`` (a ``datetime``)."""
    return float(entry.modified_time.timestamp())


def _is_dir(entry: EntryInfo) -> bool:
    """True if an e2b entry is a directory (``entry.type`` is a ``FileType``)."""
    return str(getattr(entry.type, "value", entry.type)) == "dir"


def _wire(path: Path) -> str:
    """Render a path as the absolute POSIX string the sandbox expects."""
    return str(PurePosixPath(path))


def _wrap_stdin(command: str, stdin: bytes | None) -> str:
    """
    Deliver ``stdin`` (with EOF) to ``command`` over the remote shell.

    E2B's ``commands.run`` has no stdin-data argument, so feed it via a
    base64 pipe into a subshell — robust and EOF-correct, unlike streaming
    bytes to a background pid.
    """
    if not stdin:
        return command
    b64 = base64.b64encode(stdin).decode("ascii")
    return f"printf %s {shlex.quote(b64)} | base64 -d | ({command})"


class _SandboxHandle:
    """
    Shared, mutable holder for the live e2b ``AsyncSandbox``.

    Both backends and the environment reference the *same* holder, so a
    ``restore()`` (reconnect/resume) swaps the live sandbox under the file and
    exec surfaces at once.
    """

    __slots__ = ("sandbox",)

    def __init__(self, sandbox: AsyncSandbox | None = None) -> None:
        self.sandbox: AsyncSandbox | None = sandbox

    def require(self) -> AsyncSandbox:
        if self.sandbox is None:
            raise RuntimeError(
                "E2B environment is not entered. Use `async with env:` (the "
                "context manager creates the sandbox) before any file/exec call."
            )
        return self.sandbox


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

    def __init__(self, holder: _SandboxHandle, *, policy: SandboxPolicy) -> None:
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

        candidate = PurePosixPath(path)
        for root in self._allowed_roots:
            root_posix = PurePosixPath(root)
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
        info = await self._holder.require().files.get_info(_wire(path))
        return FileStat(
            mtime=_mtime(info),
            mode=int(getattr(info, "mode", 0o644) or 0o644),
            size=int(getattr(info, "size", 0) or 0),
        )

    async def exists(self, path: Path) -> bool:
        return bool(await self._holder.require().files.exists(_wire(path)))

    async def parent_exists(self, path: Path) -> bool:
        parent = PurePosixPath(path).parent
        if str(parent) == _wire(path):
            return True
        return bool(await self._holder.require().files.exists(str(parent)))

    async def read_text(self, path: Path) -> tuple[str, float]:
        sb = self._holder.require()
        content = await sb.files.read(_wire(path), "text")
        info = await sb.files.get_info(_wire(path))
        return str(content), _mtime(info)

    async def read_bytes(self, path: Path) -> tuple[bytes, float]:
        sb = self._holder.require()
        content = await sb.files.read(_wire(path), "bytes")
        info = await sb.files.get_info(_wire(path))
        return bytes(content), _mtime(info)

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
        await sb.files.write(_wire(path), data)  # pyright: ignore[reportUnknownMemberType]
        info = await sb.files.get_info(_wire(path))
        return _mtime(info)

    async def delete(self, path: Path) -> None:
        await self._holder.require().files.remove(_wire(path))

    async def mkdir(self, path: Path) -> None:
        await self._holder.require().files.make_dir(_wire(path))

    async def list_dir(self, path: Path, *, recursive: bool = False) -> list[FileEntry]:
        depth = _RECURSIVE_DEPTH if recursive else 1
        entries = await self._holder.require().files.list(_wire(path), depth)
        out: list[FileEntry] = []
        for e in entries:
            p = type(path)(str(e.path))
            out.append(
                FileEntry(
                    name=str(getattr(e, "name", p.name)),
                    path=p,
                    is_dir=_is_dir(e),
                    mtime=_mtime(e),
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
            self._holder.require(), cmd, cwd=_wire(self._allowed_roots[0])
        )
        if code >= 2:
            raise OSError(f"remote grep failed (exit {code}) for pattern {pattern!r}")
        return _parse_grep(stdout, output_mode)


class E2BExecBackend(ExecBackend, SessionCapable):
    """
    :class:`ExecBackend` over an E2B sandbox's ``commands`` API.

    Bridges E2B's callback-driven streaming onto our ``ExecChunk`` generator,
    returns non-zero exits as data (catching ``CommandExitException`` instead
    of letting it raise), and maps ``timeout=None`` to a bounded default
    (E2B would otherwise impose its own 60 s ceiling). Co-located with the
    paired :class:`E2BFileBackend` via the shared :class:`_SandboxHandle`.
    """

    def __init__(
        self,
        holder: _SandboxHandle,
        *,
        policy: SandboxPolicy,
        name: str = "e2b",
        default_timeout: float = _DEFAULT_EXEC_TIMEOUT,
        max_output_chars: int = _MAX_OUTPUT_CHARS,
    ) -> None:
        self._holder = holder
        self._policy = policy
        self._name = name
        self._default_timeout = default_timeout
        self._max_output_chars = max_output_chars

    @property
    def name(self) -> str:
        return self._name

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    async def execute(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        out: list[str] = []
        err: list[str] = []
        terminal: ExecResult | None = None
        async for item in self.stream(
            command, cwd=cwd, timeout=timeout, stdin=stdin, env=env
        ):
            if isinstance(item, ExecChunk):
                (out if item.stream == "stdout" else err).append(item.data)
            else:
                terminal = item
        if terminal is None:  # pragma: no cover - stream always yields one
            raise RuntimeError("E2B stream produced no terminal ExecResult")
        return replace(terminal, stdout="".join(out), stderr="".join(err))

    async def stream(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        sb = self._holder.require()

        wrapped = _wrap_stdin(command, stdin)
        eff_timeout = timeout if timeout is not None else self._default_timeout
        cwd_str = self._resolve_cwd(cwd)
        envs = self._merged_env(env)

        queue: asyncio.Queue[ExecChunk | None] = asyncio.Queue()
        emitted = 0
        truncated = False

        def _push(stream: Literal["stdout", "stderr"], data: str) -> None:
            nonlocal emitted, truncated
            if truncated:
                return
            budget = self._max_output_chars - emitted
            if budget <= 0:
                truncated = True
                return
            if len(data) > budget:
                data = data[:budget]
                truncated = True
            emitted += len(data)
            queue.put_nowait(ExecChunk(stream=stream, data=data))

        def on_stdout(data: str) -> None:
            _push("stdout", data)

        def on_stderr(data: str) -> None:
            _push("stderr", data)

        start = time.monotonic()
        try:
            handle = await sb.commands.run(
                wrapped,
                background=True,
                cwd=cwd_str,
                envs=envs,
                timeout=eff_timeout,
                stdin=False,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )
        except BaseException as exc:
            reason = (
                TerminationReason.OVERALL_TIMEOUT
                if _is_timeout(exc)
                else TerminationReason.SPAWN_ERROR
            )
            yield ExecChunk(stream="stderr", data=str(exc))
            yield ExecResult(
                stdout="",
                stderr="",
                returncode=-1,
                reason=reason,
                runtime_ms=(time.monotonic() - start) * 1000,
                backend=self._name,
            )
            return

        outcome: dict[str, BaseException] = {}

        async def _drain() -> None:
            try:
                await handle.wait()
            except BaseException as exc:
                outcome["exc"] = exc
            finally:
                queue.put_nowait(None)

        waiter = asyncio.create_task(_drain())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            await waiter

        yield self._terminal(
            handle,
            outcome.get("exc"),
            runtime_ms=(time.monotonic() - start) * 1000,
            truncated=truncated,
        )

    def _terminal(
        self,
        handle: AsyncCommandHandle,
        exc: BaseException | None,
        *,
        runtime_ms: float,
        truncated: bool,
    ) -> ExecResult:
        if exc is not None and _is_timeout(exc):
            return ExecResult(
                stdout="",
                stderr="",
                returncode=-1,
                reason=TerminationReason.OVERALL_TIMEOUT,
                runtime_ms=runtime_ms,
                backend=self._name,
                truncated=truncated,
            )
        if exc is not None and isinstance(exc, CommandExitException):
            rc = int(getattr(exc, "exit_code", 1) or 1)
            return ExecResult(
                stdout="",
                stderr="",
                returncode=rc,
                reason=TerminationReason.EXIT,
                runtime_ms=runtime_ms,
                backend=self._name,
                truncated=truncated,
            )
        if exc is not None:
            raise exc  # genuine sandbox/transport failure — surface it
        rc = int(handle.exit_code or 0)
        return ExecResult(
            stdout="",
            stderr="",
            returncode=rc,
            reason=TerminationReason.EXIT,
            runtime_ms=runtime_ms,
            backend=self._name,
            truncated=truncated,
        )

    def _resolve_cwd(self, cwd: Path | None) -> str:
        roots = self._policy.allowed_roots
        if not roots:
            raise PathAccessError(
                "SandboxPolicy has no allowed_roots; cannot choose a working "
                "directory for the command."
            )
        if cwd is None:
            return _wire(roots[0])
        candidate = PurePosixPath(cwd)
        for root in roots:
            root_posix = PurePosixPath(root)
            if candidate == root_posix or root_posix in candidate.parents:
                return str(candidate)
        raise PathAccessError(f"cwd {cwd} is outside the policy's allowed_roots")

    def _merged_env(self, env: Mapping[str, str] | None) -> dict[str, str] | None:
        merged = dict(self._policy.env)
        if env:
            merged.update(env)
        return merged or None

    async def open_session(
        self, *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> E2BExecSession:
        """Open a persistent shell on the sandbox (see :class:`E2BExecSession`)."""
        return E2BExecSession(
            self._holder.require,
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
            max_output_chars=self._max_output_chars,
        )


class E2BEnvironment(ExecutionEnvironment, SnapshotCapable):
    """
    An :class:`ExecutionEnvironment` over an E2B cloud sandbox, with
    :class:`SnapshotCapable` wired to E2B's native pause/resume.

    Prefer the :func:`e2b_environment` factory. The async context manager owns
    sandbox create (``__aenter__`` — where the optional ``e2b`` import happens)
    and teardown (``__aexit__`` — kill, or pause when ``pause_on_exit``). The
    ``file_backend`` / ``exec_backend`` objects exist before entry (so a
    ``RunContext`` can source them), but their I/O raises until the context is
    entered and the sandbox is live.
    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        holder: _SandboxHandle,
        file_backend: E2BFileBackend,
        exec_backend: E2BExecBackend,
        create_params: dict[str, Any] | None = None,
        owns_sandbox: bool = True,
        pause_on_exit: bool = False,
    ) -> None:
        self._policy = policy
        self._holder = holder
        self._file_backend = file_backend
        self._exec_backend = exec_backend
        self._create_params = create_params
        self._owns_sandbox = owns_sandbox
        self._pause_on_exit = pause_on_exit

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @property
    def file_backend(self) -> E2BFileBackend:
        return self._file_backend

    @property
    def exec_backend(self) -> E2BExecBackend:
        return self._exec_backend

    @classmethod
    def from_sandbox(
        cls,
        sandbox: AsyncSandbox,
        *,
        policy: SandboxPolicy,
        owns_sandbox: bool = False,
        pause_on_exit: bool = False,
    ) -> Self:
        """
        Wrap an already-created e2b ``AsyncSandbox`` (or a test double).

        The ``policy`` is the single source of ``allowed_roots`` + carve-outs
        for both surfaces. ``owns_sandbox=False`` by default: the caller created
        it, so teardown is the caller's responsibility. Used for connecting to a
        pre-existing sandbox and for tests.
        """
        holder = _SandboxHandle(sandbox)
        return cls(
            policy=policy,
            holder=holder,
            file_backend=E2BFileBackend(holder, policy=policy),
            exec_backend=E2BExecBackend(holder, policy=policy),
            create_params=None,
            owns_sandbox=owns_sandbox,
            pause_on_exit=pause_on_exit,
        )

    async def __aenter__(self) -> Self:
        if self._holder.sandbox is None:
            if self._create_params is None:
                raise RuntimeError(
                    "E2BEnvironment has no sandbox and no create params; build "
                    "it via e2b_environment(...) or E2BEnvironment.from_sandbox(...)."
                )
            self._holder.sandbox = await AsyncSandbox.create(**self._create_params)
        sandbox = self._holder.require()
        for root in self._file_backend.allowed_roots:
            await sandbox.files.make_dir(_wire(root))
        return self

    async def __aexit__(self, *exc: object) -> None:
        sandbox = self._holder.sandbox
        if sandbox is None or not self._owns_sandbox:
            return
        try:
            if self._pause_on_exit:
                await sandbox.pause()
            else:
                await sandbox.kill()
        finally:
            self._holder.sandbox = None

    async def snapshot(self) -> str:
        """
        Create a persistent E2B snapshot of the current filesystem + memory and
        return its id (the :class:`SnapshotCapable` ref).

        The snapshot is a point-in-time copy that survives sandbox deletion; the
        sandbox briefly pauses during creation, then keeps running. :meth:`restore`
        spawns a *fresh* sandbox from it — a true rewind. (To suspend/resume the
        same sandbox instead, use :meth:`pause` / :meth:`resume`.)
        """
        snap = await self._holder.require().create_snapshot()
        return str(snap.snapshot_id)

    async def restore(self, ref: str) -> None:
        """
        Rewind to a snapshot: spawn a fresh sandbox from ``ref`` (its filesystem
        + memory at snapshot time) and swap the live handle under both backends.
        The previous sandbox, if owned, is killed.
        """
        previous = self._holder.sandbox
        params = dict(self._create_params or {})
        params["template"] = ref  # create from the snapshot, not a base template
        self._holder.sandbox = await AsyncSandbox.create(**params)
        if previous is not None and self._owns_sandbox:
            with contextlib.suppress(Exception):
                await previous.kill()

    async def pause(self) -> str:
        """
        Suspend the sandbox (filesystem + memory preserved) and return its id.

        Resume the *same* sandbox later with :meth:`resume`. Distinct from
        :meth:`snapshot`: pause is suspend/resume (no rewind), whereas a snapshot
        is a persistent point-in-time copy you spawn fresh sandboxes from.
        """
        sandbox = self._holder.require()
        await sandbox.pause()
        return str(sandbox.sandbox_id)

    async def resume(self, sandbox_id: str | None = None) -> None:
        """
        Reconnect to (and auto-resume) a paused sandbox, swapping the live
        handle. Defaults to the currently-bound sandbox's id.
        """
        target = sandbox_id
        if target is None:
            current = self._holder.sandbox
            if current is None:
                raise RuntimeError(
                    "resume() needs a sandbox_id when no sandbox is bound."
                )
            target = str(current.sandbox_id)
        self._holder.sandbox = await AsyncSandbox.connect(target)


def e2b_environment(
    *,
    allowed_roots: Sequence[str] = (_DEFAULT_WORKSPACE,),
    deny_read: Sequence[str] = (),
    allow_read: Sequence[str] = (),
    deny_write: Sequence[str] = (),
    template: str | None = None,
    sandbox_timeout: int | None = None,
    network: NetworkPolicy = NetworkPolicy.NONE,
    env: Mapping[str, str] | None = None,
    metadata: Mapping[str, str] | None = None,
    api_key: str | None = None,
    domain: str | None = None,
    secure: bool = True,
    pause_on_exit: bool = False,
    default_timeout: float = _DEFAULT_EXEC_TIMEOUT,
) -> E2BEnvironment:
    """
    Build a deferred-create E2B environment; the sandbox is created on
    ``async with``.

    Args:
        allowed_roots: POSIX paths *inside* the sandbox the file tools may
            address. The first is the default working directory; all are
            ``mkdir -p``'d on entry.
        deny_read: Remote paths carved out of the readable space (tool plane).
        allow_read: Remote paths re-allowed within ``deny_read`` (allow wins).
        deny_write: Remote paths write-protected on the tool plane (deny wins).
        template: E2B sandbox template name/ID (default base template).
        sandbox_timeout: Sandbox keep-alive in seconds (E2B default 300).
        network: ``NONE`` (no internet) or ``ALL`` (internet on). ``ALLOWLIST``
            / ``LOOPBACK`` raise ``NotImplementedError`` (per-domain egress needs
            E2B network rules — future work).
        env: Base environment variables for the sandbox + every command.
        metadata: Custom sandbox metadata.
        api_key: E2B API key (defaults to the ``E2B_API_KEY`` env var).
        domain: E2B domain (defaults to the ``E2B_DOMAIN`` env var).
        secure: Token-secure the sandbox (E2B default ``True``).
        pause_on_exit: Pause instead of kill on context exit (state persists;
            reconnect later with the sandbox id).
        default_timeout: Per-command wall-clock ceiling when a call passes no
            ``timeout`` (E2B would otherwise impose its own 60 s).

    Returns:
        An :class:`E2BEnvironment` to use as ``async with``.

    """
    if network in {NetworkPolicy.ALLOWLIST, NetworkPolicy.LOOPBACK}:
        raise NotImplementedError(
            f"E2B example backend does not implement network={network.value!r} "
            "(per-domain / loopback egress needs E2B network rules). Use "
            "NetworkPolicy.NONE or NetworkPolicy.ALL."
        )

    roots = tuple(Path(r) for r in allowed_roots)
    policy = SandboxPolicy(
        allowed_roots=roots,
        deny_read=tuple(Path(p) for p in deny_read),
        allow_read=tuple(Path(p) for p in allow_read),
        deny_write=tuple(Path(p) for p in deny_write),
        network=network,
        env=dict(env) if env else {},
    )

    create_params: dict[str, Any] = {
        "template": template,
        "timeout": sandbox_timeout,
        "metadata": dict(metadata) if metadata else None,
        "envs": dict(env) if env else None,
        "secure": secure,
        "allow_internet_access": network == NetworkPolicy.ALL,
    }
    if api_key is not None:
        create_params["api_key"] = api_key
    if domain is not None:
        create_params["domain"] = domain

    holder = _SandboxHandle()
    return E2BEnvironment(
        policy=policy,
        holder=holder,
        file_backend=E2BFileBackend(holder, policy=policy),
        exec_backend=E2BExecBackend(
            holder, policy=policy, default_timeout=default_timeout
        ),
        create_params=create_params,
        owns_sandbox=True,
        pause_on_exit=pause_on_exit,
    )


# --- grep shell-out helpers --------------------------------------------------


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
    args.extend(["-e", pattern, _wire(root)])
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


__all__ = [
    "E2BEnvironment",
    "E2BExecBackend",
    "E2BFileBackend",
    "e2b_environment",
]
