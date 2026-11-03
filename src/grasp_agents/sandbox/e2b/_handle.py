"""
Shared E2B sandbox handle + low-level adapters for the e2b backends.

``SandboxHandle`` holds the single live ``AsyncSandbox`` that the file, exec,
and kernel surfaces all address (co-location by construction); the helpers
translate paths, detect command timeouts, and read entry metadata.
"""

from __future__ import annotations

import asyncio
import base64
import posixpath
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from e2b import TimeoutException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from e2b import AsyncSandbox, EntryInfo


RECURSIVE_DEPTH = 64


DEFAULT_WORKSPACE = "/home/user/workspace"


DEFAULT_EXEC_TIMEOUT = 600.0  # matches the local supervisor's overall default


MAX_OUTPUT_CHARS = 1_000_000  # per the local supervisor's per-stream cap


TRANSPORT_TIMEOUT_NAMES = frozenset(
    {"ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException"}
)


def is_timeout(exc: BaseException) -> bool:
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
        if cls.__name__ in TRANSPORT_TIMEOUT_NAMES and cls.__module__.startswith(
            ("httpcore", "httpx")
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def mtime(entry: EntryInfo) -> float:
    """Epoch seconds from an e2b ``EntryInfo.modified_time`` (a ``datetime``)."""
    return float(entry.modified_time.timestamp())


def is_dir(entry: EntryInfo) -> bool:
    """True if an e2b entry is a directory (``entry.type`` is a ``FileType``)."""
    return str(getattr(entry.type, "value", entry.type)) == "dir"


def normalize_posix(path: Path | str) -> PurePosixPath:
    """
    Lexically normalize a sandbox path (collapse ``.`` / ``..`` segments).

    Containment checks compare path prefixes; a literal ``..`` segment would
    pass them and escape ``allowed_roots`` when the VM resolves it. Lexical
    only — remote symlinks cannot be resolved from the host; the VM remains
    the hard boundary.
    """
    return PurePosixPath(posixpath.normpath(str(PurePosixPath(path))))


def wire(path: Path) -> str:
    """Render a path as the absolute POSIX string the sandbox expects."""
    return str(normalize_posix(path))


def wrap_stdin(command: str, stdin: bytes | None) -> str:
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


class SandboxHandle:
    """
    Shared, mutable holder for the live e2b ``AsyncSandbox``.

    Both backends and the environment reference the *same* holder, so a
    ``restore()`` (reconnect/resume) swaps the live sandbox under the file and
    exec surfaces at once.

    A lazily started environment *arms* the handle with a factory instead of
    creating the sandbox on entry; :meth:`ensure` then creates it on the first
    file or exec call, once, even when several calls race.
    """

    __slots__ = ("_factory", "_lock", "sandbox")

    def __init__(self, sandbox: AsyncSandbox | None = None) -> None:
        self.sandbox: AsyncSandbox | None = sandbox
        self._factory: Callable[[], Awaitable[AsyncSandbox]] | None = None
        self._lock = asyncio.Lock()

    @property
    def armed(self) -> bool:
        """True while a factory is waiting to start the sandbox on first use."""
        return self._factory is not None

    def arm(self, factory: Callable[[], Awaitable[AsyncSandbox]]) -> None:
        """Start the sandbox with ``factory`` on the first :meth:`ensure`."""
        self._factory = factory

    def disarm(self) -> None:
        """Stop starting sandboxes on demand (the environment was exited)."""
        self._factory = None

    async def ensure(self) -> AsyncSandbox:
        """Return the live sandbox, starting it first when the handle is armed."""
        if self.sandbox is not None:
            return self.sandbox
        if self._factory is None:
            return self.require()
        async with self._lock:
            if self.sandbox is None and self._factory is not None:
                self.sandbox = await self._factory()
        return self.require()

    def require(self) -> AsyncSandbox:
        if self.sandbox is None:
            raise RuntimeError(
                "E2B environment is not entered. Use `async with env:` (the "
                "context manager creates the sandbox) before any file/exec call."
            )
        return self.sandbox
