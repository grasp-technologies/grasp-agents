"""
Shared E2B sandbox handle + low-level adapters for the e2b backends.

``SandboxHandle`` holds the single live ``AsyncSandbox`` that the file, exec,
and kernel surfaces all address (co-location by construction); the helpers
translate paths, detect command timeouts, and read entry metadata.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import posixpath
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from e2b import TimeoutException

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from e2b import AsyncSandbox, EntryInfo

logger = logging.getLogger(__name__)


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


async def kill_quietly(sandbox: AsyncSandbox) -> None:
    """Kill ``sandbox`` on a cleanup path: a failure is logged, not raised."""
    try:
        await sandbox.kill()
    except Exception:
        logger.warning(
            "Failed to kill E2B sandbox %s; it may keep running (and billing) "
            "until its timeout.",
            sandbox.sandbox_id,
            exc_info=True,
        )


def _consume_outcome(start: asyncio.Task[AsyncSandbox]) -> None:
    # Callers cancelled mid-start no longer await it: retrieve its failure here
    # so asyncio does not report it as never retrieved.
    if not start.cancelled():
        start.exception()


class SandboxHandle:
    """
    Shared, mutable holder for the live e2b ``AsyncSandbox``.

    Both backends and the environment reference the *same* holder, so a
    ``restore()`` (reconnect/resume) swaps the live sandbox under the file and
    exec surfaces at once.

    A lazily started environment *arms* the handle with a factory instead of
    creating the sandbox on entry; :meth:`ensure` then creates it on the first
    file or exec call, once, even when several calls race. The start holds
    :attr:`lock`, which the environment's exit, restore, pause and resume also
    take, so those wait for a start in flight.
    """

    __slots__ = ("_factory", "_lock", "_start_task", "_waiting", "sandbox")

    def __init__(self, sandbox: AsyncSandbox | None = None) -> None:
        self.sandbox: AsyncSandbox | None = sandbox
        self._factory: Callable[[], Awaitable[AsyncSandbox]] | None = None
        self._lock = asyncio.Lock()
        self._start_task: asyncio.Task[AsyncSandbox] | None = None
        self._waiting = 0

    @property
    def lock(self) -> asyncio.Lock:
        """Held while a sandbox starts; hold it to swap or shut down the sandbox."""
        return self._lock

    @property
    def armed(self) -> bool:
        """True while a factory is waiting to start the sandbox on first use."""
        return self._factory is not None

    def arm(self, factory: Callable[[], Awaitable[AsyncSandbox]]) -> None:
        """Start the sandbox with ``factory`` on the first :meth:`ensure`."""
        self._factory = factory

    def disarm(self) -> None:
        """
        Stop starting sandboxes on demand (the environment was exited). A start
        already in flight kills its sandbox when it finishes.
        """
        self._factory = None

    async def ensure(self) -> AsyncSandbox:
        """
        Return the live sandbox, starting it first when the handle is armed.

        Concurrent first calls share one start, which runs in a task of its
        own, so cancelling a caller does not cancel it. The start keeps its
        sandbox only if a caller still awaits it and the handle is still armed
        and empty; otherwise it kills the sandbox and raises, so a sandbox it
        created is never left running unowned.
        """
        if self.sandbox is not None:
            return self.sandbox
        if self._factory is None:
            return self.require()
        start = self._start_task
        if start is None or start.done():
            start = self._start_task = asyncio.create_task(self._start())
            start.add_done_callback(_consume_outcome)
        self._waiting += 1
        try:
            return await asyncio.shield(start)
        finally:
            self._waiting -= 1

    async def _start(self) -> AsyncSandbox:
        async with self._lock:
            if self.sandbox is not None:
                return self.sandbox
            factory = self._factory
            if factory is None:
                return self.require()
            sandbox = await factory()
            if self._waiting and self._factory is factory and self.sandbox is None:
                self.sandbox = sandbox
                return sandbox
            await kill_quietly(sandbox)
            raise RuntimeError(
                "Killed a new E2B sandbox: the environment exited or changed its "
                "sandbox, or every call waiting for it was cancelled, while it "
                "was starting."
            )

    def require(self) -> AsyncSandbox:
        if self.sandbox is None:
            raise RuntimeError(
                "E2B environment is not entered. Use `async with env:` (the "
                "context manager creates the sandbox) before any file/exec call."
            )
        return self.sandbox

    def live(self, sandbox_id: str) -> AsyncSandbox | None:
        """
        The live sandbox while it is still the one with ``sandbox_id``, else
        ``None``. A resume reconnects to the same sandbox through a new client,
        so this compares ids, not client objects.
        """
        sandbox = self.sandbox
        if sandbox is None or sandbox.sandbox_id != sandbox_id:
            return None
        return sandbox
