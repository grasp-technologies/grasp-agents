"""
Shared E2B sandbox handle + low-level adapters for the e2b backends.

``SandboxHandle`` holds the single live ``AsyncSandbox`` that the file, exec,
and kernel surfaces all address (co-location by construction); the helpers
translate paths, detect command timeouts, and read entry metadata.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import posixpath
import shlex
import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from e2b import (
    SandboxException,
    SandboxNotFoundException,
    SandboxState,
    TimeoutException,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable
    from pathlib import Path

    from e2b import AsyncSandbox, EntryInfo

logger = logging.getLogger(__name__)


RECURSIVE_DEPTH = 64


DEFAULT_WORKSPACE = "/home/user/workspace"


DEFAULT_EXEC_TIMEOUT = 600.0  # matches the local supervisor's overall default


E2B_DEFAULT_KEEP_ALIVE = 300  # seconds E2B keeps a sandbox created without one


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


class E2BSandboxExpiredError(RuntimeError):
    """
    E2B no longer has the environment's sandbox: it reached its keep-alive
    timeout or lifetime cap, or was killed from elsewhere. Its files, shell
    sessions and kernels are gone. The call that found out fails with this
    error; the next call starts a new sandbox when the environment can
    create one.
    """

    def __init__(self, sandbox_id: str, *, replaceable: bool) -> None:
        self.sandbox_id = sandbox_id
        self.replaceable = replaceable
        lost = (
            f"The E2B sandbox {sandbox_id} expired: E2B no longer has it, so its "
            "files, shell sessions and kernels are gone."
        )
        super().__init__(
            f"{lost} The next call runs in a new sandbox."
            if replaceable
            else f"{lost} This environment wraps that sandbox and cannot start "
            "another one."
        )


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


def consume_outcome(task: asyncio.Task[Any]) -> None:
    """
    Retrieve the failure of a task its cancelled callers no longer await, so
    asyncio does not report it as never retrieved.
    """
    if not task.cancelled():
        task.exception()


class SandboxHandle:
    """
    Shared, mutable holder for the live e2b ``AsyncSandbox``.

    Both backends and the environment reference the *same* holder, so a
    ``restore()`` (reconnect/resume) swaps the live sandbox under the file and
    exec surfaces at once.

    A lazily started environment *arms* the handle with a factory instead of
    creating the sandbox on entry; :meth:`ensure` then creates it on the first
    file or exec call, once, even when several calls race. The start holds
    :attr:`lock`, which the environment's exit, restore, pause, resume and
    snapshot also take, so those wait for a start in flight.

    With ``pauses_on_timeout``, E2B pauses the sandbox at the end of its
    keep-alive. The environment records when each keep-alive starts
    (:meth:`keep_alive_from`), and from its end a call first resumes the
    sandbox if E2B paused it (:meth:`ready`). ``keep_alive`` is the timeout
    such a resume passes E2B (``None``: E2B's default).
    """

    __slots__ = (
        "_expired",
        "_factory",
        "_failure",
        "_held_paused",
        "_keep_alive",
        "_lock",
        "_pause_due",
        "_pauses_on_timeout",
        "_replacement",
        "_start_task",
        "sandbox",
        "stream_epoch",
    )

    def __init__(
        self,
        sandbox: AsyncSandbox | None = None,
        *,
        keep_alive: int | None = None,
        pauses_on_timeout: bool = False,
    ) -> None:
        self.sandbox: AsyncSandbox | None = sandbox
        # Bumped whenever the sandbox drops its open connections; a shell
        # that sees a newer value reattaches to its output.
        self.stream_epoch = 0
        self._factory: Callable[[], Awaitable[AsyncSandbox]] | None = None
        self._failure: Exception | None = None
        self._lock = asyncio.Lock()
        self._start_task: asyncio.Task[AsyncSandbox] | None = None
        self._replacement: Callable[[], Awaitable[AsyncSandbox]] | None = None
        self._expired: str | None = None
        self._keep_alive = keep_alive
        self._pauses_on_timeout = pauses_on_timeout
        # The sandbox whose keep-alive is tracked, and the time.monotonic()
        # reading from which E2B may have paused it.
        self._pause_due: tuple[str, float] | None = None
        # A sandbox the environment paused itself: calls leave it paused.
        self._held_paused: str | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """Held while a sandbox starts; hold it to swap or shut down the sandbox."""
        return self._lock

    @property
    def armed(self) -> bool:
        """True while a factory is waiting to start the sandbox on first use."""
        return self._factory is not None

    def arm(self, factory: Callable[[], Awaitable[AsyncSandbox]]) -> None:
        """
        Start the sandbox with ``factory`` on the first :meth:`ensure`, trying
        again even if an earlier start failed.
        """
        self._factory = factory
        self._failure = None

    def disarm(self) -> None:
        """
        Stop starting sandboxes on demand (the environment was exited). A start
        already in flight kills its sandbox when it finishes.
        """
        self._factory = None
        self._replacement = None

    def replace_expired_with(
        self, factory: Callable[[], Awaitable[AsyncSandbox]] | None
    ) -> None:
        """
        Start a sandbox with ``factory`` on the next call after E2B removed
        the live one; with ``None``, calls keep failing instead.
        """
        self._replacement = factory
        self._expired = None

    @contextlib.asynccontextmanager
    async def use(self) -> AsyncGenerator[AsyncSandbox]:
        """
        The live sandbox (see :meth:`ensure`) for one call. A failure that
        comes from E2B no longer having it raises
        :class:`E2BSandboxExpiredError`, and one that comes from E2B having
        paused it resumes it for the next call (see :meth:`recover`).
        """
        sandbox = await self.ensure()
        try:
            yield sandbox
        except Exception as exc:
            await self.recover(sandbox, exc)
            raise

    async def retrying[T](
        self, sandbox: AsyncSandbox, call: Callable[[], Awaitable[T]]
    ) -> T:
        """
        Await ``call``, and once more if it failed because E2B had paused
        ``sandbox``, now resumed (see :meth:`recover`). Only for a call that
        is harmless to repeat: one running user code may have run up to the
        pause.
        """
        try:
            return await call()
        except Exception as exc:
            if not await self.recover(sandbox, exc):
                raise
        return await call()

    async def recover(self, sandbox: AsyncSandbox, exc: Exception) -> bool:
        """
        Act on ``exc``, a failed call on ``sandbox``, and return True if it
        resumed the sandbox.

        If E2B no longer has the sandbox, drop it from the handle while it is
        still the live one (the next call then starts a new sandbox, if the
        environment can create one) and raise :class:`E2BSandboxExpiredError`
        from ``exc``. If E2B paused it, resume it (see
        :meth:`resume_if_paused`). Otherwise, or if E2B cannot tell, return
        False.

        E2B reports a call to a removed or paused sandbox the way it reports
        some timeouts, so a failure that could mean either is checked with
        the sandbox's info.
        """
        if isinstance(exc, E2BSandboxExpiredError) or (
            isinstance(exc, SandboxException)
            and not isinstance(exc, (TimeoutException, SandboxNotFoundException))
        ):
            return False
        try:
            info = await sandbox.get_info()
        except SandboxNotFoundException:
            pass
        except Exception:
            return False
        else:
            if info.state == SandboxState.PAUSED:
                return await self.resume_if_paused(sandbox)
            return False
        sandbox_id = str(sandbox.sandbox_id)
        if self.live(sandbox_id) is not None:
            self.sandbox = None
            if self._factory is None and self._replacement is not None:
                self.arm(self._replacement)
            elif self._factory is None:
                self._expired = sandbox_id
        logger.warning("E2B sandbox %s no longer exists", sandbox_id)
        raise E2BSandboxExpiredError(
            sandbox_id, replaceable=self._factory is not None
        ) from exc

    async def ensure(self) -> AsyncSandbox:
        """
        Return the live sandbox, starting it first when the handle is armed,
        or resuming it first if E2B paused it at its keep-alive (see
        :meth:`ready`).

        Concurrent first calls share one start, which runs in a task of its
        own, so cancelling a caller does not cancel it. The start keeps its
        sandbox whenever the handle is still armed with the same factory and
        holds no other sandbox, even if every caller was cancelled: the next
        call uses it, and the environment's exit kills it. Otherwise (the
        environment exited, or another sandbox was installed) it kills the
        sandbox and raises, so a sandbox it created is never left running
        unowned.

        A failed start is not retried until the handle is re-armed: later calls
        raise at once instead of creating, and billing, a sandbox each.
        """
        while (sandbox := self.sandbox) is not None:
            await self.ready(sandbox)
            if self.sandbox is sandbox:
                return sandbox
        if self._factory is None:
            return self.require()
        if self._failure is not None:
            raise RuntimeError(
                "The E2B sandbox failed to start; re-enter the environment to "
                "try again."
            ) from self._failure
        start = self._start_task
        if start is None or start.done():
            start = self._start_task = asyncio.create_task(self._start())
            start.add_done_callback(consume_outcome)
        return await asyncio.shield(start)

    async def _start(self) -> AsyncSandbox:
        async with self._lock:
            if self.sandbox is not None:
                return self.sandbox
            factory = self._factory
            if factory is None:
                return self.require()
            try:
                sandbox = await factory()
            except Exception as exc:
                self._failure = exc
                raise
            if self._factory is factory and self.sandbox is None:
                self.sandbox = sandbox
                return sandbox
            await kill_quietly(sandbox)
            raise RuntimeError(
                "Killed a new E2B sandbox: the environment exited or changed its "
                "sandbox while it was starting."
            )

    def drop_streams(self) -> None:
        """
        Record that the sandbox dropped every open connection, command
        streams included: E2B does so when it pauses the sandbox, also for
        the duration of a snapshot, and a resume reconnects through a new
        client.
        """
        self.stream_epoch += 1

    def keep_alive_from(self, sandbox_id: str, since: float) -> None:
        """
        Record that E2B started ``sandbox_id``'s keep-alive (a create or a
        resume) no earlier than ``since``, a ``time.monotonic()`` reading
        taken before the request. It is no longer held paused.
        """
        if self._held_paused == sandbox_id:
            self._held_paused = None
        if self._pauses_on_timeout:
            keep_alive = (
                E2B_DEFAULT_KEEP_ALIVE if self._keep_alive is None else self._keep_alive
            )
            self._pause_due = (sandbox_id, since + keep_alive)

    def hold_paused(self, sandbox_id: str) -> None:
        """
        The environment paused ``sandbox_id`` itself: calls leave it paused
        until :meth:`keep_alive_from` records a resume.
        """
        self._held_paused = sandbox_id

    async def ready(self, sandbox: AsyncSandbox) -> None:
        """
        Before a call on ``sandbox``: once its keep-alive is over, resume it
        if E2B paused it (see :meth:`resume_if_paused`). Before that, and for
        a keep-alive the handle does not track, return at once.
        """
        due = self._pause_due
        if (
            due is not None
            and due[0] == sandbox.sandbox_id
            and time.monotonic() >= due[1]
        ):
            await self.resume_if_paused(sandbox)

    async def resume_if_paused(self, sandbox: AsyncSandbox) -> bool:
        """
        Resume ``sandbox`` in place, with the handle's keep-alive, if E2B
        paused it, it is still the live sandbox and the environment did not
        pause it itself; return True if it did. A resume drops the sandbox's
        open streams (:meth:`drop_streams`). Holds :attr:`lock`, so it waits
        for a start, exit, restore, pause, resume or snapshot in flight.
        """
        sandbox_id = str(sandbox.sandbox_id)
        async with self._lock:
            if self.live(sandbox_id) is None or self._held_paused == sandbox_id:
                return False
            since = time.monotonic()
            try:
                info = await sandbox.get_info()
            except Exception:
                return False
            if info.state != SandboxState.PAUSED:
                if self._pauses_on_timeout:
                    left = info.end_at.timestamp() - time.time()
                    self._pause_due = (sandbox_id, time.monotonic() + max(0.0, left))
                return False
            logger.info("resuming E2B sandbox %s, which E2B paused", sandbox_id)
            try:
                await sandbox.connect(timeout=self._keep_alive)
            except Exception:
                logger.warning(
                    "Failed to resume E2B sandbox %s", sandbox_id, exc_info=True
                )
                return False
            self.drop_streams()
            self.keep_alive_from(sandbox_id, since)
            return True

    def require(self) -> AsyncSandbox:
        if self.sandbox is None and self._expired is not None:
            raise E2BSandboxExpiredError(self._expired, replaceable=False)
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
