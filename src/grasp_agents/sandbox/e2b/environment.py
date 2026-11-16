"""
``E2BEnvironment`` + the :func:`e2b_environment` factory — an
:class:`ExecutionEnvironment` over an E2B cloud sandbox, with
:class:`SnapshotCapable` wired to E2B's native snapshots.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypedDict

from e2b import AsyncSandbox

from grasp_agents.sandbox.environment import ExecutionEnvironment, SnapshotCapable
from grasp_agents.sandbox.policy import NetworkPolicy, SandboxPolicy

from ._handle import (
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_WORKSPACE,
    SandboxHandle,
    consume_outcome,
    kill_quietly,
    wire,
)
from ._names import METADATA_KEY, check_label, snapshot_name
from .exec import E2BExecBackend
from .file_backend import E2BFileBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping, Sequence

    from e2b.connection_config import ApiParams

logger = logging.getLogger(__name__)

FRESH_SNAPSHOT_REF = "e2b-fresh"
"""Snapshot ref of a sandbox that never started: restoring it starts over."""


class E2BCreateParams(TypedDict, total=False):
    """
    Keyword args splatted into ``AsyncSandbox.create(**...)``.

    Mirrors the subset of the E2B SDK's create signature the framework
    populates (in :func:`e2b_environment`). Typing it — rather than a bare
    ``dict[str, Any]`` — checks the factory's dict literal and catches a key
    typo at the splat. ``restore`` overrides ``template`` to spawn from a
    snapshot.
    """

    template: str | None
    timeout: int | None
    metadata: dict[str, str] | None
    envs: dict[str, str] | None
    secure: bool
    allow_internet_access: bool
    api_key: str
    domain: str


async def _outlive_cancellation(
    body: Callable[[asyncio.Event], Coroutine[Any, Any, None]],
    pending: set[asyncio.Task[None]],
) -> None:
    """
    Run ``body`` in a task of its own, so cancelling the caller does not stop
    it midway, after E2B may already have created or resumed a sandbox.
    ``body`` gets an event set once the caller was cancelled, and must then
    let go of what it obtained instead of installing it.
    """
    abandoned = asyncio.Event()
    task = asyncio.create_task(body(abandoned))
    pending.add(task)
    task.add_done_callback(pending.discard)
    task.add_done_callback(consume_outcome)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        abandoned.set()
        raise


def _sandbox_cls(code_interpreter: bool) -> type[AsyncSandbox]:
    """Sandbox class to create: code-interpreter (has ``run_code``) or base."""
    if not code_interpreter:
        return AsyncSandbox
    try:
        from e2b_code_interpreter import (  # ruff: ignore[import-outside-top-level]
            AsyncSandbox as CISandbox,
        )
    except ImportError as exc:  # pragma: no cover - the e2b extra installs both
        raise RuntimeError(
            "code_interpreter=True needs the e2b-code-interpreter package "
            "(install grasp-agents[e2b])."
        ) from exc
    return CISandbox


class E2BEnvironment(ExecutionEnvironment, SnapshotCapable):
    """
    An :class:`ExecutionEnvironment` over an E2B cloud sandbox, with
    :class:`SnapshotCapable` wired to E2B's native snapshots (``snapshot`` /
    ``restore`` rewind to a point-in-time copy; ``pause`` / ``resume`` suspend
    and reconnect the same sandbox).

    Prefer the :func:`e2b_environment` factory. The async context manager owns
    sandbox create (``__aenter__``, or the first file or exec call with
    ``lazy=True``) and teardown (``__aexit__`` — kill, or pause when
    ``pause_on_exit``). The ``file_backend`` / ``exec_backend`` objects exist
    before entry (so a ``SessionContext`` can source them), but their I/O raises
    until the context is entered.
    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        holder: SandboxHandle,
        file_backend: E2BFileBackend,
        exec_backend: E2BExecBackend,
        create_params: E2BCreateParams | None = None,
        owns_sandbox: bool = True,
        pause_on_exit: bool = False,
        code_interpreter: bool = False,
        lazy: bool = False,
        label: str | None = None,
    ) -> None:
        self._lazy = lazy
        self._label = check_label(label)
        self._policy = policy
        self._holder = holder
        self._file_backend = file_backend
        self._exec_backend = exec_backend
        self._create_params = create_params
        self._owns_sandbox = owns_sandbox
        self._pause_on_exit = pause_on_exit
        self._code_interpreter = code_interpreter
        self._entered = False
        # Restores and resumes that outlive their cancelled caller.
        self._pending: set[asyncio.Task[None]] = set()

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
        holder = SandboxHandle(sandbox)
        return cls(
            policy=policy,
            holder=holder,
            file_backend=E2BFileBackend(holder, policy=policy),
            exec_backend=E2BExecBackend(holder, policy=policy),
            create_params=None,
            owns_sandbox=owns_sandbox,
            pause_on_exit=pause_on_exit,
        )

    @property
    def started(self) -> bool:
        """
        True while the environment holds a sandbox: from entry until exit, or,
        with ``lazy=True``, from the first file or exec call. Restoring
        :data:`FRESH_SNAPSHOT_REF` on a lazy environment drops it until the
        next call.
        """
        return self._holder.sandbox is not None

    async def __aenter__(self) -> Self:
        # A sandbox E2B removes is replaced on the next call, when there are
        # create params to start one from.
        self._holder.replace_expired_with(
            self._start if self._create_params is not None else None
        )
        if self._holder.sandbox is not None:
            await self._prepare(self._holder.sandbox, created_here=False)
            self._entered = True
            return self
        if self._create_params is None:
            raise RuntimeError(
                "E2BEnvironment has no sandbox and no create params; build "
                "it via e2b_environment(...) or E2BEnvironment.from_sandbox(...)."
            )
        if self._lazy:
            self._holder.arm(self._start)
        else:
            self._holder.sandbox = await self._start()
        self._entered = True
        return self

    def _require_entered(self, operation: str) -> None:
        if not self._entered:
            raise RuntimeError(
                f"E2B environment is not entered: {operation}() needs "
                "`async with env:`; after exit, nothing would kill the sandbox "
                "it starts."
            )

    async def _start(self) -> AsyncSandbox:
        """Create a sandbox from the create params and prepare its roots."""
        if self._create_params is None:
            raise RuntimeError(
                "E2BEnvironment has no create params to start a fresh sandbox "
                "from; it wraps an existing one (from_sandbox)."
            )
        sandbox_cls = _sandbox_cls(self._code_interpreter)
        logger.info("creating E2B sandbox (%s)", sandbox_cls.__name__)
        t0 = time.monotonic()
        sandbox = await sandbox_cls.create(**self._create_params)
        logger.info("E2B sandbox ready in %.1fs", time.monotonic() - t0)
        await self._prepare(sandbox, created_here=True)
        return sandbox

    async def _prepare(self, sandbox: AsyncSandbox, *, created_here: bool) -> None:
        """Make the allowed roots; kill a sandbox created here if that fails."""
        try:
            for root in self._file_backend.allowed_roots:
                await sandbox.files.make_dir(wire(root))
        except BaseException:
            # Don't orphan a just-created (billed) sandbox when setup fails.
            if created_here:
                await kill_quietly(sandbox)
            raise

    async def __aexit__(self, *exc: object) -> None:
        # Disarm before waiting for the lock: a start in flight then kills its
        # sandbox when it finishes, and no new start begins.
        self._entered = False
        self._holder.disarm()
        async with self._holder.lock:
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

        A lazy environment whose sandbox has not started yet returns
        :data:`FRESH_SNAPSHOT_REF` without starting one.
        """
        if self._holder.sandbox is None and self._holder.armed:
            return FRESH_SNAPSHOT_REF
        sandbox = self._holder.require()
        try:
            name = snapshot_name(self._label, datetime.now(UTC))
            snap = await sandbox.create_snapshot(name=name)
        except Exception as exc:
            await self._holder.raise_if_expired(sandbox, exc)
            raise
        finally:
            self._holder.drop_streams()
        return str(snap.snapshot_id)

    async def restore(self, ref: str) -> None:
        """
        Rewind to a snapshot: spawn a fresh sandbox from ``ref`` (its filesystem
        + memory at snapshot time) and swap the live handle under both backends.
        The previous sandbox, if owned, is killed.

        :data:`FRESH_SNAPSHOT_REF` rewinds to a sandbox that never ran: a lazy
        environment drops its sandbox and starts a new one on next use, even
        after a failed start; an eager one starts a new one from its create
        params (its configured template) now. An environment that wraps an
        existing sandbox (:meth:`from_sandbox`) has no create params, so it
        refuses this ref.

        Cancelling the call does not cancel a create E2B may already be
        running: the sandbox it creates is killed instead of installed. Raises
        ``RuntimeError`` outside ``async with env:``, where nothing would kill
        the sandbox it creates.
        """
        await _outlive_cancellation(
            lambda abandoned: self._restore(ref, abandoned), self._pending
        )

    async def _restore(self, ref: str, abandoned: asyncio.Event) -> None:
        async with self._holder.lock:
            if abandoned.is_set():
                return
            self._require_entered("restore")
            previous = self._holder.sandbox
            if ref == FRESH_SNAPSHOT_REF and self._holder.armed:
                self._holder.sandbox = None
                # Re-arming forgets a failed start: the next call retries.
                self._holder.arm(self._start)
            else:
                restored = await self._create_from(ref)
                if abandoned.is_set():
                    await kill_quietly(restored)
                    return
                self._holder.sandbox = restored
            self._holder.drop_streams()
            if previous is not None and self._owns_sandbox:
                await kill_quietly(previous)

    async def _create_from(self, ref: str) -> AsyncSandbox:
        if ref == FRESH_SNAPSHOT_REF:
            return await self._start()
        # Create from the snapshot instead of the configured template.
        params: E2BCreateParams = {**(self._create_params or {}), "template": ref}
        return await _sandbox_cls(self._code_interpreter).create(**params)

    async def pause(self) -> str:
        """
        Suspend the sandbox (filesystem + memory preserved) and return its id.

        Resume the *same* sandbox later with :meth:`resume`. Distinct from
        :meth:`snapshot`: pause is suspend/resume (no rewind), whereas a snapshot
        is a persistent point-in-time copy you spawn fresh sandboxes from.
        """
        async with self._holder.lock:
            if self._holder.sandbox is None and self._holder.armed:
                raise RuntimeError("Nothing to pause: the sandbox was never started.")
            sandbox = self._holder.require()
            try:
                await sandbox.pause()
            finally:
                self._holder.drop_streams()
            return str(sandbox.sandbox_id)

    async def resume(self, sandbox_id: str | None = None) -> None:
        """
        Reconnect to (and auto-resume) a paused sandbox, swapping the live
        handle. Defaults to the currently-bound sandbox's id.

        Reconnects with the sandbox class the environment creates (the code
        interpreter one with ``code_interpreter=True``) and its configured
        API key, domain and keep-alive timeout. Resuming another sandbox lets
        go of the previous one as exit would: an owned one is killed, or
        paused with ``pause_on_exit``.

        Cancelling the call does not cancel a connect E2B may already be
        running: the environment lets go of the sandbox it resumed (unless it
        is the one it already holds) instead of installing it. Raises
        ``RuntimeError`` outside ``async with env:``, where nothing would kill
        or pause the sandbox it connects.
        """
        await _outlive_cancellation(
            lambda abandoned: self._resume(sandbox_id, abandoned), self._pending
        )

    async def _resume(self, sandbox_id: str | None, abandoned: asyncio.Event) -> None:
        async with self._holder.lock:
            if abandoned.is_set():
                return
            self._require_entered("resume")
            previous = self._holder.sandbox
            target = sandbox_id
            if target is None:
                if previous is None:
                    raise RuntimeError(
                        "resume() needs a sandbox_id when no sandbox is bound."
                    )
                target = str(previous.sandbox_id)
            params: E2BCreateParams = self._create_params or {}
            opts: ApiParams = {}
            if "api_key" in params:
                opts["api_key"] = params["api_key"]
            if "domain" in params:
                opts["domain"] = params["domain"]
            sandbox_cls = _sandbox_cls(self._code_interpreter)
            resumed = await sandbox_cls.connect(
                target, timeout=params.get("timeout"), **opts
            )
            replaces = previous is None or previous.sandbox_id != resumed.sandbox_id
            if abandoned.is_set():
                if replaces:
                    await self._let_go(resumed)
                return
            self._holder.sandbox = resumed
            self._holder.drop_streams()
            if previous is not None and replaces:
                await self._let_go(previous)

    async def _let_go(self, sandbox: AsyncSandbox) -> None:
        """
        Do to a sandbox the environment no longer holds what its exit would:
        nothing unless owned, else pause it (``pause_on_exit``) or kill it.
        Failures are logged, not raised.
        """
        if not self._owns_sandbox:
            return
        if not self._pause_on_exit:
            await kill_quietly(sandbox)
            return
        try:
            await sandbox.pause()
        except Exception:
            logger.warning(
                "Failed to pause E2B sandbox %s; it may keep running (and "
                "billing) until its timeout.",
                sandbox.sandbox_id,
                exc_info=True,
            )


def e2b_environment(
    *,
    allowed_roots: Sequence[str] = (DEFAULT_WORKSPACE,),
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
    code_interpreter: bool = False,
    default_timeout: float = DEFAULT_EXEC_TIMEOUT,
    lazy: bool = False,
    label: str | None = None,
) -> E2BEnvironment:
    """
    Build a deferred-create E2B environment; the sandbox is created on
    ``async with``, or on first use with ``lazy=True``.

    Args:
        allowed_roots: POSIX paths *inside* the sandbox the file tools may
            address. The first is the default working directory; all are
            ``mkdir -p``'d when the sandbox starts.
        deny_read: Remote paths carved out of the readable space (tool plane).
        allow_read: Remote paths re-allowed within ``deny_read`` (allow wins).
        deny_write: Remote paths write-protected on the tool plane (deny wins).
        template: E2B sandbox template name/ID (default base template). This is
            how the sandbox's Python environment is provisioned: build a custom
            E2B template (a Docker image with your packages — e.g. torch — baked
            in) and pass its name/ID here. The framework does not pip-install into
            E2B sandboxes; bake requirements into the template.
        sandbox_timeout: Sandbox keep-alive in seconds (E2B default 300).
        network: ``NONE`` (no internet) or ``ALL`` (internet on). ``ALLOWLIST``
            / ``LOOPBACK`` raise ``NotImplementedError`` (per-domain egress needs
            E2B network rules — future work).
        env: Base environment variables for the sandbox + every command.
        metadata: Custom sandbox metadata. The environment adds its own
            ``"grasp-agents"`` key, set to ``label``.
        api_key: E2B API key (defaults to the ``E2B_API_KEY`` env var).
        domain: E2B domain (defaults to the ``E2B_DOMAIN`` env var).
        secure: Token-secure the sandbox (E2B default ``True``).
        pause_on_exit: Pause instead of kill on context exit (state persists;
            reconnect later with the sandbox id).
        code_interpreter: Create a **code-interpreter** sandbox (via
            ``e2b-code-interpreter``) so ``RunCell`` / notebook execution works
            (``E2BExecBackend.open_kernel``). The base template is shell + files
            only. Files + commands behave identically either way.
        default_timeout: Per-command wall-clock ceiling when a call passes no
            ``timeout``. The backend enforces it: a command that runs past it
            is reported as ``OVERALL_TIMEOUT`` and the shell running it is
            killed; processes that shell started may keep running.
        lazy: Start the sandbox on the first file or exec call instead of on
            ``async with``, so a session that never needs it never pays for
            one. Concurrent first calls share one sandbox; ``snapshot()``
            before the start returns :data:`FRESH_SNAPSHOT_REF`. A failed
            start fails like a failed entry: later calls raise without
            creating another sandbox until the environment is re-entered (or
            restored to :data:`FRESH_SNAPSHOT_REF`).
        label: Up to 32 lowercase letters, digits or dashes marking what this
            environment creates: the sandbox metadata's ``"grasp-agents"``
            value and part of every snapshot's name (``"default"`` when
            ``None``). Deployments that share an E2B team give each its own,
            so cleaning up after one never touches another's.

    Returns:
        An :class:`E2BEnvironment` to use as ``async with``.

    """
    if network in {NetworkPolicy.ALLOWLIST, NetworkPolicy.LOOPBACK}:
        raise NotImplementedError(
            f"E2B example backend does not implement network={network.value!r} "
            "(per-domain / loopback egress needs E2B network rules). Use "
            "NetworkPolicy.NONE or NetworkPolicy.ALL."
        )
    label = check_label(label)

    roots = tuple(Path(r) for r in allowed_roots)
    policy = SandboxPolicy(
        allowed_roots=roots,
        deny_read=tuple(Path(p) for p in deny_read),
        allow_read=tuple(Path(p) for p in allow_read),
        deny_write=tuple(Path(p) for p in deny_write),
        network=network,
        env=dict(env) if env else {},
    )

    create_params: E2BCreateParams = {
        "template": template,
        "timeout": sandbox_timeout,
        "metadata": {**(metadata or {}), METADATA_KEY: label},
        "envs": dict(env) if env else None,
        "secure": secure,
        "allow_internet_access": network == NetworkPolicy.ALL,
    }
    if api_key is not None:
        create_params["api_key"] = api_key
    if domain is not None:
        create_params["domain"] = domain

    holder = SandboxHandle()
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
        code_interpreter=code_interpreter,
        lazy=lazy,
        label=label,
    )
