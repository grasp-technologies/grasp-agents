"""
``E2BEnvironment`` + the :func:`e2b_environment` factory — an
:class:`ExecutionEnvironment` over an E2B cloud sandbox, with
:class:`SnapshotCapable` wired to E2B's native pause/resume.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict

from e2b import AsyncSandbox

from grasp_agents.sandbox.environment import ExecutionEnvironment, SnapshotCapable
from grasp_agents.sandbox.policy import NetworkPolicy, SandboxPolicy

from ._handle import (
    DEFAULT_EXEC_TIMEOUT,
    DEFAULT_WORKSPACE,
    SandboxHandle,
    wire,
)
from .exec import E2BExecBackend
from .file_backend import E2BFileBackend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


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


def _sandbox_cls(code_interpreter: bool) -> type[AsyncSandbox]:
    """Sandbox class to create: code-interpreter (has ``run_code``) or base."""
    if not code_interpreter:
        return AsyncSandbox
    try:
        from e2b_code_interpreter import AsyncSandbox as CISandbox  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - the e2b extra installs both
        raise RuntimeError(
            "code_interpreter=True needs the e2b-code-interpreter package "
            "(install grasp-agents[e2b])."
        ) from exc
    return CISandbox


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
        holder: SandboxHandle,
        file_backend: E2BFileBackend,
        exec_backend: E2BExecBackend,
        create_params: E2BCreateParams | None = None,
        owns_sandbox: bool = True,
        pause_on_exit: bool = False,
        code_interpreter: bool = False,
    ) -> None:
        self._policy = policy
        self._holder = holder
        self._file_backend = file_backend
        self._exec_backend = exec_backend
        self._create_params = create_params
        self._owns_sandbox = owns_sandbox
        self._pause_on_exit = pause_on_exit
        self._code_interpreter = code_interpreter

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

    async def __aenter__(self) -> Self:
        created_here = False
        if self._holder.sandbox is None:
            if self._create_params is None:
                raise RuntimeError(
                    "E2BEnvironment has no sandbox and no create params; build "
                    "it via e2b_environment(...) or E2BEnvironment.from_sandbox(...)."
                )
            sandbox_cls = _sandbox_cls(self._code_interpreter)
            self._holder.sandbox = await sandbox_cls.create(**self._create_params)
            created_here = True
        sandbox = self._holder.require()
        try:
            for root in self._file_backend.allowed_roots:
                await sandbox.files.make_dir(wire(root))
        except BaseException:
            # Don't orphan a just-created (billed) sandbox when setup fails.
            if created_here:
                self._holder.sandbox = None
                try:
                    await sandbox.kill()
                except Exception:
                    logger.warning(
                        "Failed to kill E2B sandbox after setup failure; "
                        "it may keep running (and billing) until its timeout.",
                        exc_info=True,
                    )
            raise
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
        # Create from the snapshot, not a base template.
        params: E2BCreateParams = {**(self._create_params or {}), "template": ref}
        sandbox_cls = _sandbox_cls(self._code_interpreter)
        self._holder.sandbox = await sandbox_cls.create(**params)
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
        metadata: Custom sandbox metadata.
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

    create_params: E2BCreateParams = {
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
    )
