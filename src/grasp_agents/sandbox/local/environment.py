"""
``LocalEnvironment`` + the :func:`local_environment` factory — backend #1 of
Phase D: a co-located host filesystem + host subprocess pair sharing one
:class:`SandboxPolicy`.

The factory is the co-location guarantee: it unpacks a single policy into a
:class:`~grasp_agents.tools.file_backend.local.LocalFileBackend` and an
exec backend that address the *same* host filesystem, so a host cannot
accidentally point the two surfaces at different locations. Wire the result onto
:attr:`RunContext.environment`; the ``RunContext`` validator sources
``file_backend`` from it, and ``ctx.exec_backend`` is the read-only property off
it (the environment is the sole grantor of an exec surface).

Confinement: ``"none"`` is a plain subprocess with **no** OS isolation (see
:class:`LocalExecBackend`); ``"seatbelt"`` (macOS) wraps every command in a
generated Seatbelt profile confining writes + spawned processes (see
:class:`~grasp_agents.sandbox.seatbelt.SeatbeltExecBackend`); ``"auto"`` selects
Seatbelt on macOS; ``"bwrap"`` (Linux) is not built yet. ``readonly_roots`` are
write-denied on the file-tool plane under every confinement; on the exec plane
only Seatbelt OS-denies them (under ``"none"`` shell commands can still write
there, and ``network`` is recorded-not-enforced).
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from ...tools.file_backend.local import LocalFileBackend
from ..environment import ExecutionEnvironment
from ..policy import NetworkPolicy, SandboxPolicy
from .exec import LocalExecBackend

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...tools.file_backend.base import FileBackend
    from ..exec_backend import ExecBackend
    from .supervisor import ProcessSupervisor


class LocalEnvironment(ExecutionEnvironment):
    """
    An :class:`~grasp_agents.sandbox.environment.ExecutionEnvironment` over the
    host: a :class:`LocalFileBackend` and a :class:`LocalExecBackend` sharing
    one :class:`SandboxPolicy`.

    Prefer the :func:`local_environment` factory, which builds the policy and
    both surfaces together. The async context-manager lifecycle is a no-op
    here (nothing to create or tear down on the host); it exists so callers can
    treat every environment uniformly regardless of backend.
    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        file_backend: FileBackend,
        exec_backend: ExecBackend | None,
    ) -> None:
        self._policy = policy
        self._file_backend = file_backend
        self._exec_backend = exec_backend

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @property
    def file_backend(self) -> FileBackend:
        return self._file_backend

    @property
    def exec_backend(self) -> ExecBackend | None:
        return self._exec_backend

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def local_environment(
    *,
    allowed_roots: Sequence[Path | str],
    readonly_roots: Sequence[Path | str] = (),
    deny_read: Sequence[Path | str] = (),
    allow_read: Sequence[Path | str] = (),
    deny_write: Sequence[Path | str] = (),
    confinement: Literal["none", "seatbelt", "bwrap", "srt", "auto"] = "none",
    network: NetworkPolicy = NetworkPolicy.NONE,
    allowed_domains: Sequence[str] = (),
    denied_domains: Sequence[str] = (),
    include_dotfile_denylist: bool = True,
    env: Mapping[str, str] | None = None,
    env_scrub: Sequence[str] = (),
    inherit_host_env: bool = True,
    supervisor: ProcessSupervisor | None = None,
) -> LocalEnvironment:
    """
    Build a co-located host filesystem + exec pair sharing one policy.

    Args:
        allowed_roots: Read+write address space. The first entry is the
            default working directory for commands.
        readonly_roots: Additional readable locations. The file tools deny
            writes to them under every confinement; on the exec plane they
            are OS-write-denied under Seatbelt (not in the writable root
            set) but unprotected under ``"none"``.
        deny_read: Carve unreadable regions out of the readable space.
        allow_read: Re-allow reads within ``deny_read`` regions (allow wins).
        deny_write: Carve write-protected regions out of ``allowed_roots``
            (deny wins). Enforced on both planes (file tools + exec).
        confinement: ``"none"`` (plain subprocess, no isolation), ``"seatbelt"``
            (macOS OS confinement), ``"srt"`` (delegate to Anthropic's
            sandbox-runtime CLI if installed — adds domain-allowlisted network
            egress our native Seatbelt lacks), or ``"auto"`` (Seatbelt on macOS,
            else a warned fallback to ``"none"``). ``"bwrap"`` (Linux) is not
            built yet.
        network: Recorded on the policy. Not enforced under ``"none"``; under
            Seatbelt, ``NONE`` and ``ALL`` are enforced (loopback / allowlist
            need a proxy — deferred).
        allowed_domains: Egress allowlist (proxy-backed backends only — srt /
            hosted; Seatbelt egress is all-or-nothing).
        denied_domains: Egress denylist, checked first (proxy-backed only).
        include_dotfile_denylist: Stack the credential-dotfile deny list onto
            the file backend's path policy.
        env: Base environment overlaid onto the (optionally inherited) host
            environment for subprocesses.
        env_scrub: ``fnmatch`` patterns (e.g. ``"*_API_KEY"``) removed from the
            inherited host environment so secrets do not leak into commands.
        inherit_host_env: Subprocesses inherit ``os.environ`` as their base so
            ``PATH`` resolves. Set False for a minimal environment.
        supervisor: Shared :class:`ProcessSupervisor` for the exec backend.

    Returns:
        A :class:`LocalEnvironment` exposing ``file_backend`` + ``exec_backend``.

    """
    roots = tuple(Path(r).expanduser() for r in allowed_roots)
    ro = tuple(Path(r).expanduser() for r in readonly_roots)
    dr = tuple(Path(p).expanduser() for p in deny_read)
    ar = tuple(Path(p).expanduser() for p in allow_read)
    dw = tuple(Path(p).expanduser() for p in deny_write)
    policy = SandboxPolicy(
        allowed_roots=roots,
        readonly_roots=ro,
        deny_read=dr,
        allow_read=ar,
        deny_write=dw,
        include_dotfile_denylist=include_dotfile_denylist,
        network=network,
        allowed_domains=tuple(allowed_domains),
        denied_domains=tuple(denied_domains),
        env=env or {},
        env_scrub=tuple(env_scrub),
    )

    file_backend = LocalFileBackend(
        allowed_roots=list(roots),
        readonly_roots=list(ro),
        include_dotfiles=include_dotfile_denylist,
        deny_read=list(dr),
        allow_read=list(ar),
        deny_write=list(dw),
    )
    exec_backend = _build_exec_backend(
        confinement,
        policy=policy,
        supervisor=supervisor,
        inherit_host_env=inherit_host_env,
    )
    return LocalEnvironment(
        policy=policy, file_backend=file_backend, exec_backend=exec_backend
    )


def _resolve_confinement(confinement: str) -> str:
    """Resolve ``"auto"`` to a concrete backend for the current platform."""
    if confinement != "auto":
        return confinement
    if sys.platform == "darwin":
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap") is not None:
        return "bwrap"
    warnings.warn(
        "confinement='auto' found no OS sandbox (need macOS for Seatbelt or "
        "bwrap on Linux); falling back to 'none' — this is NOT a sandbox.",
        RuntimeWarning,
        stacklevel=3,
    )
    return "none"


def _build_exec_backend(
    confinement: str,
    *,
    policy: SandboxPolicy,
    supervisor: ProcessSupervisor | None,
    inherit_host_env: bool,
) -> ExecBackend:
    resolved = _resolve_confinement(confinement)
    if resolved == "none":
        return LocalExecBackend(
            policy=policy,
            supervisor=supervisor,
            name="local",
            inherit_host_env=inherit_host_env,
        )
    if resolved == "seatbelt":
        if sys.platform != "darwin":
            raise RuntimeError(
                "confinement='seatbelt' requires macOS (it shells out to sandbox-exec)."
            )
        if shutil.which("sandbox-exec") is None:
            raise RuntimeError(
                "sandbox-exec was not found; cannot use confinement='seatbelt'."
            )
        from .seatbelt import SeatbeltExecBackend  # noqa: PLC0415

        return SeatbeltExecBackend(
            policy=policy,
            supervisor=supervisor,
            inherit_host_env=inherit_host_env,
        )
    if resolved == "srt":
        if shutil.which("srt") is None:
            raise RuntimeError(
                "confinement='srt' requires the srt CLI on PATH "
                "(npm install -g @anthropic-ai/sandbox-runtime). It adds OS "
                "confinement + domain-allowlisted network egress; otherwise use "
                "'seatbelt' or 'none'."
            )
        from .srt import SrtExecBackend  # noqa: PLC0415

        return SrtExecBackend(
            policy=policy,
            supervisor=supervisor,
            inherit_host_env=inherit_host_env,
        )
    if resolved == "bwrap":
        raise NotImplementedError(
            "confinement='bwrap' (Linux) is not implemented yet; it lands after "
            "Seatbelt. Use 'none', or run on macOS with 'seatbelt'."
        )
    raise ValueError(f"unknown confinement: {confinement!r}")


__all__ = ["LocalEnvironment", "local_environment"]
