"""
``LocalExecBackend`` — runs commands as host subprocesses.

The exec sibling of the local ``FileBackend``: both address the host
filesystem, so a command's view of files matches what the file tools read and
write (co-location by construction).

**Boundary doc (the three questions every backend answers):**

1. *What can this backend contain?* With ``confinement="none"`` — **nothing at
   the OS level.** It is a plain subprocess: the supervisor bounds runtime
   (timeouts), output size, and kills the process group on exit, but the
   command sees the whole host filesystem and the inherited environment. The
   shared :class:`SandboxPolicy` constrains only ``cwd`` selection and the base
   env here; real OS confinement arrives with the ``seatbelt`` / ``bwrap``
   variants.
2. *What is outside the boundary / trusted to the host?* Everything: reads,
   writes outside ``allowed_roots``, network, signals, other processes. **Local
   is not a sandbox.** Use it only for trusted/personal/dev work, or pair the
   policy with Seatbelt/bwrap/Docker.
3. *What did we configure and why?* ``inherit_host_env=True`` so ordinary tools
   resolve on ``PATH`` (the unconfined path is meant to just work); the
   credential-dotfile deny list still governs the *file* tools via the paired
   ``LocalFileBackend``.

POSIX only — see :class:`ProcessSupervisor`.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import replace
from typing import TYPE_CHECKING

from ..tools.file_edit.paths import PathAccessError, resolve_safe
from .exec_backend import ExecChunk, ExecResult
from .supervisor import ExecSpec, ProcessSupervisor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from .policy import SandboxPolicy
    from .supervisor import SupervisorLimits


class LocalExecBackend:
    """
    Host-subprocess :class:`~grasp_agents.sandbox.exec_backend.ExecBackend`.

    Args:
        policy: The shared :class:`SandboxPolicy`. Its ``allowed_roots`` bound
            the working directory; its ``env`` is layered onto the base
            environment.
        supervisor: The :class:`ProcessSupervisor` that owns the spawn/kill
            lifecycle. A fresh one is created when omitted.
        name: Confinement label surfaced on :class:`ExecResult` (``"local"``
            for the unconfined path; subclasses set ``"seatbelt"`` / etc.).
        inherit_host_env: When True (default for the unconfined path), the
            subprocess inherits ``os.environ`` as its base — necessary for
            ``PATH`` to resolve. Confined backends override to a scrubbed,
            minimal env.

    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        supervisor: ProcessSupervisor | None = None,
        name: str = "local",
        inherit_host_env: bool = True,
    ) -> None:
        self._policy = policy
        self._supervisor = supervisor or ProcessSupervisor()
        self._name = name
        self._inherit_host_env = inherit_host_env

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
        spec = self._build_spec(command, cwd=cwd, env=env)
        limits = self._limits_for(timeout)
        out: list[str] = []
        err: list[str] = []
        terminal: ExecResult | None = None
        async for item in self._supervisor.run(spec, stdin=stdin, limits=limits):
            if isinstance(item, ExecChunk):
                (out if item.stream == "stdout" else err).append(item.data)
            else:
                terminal = item
        if terminal is None:  # pragma: no cover - supervisor always yields one
            raise RuntimeError("supervisor produced no terminal ExecResult")
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
        spec = self._build_spec(command, cwd=cwd, env=env)
        limits = self._limits_for(timeout)
        async for item in self._supervisor.run(spec, stdin=stdin, limits=limits):
            yield item

    # --- spec construction (the override point for confined backends) ------

    def _build_spec(
        self, command: str, *, cwd: Path | None, env: Mapping[str, str] | None
    ) -> ExecSpec:
        return ExecSpec(
            argv=("/bin/sh", "-c", command),
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
        )

    def _resolve_cwd(self, cwd: Path | None) -> Path:
        roots = [*self._policy.allowed_roots, *self._policy.readonly_roots]
        if not roots:
            raise PathAccessError(
                "SandboxPolicy has no allowed_roots; cannot choose a working "
                "directory for the command."
            )
        if cwd is None:
            return roots[0]
        return resolve_safe(cwd, roots, must_exist=True)

    def _merged_env(self, env: Mapping[str, str] | None) -> dict[str, str]:
        scrub = self._policy.env_scrub
        if self._inherit_host_env:
            merged = {
                k: v
                for k, v in os.environ.items()
                if not any(fnmatch.fnmatchcase(k, pat) for pat in scrub)
            }
        else:
            merged = {}
        # Deliberately-set env wins and is never scrubbed.
        merged.update(self._policy.env)
        if env:
            merged.update(env)
        return merged

    def _limits_for(self, timeout: float | None) -> SupervisorLimits:
        base = self._supervisor.limits
        if timeout is None:
            return base
        return replace(base, overall_timeout=timeout)


__all__ = ["LocalExecBackend"]
