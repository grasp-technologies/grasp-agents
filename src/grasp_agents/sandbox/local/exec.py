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
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from grasp_agents.file_backend.paths import PathAccessError, resolve_safe
from grasp_agents.sandbox.exec_backend import (
    ExecBackend,
    ExecChunk,
    ExecResult,
    SessionCapable,
)
from grasp_agents.sandbox.kernel import KernelCapable

from .kernel import LocalKernel
from .session import LocalExecSession
from .supervisor import ExecSpec, ProcessSupervisor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from grasp_agents.sandbox.policy import SandboxPolicy

    from .supervisor import SupervisorLimits


def resolve_python(python: str | Path | None) -> str:
    """
    Resolve a configured interpreter to a launchable path (default
    ``sys.executable``).

    Accepts an interpreter path (e.g. ``"<venv>/bin/python"``) or a bare command
    name resolved on ``PATH``. Raises ``ValueError`` if neither resolves.
    """
    if python is None:
        return sys.executable
    candidate = Path(python).expanduser()
    if candidate.is_file():
        # Keep the given path; do NOT resolve symlinks — a venv's bin/python is
        # a symlink to the base interpreter, and resolving it would launch the
        # base env (no venv site-packages → no ipykernel).
        return str(candidate)
    found = shutil.which(str(python))
    if found is not None:
        return found
    raise ValueError(
        f"configured python interpreter {python!r} was not found "
        "(expected an existing interpreter path like '<venv>/bin/python', "
        "or a command available on PATH)."
    )


class LocalExecBackend(ExecBackend, SessionCapable, KernelCapable):
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
        python: Interpreter the Jupyter kernel launches with (and whose bin dir
            leads ``PATH`` for shells), defaulting to ``sys.executable``. Point
            it at a venv/conda interpreter (e.g. ``"<venv>/bin/python"``) to run
            the code interpreter in that environment; it must have ``ipykernel``.

    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        supervisor: ProcessSupervisor | None = None,
        name: str = "local",
        inherit_host_env: bool = True,
        python: str | Path | None = None,
    ) -> None:
        self._policy = policy
        self._supervisor = supervisor or ProcessSupervisor()
        self._name = name
        self._inherit_host_env = inherit_host_env
        self._python = resolve_python(python)
        # When a non-default interpreter is configured, lead PATH with its bin
        # dir so a shell's `python` / `pip` resolve to the same env as the kernel.
        self._python_path_prepend = (
            str(Path(self._python).parent) if python is not None else None
        )

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
        # Case-insensitive scrub: a lowercase secret var must not slip past
        # uppercase patterns.
        scrub = [pat.upper() for pat in self._policy.env_scrub]
        if self._inherit_host_env:
            merged = {
                k: v
                for k, v in os.environ.items()
                if not any(fnmatch.fnmatchcase(k.upper(), pat) for pat in scrub)
            }
        else:
            merged = {}
        # Deliberately-set env wins and is never scrubbed.
        merged.update(self._policy.env)
        if env:
            merged.update(env)
        # A configured interpreter's bin dir leads PATH so a shell's `python` /
        # `pip` match the kernel's interpreter.
        if self._python_path_prepend:
            existing = merged.get("PATH", "")
            merged["PATH"] = (
                f"{self._python_path_prepend}{os.pathsep}{existing}"
                if existing
                else self._python_path_prepend
            )
        return merged

    def _limits_for(self, timeout: float | None) -> SupervisorLimits:
        base = self._supervisor.limits
        if timeout is None:
            return base
        return replace(base, overall_timeout=timeout)

    # --- persistent sessions (SessionCapable) ------------------------------

    def _session_argv(self) -> tuple[str, ...]:
        """Argv that launches the persistent shell; confined backends wrap it."""
        return ("/bin/sh",)

    async def open_session(
        self, *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> LocalExecSession:
        """
        Open a persistent shell that keeps ``cd`` / env / variables across
        commands (see :class:`LocalExecSession`). Same confinement as one-shot
        ``stream`` — the wrapper is applied to the shell itself.
        """
        return LocalExecSession(
            argv=self._session_argv(),
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
            limits=self._supervisor.limits,
        )

    # --- kernels (KernelCapable) -------------------------------------------

    def _kernel_launch_argv(self, connection_file: str) -> tuple[str, ...]:
        """
        Argv that launches a Jupyter kernel; confined backends wrap it (the
        same override point as :meth:`_session_argv` for shells).
        """
        return (self._python, "-m", "ipykernel_launcher", "-f", connection_file)

    async def open_kernel(
        self,
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        context_id: str | None = None,
    ) -> LocalKernel:
        """
        Open a live Jupyter kernel co-located with this backend's filesystem.
        Same confinement as one-shot ``stream`` — the wrapper is applied to the
        kernel process via :meth:`_kernel_launch_argv`.

        ``context_id`` is accepted for protocol parity but ignored: a local
        kernel cannot persist state across a restart, so there is nothing to
        re-attach to on resume.
        """
        del context_id
        return LocalKernel(
            launch_argv=self._kernel_launch_argv,
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
            setup_code=self._policy.kernel_setup_code,
            startup_timeout=self._policy.kernel_startup_timeout,
        )


__all__ = ["LocalExecBackend"]
