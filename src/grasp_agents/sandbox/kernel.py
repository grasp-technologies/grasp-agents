"""
The kernel surface: a :class:`KernelCapable` exec backend can open a
:class:`KernelSession` — a live language kernel that keeps in-memory state
(variables, imports) across :meth:`execute` calls and yields **rich MIME
bundles** (text, ``image/png`` plots, HTML) instead of the plain stdout/stderr
text an :class:`~grasp_agents.sandbox.exec_backend.ExecSession` produces.

It is the execution half of the notebook tools: parallel to
:class:`~grasp_agents.sandbox.exec_backend.SessionCapable` /
:class:`~grasp_agents.sandbox.exec_backend.ExecSession`, detected the same way
(``isinstance(backend, KernelCapable)``), and reached through the *same*
environment as ``Bash`` — so the kernel inherits the co-located ``FileBackend``
+ shared ``SandboxPolicy`` (confinement) for free.

This module is kernel-protocol-agnostic: it defines the contract and the output
shape (:class:`CellOutput` mirrors the nbformat output union) but knows nothing
about nbformat or the agent's content parts. The local Jupyter implementation
lives in :mod:`.local_kernel`; the notebook layer
(:mod:`grasp_agents.tools.notebook_exec`) converts a :class:`CellOutput` to an
nbformat output for write-back and to an ``InputImage`` / ``InputText`` for the
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

CellStatus = Literal["ok", "error", "abort"]
CellOutputType = Literal["stream", "execute_result", "display_data", "error"]


@dataclass(frozen=True)
class CellOutput:
    """
    One output from a running cell, in nbformat shape (kernel-agnostic).

    Mirrors the nbformat output union so the notebook layer can both store it
    in the ``.ipynb`` and render it for the model:

    * ``stream`` — ``name`` (``stdout`` / ``stderr``) + ``text``;
    * ``execute_result`` / ``display_data`` — ``data`` (a MIME→payload bundle,
      e.g. ``text/plain``, ``image/png``) [+ ``execution_count``];
    * ``error`` — ``ename`` + ``evalue`` + ``traceback``.
    """

    output_type: CellOutputType
    name: str | None = None
    text: str | None = None
    data: Mapping[str, Any] | None = None
    execution_count: int | None = None
    ename: str | None = None
    evalue: str | None = None
    traceback: tuple[str, ...] = ()


@dataclass(frozen=True)
class CellResult:
    """Terminal outcome of one :meth:`KernelSession.execute` call."""

    status: CellStatus
    execution_count: int | None = None
    runtime_ms: float = 0.0
    timed_out: bool = False


@runtime_checkable
class KernelSession(Protocol):
    """
    A live language kernel opened from a :class:`KernelCapable` backend.

    Like :class:`~grasp_agents.sandbox.exec_backend.ExecSession` it is stateful
    (variables/imports persist across :meth:`execute`), serial (one execution at
    a time), and process-local (it dies with the host; nothing survives a
    restart — the ``.ipynb`` is the durable artifact, re-run on resume).
    """

    @property
    def closed(self) -> bool:
        """True once the kernel has exited or been closed; reopen via the backend."""
        ...

    def execute(
        self, code: str, *, timeout: float | None = None
    ) -> AsyncIterator[CellOutput | CellResult]:
        """
        Run ``code`` in the kernel and stream its outputs: a sequence of
        :class:`CellOutput` items followed by one terminal :class:`CellResult`.
        ``timeout`` interrupts the running cell (and, if it ignores the
        interrupt, closes the kernel).
        """
        ...

    async def interrupt(self) -> None:
        """Interrupt the currently-running cell (kernel stays alive)."""
        ...

    async def restart(self) -> None:
        """Restart the kernel, clearing all in-memory state."""
        ...

    async def close(self) -> None:
        """Terminate the kernel and release it."""
        ...


@runtime_checkable
class KernelCapable(Protocol):
    """
    An :class:`~grasp_agents.sandbox.exec_backend.ExecBackend` that can open a
    :class:`KernelSession`.

    Detect with ``isinstance(backend, KernelCapable)``; backends that cannot host
    a kernel simply do not implement it. The kernel launches under the backend's
    own confinement (the same wrapper it applies to a shell), so it shares the
    environment's ``allowed_roots`` / network / timeout contract.
    """

    async def open_kernel(
        self, *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> KernelSession: ...


__all__ = [
    "CellOutput",
    "CellOutputType",
    "CellResult",
    "CellStatus",
    "KernelCapable",
    "KernelSession",
]
