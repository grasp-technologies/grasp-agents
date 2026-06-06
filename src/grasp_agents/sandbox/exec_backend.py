"""
The exec surface: an :class:`ExecBackend` runs shell commands inside an
:class:`~grasp_agents.sandbox.environment.ExecutionEnvironment`, co-located
with the environment's :class:`~grasp_agents.tools.file_backend.base.FileBackend`.

The contract is non-interactive — no TTY, no persistent shell state between
calls. Concrete backends (host subprocess + Seatbelt/bwrap confinement,
Docker, remote providers) implement it; a single process supervisor owns the
spawn/timeout/kill lifecycle so per-backend code stays thin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from pathlib import Path

    from .policy import SandboxPolicy


class TerminationReason(StrEnum):
    """Why a command stopped running."""

    EXIT = "exit"  # ran to completion (any return code)
    OVERALL_TIMEOUT = "overall_timeout"  # exceeded the overall wall-clock limit
    NO_OUTPUT_TIMEOUT = "no_output_timeout"  # produced no output for too long
    MANUAL_CANCEL = "manual_cancel"  # cancelled by the caller / turn abort
    SIGNAL = "signal"  # killed by a signal
    SPAWN_ERROR = "spawn_error"  # never started


@dataclass(frozen=True)
class ExecChunk:
    """One streamed output fragment from a running command."""

    stream: Literal["stdout", "stderr"]
    data: str


@dataclass(frozen=True)
class ExecResult:
    """
    Outcome of a single :meth:`ExecBackend.execute` call.

    ``backend`` names the isolation that actually applied (``"local"`` /
    ``"seatbelt"`` / ``"bwrap"`` / ``"docker"`` / ...), so callers and
    per-backend boundary docs can tell what confinement was in force.
    """

    stdout: str
    stderr: str
    returncode: int
    reason: TerminationReason
    runtime_ms: float
    backend: str
    truncated: bool = False

    @property
    def timed_out(self) -> bool:
        return self.reason in {
            TerminationReason.OVERALL_TIMEOUT,
            TerminationReason.NO_OUTPUT_TIMEOUT,
        }


class ExecBackend(ABC):
    """
    Command-execution surface, co-located with a ``FileBackend``.

    A backend is bound to one :class:`SandboxPolicy` and one *location* (host /
    container / remote) and MUST execute against the same filesystem the paired
    ``FileBackend`` addresses — co-location is guaranteed at construction, not
    checked at runtime.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def policy(self) -> SandboxPolicy: ...

    @abstractmethod
    async def execute(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        """
        Run ``command`` to completion and return its result.

        ``cwd`` must resolve under the policy's roots. ``timeout`` is the
        overall wall-clock ceiling in seconds (``None`` uses the backend
        default). ``env`` is merged onto :attr:`SandboxPolicy.env`.
        """
        ...

    @abstractmethod
    def stream(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        """
        Stream output as it arrives: a sequence of :class:`ExecChunk` items
        followed by a single terminal :class:`ExecResult`.
        """
        ...


@runtime_checkable
class ExecSession(Protocol):
    """
    A persistent, stateful shell opened from a :class:`SessionCapable` backend.

    Unlike :meth:`ExecBackend.stream` — which spawns a fresh process per call —
    a session is one long-lived shell: ``cd``, environment changes, and shell
    variables persist across :meth:`run` calls. It is serial (commands run one
    at a time) and process-local (it dies with the host; nothing survives a
    restart).
    """

    @property
    def backend(self) -> str: ...

    @property
    def closed(self) -> bool:
        """True once the shell has exited or been closed; reopen via the backend."""
        ...

    def run(
        self, command: str, *, timeout: float | None = None
    ) -> AsyncIterator[ExecChunk | ExecResult]:
        """
        Run ``command`` in the shell and stream its output exactly like
        :meth:`ExecBackend.stream` (``ExecChunk`` items then one terminal
        ``ExecResult``) — but in the persistent shell, so state carries over.
        ``timeout`` closes the whole session (there is no per-command interrupt).
        """
        ...

    async def close(self) -> None:
        """Terminate the shell and release it."""
        ...


@runtime_checkable
class SessionCapable(Protocol):
    """
    An :class:`ExecBackend` that can open a persistent :class:`ExecSession`.

    Detect with ``isinstance(backend, SessionCapable)``; backends that cannot
    hold a long-lived shell simply do not implement it.
    """

    async def open_session(
        self, *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> ExecSession: ...


__all__ = [
    "ExecBackend",
    "ExecChunk",
    "ExecResult",
    "ExecSession",
    "SessionCapable",
    "TerminationReason",
]
