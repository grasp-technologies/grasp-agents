"""
The exec surface: an :class:`ExecBackend` runs shell commands inside an
:class:`~grasp_agents.sandbox.environment.ExecutionEnvironment`, co-located
with the environment's :class:`~grasp_agents.tools.file_edit.backend.FileBackend`.

The contract is non-interactive — no TTY, no persistent shell state between
calls. Concrete backends (host subprocess + Seatbelt/bwrap confinement,
Docker, remote providers) implement it; a single process supervisor owns the
spawn/timeout/kill lifecycle so per-backend code stays thin.
"""

from __future__ import annotations

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


@runtime_checkable
class ExecBackend(Protocol):
    """
    Command-execution surface, co-located with a ``FileBackend``.

    A backend is bound to one :class:`SandboxPolicy` and one *location* (host /
    container / remote) and MUST execute against the same filesystem the paired
    ``FileBackend`` addresses — co-location is guaranteed at construction, not
    checked at runtime.
    """

    @property
    def name(self) -> str: ...

    @property
    def policy(self) -> SandboxPolicy: ...

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


__all__ = [
    "ExecBackend",
    "ExecChunk",
    "ExecResult",
    "TerminationReason",
]
