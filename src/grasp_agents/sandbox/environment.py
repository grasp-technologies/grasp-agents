"""
:class:`ExecutionEnvironment` — one boundary owning two co-located surfaces
(filesystem + exec) and a shared :class:`~grasp_agents.sandbox.policy.SandboxPolicy`,
plus the optional :class:`SnapshotCapable` durability capability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from ..tools.file_backend.base import FileBackend
    from .exec_backend import ExecBackend
    from .policy import SandboxPolicy


@runtime_checkable
class SnapshotCapable(Protocol):
    """
    Optional capability for a backend that owns versioned snapshots of its
    state.

    The framework delegates filesystem-state durability to whoever implements
    this — an MCP server's append-only versioning, a sandbox provider's native
    snapshot, or a local shadow-git / tar store. :meth:`snapshot` returns an
    opaque ref (DB version, blob id, git sha) recorded in the checkpoint
    manifest; :meth:`restore` rewinds to it. Backends that do not implement it
    fall back to the per-edit transcript pre-image.
    """

    async def snapshot(self) -> str: ...

    async def restore(self, ref: str) -> None: ...


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """
    One boundary owning two co-located surfaces plus a shared policy.

    :attr:`file_backend` and :attr:`exec_backend` address the same filesystem
    by construction. :attr:`policy` is consumed by both — the file tools
    enforce it on I/O, the exec backend generates its OS confinement from it.
    The async context-manager lifecycle is a no-op for local backends and owns
    container / remote create + teardown for the rest. :attr:`exec_backend` is
    ``None`` for filesystem-only environments (e.g. memory authoring).
    """

    @property
    def policy(self) -> SandboxPolicy: ...

    @property
    def file_backend(self) -> FileBackend: ...

    @property
    def exec_backend(self) -> ExecBackend | None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc: object) -> None: ...


__all__ = ["ExecutionEnvironment", "SnapshotCapable"]
