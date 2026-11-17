"""
:class:`ExecutionEnvironment` — one boundary owning two co-located surfaces
(filesystem + exec) and a shared :class:`~grasp_agents.sandbox.policy.SandboxPolicy`,
plus the optional :class:`SnapshotCapable` durability capability and
:class:`NetworkUpdatable`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.file_backend.base import FileBackend

    from .exec_backend import ExecBackend
    from .policy import NetworkPolicy, SandboxPolicy


@runtime_checkable
class SnapshotCapable(Protocol):
    """
    Optional capability for an environment that can snapshot and rewind its
    filesystem (e.g. :class:`~grasp_agents.sandbox.E2BEnvironment`).

    :meth:`snapshot` captures the current state and returns an opaque ref, which
    the framework records in the session checkpoint and per-step watermarks;
    :meth:`restore` rewinds the environment to a ref on resume or step rollback.

    There is no fallback for environments that do not implement it:
    ``SessionContext.fs_snapshot_policy`` must stay ``"off"`` (any other value
    raises), checkpoints record no filesystem state, and resuming or rolling
    back a session leaves the files as they are.
    """

    async def snapshot(self) -> str: ...

    async def restore(self, ref: str) -> None: ...


@runtime_checkable
class NetworkUpdatable(Protocol):
    """
    Optional capability for an environment whose egress can change while it
    runs (e.g. :class:`~grasp_agents.sandbox.E2BEnvironment`).

    :meth:`update_network` applies a new ``network`` policy, with its domain
    lists, to the live sandbox and to every sandbox the environment creates
    after it, and the environment's and exec backend's ``policy`` report it
    from then on. Local environments fix their egress when they are built:
    build a new one to change it.
    """

    async def update_network(
        self,
        network: NetworkPolicy,
        *,
        allowed_domains: Sequence[str] = (),
        denied_domains: Sequence[str] = (),
    ) -> None: ...


class ExecutionEnvironment(ABC):
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
    @abstractmethod
    def policy(self) -> SandboxPolicy: ...

    @property
    @abstractmethod
    def file_backend(self) -> FileBackend: ...

    @property
    @abstractmethod
    def exec_backend(self) -> ExecBackend | None: ...

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, *exc: object) -> None: ...


__all__ = ["ExecutionEnvironment", "NetworkUpdatable", "SnapshotCapable"]
