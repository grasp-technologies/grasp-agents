"""
Sandbox / execution-environment abstractions: the shared
:class:`~grasp_agents.sandbox.policy.SandboxPolicy`, the :class:`ExecBackend`
exec surface, the :class:`ExecutionEnvironment` that co-locates it with a
``FileBackend``, and the optional :class:`SnapshotCapable` durability
capability.

The lightweight protocols + value types are imported eagerly (they are leaf
modules — no cross-package imports). The concrete backends + factory
(:class:`LocalExecBackend`, :class:`LocalEnvironment`, :func:`local_environment`,
:class:`ProcessSupervisor`) are lazy (PEP 562) so importing this package — which
``RunContext`` does at construction — does not eagerly pull in the file-tool
stack.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .environment import ExecutionEnvironment, SnapshotCapable
from .exec_backend import ExecBackend, ExecChunk, ExecResult, TerminationReason
from .policy import NetworkPolicy, SandboxPolicy

if TYPE_CHECKING:
    from .local_env import LocalEnvironment, local_environment
    from .local_exec import LocalExecBackend
    from .seatbelt import (
        SeatbeltExecBackend,
        build_seatbelt_profile,
        seatbelt_argv,
    )
    from .supervisor import ExecSpec, ProcessSupervisor, SupervisorLimits


_LAZY: dict[str, str] = {
    "ExecSpec": "supervisor",
    "ProcessSupervisor": "supervisor",
    "SupervisorLimits": "supervisor",
    "LocalExecBackend": "local_exec",
    "LocalEnvironment": "local_env",
    "local_environment": "local_env",
    "SeatbeltExecBackend": "seatbelt",
    "build_seatbelt_profile": "seatbelt",
    "seatbelt_argv": "seatbelt",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for next access
    return attr


__all__ = [
    "ExecBackend",
    "ExecChunk",
    "ExecResult",
    "ExecSpec",
    "ExecutionEnvironment",
    "LocalEnvironment",
    "LocalExecBackend",
    "NetworkPolicy",
    "ProcessSupervisor",
    "SandboxPolicy",
    "SeatbeltExecBackend",
    "SnapshotCapable",
    "SupervisorLimits",
    "TerminationReason",
    "build_seatbelt_profile",
    "local_environment",
    "seatbelt_argv",
]
