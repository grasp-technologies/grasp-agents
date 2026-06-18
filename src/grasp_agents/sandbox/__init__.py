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
    from .config import (
        EnvironmentConfig,
        ExecConfig,
        FilesystemConfig,
        NetworkConfig,
        load_environment_config,
    )
    from .e2b.config import (
        E2BEnvironmentConfig,
        E2BTemplateConfig,
        load_e2b_config,
    )
    from .e2b.environment import E2BEnvironment, e2b_environment
    from .e2b.exec import E2BExecBackend
    from .e2b.file_backend import E2BFileBackend
    from .local.environment import LocalEnvironment, local_environment
    from .local.exec import LocalExecBackend
    from .local.seatbelt import (
        SeatbeltExecBackend,
        build_seatbelt_profile,
        seatbelt_argv,
    )
    from .local.srt import SrtExecBackend, build_srt_settings, srt_argv
    from .local.supervisor import ExecSpec, ProcessSupervisor, SupervisorLimits


_LAZY: dict[str, str] = {
    "EnvironmentConfig": "config",
    "ExecConfig": "config",
    "FilesystemConfig": "config",
    "NetworkConfig": "config",
    "load_environment_config": "config",
    "E2BEnvironment": "e2b.environment",
    "E2BEnvironmentConfig": "e2b.config",
    "E2BTemplateConfig": "e2b.config",
    "load_e2b_config": "e2b.config",
    "E2BExecBackend": "e2b.exec",
    "E2BFileBackend": "e2b.file_backend",
    "e2b_environment": "e2b.environment",
    "ExecSpec": "local.supervisor",
    "ProcessSupervisor": "local.supervisor",
    "SupervisorLimits": "local.supervisor",
    "LocalExecBackend": "local.exec",
    "LocalEnvironment": "local.environment",
    "local_environment": "local.environment",
    "SeatbeltExecBackend": "local.seatbelt",
    "build_seatbelt_profile": "local.seatbelt",
    "seatbelt_argv": "local.seatbelt",
    "SrtExecBackend": "local.srt",
    "build_srt_settings": "local.srt",
    "srt_argv": "local.srt",
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
    "E2BEnvironment",
    "E2BEnvironmentConfig",
    "E2BExecBackend",
    "E2BFileBackend",
    "E2BTemplateConfig",
    "EnvironmentConfig",
    "ExecBackend",
    "ExecChunk",
    "ExecConfig",
    "ExecResult",
    "ExecSpec",
    "ExecutionEnvironment",
    "FilesystemConfig",
    "LocalEnvironment",
    "LocalExecBackend",
    "NetworkConfig",
    "NetworkPolicy",
    "ProcessSupervisor",
    "SandboxPolicy",
    "SeatbeltExecBackend",
    "SnapshotCapable",
    "SrtExecBackend",
    "SupervisorLimits",
    "TerminationReason",
    "build_seatbelt_profile",
    "build_srt_settings",
    "e2b_environment",
    "load_e2b_config",
    "load_environment_config",
    "local_environment",
    "seatbelt_argv",
    "srt_argv",
]
