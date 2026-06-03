"""
Shared sandbox policy: the single source of truth for what an
:class:`~grasp_agents.sandbox.environment.ExecutionEnvironment` may touch.

A :class:`SandboxPolicy` is consumed by *both* surfaces of an environment.
The file tools enforce it on every I/O call; the exec backend generates its
OS-level confinement (a macOS Seatbelt profile, ``bwrap`` arguments, or
container mounts) from the same fields. One allowlist, one deny list, one
network rule — never two that can drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class NetworkPolicy(StrEnum):
    """Egress policy for an environment's exec surface."""

    NONE = "none"  # no egress (default; right for code-illustration / notebooks)
    LOOPBACK = "loopback"  # only localhost — e.g. a host-side filtering proxy
    ALLOWLIST = "allowlist"  # only SandboxPolicy.allowed_domains, via a proxy
    ALL = "all"  # unrestricted


@dataclass(frozen=True)
class SandboxPolicy:
    """
    The shared address-space + network policy for an environment.

    ``allowed_roots`` is the read+write address space; ``readonly_roots`` adds
    read-only locations. ``include_dotfile_denylist`` stacks the credential
    deny list (``.ssh`` / ``.aws`` / ``.env`` / ...) on top; ``dotfile_overrides``
    are explicit opt-ins that bypass it for this environment. ``network`` +
    ``allowed_domains`` govern the exec surface's egress; ``env`` is the base
    environment exposed to subprocesses.
    """

    allowed_roots: tuple[Path, ...]
    readonly_roots: tuple[Path, ...] = ()
    include_dotfile_denylist: bool = True
    dotfile_overrides: frozenset[Path] = field(default_factory=frozenset["Path"])
    network: NetworkPolicy = NetworkPolicy.NONE
    allowed_domains: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict[str, str])


__all__ = ["NetworkPolicy", "SandboxPolicy"]
