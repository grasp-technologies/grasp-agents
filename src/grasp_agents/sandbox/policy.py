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


# Inherited host env vars matching these ``fnmatch`` patterns
# (case-insensitively — lowercase secret vars must not slip through) are
# scrubbed from a command's environment by default, so an unconfined
# subprocess (``confinement="none"``) the model runs cannot read the host's
# credentials — API keys, tokens, cloud secrets. Security-first and deliberately
# broad; pass ``env_scrub=()`` to disable or an explicit list to override.
# Explicitly-provided ``env`` (and ``policy.env``) is never scrubbed.
DEFAULT_ENV_SCRUB: tuple[str, ...] = (
    "*_API_KEY",
    "*_APIKEY",
    "*_ACCESS_KEY",
    "*_ACCESS_KEY_ID",
    "*_SECRET",
    "*_SECRET_KEY",
    "*SECRET*",
    "*_TOKEN",
    "*TOKEN",
    "*_PASSWORD",
    "*PASSWORD*",
    "*_PRIVATE_KEY",
    "*_CREDENTIALS",
    "*_CREDS",
    "*_KEY",
    "DATABASE_URL",
    "REDIS_URL",
    "MONGODB_URI",
    "SSH_AUTH_SOCK",
    "KUBECONFIG",
    "DOCKER_AUTH_CONFIG",
    "SENTRY_DSN",
    "SLACK_WEBHOOK_URL",
    "OP_SESSION_*",
)


@dataclass(frozen=True)
class SandboxPolicy:
    """
    The shared address-space + network policy for an environment.

    Consumed by **both** enforcement planes: the file tools enforce it in
    ``FileBackend.validate_path`` (host-process I/O), and the exec backend
    generates its OS confinement (Seatbelt profile / srt settings) from the
    same fields (subprocess I/O). A path/domain is confined on a given surface
    only if that surface's plane enforces it.

    Filesystem:

    - ``allowed_roots`` — the read+write workspace (an allowlist). This **is**
      the write allowlist (srt's ``allowWrite``); there is no separate
      ``allow_write`` field.
    - ``readonly_roots`` — readable but not writable additions.
    - ``deny_read`` / ``allow_read`` — carve unreadable regions out of the
      readable space; ``allow_read`` re-allows within them (**allow wins**).
    - ``deny_write`` — carve write-protected regions out of ``allowed_roots``
      (**deny wins**). There is no write carve-back: ``deny_write`` always
      wins, so protect a sub-region by denying it rather than re-allowing
      within a denied one.
    - ``include_dotfile_denylist`` stacks the credential deny set (``.ssh`` /
      ``.aws`` / ``.env`` / ...) on top; ``dotfile_overrides`` are explicit
      opt-ins that bypass it.

    The allow/deny precedence is **reversed for reads vs writes**:
    ``allow_read`` overrides ``deny_read`` (carve readable regions back within
    denied areas), while ``deny_write`` overrides ``allowed_roots`` (carve
    protected regions within the writable workspace). Reads also cannot be
    positively allowlisted at the OS plane (dev tooling needs broad reads), so
    under Seatbelt a subprocess sees "everything readable minus
    ``deny_read``", while the file tools stay confined to ``allowed_roots`` —
    same policy, stricter on the tool plane.

    Network / env (exec surface only):

    - ``network`` + ``allowed_domains`` + ``denied_domains`` govern egress;
      ``denied_domains`` is checked first (**denied wins**). Per-domain rules
      are enforced only by proxy-backed backends (``srt`` / hosted); Seatbelt
      egress is all-or-nothing.
    - ``env`` is the base environment set for subprocesses; ``env_scrub`` is a
      list of ``fnmatch`` patterns (e.g. ``"*_API_KEY"``) removed from the
      *inherited* host environment so secrets do not leak into commands.

    Kernel setup code (exec surface only):

    - ``kernel_setup_code`` is a snippet (e.g. ``%matplotlib inline``) run in a
      RunPython kernel at startup to configure the Python environment.
    """

    allowed_roots: tuple[Path, ...]
    readonly_roots: tuple[Path, ...] = ()
    deny_read: tuple[Path, ...] = ()
    allow_read: tuple[Path, ...] = ()
    deny_write: tuple[Path, ...] = ()
    include_dotfile_denylist: bool = True
    dotfile_overrides: frozenset[Path] = field(default_factory=frozenset["Path"])
    network: NetworkPolicy = NetworkPolicy.NONE
    allowed_domains: tuple[str, ...] = ()
    denied_domains: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    env_scrub: tuple[str, ...] = DEFAULT_ENV_SCRUB
    kernel_setup_code: str = ""


__all__ = ["NetworkPolicy", "SandboxPolicy"]
