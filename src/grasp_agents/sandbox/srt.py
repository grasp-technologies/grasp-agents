"""
``SrtExecBackend`` — delegate confinement to Anthropic's ``sandbox-runtime``
(`srt`) CLI when it is installed.

`srt` is a production-grade OS sandbox (macOS Seatbelt / Linux bwrap + seccomp)
with a host-side HTTP/SOCKS proxy for **domain-allowlisted network egress** —
the one capability our native :class:`SeatbeltExecBackend` deliberately does not
reproduce. Rather than reimplement srt's ~20k LOC, this backend shells out to
its CLI exactly the way the local backend shells out to ``sandbox-exec``: it
maps our :class:`SandboxPolicy` to an srt settings file and runs
``srt --settings <file> -c <command>`` (srt's ``-c`` mode runs the string via
``bash -c``).

**Boundary doc (the three questions):**

1. *What can this backend contain?* Whatever `srt` enforces: writes confined to
   the policy's roots, srt's automatic credential/system deny set, curated
   mach/sysctl/IOKit allowlists, move/symlink-bypass hardening, and —
   uniquely — **network egress restricted to an allowlist** (``network=ALLOWLIST``)
   via srt's proxy. Stronger than our native Seatbelt profile.
2. *What is outside the boundary / trusted?* srt's own documented limits
   (domain-fronting can bypass the allowlist; unix-socket grants can escalate;
   the Node runtime + the `srt` binary are trusted host code). Requires `srt`
   on ``PATH`` (a Node CLI) — an opt-in, like Docker.
3. *What did we configure and why?* A settings file generated from the shared
   :class:`SandboxPolicy`: ``filesystem.allowWrite`` = the policy roots,
   ``network.allowedDomains`` = the policy's allowlist. srt applies its own
   mandatory credential/system denies on top.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

from .local_exec import LocalExecBackend
from .policy import NetworkPolicy
from .supervisor import ExecSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .policy import SandboxPolicy
    from .supervisor import ProcessSupervisor


def _resolved(paths: tuple[Path, ...]) -> list[str]:
    return [str(Path(p).expanduser().resolve()) for p in paths]


def build_srt_settings(policy: SandboxPolicy) -> dict[str, object]:
    """
    Map a :class:`SandboxPolicy` to an `srt` settings dict.

    ``allowed_roots`` become ``filesystem.allowWrite`` (realpath'd); reads are
    left broad (srt allows reads by default and applies its own credential
    denies). Network: ``NONE`` → empty allowlist (srt = no egress); ``ALLOWLIST``
    → the policy's domains. ``ALL`` / ``LOOPBACK`` are not expressible in srt's
    allowlist model and raise.
    """
    if policy.network in {NetworkPolicy.ALL, NetworkPolicy.LOOPBACK}:
        raise NotImplementedError(
            f"network={policy.network.value!r} is not expressible for the srt "
            "backend (srt is allowlist-oriented). Use NetworkPolicy.NONE, "
            "NetworkPolicy.ALLOWLIST with allowed_domains, or confinement='none' "
            "for unrestricted egress."
        )
    allowed_domains = (
        list(policy.allowed_domains)
        if policy.network == NetworkPolicy.ALLOWLIST
        else []
    )
    return {
        "network": {
            "allowedDomains": allowed_domains,
            "deniedDomains": list(policy.denied_domains),
        },
        "filesystem": {
            "denyRead": _resolved(policy.deny_read),
            "allowRead": _resolved(policy.allow_read),
            "allowWrite": _resolved(policy.allowed_roots),
            "denyWrite": _resolved(policy.deny_write),
        },
    }


def srt_argv(srt_path: str, settings_path: str, command: str) -> tuple[str, ...]:
    """Build the ``srt`` argv. ``-c`` runs ``command`` via ``bash -c``, unescaped."""
    return (srt_path, "--settings", settings_path, "-c", command)


def _unlink_quietly(path: str) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


class SrtExecBackend(LocalExecBackend):
    """
    :class:`LocalExecBackend` that runs each command under the ``srt`` CLI.

    Writes the policy-derived settings to a temp file once (cleaned up when the
    backend is garbage-collected) and wraps every command in ``srt --settings
    <file> -c <command>``. Requires ``srt`` on ``PATH``; the caller (the factory)
    verifies this.
    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        supervisor: ProcessSupervisor | None = None,
        inherit_host_env: bool = True,
        srt_path: str | None = None,
    ) -> None:
        super().__init__(
            policy=policy,
            supervisor=supervisor,
            name="srt",
            inherit_host_env=inherit_host_env,
        )
        resolved = srt_path or shutil.which("srt")
        if resolved is None:
            raise RuntimeError(
                "srt was not found on PATH; install Anthropic's sandbox-runtime "
                "CLI to use confinement='srt'."
            )
        self._srt = resolved

        fd, path = tempfile.mkstemp(prefix="srt-settings-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(build_srt_settings(policy), handle)
        self._settings_path = path
        weakref.finalize(self, _unlink_quietly, path)

    @property
    def settings_path(self) -> str:
        return self._settings_path

    def _build_spec(
        self, command: str, *, cwd: Path | None, env: Mapping[str, str] | None
    ) -> ExecSpec:
        return ExecSpec(
            argv=srt_argv(self._srt, self._settings_path, command),
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
        )

    def _session_argv(self) -> tuple[str, ...]:
        # `-c /bin/sh` runs a persistent shell (reading stdin) under srt.
        return srt_argv(self._srt, self._settings_path, "/bin/sh")


__all__ = ["SrtExecBackend", "build_srt_settings", "srt_argv"]
