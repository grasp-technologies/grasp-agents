"""
macOS Seatbelt confinement: a pure-Python SBPL emitter + ``SeatbeltExecBackend``.

The backend wraps the unconfined :class:`LocalExecBackend` argv in
``sandbox-exec -p <profile> -D WS_i=<root> ... /bin/sh -c <command>``. The
profile is generated from the shared :class:`SandboxPolicy` — the *same* policy
the file tools enforce — so the credential/system deny set in
:mod:`..tools.file_backend.paths` becomes the profile's mandatory denies (one
source, two enforcement points). No Node / ``srt`` runtime dependency.

**Boundary doc (the three questions every backend answers):**

1. *What can this backend contain?* **Writes** — denied everywhere except the
   policy's ``allowed_roots`` (passed as ``-D`` params, never interpolated),
   scratch temp, and standard write-devices. Spawned child processes inherit
   the same confinement (no escape via ``fork``/``exec``). Credential paths
   (``.ssh`` / ``.aws`` / ...) and sensitive system paths (``/etc`` / ...) are
   explicitly denied. **Network** is denied unless ``policy.network`` is
   ``ALL``.
2. *What is outside the boundary / trusted to the host?* **Reads** — Seatbelt
   cannot meaningfully confine reads (Apple/Codex), so dev tooling gets broad
   read access and read-scoping stays a *tool-layer* concern (the file tools'
   ``allowed_roots`` on ``Read``). Credential paths are still read-denied as
   defense-in-depth, but general read isolation is *not* claimed. Also trusted:
   raw ``socket()`` and Mach IPC have documented kernel holes every
   ``sandbox-exec`` consumer lives with; the deprecation warning it prints is
   cosmetic.
3. *What did we configure and why?* ``(deny default)`` + ``(import
   "system.sb")`` (so ordinary tools run) + broad reads + write confinement to
   the policy roots, generated from one ``SandboxPolicy``. macOS only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from ...tools.file_backend.paths import sensitive_path_rules
from ..policy import NetworkPolicy
from .exec import LocalExecBackend
from .supervisor import ExecSpec

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..policy import SandboxPolicy
    from .supervisor import ProcessSupervisor

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Scratch space real tools expect to write to (compilers, package managers).
# Narrower than the system deny set: /private/var/folders is the per-user
# TMPDIR, deliberately NOT in the paths.py system deny list.
_TMP_WRITABLE_SUBPATHS: tuple[str, ...] = (
    "/tmp",
    "/private/tmp",
    "/private/var/folders",
)

# Standard write-only devices; ``(deny default)`` blocks these otherwise.
_WRITABLE_DEVICES: tuple[str, ...] = (
    "/dev/null",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/dtracehelper",
    "/dev/tty",
)


def _quote(path: str) -> str:
    """Quote a trusted path constant as an SBPL string literal."""
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_seatbelt_profile(policy: SandboxPolicy) -> tuple[str, dict[str, str]]:
    """
    Generate an SBPL profile + its ``-D`` defines from a :class:`SandboxPolicy`.

    Returns ``(profile_text, defines)`` where ``defines`` maps ``-D`` parameter
    names to realpath'd writable roots. Rules are ordered so the mandatory
    denies come *after* the broad allows (SBPL is last-match-wins). Writable
    roots are passed as params — never string-interpolated into the profile —
    so a path containing SBPL metacharacters cannot break out.
    """
    rules = sensitive_path_rules()
    defines: dict[str, str] = {}

    lines: list[str] = [
        "(version 1)",
        "(deny default)",
        '(import "system.sb")',
        # Baseline needed for ordinary tools to run under (deny default).
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        # POSIX shared memory + semaphores — Python's multiprocessing needs
        # these; (deny default) blocks them otherwise.
        "(allow ipc-posix-shm)",
        "(allow ipc-posix-sem)",
        # Reads broad (see the boundary doc): not OS-confinable; tools need them.
        "(allow file-read*)",
    ]

    write_filters: list[str] = []
    for i, root in enumerate(policy.allowed_roots):
        name = f"WS_{i}"
        defines[name] = str(Path(root).expanduser().resolve())
        write_filters.append(f'(subpath (param "{name}"))')
    write_filters += [f"(subpath {_quote(p)})" for p in _TMP_WRITABLE_SUBPATHS]
    device_filters = "\n    ".join(f"(literal {_quote(d)})" for d in _WRITABLE_DEVICES)
    lines.extend(
        (
            "(allow file-write*\n    " + "\n    ".join(write_filters) + ")",
            "(allow file-write-data\n    " + device_filters + ")",
        )
    )

    # User read/write carve-outs — paths passed as -D params (never
    # interpolated). Ordered so allow_read overrides deny_read; the mandatory
    # system + credential denies below come AFTER these and override them, so
    # allow_read can never re-expose a credential path.
    for i, path in enumerate(policy.deny_read):
        name = f"DR_{i}"
        defines[name] = str(Path(path).expanduser().resolve())
        lines.append(f'(deny file-read* (subpath (param "{name}")))')
    for i, path in enumerate(policy.allow_read):
        name = f"AR_{i}"
        defines[name] = str(Path(path).expanduser().resolve())
        lines.append(f'(allow file-read* (subpath (param "{name}")))')
    for i, path in enumerate(policy.deny_write):
        name = f"DW_{i}"
        defines[name] = str(Path(path).expanduser().resolve())
        lines.append(f'(deny file-write* (subpath (param "{name}")))')

    # Mandatory denies — AFTER every allow (incl. user allow_read) so they win.
    system_denies = [
        f"(subpath {_quote(p.rstrip('/'))})" for p in rules.system_write_prefixes
    ]
    system_denies += [f"(literal {_quote(p)})" for p in rules.system_exact]
    if system_denies:
        lines.append("(deny file-write*\n    " + "\n    ".join(system_denies) + ")")

    if policy.include_dotfile_denylist:
        cred_filters: list[str] = []
        if rules.credential_dir_names:
            alt = "|".join(
                re.escape(n.removeprefix(".")) for n in rules.credential_dir_names
            )
            cred_filters.append(f'(regex #"/\\.({alt})(/|$)")')
        cred_filters += [
            f'(regex #"/{re.escape(nm)}($|\\.)")' for nm in rules.credential_file_names
        ]
        if cred_filters:
            lines.append(
                "(deny file-read* file-write*\n    " + "\n    ".join(cred_filters) + ")"
            )

    if policy.network == NetworkPolicy.ALL:
        lines.append("(allow network*)")
    elif policy.network == NetworkPolicy.NONE:
        lines.append("(deny network*)")
    else:
        raise NotImplementedError(
            f"network={policy.network.value!r} is not enforceable under Seatbelt "
            "(loopback / allowlist need a host-side proxy — deferred). Use "
            "NetworkPolicy.NONE or NetworkPolicy.ALL."
        )

    return "\n".join(lines) + "\n", defines


def seatbelt_argv(
    profile: str, defines: Mapping[str, str], inner_argv: Sequence[str]
) -> tuple[str, ...]:
    """Build the ``sandbox-exec`` argv wrapping ``inner_argv``."""
    argv: list[str] = [_SANDBOX_EXEC, "-p", profile]
    for key, value in defines.items():
        argv += ["-D", f"{key}={value}"]
    argv.extend(inner_argv)
    return tuple(argv)


class SeatbeltExecBackend(LocalExecBackend):
    """
    :class:`LocalExecBackend` whose commands run under a macOS Seatbelt profile.

    Generates a fresh profile per command from the shared policy and wraps the
    ``/bin/sh -c`` invocation in ``sandbox-exec``. macOS only — the caller (the
    :func:`local_environment` factory) verifies the platform and that
    ``sandbox-exec`` is present.
    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        supervisor: ProcessSupervisor | None = None,
        inherit_host_env: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            supervisor=supervisor,
            name="seatbelt",
            inherit_host_env=inherit_host_env,
        )

    def _build_spec(
        self, command: str, *, cwd: Path | None, env: Mapping[str, str] | None
    ) -> ExecSpec:
        profile, defines = build_seatbelt_profile(self._policy)
        inner = ("/bin/sh", "-c", command)
        return ExecSpec(
            argv=seatbelt_argv(profile, defines, inner),
            cwd=self._resolve_cwd(cwd),
            env=self._merged_env(env),
            backend=self._name,
        )

    def _session_argv(self) -> tuple[str, ...]:
        profile, defines = build_seatbelt_profile(self._policy)
        return seatbelt_argv(profile, defines, ("/bin/sh",))

    def _kernel_launch_argv(self, connection_file: str) -> tuple[str, ...]:
        profile, defines = build_seatbelt_profile(self._policy)
        inner = (sys.executable, "-m", "ipykernel_launcher", "-f", connection_file)
        return seatbelt_argv(profile, defines, inner)


__all__ = ["SeatbeltExecBackend", "build_seatbelt_profile", "seatbelt_argv"]
