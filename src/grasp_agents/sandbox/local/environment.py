"""
``LocalEnvironment`` + the :func:`local_environment` factory — a co-located
host filesystem + host subprocess pair sharing one :class:`SandboxPolicy`.

The factory is the co-location guarantee: it unpacks a single policy into a
:class:`~grasp_agents.file_backend.local.LocalFileBackend` and an
exec backend that address the *same* host filesystem, so a host cannot
accidentally point the two surfaces at different locations. Wire the result onto
:attr:`SessionContext.environment`; the ``SessionContext`` validator sources
``file_backend`` from it, and ``ctx.exec_backend`` is the read-only property off
it (the environment is the sole grantor of an exec surface).

Confinement: ``"none"`` is a plain subprocess with **no** OS isolation (see
:class:`LocalExecBackend`); ``"seatbelt"`` (macOS) wraps every command in a
generated Seatbelt profile confining writes + spawned processes (see
:class:`~grasp_agents.sandbox.seatbelt.SeatbeltExecBackend`); ``"auto"`` selects
Seatbelt on macOS; ``"bwrap"`` (Linux) is not built yet. ``readonly_roots`` are
write-denied on the file-tool plane under every confinement; on the exec plane
Seatbelt and srt OS-deny them (they're excluded from the writable root set),
while ``"none"`` leaves them writable and ``network`` is recorded-not-enforced.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess  # noqa: S404 - trusted interpreter + literal script, no shell
import sys
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.sandbox.environment import ExecutionEnvironment
from grasp_agents.sandbox.policy import DEFAULT_ENV_SCRUB, NetworkPolicy, SandboxPolicy

from .exec import LocalExecBackend, resolve_python
from .kernel import DEFAULT_KERNEL_STARTUP_TIMEOUT
from .supervisor import ProcessSupervisor, ResourceLimits, SupervisorLimits

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from grasp_agents.file_backend.base import FileBackend
    from grasp_agents.sandbox.exec_backend import ExecBackend

logger = logging.getLogger(__name__)


class LocalEnvironment(ExecutionEnvironment):
    """
    An :class:`~grasp_agents.sandbox.environment.ExecutionEnvironment` over the
    host: a :class:`LocalFileBackend` and a :class:`LocalExecBackend` sharing
    one :class:`SandboxPolicy`.

    Prefer the :func:`local_environment` factory, which builds the policy and
    both surfaces together. The async context-manager lifecycle is a no-op
    here (nothing to create or tear down on the host); it exists so callers can
    treat every environment uniformly regardless of backend.
    """

    def __init__(
        self,
        *,
        policy: SandboxPolicy,
        file_backend: FileBackend,
        exec_backend: ExecBackend | None,
    ) -> None:
        self._policy = policy
        self._file_backend = file_backend
        self._exec_backend = exec_backend

    @property
    def policy(self) -> SandboxPolicy:
        return self._policy

    @property
    def file_backend(self) -> FileBackend:
        return self._file_backend

    @property
    def exec_backend(self) -> ExecBackend | None:
        return self._exec_backend

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def local_environment(
    *,
    allowed_roots: Sequence[Path | str],
    readonly_roots: Sequence[Path | str] = (),
    deny_read: Sequence[Path | str] = (),
    allow_read: Sequence[Path | str] = (),
    deny_write: Sequence[Path | str] = (),
    confinement: Literal["none", "seatbelt", "bwrap", "srt", "auto"] = "none",
    network: NetworkPolicy = NetworkPolicy.NONE,
    allowed_domains: Sequence[str] = (),
    denied_domains: Sequence[str] = (),
    include_dotfile_denylist: bool = True,
    env: Mapping[str, str] | None = None,
    env_scrub: Sequence[str] = DEFAULT_ENV_SCRUB,
    inherit_host_env: bool = True,
    python: str | Path | None = None,
    packages: Sequence[str] = (),
    provision: bool = False,
    kernel_setup_code: str = "",
    kernel_startup_timeout: float = DEFAULT_KERNEL_STARTUP_TIMEOUT,
    limits: ResourceLimits | None = None,
    supervisor: ProcessSupervisor | None = None,
) -> LocalEnvironment:
    """
    Build a co-located host filesystem + exec pair sharing one policy.

    Args:
        allowed_roots: Read+write address space. The first entry is the
            default working directory for commands.
        readonly_roots: Additional readable locations. The file tools deny
            writes to them under every confinement; on the exec plane they
            are OS-write-denied under Seatbelt and srt (not in the writable
            root set) but unprotected under ``"none"``.
        deny_read: Carve unreadable regions out of the readable space.
        allow_read: Re-allow reads within ``deny_read`` regions (allow wins).
        deny_write: Carve write-protected regions out of ``allowed_roots``
            (deny wins). Enforced on both planes (file tools + exec).
        confinement: ``"none"`` (plain subprocess, no isolation), ``"seatbelt"``
            (macOS OS confinement), ``"srt"`` (delegates to the ``srt`` CLI if
            installed — adds domain-allowlisted network egress that native
            Seatbelt lacks), or ``"auto"`` (Seatbelt on macOS, else a warned
            fallback to ``"none"``). ``"bwrap"`` (Linux) is not built yet.
        network: Recorded on the policy. Not enforced under ``"none"``; under
            Seatbelt, ``NONE`` and ``ALL`` are enforced (loopback / allowlist
            need a proxy — deferred).
        allowed_domains: Egress allowlist (proxy-backed backends only — srt /
            hosted; Seatbelt egress is all-or-nothing).
        denied_domains: Egress denylist, checked first (proxy-backed only).
        include_dotfile_denylist: Stack the credential-dotfile deny list onto
            the file backend's path policy.
        env: Base environment overlaid onto the (optionally inherited) host
            environment for subprocesses.
        env_scrub: ``fnmatch`` patterns (e.g. ``"*_API_KEY"``) removed from the
            inherited host environment so secrets do not leak into commands.
            Defaults to :data:`DEFAULT_ENV_SCRUB`, a credential denylist —
            critical under ``confinement="none"``, where the subprocess is
            otherwise the model's window onto every host secret. Pass ``()`` to
            disable. Explicitly-provided ``env`` is never scrubbed.
        inherit_host_env: Subprocesses inherit ``os.environ`` as their base so
            ``PATH`` resolves. Set False for a minimal environment.
        python: Interpreter the code-interpreter kernel launches with (and whose
            bin dir leads ``PATH``), defaulting to ``sys.executable``. Point it
            at a venv/conda interpreter (e.g. ``"<venv>/bin/python"``, which must
            have ``ipykernel``) to run experiments in that environment. Under a
            confined backend the interpreter stays readable, but writes (e.g.
            ``pip install``) only land if its tree is within ``allowed_roots``.
        packages: Distribution names required in the ``python`` environment
            (e.g. ``["torch", "datasets"]``; version specifiers / extras are
            allowed and ignored for the check). With ``provision=False``
            (default) they are **verified** present at setup — a missing one
            raises with the ``pip install`` command to run; the framework does
            not mutate your env. With ``provision=True`` the framework installs
            the missing ones (see ``provision``).
        provision: Give the agent its own dedicated, isolated venv. Off by
            default — the framework then never mutates a configured environment
            (``packages`` is only verified). When on, a venv is created at
            ``<first allowed_root>/.venv`` and missing ``packages`` are
            ``pip install``-ed into it (so the agent's installs never touch the
            host / project environment). ``python``, if it points to a real
            interpreter, selects the base to clone from — i.e. the venv's
            Python version — otherwise ``sys.executable`` is used (logged).
            Idempotent: an existing venv at that path is reused. To run against
            an environment you manage yourself instead, leave ``provision``
            off and point ``python`` at it.
        kernel_setup_code: Code run once at code-interpreter (``RunPython``)
            kernel startup, in its own execution — empty by default (the
            framework imposes nothing). Pass e.g. ``"%matplotlib inline"`` to
            opt into the inline plotting backend.
        kernel_startup_timeout: Seconds a ``RunPython`` kernel may take to become
            ready before launch is given up. The default has headroom for a cold
            first launch under confinement (which compiles ``.pyc`` + warms
            caches; a later launch is fast); raise it for a heavier base venv.
        limits: Convenient per-command resource ceilings (CPU-seconds, memory,
            file size) applied via ``setrlimit`` — a shortcut for a
            :class:`ProcessSupervisor` built with these :class:`SupervisorLimits`.
            Mutually exclusive with ``supervisor``. See :class:`ResourceLimits`.
            (Local family only; E2B allocates resources per-VM at template build.)
        supervisor: Shared :class:`ProcessSupervisor` for the exec backend.

    Returns:
        A :class:`LocalEnvironment` exposing ``file_backend`` + ``exec_backend``.

    """
    roots = tuple(Path(r).expanduser() for r in allowed_roots)
    ro = tuple(Path(r).expanduser() for r in readonly_roots)
    dr = tuple(Path(p).expanduser() for p in deny_read)
    ar = tuple(Path(p).expanduser() for p in allow_read)
    dw = tuple(Path(p).expanduser() for p in deny_write)
    policy = SandboxPolicy(
        allowed_roots=roots,
        readonly_roots=ro,
        deny_read=dr,
        allow_read=ar,
        deny_write=dw,
        include_dotfile_denylist=include_dotfile_denylist,
        network=network,
        allowed_domains=tuple(allowed_domains),
        denied_domains=tuple(denied_domains),
        env=env or {},
        env_scrub=tuple(env_scrub),
        kernel_setup_code=kernel_setup_code,
        kernel_startup_timeout=kernel_startup_timeout,
    )

    file_backend = LocalFileBackend(
        allowed_roots=list(roots),
        readonly_roots=list(ro),
        include_dotfiles=include_dotfile_denylist,
        deny_read=list(dr),
        allow_read=list(ar),
        deny_write=list(dw),
    )
    if limits is not None:
        if supervisor is not None:
            raise ValueError(
                "Pass either `limits` (convenient per-command ceilings) or a "
                "fully configured `supervisor`, not both."
            )
        supervisor = ProcessSupervisor(SupervisorLimits(**limits))
    resolved_python: str | Path | None = python
    if provision:
        if not roots:
            raise ValueError("provision=True requires at least one allowed_root")
        resolved_python = _provision_python_env(python, packages, roots[0] / ".venv")
    exec_backend = _build_exec_backend(
        confinement,
        policy=policy,
        supervisor=supervisor,
        inherit_host_env=inherit_host_env,
        python=resolved_python,
    )
    if packages:
        _verify_packages(resolve_python(resolved_python), packages)
    return LocalEnvironment(
        policy=policy, file_backend=file_backend, exec_backend=exec_backend
    )


def _resolve_confinement(confinement: str) -> str:
    """Resolve ``"auto"`` to a concrete backend for the current platform."""
    if confinement != "auto":
        return confinement
    if sys.platform == "darwin":
        return "seatbelt"
    if sys.platform.startswith("linux") and shutil.which("bwrap") is not None:
        return "bwrap"
    warnings.warn(
        "confinement='auto' found no OS sandbox (need macOS for Seatbelt or "
        "bwrap on Linux); falling back to 'none' — this is NOT a sandbox.",
        RuntimeWarning,
        stacklevel=3,
    )
    return "none"


def _build_exec_backend(
    confinement: str,
    *,
    policy: SandboxPolicy,
    supervisor: ProcessSupervisor | None,
    inherit_host_env: bool,
    python: str | Path | None,
) -> ExecBackend:
    resolved = _resolve_confinement(confinement)
    if resolved == "none":
        return LocalExecBackend(
            policy=policy,
            supervisor=supervisor,
            name="local",
            inherit_host_env=inherit_host_env,
            python=python,
        )
    if resolved == "seatbelt":
        if sys.platform != "darwin":
            raise RuntimeError(
                "confinement='seatbelt' requires macOS (it shells out to sandbox-exec)."
            )
        if shutil.which("sandbox-exec") is None:
            raise RuntimeError(
                "sandbox-exec was not found; cannot use confinement='seatbelt'."
            )
        # Fail at construction, not on the first command: Seatbelt has no
        # proxy layer, so per-domain / loopback policies cannot be enforced.
        if policy.network not in {NetworkPolicy.NONE, NetworkPolicy.ALL}:
            raise ValueError(
                f"network={policy.network.value!r} is not enforceable under "
                "Seatbelt (no proxy layer): use NetworkPolicy.NONE or "
                "NetworkPolicy.ALL, or confinement='srt' for a domain "
                "allowlist."
            )
        from .seatbelt import SeatbeltExecBackend  # noqa: PLC0415

        return SeatbeltExecBackend(
            policy=policy,
            supervisor=supervisor,
            inherit_host_env=inherit_host_env,
            python=python,
        )
    if resolved == "srt":
        if shutil.which("srt") is None:
            raise RuntimeError(
                "confinement='srt' requires the srt CLI on PATH "
                "(npm install -g @anthropic-ai/sandbox-runtime). It adds OS "
                "confinement + domain-allowlisted network egress; otherwise use "
                "'seatbelt' or 'none'."
            )
        from .srt import SrtExecBackend  # noqa: PLC0415

        return SrtExecBackend(
            policy=policy,
            supervisor=supervisor,
            inherit_host_env=inherit_host_env,
            python=python,
        )
    if resolved == "bwrap":
        raise NotImplementedError(
            "confinement='bwrap' (Linux) is not implemented yet; it lands after "
            "Seatbelt. Use 'none', or run on macOS with 'seatbelt'."
        )
    raise ValueError(f"unknown confinement: {confinement!r}")


def _dist_name(spec: str) -> str:
    """Distribution name from a requirement spec (strip version / extras / marker)."""
    return re.split(r"[<>=!~;\[ ]", spec.strip(), maxsplit=1)[0].strip()


def _missing_packages(python: str, packages: Sequence[str]) -> list[str]:
    """Distribution names in ``packages`` not installed in the ``python`` env."""
    names = [_dist_name(p) for p in packages if p.strip()]
    if not names:
        return []
    check = (
        "import importlib.metadata as m, sys\n"
        "missing = []\n"
        "for n in sys.argv[1:]:\n"
        "    try:\n"
        "        m.distribution(n)\n"
        "    except Exception:\n"
        "        missing.append(n)\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", check, *names],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"could not verify packages with interpreter {python!r}: {exc}"
        ) from exc
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _verify_packages(python: str, packages: Sequence[str]) -> None:
    """
    Verify each distribution in ``packages`` is installed in the ``python``
    environment; raise with an install hint if any are missing. Does not install
    (this environment is the caller's to manage unless ``provision=True``).
    """
    missing = _missing_packages(python, packages)
    if missing:
        raise ValueError(
            f"the configured Python environment ({python}) is missing required "
            f"packages: {', '.join(missing)}. Install them into that environment, "
            f"e.g. `{python} -m pip install {' '.join(missing)}`."
        )


def _venv_interpreter(venv_dir: Path) -> Path:
    """The interpreter path inside a venv created at ``venv_dir``."""
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _install_missing(python: str, packages: Sequence[str]) -> list[str]:
    """
    ``pip install`` the ``packages`` not already present in ``python``.

    Returns the packages it installed (empty if all were already present).
    """
    missing = _missing_packages(python, packages)
    if not missing:
        return []
    logger.info("provision: installing %s into %s", ", ".join(missing), python)
    try:
        subprocess.run([python, "-m", "pip", "install", *missing], check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            f"failed to install {', '.join(missing)} into {python}: {exc}"
        ) from exc
    return missing


# Import the venv's packages once (best-effort) via a throwaway interpreter run.
# Resolves each distribution's top-level modules from its metadata; ``sys.argv``
# carries the distribution names so no list is interpolated into the source.
_WARMUP_SNIPPET = """\
import importlib, importlib.metadata as md, sys
mods = {"ipykernel"}
for dist in sys.argv[1:]:
    try:
        top = md.distribution(dist).read_text("top_level.txt") or ""
        mods.update(top.split())
    except Exception:
        pass
for name in mods:
    try:
        importlib.import_module(name)
    except Exception:
        pass
"""


def _warmup_venv(python: str, packages: Sequence[str]) -> None:
    """
    Pre-import the venv's packages once so the first ``RunPython`` kernel launch
    and first cell don't pay the cold tax.

    A freshly-provisioned venv's first interpreter exec under a confined backend
    is slow — the OS verifies the new interpreter + freshly-installed native
    extensions, and ``.pyc`` / page cache are cold. Left unpaid, the kernel's
    readiness handshake can race that first exec and the launch fails ("kernel
    died before replying"); and the first heavy ``import`` (numpy/pandas/…) can
    take tens of seconds. One throwaway import here, during setup, warms all of
    it. Best-effort: a warm-up failure never blocks provisioning.
    """
    logger.info("provision: warming up imports in %s", python)
    t0 = time.monotonic()
    try:
        subprocess.run(
            [python, "-c", _WARMUP_SNIPPET, *packages],
            check=False,
            timeout=600,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("provision: import warm-up skipped (%s)", exc)
        return
    logger.info("provision: import warm-up done in %.1fs", time.monotonic() - t0)


def _provision_python_env(
    python: str | Path | None, packages: Sequence[str], venv_dir: Path
) -> str:
    """
    Create the agent's dedicated venv at ``venv_dir`` and return its interpreter.

    ``provision=True`` always gives the agent its OWN isolated venv (so its
    ``pip install``s never touch the host / project environment). ``python``,
    if it points to a real interpreter, only selects the base to clone from
    (i.e. the venv's Python version); otherwise ``sys.executable`` is used (and
    the unusable value is logged). Missing ``packages`` are then installed into
    the venv. Idempotent: an existing venv at ``venv_dir`` is reused.
    """
    interpreter = _venv_interpreter(venv_dir)
    created = not interpreter.is_file()
    if not created:
        # Reuse an existing venv as-is (e.g. on resume); only top up packages.
        logger.info("provision: reusing the existing venv at %s", venv_dir)
    else:
        base = sys.executable
        if python is not None:
            candidate = Path(python).expanduser()
            if candidate.is_file():
                base = str(candidate)
            else:
                logger.warning(
                    "provision: configured python %r does not exist; using %s "
                    "as the base for the agent's venv.",
                    str(python),
                    sys.executable,
                )
        logger.info(
            "provision: creating the agent's venv at %s from %s", venv_dir, base
        )
        try:
            subprocess.run([base, "-m", "venv", str(venv_dir)], check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError(f"failed to create venv at {venv_dir}: {exc}") from exc
    installed = _install_missing(str(interpreter), packages)
    # Warm imports only when the venv is fresh or just changed (the cold,
    # crash-prone case); a pure reuse (resume) already ran the interpreter in a
    # prior session, so skip the warm-up to keep resumes fast.
    if packages and (created or installed):
        _warmup_venv(str(interpreter), packages)
    return str(interpreter)


__all__ = ["LocalEnvironment", "local_environment"]
