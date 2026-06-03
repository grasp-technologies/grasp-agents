"""
Tests for backend #2 (macOS Seatbelt): the SBPL emitter + ``SeatbeltExecBackend``
+ factory wiring.

Three tiers:

* **Emitter** (runs anywhere) — assert the generated profile + ``-D`` defines
  have the right rules, and that writable roots are passed as params, never
  string-interpolated.
* **Compile-validation** (``skipif`` not macOS / no ``sandbox-exec``) — actually
  feed the generated profile to ``sandbox-exec``. A *valid* profile fails only
  at ``sandbox_apply`` (blocked when already sandboxed); a malformed one fails
  earlier at parse. This runs in-session and catches real SBPL bugs.
* **Live confinement** (``skipif`` ``sandbox-exec`` can't apply — i.e. nested /
  non-macOS) — prove writes are actually confined. Skips inside the agent
  sandbox; runs on a real macOS host.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from grasp_agents.sandbox import (
    NetworkPolicy,
    SandboxPolicy,
    SeatbeltExecBackend,
    build_seatbelt_profile,
    local_environment,
    seatbelt_argv,
)

pytestmark = pytest.mark.asyncio

_DARWIN = sys.platform == "darwin"
_HAS_SANDBOX_EXEC = shutil.which("sandbox-exec") is not None
_CAN_COMPILE = _DARWIN and _HAS_SANDBOX_EXEC


def _seatbelt_can_apply() -> bool:
    """True only on a real (non-nested) macOS host where apply succeeds."""
    if not _CAN_COMPILE:
        return False
    proc = subprocess.run(
        ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


_CAN_APPLY = _seatbelt_can_apply()


def _profile(tmp_path: Path, **kw: Any) -> tuple[str, dict[str, str]]:
    policy = SandboxPolicy(allowed_roots=(tmp_path,), **kw)
    return build_seatbelt_profile(policy)


def _assert_compiles(profile: str, defines: dict[str, str]) -> None:
    """Assert the profile parses + resolves params under sandbox-exec."""
    argv = list(seatbelt_argv(profile, defines, ("/usr/bin/true",)))
    proc = subprocess.run(argv, capture_output=True, text=True, check=False)
    err = proc.stderr
    for bad in ("syntax error", "unbound variable", "invalid data type"):
        msg = f"profile did not compile ({bad}):\n{err}\n---\n{profile}"
        assert bad not in err, msg
    # Either it applied cleanly (real host) or only the apply step was blocked.
    assert proc.returncode == 0 or "sandbox_apply" in err, err


# --- emitter -----------------------------------------------------------------


async def test_profile_core_structure(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path)
    assert profile.startswith("(version 1)\n(deny default)\n")
    assert '(import "system.sb")' in profile
    assert "(allow file-read*)" in profile
    assert "(allow process-exec*)" in profile


async def test_writable_root_is_a_param_not_interpolated(tmp_path: Path) -> None:
    # A root with a space + quote must NOT appear literally in the profile —
    # it goes through a -D param, defeating SBPL injection.
    tricky = tmp_path / 'we ird"dir'
    tricky.mkdir()
    profile, defines = build_seatbelt_profile(
        SandboxPolicy(allowed_roots=(tricky,))
    )
    assert '(subpath (param "WS_0"))' in profile
    assert str(tricky) not in profile
    assert defines["WS_0"] == str(tricky.resolve())


async def test_multiple_roots_get_indexed_params(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    profile, defines = build_seatbelt_profile(SandboxPolicy(allowed_roots=(a, b)))
    assert '(param "WS_0")' in profile
    assert '(param "WS_1")' in profile
    assert set(defines) == {"WS_0", "WS_1"}


async def test_network_none_denies(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path, network=NetworkPolicy.NONE)
    assert "(deny network*)" in profile
    assert "(allow network*)" not in profile


async def test_network_all_allows(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path, network=NetworkPolicy.ALL)
    assert "(allow network*)" in profile


async def test_network_loopback_unsupported(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="proxy"):
        _profile(tmp_path, network=NetworkPolicy.LOOPBACK)


async def test_credential_and_system_denies_present(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path)
    # credential dirs (alternation) + dotfile names + system paths + docker sock
    assert r"/\.(aws|docker|gnupg|kube|ssh)(/|$)" in profile
    assert r"/\.env($|\.)" in profile
    assert '(subpath "/etc")' in profile
    assert '(literal "/var/run/docker.sock")' in profile


async def test_dotfile_denylist_toggle(tmp_path: Path) -> None:
    on, _ = _profile(tmp_path, include_dotfile_denylist=True)
    off, _ = _profile(tmp_path, include_dotfile_denylist=False)
    assert "(aws|docker|gnupg|kube|ssh)" in on
    assert "(aws|docker|gnupg|kube|ssh)" not in off
    # system write-denies stay regardless of the dotfile toggle
    assert '(subpath "/etc")' in off


async def test_denies_come_after_allows(tmp_path: Path) -> None:
    # last-match-wins: credential/system denies must follow the broad allows
    profile, _ = _profile(tmp_path)
    assert profile.index("(allow file-write*") < profile.index("(deny file-write*")
    assert profile.index("(allow file-read*)") < profile.index(
        "(deny file-read* file-write*"
    )


async def test_user_carveouts_in_profile(tmp_path: Path) -> None:
    profile, defines = build_seatbelt_profile(
        SandboxPolicy(
            allowed_roots=(tmp_path,),
            deny_read=(tmp_path / "secret",),
            allow_read=(tmp_path / "secret" / "shared",),
            deny_write=(tmp_path / "protected",),
        )
    )
    assert '(deny file-read* (subpath (param "DR_0")))' in profile
    assert '(allow file-read* (subpath (param "AR_0")))' in profile
    assert '(deny file-write* (subpath (param "DW_0")))' in profile
    assert defines["DR_0"] == str((tmp_path / "secret").resolve())
    # deny_read before allow_read (allow wins); allow_read before the credential
    # deny (so allow_read can never re-expose a credential path).
    assert profile.index('(param "DR_0")') < profile.index('(param "AR_0")')
    assert profile.index('(param "AR_0")') < profile.index(
        "(deny file-read* file-write*"
    )


async def test_seatbelt_argv_shape(tmp_path: Path) -> None:
    profile, defines = _profile(tmp_path)
    argv = seatbelt_argv(profile, defines, ("/bin/sh", "-c", "echo hi"))
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[1] == "-p"
    assert argv[2] == profile
    assert "-D" in argv
    assert argv[-3:] == ("/bin/sh", "-c", "echo hi")


# --- compile-validation (runs in-session on macOS) --------------------------


@pytest.mark.skipif(not _CAN_COMPILE, reason="needs macOS + sandbox-exec")
async def test_generated_profiles_compile(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for policy in (
        SandboxPolicy(allowed_roots=(tmp_path,)),
        SandboxPolicy(allowed_roots=(a, b), network=NetworkPolicy.ALL),
        SandboxPolicy(allowed_roots=(tmp_path,), include_dotfile_denylist=False),
        SandboxPolicy(allowed_roots=(tmp_path / "sp ace",)),
        SandboxPolicy(
            allowed_roots=(tmp_path,),
            deny_read=(tmp_path / "secret",),
            allow_read=(tmp_path / "secret" / "shared",),
            deny_write=(tmp_path / "protected",),
        ),
    ):
        profile, defines = build_seatbelt_profile(policy)
        _assert_compiles(profile, defines)


# --- factory wiring ----------------------------------------------------------


async def test_factory_bwrap_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="bwrap"):
        local_environment(allowed_roots=[tmp_path], confinement="bwrap")


@pytest.mark.skipif(not _CAN_COMPILE, reason="needs macOS + sandbox-exec")
async def test_factory_seatbelt_backend(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    assert env.exec_backend is not None
    assert env.exec_backend.name == "seatbelt"
    assert isinstance(env.exec_backend, SeatbeltExecBackend)


@pytest.mark.skipif(not _DARWIN, reason="auto resolves to seatbelt only on macOS")
async def test_factory_auto_is_seatbelt_on_macos(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="auto")
    assert env.exec_backend is not None
    assert env.exec_backend.name == "seatbelt"


# --- live confinement (skips in-session; runs on a real macOS host) ---------


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_confines_writes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    env = local_environment(allowed_roots=[work], confinement="seatbelt")
    backend = env.exec_backend
    assert backend is not None

    inside = await backend.execute("echo hi > in.txt", cwd=work)
    assert inside.returncode == 0
    assert (work / "in.txt").exists()

    blocked = await backend.execute(f"echo no > {outside / 'out.txt'}", cwd=work)
    assert blocked.returncode != 0
    assert not (outside / "out.txt").exists()
