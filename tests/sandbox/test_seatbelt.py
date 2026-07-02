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
  earlier at parse. This catches real SBPL bugs.
* **Live confinement** (``skipif`` ``sandbox-exec`` can't apply — i.e. nested /
  non-macOS) — prove writes / network are actually confined. Caveat: pytest's
  ``tmp_path`` lives under ``/private/var/folders`` — inside the scratch temp
  the profile *deliberately* write-allows — so denied-write probes must target
  a directory outside both the workspace roots and the temp subtree (allow-
  shaped and carve-out rules, by contrast, are testable under ``tmp_path``).
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.sandbox import (
    NetworkPolicy,
    SandboxPolicy,
    SeatbeltExecBackend,
    build_seatbelt_profile,
    local_environment,
    seatbelt_argv,
)
from grasp_agents.sandbox.local import seatbelt as seatbelt_mod
from grasp_agents.session_context import SessionContext

from ._bg_harness import (
    background,
    drain_notes,
    flush,
    kill,
    make_stack,
    marker_size,
    poll_until_done,
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
    profile, defines = build_seatbelt_profile(SandboxPolicy(allowed_roots=(tricky,)))
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
    # credential dirs (alternation, case-insensitive via two-case classes —
    # APFS matches paths case-insensitively) + dotfile names + system paths +
    # docker sock
    alt = "|".join(
        "".join(f"[{c}{c.upper()}]" for c in name)
        for name in ("aws", "docker", "gnupg", "kube", "ssh")
    )
    assert rf"/\.({alt})(/|$)" in profile
    assert r"/\.[eE][nN][vV]($|\.)" in profile
    assert '(subpath "/etc")' in profile
    assert '(literal "/var/run/docker.sock")' in profile


async def test_dotfile_denylist_toggle(tmp_path: Path) -> None:
    on, _ = _profile(tmp_path, include_dotfile_denylist=True)
    off, _ = _profile(tmp_path, include_dotfile_denylist=False)
    assert "[sS][sS][hH]" in on
    assert "[sS][sS][hH]" not in off
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


# --- live confinement (needs a macOS host where sandbox-exec can apply) -----


def _outside_scratch() -> Path | None:
    """
    A user-writable dir genuinely OUTSIDE the profile's always-writable temp.

    ``tmp_path`` can't serve as "outside": it sits under
    ``/private/var/folders``, which the profile write-allows as scratch temp.
    The repo cwd qualifies on dev hosts; returns None if cwd itself is temp.
    """
    cwd = Path.cwd().resolve()
    temp_prefixes = seatbelt_mod._TMP_WRITABLE_SUBPATHS
    if any(str(cwd).startswith(prefix) for prefix in temp_prefixes):
        return None
    return Path(tempfile.mkdtemp(prefix="sb_outside_", dir=cwd))


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_confines_writes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    env = local_environment(allowed_roots=[work], confinement="seatbelt")
    backend = env.exec_backend
    assert backend is not None

    inside = await backend.execute("echo hi > in.txt", cwd=work)
    assert inside.returncode == 0
    assert (work / "in.txt").exists()

    outside = _outside_scratch()
    if outside is None:
        pytest.skip("cwd is inside the always-writable temp subtree")
    try:
        blocked = await backend.execute(f"echo no > {outside / 'out.txt'}", cwd=work)
        assert blocked.returncode != 0
        assert not (outside / "out.txt").exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_deny_write_carveout_live(tmp_path: Path) -> None:
    # deny_write is emitted AFTER the workspace/temp allows (last-match-wins),
    # so it is enforceable even under the always-writable temp subtree.
    work = tmp_path / "work"
    protected = work / "protected"
    protected.mkdir(parents=True)
    env = local_environment(
        allowed_roots=[work], deny_write=[protected], confinement="seatbelt"
    )
    backend = env.exec_backend
    assert backend is not None

    ok = await backend.execute("echo hi > ok.txt", cwd=work)
    assert ok.returncode == 0
    blocked = await backend.execute(f"echo no > {protected / 'x.txt'}", cwd=work)
    assert blocked.returncode != 0
    assert not (protected / "x.txt").exists()


# bind() on 127.0.0.1:0 never fails on an unconfined host but is a network op
# under (deny network*) — a deterministic probe needing no external egress.
_BIND_PROBE = "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0))"


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_network_none_blocks_live(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    backend = env.exec_backend
    assert backend is not None
    res = await backend.execute(f'python3 -c "{_BIND_PROBE}"', cwd=tmp_path)
    assert res.returncode != 0
    assert "PermissionError" in res.stderr or "Operation not permitted" in res.stderr


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_network_all_allows_live(tmp_path: Path) -> None:
    env = local_environment(
        allowed_roots=[tmp_path], network=NetworkPolicy.ALL, confinement="seatbelt"
    )
    backend = env.exec_backend
    assert backend is not None
    res = await backend.execute(f'python3 -c "{_BIND_PROBE}"', cwd=tmp_path)
    assert res.returncode == 0, res.stderr


# --- live backgrounding (manager + Bash + KillTask) --------------------------
#
# The same flow test_bash_polish exercises against an unconfined local
# subprocess, here under a real Seatbelt profile: a long Bash command outlives
# its deadline → the manager sidelines it → it completes (output read off its
# buffered events) / is killed (the process group dies via the shared
# supervisor's killpg, same as the unconfined backend).


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_background_poll_and_complete(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    ctx: SessionContext[None] = SessionContext(environment=env)
    agent_ctx, mgr = make_stack()

    note, task_id = await background(
        mgr, ctx, agent_ctx, "echo early && sleep 2 && echo late", abg=0.3
    )
    assert "moved to the background" in note
    assert task_id is not None

    collected, out = await poll_until_done(mgr, task_id)
    assert out.status == "completed"
    assert out.result is not None
    assert out.result.returncode == 0
    assert "early" in collected
    assert "late" in collected


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_background_kill_terminates(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    ctx: SessionContext[None] = SessionContext(environment=env)
    agent_ctx, mgr = make_stack()

    marker = tmp_path / "ticks.txt"
    _note, task_id = await background(
        mgr,
        ctx,
        agent_ctx,
        f"while true; do echo tick >> {marker}; sleep 0.1; done",
        abg=0.3,
        timeout=60,
    )
    assert task_id is not None

    killed = await kill(mgr, task_id)
    assert killed.status == "cancelled"  # killed mid-run, not finished

    # process group killed under Seatbelt: the marker stops growing
    size_a = await marker_size(env, str(marker))
    await asyncio.sleep(0.6)
    size_b = await marker_size(env, str(marker))
    assert size_a == size_b

    with pytest.raises(ValueError, match="Unknown background task id"):
        await kill(mgr, task_id)


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_background_small_result_inlined(tmp_path: Path) -> None:
    # Under confinement too, a finished command's output is delivered inline in
    # its completion note and the task is dropped.
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    ctx: SessionContext[None] = SessionContext(environment=env)
    agent_ctx, mgr = make_stack()

    _note, task_id = await background(
        mgr, ctx, agent_ctx, "echo hello && sleep 1.5", abg=0.3
    )
    assert task_id is not None

    await mgr.wait_idle()
    notes = await drain_notes(mgr, ctx)
    assert len(notes) == 1
    assert "completed" in notes[0]
    assert "hello" in notes[0]
    assert "omitted" not in notes[0]  # small result inlined whole, not excerpted
    assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_background_large_result_excerpted(tmp_path: Path) -> None:
    # Cap-and-defer under Seatbelt: a large result is excerpted in the note,
    # which points at the task's .grasp log holding the full output.
    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    ctx: SessionContext[None] = SessionContext(environment=env)
    agent_ctx, mgr = make_stack()

    _note, task_id = await background(
        mgr,
        ctx,
        agent_ctx,
        "head -c 5000 /dev/zero | tr '\\0' 'A'; echo END; sleep 1.5",
        abg=0.3,
        max_inline_result_chars=200,
    )
    assert task_id is not None

    await mgr.wait_idle()
    notes = await drain_notes(mgr, ctx)
    assert len(notes) == 1
    assert "completed" in notes[0]
    assert "chars omitted" in notes[0]
    assert "<output_file>" in notes[0]
    assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]

    # The full, untruncated output is in the .grasp log the note points at.
    match = re.search(r"<output_file>(.+?)</output_file>", notes[0], re.DOTALL)
    assert match is not None
    log_text, _ = await env.file_backend.read_text(Path(match.group(1).strip()))
    assert log_text.count("A") == 5000


@pytest.mark.skipif(not _CAN_APPLY, reason="sandbox-exec cannot apply here")
async def test_seatbelt_background_writes_greppable_log(tmp_path: Path) -> None:
    # The greppable .grasp/tasks log is written under confinement too.
    from grasp_agents.durability.checkpoint_store import InMemoryCheckpointStore
    from grasp_agents.durability.store_keys import task_prefix
    from grasp_agents.durability.task_record import TaskRecord

    env = local_environment(allowed_roots=[tmp_path], confinement="seatbelt")
    store = InMemoryCheckpointStore()
    ctx: SessionContext[None] = SessionContext(
        environment=env, checkpoint_store=store, session_key="s1"
    )
    agent_ctx, mgr = make_stack()

    _note, task_id = await background(
        mgr, ctx, agent_ctx, "echo HELLO && sleep 5", abg=0.3
    )
    assert task_id is not None

    await asyncio.sleep(0.3)
    await flush(mgr, ctx)

    keys = await store.list_keys(task_prefix("s1"))
    rec = TaskRecord.model_validate_json((await store.load(keys[0])) or b"{}")
    assert rec.output_path is not None
    assert ".grasp/tasks" in rec.output_path
    content, _ = await env.file_backend.read_text(Path(rec.output_path))
    assert "HELLO" in content

    await kill(mgr, task_id)
    assert mgr._tasks == {}  # pyright: ignore[reportPrivateUsage]


# ---------- seatbelt rejects unenforceable network policies ----------


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="seatbelt requires macOS with sandbox-exec",
)
class TestSeatbeltNetworkValidation:
    @pytest.mark.parametrize(
        "network", [NetworkPolicy.LOOPBACK, NetworkPolicy.ALLOWLIST]
    )
    async def test_unenforceable_network_rejected_at_construction(
        self, tmp_path: Path, network: NetworkPolicy
    ) -> None:
        with pytest.raises(ValueError, match="not enforceable under"):
            local_environment(
                allowed_roots=[tmp_path],
                confinement="seatbelt",
                network=network,
                allowed_domains=["example.com"],
            )

    async def test_all_or_none_accepted(self, tmp_path: Path) -> None:
        for network in (NetworkPolicy.NONE, NetworkPolicy.ALL):
            local_environment(
                allowed_roots=[tmp_path],
                confinement="seatbelt",
                network=network,
            )
