"""
Tests for the `srt` (Anthropic sandbox-runtime) exec backend.

`srt` is an external Node CLI and is usually absent in CI, so — as with
Seatbelt — the deterministic surface (policy→settings mapping, argv shape,
settings-file writing) is unit-tested in-session, while anything that actually
spawns `srt` is `skipif`-guarded for a host that has it installed.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from grasp_agents.run_context import RunContext
from grasp_agents.sandbox import (
    NetworkPolicy,
    SandboxPolicy,
    SrtExecBackend,
    build_srt_settings,
    local_environment,
    srt_argv,
)

from ._bg_harness import background, kill, make_stack, marker_size, poll_until_done

pytestmark = pytest.mark.asyncio

_HAS_SRT = shutil.which("srt") is not None


def _srt_can_run() -> bool:
    """
    True only where `srt` can actually apply confinement and run a command —
    i.e. an unconfined host, not a nested sandbox (where its Seatbelt/bwrap
    layer can't apply). Gates the live-exec tests, mirroring seatbelt's probe.
    """
    srt = shutil.which("srt")
    if srt is None:
        return False
    with tempfile.TemporaryDirectory() as d:
        settings = build_srt_settings(SandboxPolicy(allowed_roots=(Path(d),)))
        settings_file = Path(d) / "s.json"
        settings_file.write_text(json.dumps(settings))
        argv = list(srt_argv(srt, str(settings_file), "true"))
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=30, check=False)
        except Exception:
            return False
        return proc.returncode == 0


_SRT_CAN_RUN = _srt_can_run()


# --- settings mapping --------------------------------------------------------


async def test_settings_network_none(tmp_path: Path) -> None:
    settings = build_srt_settings(SandboxPolicy(allowed_roots=(tmp_path,)))
    assert settings["network"] == {"allowedDomains": [], "deniedDomains": []}


async def test_settings_allowlist_domains(tmp_path: Path) -> None:
    settings = build_srt_settings(
        SandboxPolicy(
            allowed_roots=(tmp_path,),
            network=NetworkPolicy.ALLOWLIST,
            allowed_domains=("example.com", "*.github.com"),
        )
    )
    assert settings["network"] == {
        "allowedDomains": ["example.com", "*.github.com"],
        "deniedDomains": [],
    }


async def test_settings_allowwrite_is_resolved_roots(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    settings = build_srt_settings(SandboxPolicy(allowed_roots=(a, b)))
    assert settings["filesystem"] == {
        "denyRead": [],
        "allowRead": [],
        "allowWrite": [str(a.resolve()), str(b.resolve())],
        "denyWrite": [],
    }


async def test_settings_carveouts_and_denied_domains(tmp_path: Path) -> None:
    secret, shared, protected = (
        tmp_path / "secret",
        tmp_path / "secret" / "shared",
        tmp_path / "protected",
    )
    settings = build_srt_settings(
        SandboxPolicy(
            allowed_roots=(tmp_path,),
            deny_read=(secret,),
            allow_read=(shared,),
            deny_write=(protected,),
            network=NetworkPolicy.ALLOWLIST,
            allowed_domains=("api.example.com",),
            denied_domains=("evil.example.com",),
        )
    )
    assert settings["filesystem"] == {
        "denyRead": [str(secret.resolve())],
        "allowRead": [str(shared.resolve())],
        "allowWrite": [str(tmp_path.resolve())],
        "denyWrite": [str(protected.resolve())],
    }
    assert settings["network"] == {
        "allowedDomains": ["api.example.com"],
        "deniedDomains": ["evil.example.com"],
    }


async def test_settings_network_all_unsupported(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="allowlist-oriented"):
        build_srt_settings(
            SandboxPolicy(allowed_roots=(tmp_path,), network=NetworkPolicy.ALL)
        )


async def test_settings_network_loopback_unsupported(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        build_srt_settings(
            SandboxPolicy(allowed_roots=(tmp_path,), network=NetworkPolicy.LOOPBACK)
        )


async def test_srt_argv_shape() -> None:
    argv = srt_argv("/usr/local/bin/srt", "/tmp/s.json", "echo hi")
    assert argv == ("/usr/local/bin/srt", "--settings", "/tmp/s.json", "-c", "echo hi")


# --- backend (constructs without a real srt via srt_path override) ----------


async def test_backend_writes_settings_file(tmp_path: Path) -> None:
    policy = SandboxPolicy(allowed_roots=(tmp_path,))
    # srt_path points at a real binary so __init__ doesn't reject it; we only
    # exercise settings-file writing here, not execution.
    backend = SrtExecBackend(policy=policy, srt_path="/usr/bin/true")
    assert backend.name == "srt"
    written = json.loads(Path(backend.settings_path).read_text())
    assert written == build_srt_settings(policy)


# --- factory wiring ----------------------------------------------------------


@pytest.mark.skipif(_HAS_SRT, reason="srt is installed; the absent-path is tested")
async def test_factory_srt_requires_cli(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="srt CLI"):
        local_environment(allowed_roots=[tmp_path], confinement="srt")


@pytest.mark.skipif(not _HAS_SRT, reason="needs the srt CLI installed")
async def test_factory_srt_backend(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="srt")
    assert env.exec_backend is not None
    assert env.exec_backend.name == "srt"
    assert isinstance(env.exec_backend, SrtExecBackend)


# --- live backgrounding (manager + Bash + TaskOutput / KillTask) -------------
#
# Same flow as the seatbelt / e2b variants, here under Anthropic's srt CLI: a
# long Bash command outlives its deadline → the manager sidelines it → poll
# incremental output via TaskOutput → it completes / is killed (the srt child
# is a local process group the shared supervisor kills via killpg).


@pytest.mark.skipif(not _SRT_CAN_RUN, reason="srt cannot apply/run here")
async def test_srt_background_poll_and_complete(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="srt")
    ctx: RunContext[None] = RunContext(environment=env)
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


@pytest.mark.skipif(not _SRT_CAN_RUN, reason="srt cannot apply/run here")
async def test_srt_background_kill_terminates(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path], confinement="srt")
    ctx: RunContext[None] = RunContext(environment=env)
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
    assert killed.status == "completed"

    size_a = await marker_size(env, str(marker))
    await asyncio.sleep(0.6)
    size_b = await marker_size(env, str(marker))
    assert size_a == size_b

    with pytest.raises(ValueError, match="Unknown background task id"):
        await kill(mgr, task_id)
