"""
Tests for the `srt` (Anthropic sandbox-runtime) exec backend.

`srt` is an external Node CLI and is usually absent in CI, so — as with
Seatbelt — the deterministic surface (policy→settings mapping, argv shape,
settings-file writing) is unit-tested in-session, while anything that actually
spawns `srt` is `skipif`-guarded for a host that has it installed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from grasp_agents.sandbox import (
    NetworkPolicy,
    SandboxPolicy,
    SrtExecBackend,
    build_srt_settings,
    local_environment,
    srt_argv,
)

pytestmark = pytest.mark.asyncio

_HAS_SRT = shutil.which("srt") is not None


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
