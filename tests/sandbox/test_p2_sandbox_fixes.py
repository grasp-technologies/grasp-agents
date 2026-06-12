"""
Regression tests for the P2 sandbox / env fixes
(consolidated audit 2026-06-11, §3 items 27-28).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.sandbox.local.environment import local_environment
from grasp_agents.sandbox.local.exec import LocalExecBackend
from grasp_agents.sandbox.policy import (
    DEFAULT_ENV_SCRUB,
    NetworkPolicy,
    SandboxPolicy,
)

# ---------- Item 27: seatbelt x LOOPBACK/ALLOWLIST fails at construction ----------


@pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("sandbox-exec") is None,
    reason="seatbelt requires macOS with sandbox-exec",
)
class TestSeatbeltNetworkValidation:
    @pytest.mark.parametrize(
        "network", [NetworkPolicy.LOOPBACK, NetworkPolicy.ALLOWLIST]
    )
    def test_unenforceable_network_rejected_at_construction(
        self, tmp_path: Path, network: NetworkPolicy
    ) -> None:
        with pytest.raises(ValueError, match="not enforceable under"):
            local_environment(
                allowed_roots=[tmp_path],
                confinement="seatbelt",
                network=network,
                allowed_domains=["example.com"],
            )

    def test_all_or_none_accepted(self, tmp_path: Path) -> None:
        for network in (NetworkPolicy.NONE, NetworkPolicy.ALL):
            local_environment(
                allowed_roots=[tmp_path],
                confinement="seatbelt",
                network=network,
            )


# ---------- Item 28: env scrub coverage + case-insensitivity ----------


def _scrubbed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_env: dict[str, str]
) -> dict[str, str]:
    for key, value in host_env.items():
        monkeypatch.setenv(key, value)
    policy = SandboxPolicy(allowed_roots=(tmp_path,), env_scrub=DEFAULT_ENV_SCRUB)
    backend = LocalExecBackend(policy=policy, inherit_host_env=True)
    return backend._merged_env(None)


class TestEnvScrub:
    def test_lowercase_secrets_scrubbed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        merged = _scrubbed_env(
            tmp_path,
            monkeypatch,
            {
                "my_api_key": "s3cret",
                "stripe_token": "s3cret",
                "Db_Password": "s3cret",
            },
        )
        assert "my_api_key" not in merged
        assert "stripe_token" not in merged
        assert "Db_Password" not in merged

    def test_connection_and_session_vars_scrubbed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        merged = _scrubbed_env(
            tmp_path,
            monkeypatch,
            {
                "DATABASE_URL": "postgres://u:p@h/db",
                "REDIS_URL": "redis://h",
                "MONGODB_URI": "mongodb://h",
                "SSH_AUTH_SOCK": "/tmp/agent.sock",
                "KUBECONFIG": "/home/u/.kube/config",
                "SENTRY_DSN": "https://x@sentry.io/1",
                "SLACK_WEBHOOK_URL": "https://hooks.slack.com/x",
                "OP_SESSION_myteam": "tok",
            },
        )
        for var in (
            "DATABASE_URL",
            "REDIS_URL",
            "MONGODB_URI",
            "SSH_AUTH_SOCK",
            "KUBECONFIG",
            "SENTRY_DSN",
            "SLACK_WEBHOOK_URL",
            "OP_SESSION_myteam",
        ):
            assert var not in merged, var

    def test_benign_vars_survive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        merged = _scrubbed_env(tmp_path, monkeypatch, {"MY_PLAIN_VAR": "v"})
        assert merged.get("MY_PLAIN_VAR") == "v"


# ---------- Item 28: E2B sandbox not orphaned on setup failure ----------


class _FakeFiles:
    async def make_dir(self, path: str) -> None:
        del path
        msg = "make_dir failed"
        raise RuntimeError(msg)


class _FakeSandbox:
    def __init__(self) -> None:
        self.files = _FakeFiles()
        self.killed = False

    @classmethod
    async def create(cls, **kwargs: Any) -> _FakeSandbox:
        del kwargs
        return cls()

    async def kill(self) -> None:
        self.killed = True


class TestE2BSetupFailure:
    @pytest.mark.asyncio
    async def test_created_sandbox_killed_when_setup_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        e2b_env_mod = pytest.importorskip("grasp_agents.sandbox.e2b.environment")

        created: list[_FakeSandbox] = []

        class _Recorder(_FakeSandbox):
            @classmethod
            async def create(cls, **kwargs: Any) -> _FakeSandbox:
                sandbox = await super().create(**kwargs)
                created.append(sandbox)  # type: ignore[arg-type]
                return sandbox

        monkeypatch.setattr(e2b_env_mod, "_sandbox_cls", lambda _ci: _Recorder)

        from grasp_agents.sandbox.e2b._handle import SandboxHandle
        from grasp_agents.sandbox.e2b.exec import E2BExecBackend
        from grasp_agents.sandbox.e2b.file_backend import (
            E2BFileBackend,
        )

        policy = SandboxPolicy(allowed_roots=(Path("/home/user/workspace"),))
        holder = SandboxHandle(None)
        env = e2b_env_mod.E2BEnvironment(
            policy=policy,
            holder=holder,
            file_backend=E2BFileBackend(holder, policy=policy),
            exec_backend=E2BExecBackend(holder, policy=policy),
            create_params={},
            owns_sandbox=True,
        )

        with pytest.raises(RuntimeError, match="make_dir failed"):
            await env.__aenter__()

        assert len(created) == 1
        assert created[0].killed is True
        assert holder.sandbox is None
