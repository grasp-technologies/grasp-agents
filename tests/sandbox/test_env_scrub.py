"""
The default ``env_scrub`` denylist keeps host credentials out of the
subprocess environment under ``confinement="none"`` (the model's window onto
host secrets if left unscrubbed).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.sandbox.local.environment import (
    DEFAULT_ENV_SCRUB,
    local_environment,
)
from grasp_agents.sandbox.local.exec import LocalExecBackend
from grasp_agents.sandbox.policy import SandboxPolicy


def test_default_env_scrub_flows_to_policy(tmp_path: Path) -> None:
    env = local_environment(allowed_roots=[tmp_path])
    assert env.policy.env_scrub == DEFAULT_ENV_SCRUB


def test_secrets_scrubbed_from_subprocess_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("MY_SERVICE_TOKEN", "t0ken")
    monkeypatch.setenv("DB_PASSWORD", "hunter2")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GRASP_SAFE_VAR", "ok")  # not a secret pattern

    env = local_environment(allowed_roots=[tmp_path])
    backend = env.exec_backend
    assert backend is not None
    merged = backend._merged_env(None)  # env handed to subprocesses

    assert "OPENAI_API_KEY" not in merged
    assert "MY_SERVICE_TOKEN" not in merged
    assert "DB_PASSWORD" not in merged
    assert "AWS_SECRET_ACCESS_KEY" not in merged
    assert merged.get("GRASP_SAFE_VAR") == "ok"  # non-secret passes through
    assert "PATH" in merged  # PATH still resolves


def test_explicit_env_is_never_scrubbed(tmp_path: Path) -> None:
    # A key the caller deliberately sets wins even if its name matches a pattern.
    env = local_environment(
        allowed_roots=[tmp_path], env={"OPENAI_API_KEY": "explicit"}
    )
    backend = env.exec_backend
    assert backend is not None
    assert backend._merged_env(None)["OPENAI_API_KEY"] == "explicit"


def test_env_scrub_disablable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    env = local_environment(allowed_roots=[tmp_path], env_scrub=())
    backend = env.exec_backend
    assert backend is not None
    assert backend._merged_env(None).get("OPENAI_API_KEY") == "sk-secret"


# ---------- case-insensitivity + connection/session vars ----------


def _scrubbed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_env: dict[str, str]
) -> dict[str, str]:
    for key, value in host_env.items():
        monkeypatch.setenv(key, value)
    policy = SandboxPolicy(allowed_roots=(tmp_path,), env_scrub=DEFAULT_ENV_SCRUB)
    backend = LocalExecBackend(policy=policy, inherit_host_env=True)
    return backend._merged_env(None)


def test_lowercase_secrets_scrubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged = _scrubbed_env(
        tmp_path,
        monkeypatch,
        {"my_api_key": "s3cret", "stripe_token": "s3cret", "Db_Password": "s3cret"},
    )
    assert "my_api_key" not in merged
    assert "stripe_token" not in merged
    assert "Db_Password" not in merged


def test_connection_and_session_vars_scrubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
