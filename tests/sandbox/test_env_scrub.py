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
