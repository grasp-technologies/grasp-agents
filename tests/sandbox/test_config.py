"""
Tests for the JSON-friendly :class:`EnvironmentConfig` loader.

Covers deserialization (incl. the ``exec`` alias + ``extra="forbid"``), the
policy mapping, file loading, and that ``build()`` yields a working environment.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from grasp_agents.sandbox import EnvironmentConfig, load_environment_config

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


async def test_config_maps_to_policy(tmp_path: Path) -> None:
    cfg = EnvironmentConfig.model_validate(
        {
            "confinement": "none",
            "filesystem": {
                "allowed_roots": [str(tmp_path)],
                "deny_write": [str(tmp_path / "protected")],
            },
            "network": {"policy": "none"},
            "exec": {"env_scrub": ["*_API_KEY"], "cpu_timeout": 5},
        }
    )
    env = cfg.build()
    assert env.policy.allowed_roots == (tmp_path,)
    assert env.policy.deny_write == (tmp_path / "protected",)
    assert env.policy.env_scrub == ("*_API_KEY",)
    assert env.exec_backend is not None
    assert env.exec_backend.name == "local"


async def test_config_exec_alias() -> None:
    cfg = EnvironmentConfig.model_validate({"exec": {"cpu_timeout": 3}})
    assert cfg.exec_.cpu_timeout == 3


async def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate({"filesystem": {"bogus": 1}})


async def test_config_rejects_bad_confinement() -> None:
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate({"confinement": "nope"})


async def test_config_loads_from_file_and_runs(tmp_path: Path) -> None:
    cfg_path = tmp_path / "sandbox.json"
    cfg_path.write_text(
        json.dumps({"filesystem": {"allowed_roots": [str(tmp_path)]}})
    )
    env = load_environment_config(cfg_path).build()
    assert env.exec_backend is not None
    result = await env.exec_backend.execute("echo configured")
    assert result.stdout.strip() == "configured"
