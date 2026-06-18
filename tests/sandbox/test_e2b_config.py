"""
E2B template / environment config (``sandbox/e2b/config.py``).

Schema validation + the config -> Dockerfile mapping run **offline** — no E2B
key, no build, no network. Gated only on the optional ``e2b`` package being
importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("e2b")

import json

from pydantic import ValidationError

from grasp_agents.sandbox.e2b.config import (
    AptInstall,
    E2BEnvironmentConfig,
    E2BTemplateConfig,
    GitClone,
    PipInstall,
    RunCmd,
    SetEnvs,
    SetWorkdir,
    TemplateBase,
    load_e2b_config,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_to_dockerfile_maps_base_and_steps() -> None:
    c = E2BTemplateConfig(
        base=TemplateBase(python="3.11"),
        steps=[
            AptInstall(op="apt_install", packages=["ffmpeg"]),
            PipInstall(op="pip_install", packages=["torch", "datasets"]),
            GitClone(op="git_clone", url="https://example.com/x.git", path="/app/x"),
            SetEnvs(op="set_envs", envs={"HF_HOME": "/data/hf"}),
            SetWorkdir(op="set_workdir", workdir="/app"),
            RunCmd(op="run_cmd", command="echo ready"),
        ],
        alias="ml-research",
    )
    df = c.to_dockerfile()
    assert "FROM python:3.11" in df
    assert "ffmpeg" in df
    assert "pip install torch datasets" in df
    assert "x.git" in df
    assert "HF_HOME=/data/hf" in df
    assert "WORKDIR /app" in df
    assert "echo ready" in df


def test_raw_dockerfile_base_is_escape_hatch() -> None:
    c = E2BTemplateConfig(
        base=TemplateBase(dockerfile="FROM python:3.12\nRUN echo custom\n")
    )
    df = c.to_dockerfile()
    assert "python:3.12" in df
    assert "echo custom" in df


def test_base_rejects_two_images() -> None:
    with pytest.raises(ValidationError):
        TemplateBase(python="3.11", ubuntu="22.04")


def test_start_cmd_requires_ready_cmd() -> None:
    with pytest.raises(ValidationError):
        E2BTemplateConfig(start_cmd="run.sh")


def test_env_config_rejects_both_template_sources() -> None:
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template="prebuilt", template_build=E2BTemplateConfig())


def test_load_e2b_config(tmp_path: Path) -> None:
    p = tmp_path / "e2b.json"
    p.write_text(
        json.dumps(
            {
                "template_build": {
                    "base": {"python": "3.11"},
                    "steps": [{"op": "pip_install", "packages": ["numpy"]}],
                    "alias": "demo",
                    "cpu_count": 4,
                    "memory_mb": 4096,
                },
                "allowed_roots": ["/home/user/ws"],
                "code_interpreter": True,
            }
        ),
        encoding="utf-8",
    )
    conf = load_e2b_config(p)
    assert conf.template_build is not None
    assert conf.template_build.alias == "demo"
    assert conf.template_build.cpu_count == 4
    assert conf.code_interpreter is True
    assert conf.allowed_roots == ["/home/user/ws"]
    # the inline template renders to a Dockerfile offline
    assert "numpy" in conf.template_build.to_dockerfile()
