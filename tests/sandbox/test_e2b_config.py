"""
E2B template / environment config (``sandbox/e2b/config.py``).

Schema validation, the config -> Dockerfile mapping and what ``build()`` does
for each state of a template's latest build run **offline**, against
``FakeTemplateAPI`` — no E2B key, no build, no network. Gated only on the
optional ``e2b`` package being importable.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("e2b")

import json

from e2b import BuildException, Template
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
from grasp_agents.sandbox.e2b.templates import TemplateBuildState
from tests.sandbox._e2b_fakes import FakeTemplateAPI

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
    with pytest.raises(ValidationError):
        TemplateBase(template="code-interpreter-v1", python="3.11")


def test_start_cmd_requires_ready_cmd() -> None:
    with pytest.raises(ValidationError):
        E2BTemplateConfig(start_cmd="run.sh")


def test_env_config_rejects_both_template_sources() -> None:
    with pytest.raises(ValidationError):
        E2BEnvironmentConfig(template="prebuilt", template_build=E2BTemplateConfig())


def test_env_config_rejects_a_label_e2b_cannot_name_snapshots_with() -> None:
    with pytest.raises(ValidationError, match="should match pattern"):
        E2BEnvironmentConfig(label="Prod.EU")


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


# --- build(): what it does for each state of the latest build ---------------

BUILDING = TemplateBuildState("building")
READY = TemplateBuildState("ready")


def _demo() -> E2BTemplateConfig:
    return E2BTemplateConfig(
        base=TemplateBase(python="3.12"),
        steps=[PipInstall(op="pip_install", packages=["six==1.16.0"])],
        alias="demo",
        tags=["v1"],
        cpu_count=4,
        memory_mb=2048,
    )


@pytest.mark.asyncio
async def test_build_reuses_a_ready_template() -> None:
    templates = FakeTemplateAPI(READY)
    assert await _demo().build(templates=templates) == "demo"
    assert templates.builds == []


@pytest.mark.asyncio
async def test_build_builds_a_template_no_one_has_built() -> None:
    templates = FakeTemplateAPI(None)
    assert await _demo().build(templates=templates) == "demo"
    [build] = templates.builds
    assert build["name"] == "demo"
    assert build["tags"] == ["v1"]
    assert (build["cpu_count"], build["memory_mb"]) == (4, 2048)
    assert build["skip_cache"] is False


@pytest.mark.asyncio
async def test_build_waits_for_a_build_under_way() -> None:
    templates = FakeTemplateAPI(BUILDING, BUILDING, READY)
    assert await _demo().build(templates=templates, poll_interval=0) == "demo"
    assert templates.lookups == 3
    assert templates.builds == []


@pytest.mark.asyncio
async def test_build_rebuilds_a_template_whose_last_build_failed() -> None:
    templates = FakeTemplateAPI(TemplateBuildState("failed", "exit status 3"))
    assert await _demo().build(templates=templates) == "demo"
    assert len(templates.builds) == 1


@pytest.mark.asyncio
async def test_build_raises_e2bs_reason_when_the_awaited_build_fails() -> None:
    templates = FakeTemplateAPI(
        BUILDING, TemplateBuildState("failed", "failed to run command: exit status 3")
    )
    with pytest.raises(BuildException, match="failed to run command: exit status 3"):
        await _demo().build(templates=templates, poll_interval=0)
    assert templates.builds == []


@pytest.mark.asyncio
async def test_build_stops_waiting_after_wait_timeout() -> None:
    templates = FakeTemplateAPI(BUILDING)
    with pytest.raises(TimeoutError, match="'demo' is still building"):
        await _demo().build(templates=templates, wait_timeout=0, poll_interval=0)
    assert templates.builds == []


@pytest.mark.asyncio
async def test_build_raises_e2bs_error_when_its_own_build_fails() -> None:
    templates = FakeTemplateAPI(None, build_error="failed to run command")
    with pytest.raises(BuildException, match="failed to run command"):
        await _demo().build(templates=templates)
    # The next call finds the failed build and builds again.
    with pytest.raises(BuildException):
        await _demo().build(templates=templates)
    assert len(templates.builds) == 2


@pytest.mark.asyncio
async def test_skip_cache_builds_without_looking_at_the_last_build() -> None:
    templates = FakeTemplateAPI(READY)
    config = _demo().model_copy(update={"skip_cache": True})
    assert await config.build(templates=templates) == "demo"
    assert templates.lookups == 0
    [build] = templates.builds
    assert build["skip_cache"] is True


@pytest.mark.asyncio
async def test_concurrent_builds_of_one_name_in_a_process_build_once() -> None:
    templates = FakeTemplateAPI(None)
    refs = await asyncio.gather(
        _demo().build(templates=templates), _demo().build(templates=templates)
    )
    assert refs == ["demo", "demo"]
    assert len(templates.builds) == 1


@pytest.mark.asyncio
async def test_build_needs_an_alias() -> None:
    templates = FakeTemplateAPI(None)
    with pytest.raises(ValueError, match="alias"):
        await E2BTemplateConfig().build(templates=templates)
    assert templates.builds == []


@pytest.mark.asyncio
async def test_template_base_builds_on_an_existing_e2b_template() -> None:
    config = E2BTemplateConfig(
        base=TemplateBase(template="code-interpreter-v1"),
        steps=[PipInstall(op="pip_install", packages=["six==1.16.0"])],
        alias="demo",
    )
    templates = FakeTemplateAPI(None)
    await config.build(templates=templates)
    [build] = templates.builds
    built = json.loads(Template.to_json(build["template"]))
    assert built["fromTemplate"] == "code-interpreter-v1"
    assert "pip install six==1.16.0" in json.dumps(built["steps"])
    with pytest.raises(ValueError, match="another template"):
        config.to_dockerfile()
