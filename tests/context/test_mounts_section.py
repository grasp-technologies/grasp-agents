"""Tests for the ``mounts`` system-prompt section."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from grasp_agents import LLMAgent
from grasp_agents.context import MOUNTS_SECTION_NAME, make_mounts_section
from grasp_agents.file_backend import CompositeFileBackend, InMemoryFileBackend, Mount
from grasp_agents.sandbox import local_environment
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.file_edit import CopyTool
from tests._helpers import MockLLM, _make_agent_ctx

pytestmark = pytest.mark.asyncio

MEMORY = Path("/learner/memory")
NOTES = Path("/learner/notes")
SCRATCH = Path("/scratch")


def _mounts() -> list[Mount]:
    return [
        Mount(
            MEMORY,
            InMemoryFileBackend([MEMORY]),
            description="what you know about the learner",
        ),
        Mount(
            NOTES,
            InMemoryFileBackend([NOTES]),
            read_only=True,
            description="the learner's notes",
        ),
        Mount(SCRATCH, InMemoryFileBackend([SCRATCH])),
    ]


async def _render(ctx: SessionContext[Any], **kwargs: Any) -> str | None:
    result = make_mounts_section().compute(ctx=ctx, exec_id=None, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


async def test_lists_each_mount_with_its_access() -> None:
    ctx: SessionContext[Any] = SessionContext(
        file_backend=CompositeFileBackend(_mounts())
    )

    text = await _render(ctx)

    assert text is not None
    assert text.startswith("<mounts>")
    assert "- `/learner/memory`: what you know about the learner (read-write)" in text
    assert "- `/learner/notes`: the learner's notes (read-only)" in text
    assert "- `/scratch` (read-write)" in text
    assert "Bash" not in text
    assert "Copy" not in text


async def test_mentions_copy_when_the_agent_has_it() -> None:
    ctx: SessionContext[Any] = SessionContext(
        file_backend=CompositeFileBackend(_mounts())
    )
    agent_ctx = _make_agent_ctx(tools={"Copy": CopyTool()})

    text = await _render(ctx, agent_ctx=agent_ctx)

    assert text is not None
    assert "Use Copy to move files between mounts" in text


async def test_marks_the_mount_bash_can_see(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    env = local_environment(allowed_roots=[root])
    composite = CompositeFileBackend(
        [*_mounts(), Mount(root, env.file_backend, description="sandbox")],
        scratch=SCRATCH,
    )
    ctx: SessionContext[Any] = SessionContext(environment=env, file_backend=composite)

    text = await _render(ctx)

    assert text is not None
    assert (
        f"- `{root}`: sandbox (read-write, visible to Bash and code execution)" in text
    )
    assert f"only `{root}` exists" in text


async def test_renders_nothing_without_a_composite() -> None:
    ctx: SessionContext[Any] = SessionContext(
        file_backend=InMemoryFileBackend([SCRATCH])
    )

    assert await _render(ctx) is None
    assert await _render(SessionContext()) is None


async def test_llm_agent_registers_the_section() -> None:
    agent = LLMAgent[str, str, None](
        name="t", llm=MockLLM(responses_queue=[]), stream_llm=True, env_info=False
    )

    names = [
        s.name
        for s in agent._prompt_builder.system_prompt_sections  # pyright: ignore[reportPrivateUsage]
    ]

    assert MOUNTS_SECTION_NAME in names
