"""Tests: memory section auto-attaches on LLMAgent + reads RunContext.memory."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.prompt_builder import SystemPromptSection
from grasp_agents.memory import InMemoryMemoryProvider
from grasp_agents.run_context import RunContext

from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
)


class _State(BaseModel):
    pass


def _make_agent(
    *,
    sys_prompt: str | None = None,
    env_info: bool = False,
) -> LLMAgent[str, str, _State]:
    return LLMAgent[str, str, _State](
        name="memory_test_agent",
        llm=MockLLM(responses_queue=[]),
        sys_prompt=sys_prompt,
        stream_llm=True,
        env_info=env_info,
    )


class TestAutoAttach:
    def test_memory_section_auto_registered(self) -> None:
        agent = _make_agent()
        assert any(s.name == "memory" for s in agent.system_prompt_sections)

    def test_default_sections_present(self) -> None:
        agent = _make_agent(env_info=True)
        names = {s.name for s in agent.system_prompt_sections}
        assert {"skills", "memory", "env_info", "mcp_instructions"} <= names

    def test_env_info_off(self) -> None:
        agent = _make_agent(env_info=False)
        names = {s.name for s in agent.system_prompt_sections}
        assert "env_info" not in names


class TestSystemPromptIntegration:
    @pytest.mark.anyio
    async def test_no_provider_no_memory_block(self) -> None:
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(state=_State())
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is None

    @pytest.mark.anyio
    async def test_provider_renders_memory_block(self) -> None:
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            memory=InMemoryMemoryProvider(index="# index body\n"),
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is not None
        assert "# memory" in prompt
        assert "index body" in prompt

    @pytest.mark.anyio
    async def test_combines_with_user_sys_prompt(self) -> None:
        agent = _make_agent(sys_prompt="You are a helper.")
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            memory=InMemoryMemoryProvider(index="# idx\n"),
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is not None
        assert prompt.startswith("You are a helper.")
        assert "# memory" in prompt

    @pytest.mark.anyio
    async def test_empty_provider_drops_block(self) -> None:
        agent = _make_agent(sys_prompt="head.")
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            memory=InMemoryMemoryProvider(),
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt == "head."


class TestUserOverride:
    def test_add_section_with_same_name_replaces(self) -> None:
        agent = _make_agent()
        before = sum(
            1 for s in agent.system_prompt_sections if s.name == "memory"
        )
        assert before == 1

        def _override(**_: Any) -> str:
            return "OVERRIDE"

        custom = SystemPromptSection(name="memory", compute=_override)
        agent.add_system_prompt_section(custom)
        sections = [s for s in agent.system_prompt_sections if s.name == "memory"]
        assert len(sections) == 1
        assert sections[0] is custom
