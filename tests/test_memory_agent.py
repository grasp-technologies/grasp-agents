"""Tests: memory section auto-attaches on LLMAgent + reads RunContext.memory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.prompt_builder import SystemPromptSection
from grasp_agents.memory import (
    InMemoryMemoryProvider,
    MemoryEntry,
)
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
    agentic_mode: bool = False,
    enable_memory: bool = True,
) -> LLMAgent[str, str, _State]:
    return LLMAgent[str, str, _State](
        name="memory_test_agent",
        llm=MockLLM(responses_queue=[]),
        sys_prompt=sys_prompt,
        stream_llm=True,
        env_info=env_info,
        enable_memory=enable_memory,
        tools=[] if agentic_mode else None,
    )


class TestEnableMemory:
    def test_memory_section_registered_when_enabled(self) -> None:
        agent = _make_agent()  # helper passes enable_memory=True
        assert any(s.name == "memory" for s in agent.system_prompt_sections)

    def test_memory_section_dropped_when_disabled(self) -> None:
        agent = _make_agent(enable_memory=False)
        names = {s.name for s in agent.system_prompt_sections}
        assert "memory" not in names

    def test_default_off_means_no_memory_section(self) -> None:
        # Framework default is enable_memory=False — agents don't
        # silently gain memory.
        agent = LLMAgent[str, str, _State](
            name="default",
            llm=MockLLM(responses_queue=[]),
            stream_llm=True,
            env_info=False,
        )
        names = {s.name for s in agent.system_prompt_sections}
        assert "memory" not in names
        # And no memory_relevance attachment either.
        attachment_names = [
            a.name for a in agent._prompt_builder.input_attachments  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        ]
        assert "memory_relevance" not in attachment_names

    def test_env_info_off(self) -> None:
        agent = _make_agent(env_info=False)
        names = {s.name for s in agent.system_prompt_sections}
        assert "env_info" not in names

    def test_enable_memory_does_not_auto_attach_tools(self) -> None:
        """
        Memory authoring now goes through generic file tools, NOT a
        specialized save/load/delete surface. So the agent's tool set
        stays exactly what the caller passed — no quiet additions.
        """
        agent = _make_agent(agentic_mode=True)
        assert list(agent.tools) == []


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
        assert "<memory-index>" in prompt
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
        assert "<memory-index>" in prompt

    @pytest.mark.anyio
    async def test_empty_provider_emits_instructions_only(self) -> None:
        agent = _make_agent(sys_prompt="head.")
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            memory=InMemoryMemoryProvider(),
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        # Empty provider → no MEMORY.md index sub-block, but the
        # instructions still render (memory IS configured; the agent
        # should know how to use it once it has anything to load).
        assert prompt is not None
        assert prompt.startswith("head.")
        assert "# Memory" in prompt
        # No index sub-block.
        assert "</memory-index>" not in prompt


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


class TestAutoMemoryInstructions:
    @pytest.mark.anyio
    async def test_instructions_render_whenever_provider_set(self) -> None:
        # No matter which tools are wired, when ctx.memory is configured
        # the instructions block describes the substrate (taxonomy +
        # MEMORY.md format + edit loop). The prompt is cache-stable
        # across runs with different tool sets.
        agent = _make_agent()  # no explicit tools
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            memory=InMemoryMemoryProvider(index="# idx body\n"),
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is not None
        assert "<memory-index>" in prompt  # index sub-block
        assert "# Memory" in prompt  # instructions sub-block
        # Taxonomy names appear inside <type><name>…</name>…</type> blocks.
        assert "<name>user</name>" in prompt
        assert "<name>feedback</name>" in prompt
        # File-tool authoring is mentioned generically; no specialized
        # save_memory / list_memories names appear.
        assert "file tools" in prompt

    @pytest.mark.anyio
    async def test_instructions_do_not_enumerate_tools(self) -> None:
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            memory=InMemoryMemoryProvider(index="# idx\n"),
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is not None
        # Specialized memory-tool names from the old surface must not
        # appear — they no longer exist.
        assert "save_memory" not in prompt
        assert "list_memories" not in prompt
        assert "update_memory_index" not in prompt
        assert "delete_memory" not in prompt

    @pytest.mark.anyio
    async def test_selector_adds_per_turn_note(self) -> None:
        provider = InMemoryMemoryProvider(index="# idx\n")

        def keep_all(
            *, entries: Sequence[MemoryEntry], **_: Any
        ) -> Sequence[MemoryEntry]:
            return entries

        provider.set_selector(keep_all)

        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(state=_State(), memory=provider)
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is not None
        assert "surfaced into each turn" in prompt

    @pytest.mark.anyio
    async def test_no_memory_provider_no_section(self) -> None:
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(state=_State())
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        # No ctx.memory → memory section drops entirely.
        assert prompt is None

    @pytest.mark.anyio
    async def test_empty_provider_emits_instructions_only(self) -> None:
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(
            state=_State(), memory=InMemoryMemoryProvider()
        )
        prompt = await agent.build_system_prompt(ctx, exec_id="e1")
        assert prompt is not None
        # No rendered <memory-index> sub-block when the index is empty.
        # (The instructions text REFERS to <memory-index> as a forward
        # reference; checking the closing tag is the precise marker for
        # an actually-rendered block.)
        assert "</memory-index>" not in prompt
        # Instructions still render — the substrate is configured.
        assert "# Memory" in prompt
        assert "[name](file.md)" in prompt
