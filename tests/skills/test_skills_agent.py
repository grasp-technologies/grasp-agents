"""Tests: skills section auto-attaches on LLMAgent + reads RunContext.skills."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.run_context import RunContext
from grasp_agents.skills import (
    SkillRegistry,
    list_skills,
    load_skill,
)
from grasp_agents.types.events import ToolErrorInfo
from tests.durability.test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class _State(BaseModel):
    pass


def _write_skill(
    root: Path,
    name: str,
    description: str,
    body: str = "Body content.",
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_lines = [f"name: {name}", f"description: {description}"]
    if extra_metadata is not None:
        import yaml

        meta_yaml = yaml.safe_dump(
            {"metadata": extra_metadata}, sort_keys=False
        ).strip()
        fm_lines.append(meta_yaml)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        "---\n" + "\n".join(fm_lines) + "\n---\n" + body + "\n",
        encoding="utf-8",
    )
    return skill_md


def _make_agent(
    *,
    env_info: bool = False,
    with_skill_tools: bool = False,
    enable_skills: bool = True,
    ctx: RunContext[_State] | None = None,
) -> LLMAgent[str, str, _State]:
    return LLMAgent[str, str, _State](
        name="skills_test_agent",
        ctx=ctx,
        llm=MockLLM(responses_queue=[]),
        stream_llm=True,
        env_info=env_info,
        enable_skills=enable_skills,
        tools=[load_skill, list_skills] if with_skill_tools else None,
    )


async def _build_system_prompt(
    agent: LLMAgent[str, str, _State], ctx: RunContext[_State]
) -> str | None:
    agent.on_adopted(ctx=ctx)
    return await agent.build_system_prompt(exec_id="e1")


# ---------- Auto-attached skills section ----------


class TestEnableSkills:
    def test_section_registered_when_enabled(self) -> None:
        agent = _make_agent()  # helper passes enable_skills=True
        names = {s.name for s in agent.system_prompt_sections}
        assert "skills" in names

    def test_section_dropped_when_disabled(self) -> None:
        agent = _make_agent(enable_skills=False)
        names = {s.name for s in agent.system_prompt_sections}
        assert "skills" not in names

    def test_default_off_means_no_section(self) -> None:
        # The framework default is enable_skills=False — agents don't
        # silently gain the skills section.
        agent = LLMAgent[str, str, _State](
            name="default",
            llm=MockLLM(responses_queue=[]),
            stream_llm=True,
            env_info=False,
        )
        names = {s.name for s in agent.system_prompt_sections}
        assert "skills" not in names
        assert "load_skill" not in agent.tools

    def test_load_skill_auto_attaches_in_agentic_mode(self) -> None:
        agent = LLMAgent[str, str, _State](
            name="agentic",
            llm=MockLLM(responses_queue=[]),
            stream_llm=True,
            env_info=False,
            enable_skills=True,
            tools=[],
        )
        # load_skill rides in; list_skills stays opt-in (catalog is in
        # the system prompt — calling it again per-turn is bloat).
        assert "load_skill" in agent.tools
        assert "list_skills" not in agent.tools

    def test_enable_skills_implies_agentic_without_tools(self) -> None:
        # enable_skills attaches load_skill, which the model must be able to
        # call — so it implies agentic mode even with tools=None (regression:
        # tools=None previously left the agent non-agentic and the loader
        # unattached, so the model could never load a skill).
        agent = LLMAgent[str, str, _State](
            name="skills_no_tools",
            llm=MockLLM(responses_queue=[]),
            stream_llm=True,
            env_info=False,
            enable_skills=True,
            tools=None,
        )
        assert "load_skill" in agent.tools
        # agentic ⇒ no default structured-output schema was applied
        assert agent._used_default_llm_output_schema is False

    def test_explicit_list_skills_via_tools_kwarg(self) -> None:
        agent = _make_agent(with_skill_tools=True)
        assert "load_skill" in agent.tools
        assert "list_skills" in agent.tools


# ---------- System prompt section ----------


class TestSystemPromptSection:
    @pytest.mark.asyncio
    async def test_empty_registry_no_block(self, tmp_path: Path) -> None:
        del tmp_path
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(state=_State())
        # No ctx.skills set
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is None

    @pytest.mark.asyncio
    async def test_section_renders_catalog(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha skill description.")
        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            skills=SkillRegistry.from_path(tmp_path),
        )
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert "<available_skills>" in prompt
        assert "<name>alpha</name>" in prompt
        assert "Alpha skill description." in prompt

    @pytest.mark.asyncio
    async def test_combines_with_user_sys_prompt(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        agent = LLMAgent[str, str, _State](
            name="combo",
            llm=MockLLM(responses_queue=[]),
            sys_prompt="You are a helper.",
            stream_llm=True,
            env_info=False,
            enable_skills=True,
        )
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            skills=SkillRegistry.from_path(tmp_path),
        )
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert prompt.startswith("You are a helper.")
        assert "<available_skills>" in prompt

    @pytest.mark.asyncio
    async def test_combines_with_dynamic_builder(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        agent = _make_agent()

        def dyn(*, exec_id: str) -> str:
            del exec_id
            return "Dynamic header."

        agent.add_system_prompt_builder(dyn)

        ctx: RunContext[_State] = RunContext(
            state=_State(),
            skills=SkillRegistry.from_path(tmp_path),
        )
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert prompt.startswith("Dynamic header.")
        assert "<available_skills>" in prompt


# ---------- Catalog selector hook ----------


class TestCatalogSelectorHelper:
    """
    The selector is NOT applied at ``build_system_prompt`` time — the
    catalog stays cache-stable. The selector API is still consulted
    elsewhere (e.g. via :meth:`SkillRegistry.select_relevant` directly, or
    in future ``InputAttachment``-style per-turn surfacing).
    """

    @pytest.mark.asyncio
    async def test_select_relevant_filters(self, tmp_path: Path) -> None:
        from grasp_agents.skills import Skill

        _write_skill(tmp_path, "alpha", "Alpha")
        _write_skill(tmp_path, "beta", "Beta")
        registry = SkillRegistry.from_path(tmp_path)

        def keep_alpha(*, entries: Sequence[Skill], **_: Any) -> Sequence[Skill]:
            return [s for s in entries if s.name == "alpha"]

        registry.set_selector(keep_alpha)
        kept = await registry.select_relevant()
        assert [s.name for s in kept] == ["alpha"]

    @pytest.mark.asyncio
    async def test_system_prompt_ignores_selector(self, tmp_path: Path) -> None:

        from grasp_agents.skills import Skill

        _write_skill(tmp_path, "alpha", "Alpha")
        _write_skill(tmp_path, "beta", "Beta")
        registry = SkillRegistry.from_path(tmp_path)

        def keep_alpha(*, entries: Sequence[Skill], **_: Any) -> Sequence[Skill]:
            return [s for s in entries if s.name == "alpha"]

        registry.set_selector(keep_alpha)

        agent = _make_agent()
        ctx: RunContext[_State] = RunContext(state=_State(), skills=registry)
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        # Both skills still in the catalog — selector is NOT applied.
        assert "<name>alpha</name>" in prompt
        assert "<name>beta</name>" in prompt


# ---------- load_skill / list_skills tools ----------


class TestLoadSkillTool:
    @pytest.mark.asyncio
    async def test_returns_body(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x", body="ALPHA BODY")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await load_skill(name="alpha", ctx=ctx)
        assert isinstance(result, str)
        assert "ALPHA BODY" in result

    @pytest.mark.asyncio
    async def test_strips_argument_placeholder(self, tmp_path: Path) -> None:
        # A model-invoked load has no slash-command args — the content is in the
        # conversation, so a placeholder on its own line is dropped, not left as
        # an empty inline slot.
        _write_skill(
            tmp_path, "alpha", "x", body="Proofread the user's text.\n\n$ARGUMENTS"
        )
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await load_skill(name="alpha", ctx=ctx)
        assert isinstance(result, str)
        assert "$ARGUMENTS" not in result
        assert result.strip() == "Proofread the user's text."

    @pytest.mark.asyncio
    async def test_strips_named_argument_placeholder(self, tmp_path: Path) -> None:
        # Named placeholders ($ARG_NAME) are stripped too — they'd otherwise
        # leak as literals on a model-invoked load.
        _write_skill(tmp_path, "alpha", "x", body="Search the index.\n\n$QUERY")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await load_skill(name="alpha", ctx=ctx)
        assert isinstance(result, str)
        assert "$QUERY" not in result
        assert result.strip() == "Search the index."

    @pytest.mark.asyncio
    async def test_no_registry_errors(self) -> None:
        ctx: RunContext[_State] = RunContext(state=_State())
        result = await load_skill(name="nope", ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "no skills" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unknown_skill_errors(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await load_skill(name="nonexistent", ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "not available" in result.error.lower()

    @pytest.mark.asyncio
    async def test_mid_session_edit_visible(self, tmp_path: Path) -> None:
        skill_md = _write_skill(tmp_path, "alpha", "x", body="ORIGINAL BODY")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )

        first = await load_skill(name="alpha", ctx=ctx)
        assert isinstance(first, str)
        assert "ORIGINAL BODY" in first

        skill_md.write_text(
            "---\nname: alpha\ndescription: x\n---\nUPDATED BODY\n",
            encoding="utf-8",
        )

        second = await load_skill(name="alpha", ctx=ctx)
        assert isinstance(second, str)
        assert "UPDATED BODY" in second
        assert "ORIGINAL BODY" not in second


class TestListSkillsTool:
    @pytest.mark.asyncio
    async def test_no_registry(self) -> None:
        ctx: RunContext[_State] = RunContext(state=_State())
        result = await list_skills(ctx=ctx)
        assert isinstance(result, str)
        assert "no skills" in result.lower()

    @pytest.mark.asyncio
    async def test_returns_catalog(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha skill.")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await list_skills(ctx=ctx)
        assert isinstance(result, str)
        assert "<available_skills>" in result
        assert "<name>alpha</name>" in result

    @pytest.mark.asyncio
    async def test_refresh_picks_up_new_skill(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha skill.")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )

        first = await list_skills(ctx=ctx)
        assert isinstance(first, str)
        assert "<name>alpha</name>" in first
        assert "<name>beta</name>" not in first

        # Author a new skill, then refresh.
        _write_skill(tmp_path, "beta", "Beta skill.")
        second = await list_skills(refresh=True, ctx=ctx)
        assert isinstance(second, str)
        assert "<name>beta</name>" in second
