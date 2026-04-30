"""End-to-end tests: skills attached to LLMAgent via RunContext."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.run_context import RunContext
from grasp_agents.skills import (
    SkillRegistry,
    attach_skills,
    list_skills,
    load_skill,
)
from grasp_agents.types.events import ToolErrorInfo

from .test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
)

if TYPE_CHECKING:
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
        import yaml  # noqa: PLC0415

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


def _make_agent() -> LLMAgent[str, str, _State]:
    return LLMAgent[str, str, _State](
        name="skills_test_agent",
        llm=MockLLM(responses_queue=[]),
        stream_llm=True,
    )


async def _build_system_prompt(
    agent: LLMAgent[str, str, _State], ctx: RunContext[_State]
) -> str | None:
    return await agent.build_system_prompt(ctx, exec_id="e1")


# ---------- attach_skills wiring ----------


class TestAttachSkills:
    def test_no_skills_no_section_no_tools(self) -> None:
        agent = _make_agent()
        assert "load_skill" not in agent.tools
        assert "list_skills" not in agent.tools
        assert agent.system_prompt_sections == ()

    def test_attach_adds_tools_and_section(self) -> None:
        agent = _make_agent()
        attach_skills(agent)
        assert "load_skill" in agent.tools
        assert "list_skills" in agent.tools
        assert any(s.name == "skills" for s in agent.system_prompt_sections)

    def test_attach_is_idempotent(self) -> None:
        agent = _make_agent()
        attach_skills(agent)
        attach_skills(agent)
        # Tools list intact, but section may be re-registered (consumer-controlled).
        assert "load_skill" in agent.tools
        assert "list_skills" in agent.tools


# ---------- System prompt section ----------


class TestSystemPromptSection:
    @pytest.mark.anyio
    async def test_empty_registry_no_block(self, tmp_path: Path) -> None:
        del tmp_path
        agent = _make_agent()
        attach_skills(agent)
        ctx: RunContext[_State] = RunContext(state=_State())
        # No ctx.skills set
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is None

    @pytest.mark.anyio
    async def test_section_renders_catalog(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha skill description.")
        agent = _make_agent()
        attach_skills(agent)
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            skills=SkillRegistry.from_path(tmp_path),
        )
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert "<available_skills>" in prompt
        assert "<name>alpha</name>" in prompt
        assert "Alpha skill description." in prompt

    @pytest.mark.anyio
    async def test_combines_with_user_sys_prompt(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        agent = LLMAgent[str, str, _State](
            name="combo",
            llm=MockLLM(responses_queue=[]),
            sys_prompt="You are a helper.",
            stream_llm=True,
        )
        attach_skills(agent)
        ctx: RunContext[_State] = RunContext(
            state=_State(),
            skills=SkillRegistry.from_path(tmp_path),
        )
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert prompt.startswith("You are a helper.")
        assert "<available_skills>" in prompt

    @pytest.mark.anyio
    async def test_combines_with_dynamic_builder(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        agent = _make_agent()
        attach_skills(agent)

        def dyn(*, ctx: RunContext[_State], exec_id: str) -> str:
            del ctx, exec_id
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


# ---------- Relevance filter hook ----------


class TestRelevanceFilter:
    @pytest.mark.anyio
    async def test_sync_filter(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha")
        _write_skill(tmp_path, "beta", "Beta")
        registry = SkillRegistry.from_path(tmp_path)

        def keep_alpha(*, skills: list[Any], **_: Any) -> list[Any]:
            return [s for s in skills if s.name == "alpha"]

        registry.set_filter(keep_alpha)

        agent = _make_agent()
        attach_skills(agent)
        ctx: RunContext[_State] = RunContext(state=_State(), skills=registry)
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert "<name>alpha</name>" in prompt
        assert "<name>beta</name>" not in prompt

    @pytest.mark.anyio
    async def test_async_filter_receives_ctx(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha")
        _write_skill(tmp_path, "beta", "Beta")
        registry = SkillRegistry.from_path(tmp_path)

        seen_ctx: list[RunContext[_State] | None] = []

        async def filter_fn(  # noqa: RUF029
            *,
            skills: list[Any],
            ctx: RunContext[_State] | None = None,
            exec_id: str | None = None,
        ) -> list[Any]:
            del exec_id
            seen_ctx.append(ctx)
            return [s for s in skills if s.name != "beta"]

        registry.set_filter(filter_fn)

        agent = _make_agent()
        attach_skills(agent)
        ctx: RunContext[_State] = RunContext(state=_State(), skills=registry)
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is not None
        assert "<name>beta</name>" not in prompt
        assert seen_ctx == [ctx]

    @pytest.mark.anyio
    async def test_filter_returning_empty_drops_section(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha")
        registry = SkillRegistry.from_path(tmp_path)

        def empty_filter(**_: Any) -> list[Any]:
            return []

        registry.set_filter(empty_filter)

        agent = _make_agent()
        attach_skills(agent)
        ctx: RunContext[_State] = RunContext(state=_State(), skills=registry)
        prompt = await _build_system_prompt(agent, ctx)
        assert prompt is None


# ---------- load_skill / list_skills tools ----------


class TestLoadSkillTool:
    @pytest.mark.anyio
    async def test_returns_body(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x", body="ALPHA BODY")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await load_skill(name="alpha", ctx=ctx)
        assert isinstance(result, str)
        assert "ALPHA BODY" in result

    @pytest.mark.anyio
    async def test_no_registry_errors(self) -> None:
        ctx: RunContext[_State] = RunContext(state=_State())
        result = await load_skill(name="nope", ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "no skills" in result.error.lower()

    @pytest.mark.anyio
    async def test_unknown_skill_errors(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await load_skill(name="nonexistent", ctx=ctx)
        assert isinstance(result, ToolErrorInfo)
        assert "not available" in result.error.lower()

    @pytest.mark.anyio
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
    @pytest.mark.anyio
    async def test_no_registry(self) -> None:
        ctx: RunContext[_State] = RunContext(state=_State())
        result = await list_skills(ctx=ctx)
        assert isinstance(result, str)
        assert "no skills" in result.lower()

    @pytest.mark.anyio
    async def test_returns_catalog(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha skill.")
        ctx: RunContext[_State] = RunContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        result = await list_skills(ctx=ctx)
        assert isinstance(result, str)
        assert "<available_skills>" in result
        assert "<name>alpha</name>" in result

    @pytest.mark.anyio
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
