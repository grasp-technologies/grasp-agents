"""Tests: per-agent ``SkillFilter`` scopes the session-shared skill catalog."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from grasp_agents.agent.agent_context import AgentContext
from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.agent.llm_agent_transcript import LLMAgentTranscript
from grasp_agents.session_context import SessionContext
from grasp_agents.skills import (
    Skill,
    SkillFilter,
    SkillFrontmatter,
    SkillRegistry,
    list_skills,
    load_skill,
)
from grasp_agents.tools.function_tool import function_tool
from grasp_agents.types.events import ToolErrorInfo
from tests.durability.test_sessions import (  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateUsage]
    MockLLM,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class _State(BaseModel):
    pass


def _skill(name: str, *, description: str = "desc", body: str = "Body.") -> Skill:
    fm = SkillFrontmatter(name=name, description=description)
    return Skill(frontmatter=fm, body=body, path=Path(f"/skills/{name}/SKILL.md"))


def _write_skill(
    root: Path, name: str, description: str, *, body: str = "Body."
) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )


def _registry() -> SkillRegistry:
    return SkillRegistry([_skill("alpha"), _skill("beta")])


def _agent_ctx(skill_filter: SkillFilter | None) -> AgentContext:
    return AgentContext.create(
        transcript=LLMAgentTranscript(), tools={}, skill_filter=skill_filter
    )


def _make_agent(
    *,
    name: str = "filt_agent",
    skill_include: Iterable[str] | None = None,
    skill_exclude: Iterable[str] | None = None,
) -> LLMAgent[str, str, _State]:
    return LLMAgent[str, str, _State](
        name=name,
        llm=MockLLM(responses_queue=[]),
        stream_llm=True,
        env_info=False,
        enable_skills=True,
        skill_include=skill_include,
        skill_exclude=skill_exclude,
    )


async def _prompt(
    agent: LLMAgent[str, str, _State], ctx: SessionContext[_State]
) -> str | None:
    agent.on_adopted(ctx=ctx)
    return await agent.build_system_prompt(exec_id="e1")


# ---------- SkillFilter value object ----------


class TestSkillFilter:
    def test_build_is_none_when_unset(self) -> None:
        assert SkillFilter.build() is None
        assert SkillFilter.build(None, None) is None

    def test_include_is_an_allowlist(self) -> None:
        f = SkillFilter.build(include=["a", "b"])
        assert f is not None
        assert f.allows("a")
        assert f.allows("b")
        assert not f.allows("c")

    def test_exclude_is_a_blocklist(self) -> None:
        f = SkillFilter.build(exclude=["a"])
        assert f is not None
        assert not f.allows("a")
        assert f.allows("b")

    def test_both_apply_intersection(self) -> None:
        f = SkillFilter.build(include=["a", "b"], exclude=["b"])
        assert f is not None
        assert f.allows("a")
        assert not f.allows("b")  # exclude wins over include

    def test_empty_include_blocks_everything(self) -> None:
        f = SkillFilter.build(include=[])
        assert f is not None
        assert not f.allows("a")
        assert f.apply([_skill("a"), _skill("b")]) == []

    def test_apply_filters_skill_list(self) -> None:
        f = SkillFilter.build(include=["a"])
        assert f is not None
        assert [s.name for s in f.apply([_skill("a"), _skill("b")])] == ["a"]


# ---------- function_tool agent_ctx pass-through ----------


class TestFunctionToolAgentCtx:
    def test_agent_ctx_excluded_from_input_schema(self) -> None:
        @function_tool
        async def with_actx(x: int, *, agent_ctx: Any = None) -> int:
            del agent_ctx
            return x

        # ``agent_ctx`` is an executor pass-through, never an LLM-facing arg.
        assert "agent_ctx" not in with_actx.in_type.model_fields
        assert "x" in with_actx.in_type.model_fields

    def test_load_skill_does_not_expose_agent_ctx_to_model(self) -> None:
        assert "agent_ctx" not in load_skill.in_type.model_fields
        assert "agent_ctx" not in list_skills.in_type.model_fields


# ---------- Catalog section honors the per-agent filter ----------


class TestSectionRespectsFilter:
    @pytest.mark.asyncio
    async def test_no_filter_sees_all(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        prompt = await _prompt(_make_agent(), ctx)
        assert prompt is not None
        assert "<name>alpha</name>" in prompt
        assert "<name>beta</name>" in prompt

    @pytest.mark.asyncio
    async def test_include_scopes_catalog(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        prompt = await _prompt(_make_agent(skill_include=["alpha"]), ctx)
        assert prompt is not None
        assert "<name>alpha</name>" in prompt
        assert "<name>beta</name>" not in prompt

    @pytest.mark.asyncio
    async def test_exclude_scopes_catalog(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        prompt = await _prompt(_make_agent(skill_exclude=["alpha"]), ctx)
        assert prompt is not None
        assert "<name>alpha</name>" not in prompt
        assert "<name>beta</name>" in prompt

    @pytest.mark.asyncio
    async def test_two_agents_share_one_registry_with_different_views(self) -> None:
        # The headline case: one session-shared catalog, per-agent views.
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        a = _make_agent(name="agent_a", skill_include=["alpha"])
        b = _make_agent(name="agent_b", skill_include=["beta"])
        pa = await _prompt(a, ctx)
        pb = await _prompt(b, ctx)
        assert pa is not None
        assert pb is not None
        assert "<name>alpha</name>" in pa
        assert "<name>beta</name>" not in pa
        assert "<name>beta</name>" in pb
        assert "<name>alpha</name>" not in pb


# ---------- load_skill / list_skills honor the filter ----------


class TestToolsRespectFilter:
    @pytest.mark.asyncio
    async def test_load_skill_rejects_out_of_filter(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        actx = _agent_ctx(SkillFilter.build(include=["alpha"]))
        result = await load_skill.run(
            load_skill.in_type(name="beta"), ctx=ctx, agent_ctx=actx
        )
        assert isinstance(result, ToolErrorInfo)
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_load_skill_allows_in_filter(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "Alpha desc", body="ALPHA BODY")
        _write_skill(tmp_path, "beta", "Beta desc", body="BETA BODY")
        ctx: SessionContext[_State] = SessionContext(
            state=_State(), skills=SkillRegistry.from_path(tmp_path)
        )
        actx = _agent_ctx(SkillFilter.build(include=["alpha"]))
        ok = await load_skill.run(
            load_skill.in_type(name="alpha"), ctx=ctx, agent_ctx=actx
        )
        assert isinstance(ok, str)
        assert "ALPHA BODY" in ok
        rejected = await load_skill.run(
            load_skill.in_type(name="beta"), ctx=ctx, agent_ctx=actx
        )
        assert isinstance(rejected, ToolErrorInfo)

    @pytest.mark.asyncio
    async def test_list_skills_applies_filter(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        actx = _agent_ctx(SkillFilter.build(include=["alpha"]))
        result = await list_skills.run(list_skills.in_type(), ctx=ctx, agent_ctx=actx)
        assert isinstance(result, str)
        assert "<name>alpha</name>" in result
        assert "<name>beta</name>" not in result

    @pytest.mark.asyncio
    async def test_list_skills_without_filter_shows_all(self) -> None:
        ctx: SessionContext[_State] = SessionContext(state=_State(), skills=_registry())
        actx = _agent_ctx(None)
        result = await list_skills.run(list_skills.in_type(), ctx=ctx, agent_ctx=actx)
        assert isinstance(result, str)
        assert "<name>alpha</name>" in result
        assert "<name>beta</name>" in result
