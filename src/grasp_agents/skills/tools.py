"""
Top-level skill tools and the ``attach_skills`` helper.

``load_skill`` and ``list_skills`` are global :class:`FunctionTool` instances
that read ``ctx.skills`` at call time. ``attach_skills(agent)`` adds them to
the agent's tool list and registers the skills system-prompt section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agent.function_tool import function_tool
from .injection import (
    LOAD_INSTRUCTION,
    render_available_skills_block,
    skills_system_prompt_section,
)
from .loader import parse_skill_md
from .types import SkillNotFoundError

if TYPE_CHECKING:
    from ..agent.llm_agent import LLMAgent
    from ..run_context import RunContext

LOAD_SKILL_DESCRIPTION = (
    "Load the full body (markdown instructions) of an available skill by name. "
    "Use this once you have decided a skill in the <available_skills> catalog "
    "is relevant to the user's request. Returns the skill's instructions as "
    "plain text. The skill file is read fresh from disk on each call."
)

LIST_SKILLS_DESCRIPTION = (
    "List all skills currently available to this session. Returns the same "
    "<available_skills> catalog block injected into the system prompt. Pass "
    "refresh=True to re-walk the skill source directories first — useful when "
    "you suspect new skills were authored after the session started."
)


@function_tool(name="load_skill", description=LOAD_SKILL_DESCRIPTION)
async def load_skill(  # noqa: RUF029
    name: str, *, ctx: RunContext[Any] | None = None
) -> str:
    if ctx is None or ctx.skills is None:
        raise SkillNotFoundError("No skills are configured for this run.")
    skill = ctx.skills.get_optional(name)
    if skill is None or skill.disable_model_invocation:
        raise SkillNotFoundError(
            f"Skill {name!r} is not available. "
            f"Pick one from the <available_skills> catalog."
        )
    try:
        text = skill.path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillNotFoundError(f"Skill {name!r} could not be read: {exc}") from exc
    try:
        _, body = parse_skill_md(text, path=skill.path)
    except Exception:
        return skill.body
    return body


@function_tool(name="list_skills", description=LIST_SKILLS_DESCRIPTION)
async def list_skills(  # noqa: RUF029
    refresh: bool = False, *, ctx: RunContext[Any] | None = None
) -> str:
    if ctx is None or ctx.skills is None:
        return "No skills are configured for this run."
    if refresh:
        ctx.skills.refresh()
    rendered = render_available_skills_block(ctx.skills.all)
    if not rendered:
        return "No skills available."
    return f"{rendered}\n\n{LOAD_INSTRUCTION}"


def attach_skills(agent: LLMAgent[Any, Any, Any]) -> None:
    """
    Add the skill tools and the ``<available_skills>`` system-prompt section.

    The actual skill registry lives on the :class:`RunContext` you pass to
    :meth:`LLMAgent.run` (``ctx.skills``); this helper only wires the agent's
    tools and prompt section to read from it.
    """
    tools = agent.tools
    if "load_skill" not in tools:
        tools[load_skill.name] = load_skill
    if "list_skills" not in tools:
        tools[list_skills.name] = list_skills
    agent.add_system_prompt_section(skills_system_prompt_section)
