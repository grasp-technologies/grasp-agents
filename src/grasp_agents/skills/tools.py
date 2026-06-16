"""
Top-level skill tools ``load_skill`` and ``list_skills``.

Both are global :class:`FunctionTool` instances that read ``ctx.skills`` at
call time. Add them to an agent the same way you add any other tool — pass
them in the ``tools=[...]`` ctor kwarg. The ``skills`` system-prompt
section auto-attaches on every ``LLMAgent`` and consults ``ctx.skills``;
no separate ``attach_*`` step is needed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from grasp_agents.tools.function_tool import function_tool

from .injection import LOAD_INSTRUCTION, render_available_skills_block
from .loader import parse_skill_md
from .registry import substitute_args
from .types import SkillNotFoundError

if TYPE_CHECKING:
    from grasp_agents.run_context import RunContext

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
async def load_skill(
    name: Annotated[
        str, Field(description="Exact skill name from the <available_skills> catalog.")
    ],
    *,
    ctx: RunContext[Any] | None = None,
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
        text = await asyncio.to_thread(skill.path.read_text, encoding="utf-8")
    except OSError as exc:
        raise SkillNotFoundError(f"Skill {name!r} could not be read: {exc}") from exc
    try:
        _, body = parse_skill_md(text, path=skill.path)
    except Exception:
        body = skill.body

    return substitute_args(body, None)


@function_tool(name="list_skills", description=LIST_SKILLS_DESCRIPTION)
async def list_skills(
    refresh: Annotated[
        bool,
        Field(
            description="Re-walk the skill source directories first to pick up "
            "skills authored after the session started."
        ),
    ] = False,
    *,
    ctx: RunContext[Any] | None = None,
) -> str:
    if ctx is None or ctx.skills is None:
        return "No skills are configured for this run."
    if refresh:
        # The re-walk hits the filesystem — keep it off the event loop.
        await asyncio.to_thread(ctx.skills.refresh)
    rendered = render_available_skills_block(ctx.skills.all, include_license=True)
    if not rendered:
        return "No skills available."
    return f"{rendered}\n\n{LOAD_INSTRUCTION}"
