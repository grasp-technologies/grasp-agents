from __future__ import annotations

from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

from ..agent.prompt_builder import SystemPromptSection

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..run_context import RunContext
    from .types import Skill

LOAD_INSTRUCTION = (
    "To load the full body of a skill, call the `load_skill` tool with its `name`."
)


def _xml_escape(value: str) -> str:
    return escape(value)


def render_available_skills_block(
    skills: Sequence[Skill], *, include_license: bool = False
) -> str:
    """
    Render the ``<available_skills>`` XML catalog injected into the system prompt.

    Per-skill emitted elements: ``<name>``, ``<description>``, ``<location>``,
    plus ``<compatibility>`` and ``<allowed-tools>`` when set on the
    frontmatter (informational — ``allowed-tools`` is *not* enforced here;
    see :class:`BeforeToolHook` for the authoritative permission gate).
    ``<license>`` is emitted only when ``include_license=True`` — the
    system-prompt section keeps it off to stay lean; ``list_skills`` (the
    tool) turns it on.

    Skills with ``metadata.grasp.inject_body == true`` get their body
    appended after the catalog. Skills with
    ``disable_model_invocation == true`` are excluded.
    """
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""

    lines: list[str] = ["## Available skills", "", "<available_skills>"]
    for skill in visible:
        skill_lines: list[str] = [
            "  <skill>",
            f"    <name>{_xml_escape(skill.name)}</name>",
            f"    <description>{_xml_escape(skill.description)}</description>",
            f"    <location>{_xml_escape(str(skill.path))}</location>",
        ]
        if skill.frontmatter.compatibility:
            skill_lines.append(
                "    <compatibility>"
                f"{_xml_escape(skill.frontmatter.compatibility)}"
                "</compatibility>"
            )
        if skill.frontmatter.allowed_tools:
            skill_lines.append(
                "    <allowed-tools>"
                f"{_xml_escape(skill.frontmatter.allowed_tools)}"
                "</allowed-tools>"
            )
        if include_license and skill.frontmatter.license:
            skill_lines.append(
                f"    <license>{_xml_escape(skill.frontmatter.license)}</license>"
            )
        skill_lines.append("  </skill>")
        lines.extend(skill_lines)
    lines.extend(["</available_skills>", "", LOAD_INSTRUCTION])

    for skill in visible:
        if skill.inject_body:
            lines.extend(["", f"## Skill: {skill.name}", "", skill.body])

    return "\n".join(lines)


SKILLS_SECTION_NAME = "skills"


async def _compute_skills_section(
    *, ctx: RunContext[Any] | None = None, exec_id: str | None = None
) -> str | None:
    if ctx is None or ctx.skills is None or len(ctx.skills) == 0:
        return None
    skills = await ctx.skills.apply_filter(ctx=ctx, exec_id=exec_id)
    rendered = render_available_skills_block(skills)
    return rendered or None


skills_system_prompt_section = SystemPromptSection(
    name=SKILLS_SECTION_NAME,
    compute=_compute_skills_section,
)
