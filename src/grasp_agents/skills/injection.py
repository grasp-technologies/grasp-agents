from __future__ import annotations

from importlib.resources import files
from typing import TYPE_CHECKING, Any
from xml.sax.saxutils import escape

from ..agent.prompt_builder import SystemPromptSection

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..run_context import RunContext
    from .types import Skill

SKILLS_SECTION_NAME = "skills"

SKILL_INSTRUCTIONS = (
    files(__package__ or __name__)
    .joinpath("skill_instructions.md")
    .read_text(encoding="utf-8")
)

LOAD_INSTRUCTION = (
    "To load the full body of a skill, call the `load_skill` tool with its `name`."
)


def _xml_escape(value: str) -> str:
    return escape(value)


def render_skill_instructions() -> str:
    """
    Render the skill-instructions sub-block.

    Static, cache-stable copy describing how to choose, load, and follow a
    skill. Sits above the ``<available_skills>`` catalog in the system
    prompt so the agent reads the discipline before the data.
    """
    return SKILL_INSTRUCTIONS.rstrip()


def render_available_skills_block(
    skills: Sequence[Skill], *, include_license: bool = False
) -> str:
    """
    Render the ``<available_skills>`` XML catalog.

    Per-skill emitted elements: ``<name>`` and ``<description>``, plus
    ``<compatibility>`` and ``<allowed-tools>`` when set on the
    frontmatter (informational — ``allowed-tools`` is *not* enforced
    here; see :class:`BeforeToolHook` for the authoritative permission
    gate). ``<license>`` is emitted only when
    ``include_license=True`` — the system-prompt section keeps it off to
    stay lean; ``list_skills`` (the tool) turns it on.

    Skills with ``metadata.grasp.inject_body == true`` get their body
    appended after the catalog. Skills with
    ``disable_model_invocation == true`` are excluded.

    The output is just the XML (and any injected bodies) — instructions
    on how to invoke skills live in :func:`render_skill_instructions`.
    """
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""

    lines: list[str] = ["<available_skills>"]
    for skill in visible:
        skill_lines: list[str] = [
            "  <skill>",
            f"    <name>{_xml_escape(skill.name)}</name>",
            f"    <description>{_xml_escape(skill.description)}</description>",
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
    lines.append("</available_skills>")

    for skill in visible:
        if skill.inject_body:
            lines.extend(["", f"## Skill: {skill.name}", "", skill.body])

    return "\n".join(lines)


def make_skills_section(
    *, section_name: str = SKILLS_SECTION_NAME
) -> SystemPromptSection:
    """
    Build a skills :class:`SystemPromptSection`.

    Emits two sub-blocks when ``ctx.skills`` has at least one visible
    skill: the instructions block (how to choose / load / follow a skill)
    and the ``<available_skills>`` catalog. The catalog is cache-stable —
    per-turn relevance selection belongs on an :class:`InputAttachment`,
    not here.

    Returns ``None`` when ``ctx.skills`` is unset or every registered
    skill has ``disable_model_invocation = true`` (the model would have
    nothing to load, so the instructions become misleading).
    """

    async def compute(  # noqa: RUF029
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        **_: Any,
    ) -> str | None:
        del exec_id
        if ctx is None or ctx.skills is None:
            return None
        visible = ctx.skills.visible
        if not visible:
            return None
        catalog = render_available_skills_block(visible)
        if not catalog:
            return None
        return f"{render_skill_instructions()}\n\n{catalog}"

    return SystemPromptSection(name=section_name, compute=compute)


skills_system_prompt_section = make_skills_section()
