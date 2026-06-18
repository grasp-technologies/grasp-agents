"""Tests for the <available_skills> system-prompt injection block."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.run_context import RunContext
from grasp_agents.skills import (
    Skill,
    SkillFrontmatter,
    SkillRegistry,
    make_skills_section,
    render_available_skills_block,
    render_skill_instructions,
)

if TYPE_CHECKING:
    from grasp_agents.context.prompt_builder import SystemPromptSection


async def _run_section(
    section: SystemPromptSection, ctx: RunContext[Any]
) -> str | None:
    """Invoke a section's compute and resolve any awaitable return."""
    result = section.compute(ctx=ctx, exec_id="test")
    if inspect.isawaitable(result):
        result = await result
    return result


def _skill(
    name: str,
    description: str,
    *,
    body: str = "body",
    inject_body: bool = False,
    disabled: bool = False,
    path: Path | None = None,
    license_: str | None = None,
    compatibility: str | None = None,
    allowed_tools: str | None = None,
) -> Skill:
    grasp: dict[str, object] = {}
    if inject_body:
        grasp["inject_body"] = True
    if disabled:
        grasp["disable_model_invocation"] = True
    fm_data: dict[str, object] = {"name": name, "description": description}
    if license_ is not None:
        fm_data["license"] = license_
    if compatibility is not None:
        fm_data["compatibility"] = compatibility
    if allowed_tools is not None:
        fm_data["allowed-tools"] = allowed_tools
    if grasp:
        fm_data["metadata"] = {"grasp": grasp}
    fm = SkillFrontmatter.model_validate(fm_data)
    return Skill(
        frontmatter=fm,
        body=body,
        path=path or Path(f"/skills/{name}/SKILL.md"),
    )


class TestRenderAvailableSkillsBlock:
    def test_empty_list(self) -> None:
        assert not render_available_skills_block([])

    def test_all_disabled_returns_empty(self) -> None:
        block = render_available_skills_block(
            [
                _skill("hidden-a", "x", disabled=True),
                _skill("hidden-b", "y", disabled=True),
            ]
        )
        assert not block

    def test_single_visible_skill(self) -> None:
        block = render_available_skills_block([_skill("alpha", "Does alpha things.")])
        # The catalog renders the XML element only; the surrounding
        # instructions ("## Skills", "use load_skill ...") live in the
        # separate ``render_skill_instructions`` block.
        assert "## Available skills" not in block
        assert "<available_skills>" in block
        assert "<name>alpha</name>" in block
        assert "<description>Does alpha things.</description>" in block
        # <location> intentionally omitted: skills load by name, and the
        # rendered host path leaked an absolute deploy path of no use.
        assert "<location>" not in block
        assert "</available_skills>" in block

    def test_disabled_skills_filtered(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "x"), _skill("hidden", "y", disabled=True)]
        )
        assert "<name>alpha</name>" in block
        assert "<name>hidden</name>" not in block

    def test_multiple_skills_preserve_order(self) -> None:
        block = render_available_skills_block(
            [_skill("first", "1st"), _skill("second", "2nd"), _skill("third", "3rd")]
        )
        i1 = block.index("<name>first</name>")
        i2 = block.index("<name>second</name>")
        i3 = block.index("<name>third</name>")
        assert i1 < i2 < i3

    def test_inject_body_appends_body(self) -> None:
        body = "Always-on body content."
        block = render_available_skills_block(
            [_skill("alpha", "x", body=body, inject_body=True)]
        )
        assert "<name>alpha</name>" in block
        assert "## Skill: alpha" in block
        assert body in block
        # Body must appear AFTER the </available_skills> tag.
        assert block.index("</available_skills>") < block.index(body)

    def test_inject_body_only_for_eager_skills(self) -> None:
        block = render_available_skills_block(
            [
                _skill("eager", "x", body="eager body", inject_body=True),
                _skill("lazy", "y", body="lazy body"),
            ]
        )
        assert "eager body" in block
        assert "lazy body" not in block

    def test_xml_special_chars_escaped(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "Use when <tag> & similar markup")]
        )
        expected = (
            "<description>Use when &lt;tag&gt; &amp; similar markup</description>"
        )
        assert expected in block
        # The actual XML wrapper tags must remain intact.
        assert "<available_skills>" in block
        assert "</available_skills>" in block

    def test_compatibility_surfaced_when_present(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "x", compatibility="Requires Python 3.11+")]
        )
        assert "<compatibility>Requires Python 3.11+</compatibility>" in block

    def test_compatibility_omitted_when_absent(self) -> None:
        block = render_available_skills_block([_skill("alpha", "x")])
        assert "<compatibility>" not in block

    def test_allowed_tools_surfaced_when_present(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "x", allowed_tools="Bash(git:*) Read")]
        )
        assert "<allowed-tools>Bash(git:*) Read</allowed-tools>" in block

    def test_allowed_tools_omitted_when_absent(self) -> None:
        block = render_available_skills_block([_skill("alpha", "x")])
        assert "<allowed-tools>" not in block

    def test_license_omitted_by_default(self) -> None:
        block = render_available_skills_block([_skill("alpha", "x", license_="MIT")])
        assert "<license>" not in block

    def test_license_surfaced_with_include_license(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "x", license_="MIT")],
            include_license=True,
        )
        assert "<license>MIT</license>" in block

    def test_license_omitted_with_include_license_when_absent(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "x")],
            include_license=True,
        )
        assert "<license>" not in block

    def test_compatibility_special_chars_escaped(self) -> None:
        block = render_available_skills_block(
            [_skill("alpha", "x", compatibility="needs <psql> & jq")]
        )
        assert "<compatibility>needs &lt;psql&gt; &amp; jq</compatibility>" in block


class TestRenderSkillInstructions:
    def test_non_empty(self) -> None:
        text = render_skill_instructions()
        assert text

    def test_describes_load_loop(self) -> None:
        text = render_skill_instructions()
        # The instructions teach the agent to call load_skill — that's
        # the one promise the section must keep, since the catalog no
        # longer carries the load hint itself.
        assert "load_skill" in text

    def test_distinguishes_skills_from_memory(self) -> None:
        text = render_skill_instructions()
        # The "Skills vs memory" block guards against the agent reaching
        # for skills when it should reach for memory. Plans / tasks are
        # deliberately not referenced — those subsystems don't exist yet.
        assert "memory" in text.lower()
        assert "plan" not in text.lower()
        assert "tasks" not in text.lower()


class TestMakeSkillsSection:
    def _ctx(self, skills: list[Skill] | None = None) -> RunContext[Any]:
        registry: SkillRegistry | None
        registry = None if skills is None else SkillRegistry(skills)
        return RunContext[Any](skills=registry)

    @pytest.mark.asyncio
    async def test_no_skills_attr_returns_none(self) -> None:
        result = await _run_section(make_skills_section(), self._ctx())
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_registry_returns_none(self) -> None:
        result = await _run_section(make_skills_section(), self._ctx([]))
        assert result is None

    @pytest.mark.asyncio
    async def test_all_disabled_returns_none(self) -> None:
        result = await _run_section(
            make_skills_section(),
            self._ctx(
                [
                    _skill("hidden-a", "x", disabled=True),
                    _skill("hidden-b", "y", disabled=True),
                ]
            ),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_visible_skill_renders_both_blocks(self) -> None:
        result = await _run_section(
            make_skills_section(),
            self._ctx([_skill("alpha", "Does alpha things.")]),
        )
        assert result is not None
        # Instructions block (above) — agent-facing how-to.
        assert "# Skills" in result
        assert "load_skill" in result
        # Catalog block (below) — XML data.
        assert "<available_skills>" in result
        assert "<name>alpha</name>" in result
        # Order: instructions heading first, catalog payload second.
        assert result.index("# Skills") < result.index("<name>alpha</name>")

    @pytest.mark.asyncio
    async def test_section_name_overridable(self) -> None:
        section = make_skills_section(section_name="my_skills")
        assert section.name == "my_skills"
