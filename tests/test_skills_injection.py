"""Tests for the <available_skills> system-prompt injection block."""

from __future__ import annotations

from pathlib import Path

from grasp_agents.skills import Skill, SkillFrontmatter, render_available_skills_block


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
        assert "## Available skills" in block
        assert "<available_skills>" in block
        assert "<name>alpha</name>" in block
        assert "<description>Does alpha things.</description>" in block
        assert "<location>/skills/alpha/SKILL.md</location>" in block
        assert "</available_skills>" in block
        assert "load_skill" in block

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
            "<description>Use when &lt;tag&gt; &amp; "
            "similar markup</description>"
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
        block = render_available_skills_block(
            [_skill("alpha", "x", license_="MIT")]
        )
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
