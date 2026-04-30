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
) -> Skill:
    grasp: dict[str, object] = {}
    if inject_body:
        grasp["inject_body"] = True
    if disabled:
        grasp["disable_model_invocation"] = True
    fm_data: dict[str, object] = {"name": name, "description": description}
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
