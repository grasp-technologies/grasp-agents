"""Tests for SkillRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.skills import (
    Skill,
    SkillFrontmatter,
    SkillNotFoundError,
    SkillRegistry,
)


def _make_skill(
    name: str, *, disabled: bool = False, path: Path | None = None
) -> Skill:
    grasp: dict[str, object] = {}
    if disabled:
        grasp["disable_model_invocation"] = True
    fm = SkillFrontmatter.model_validate(
        {
            "name": name,
            "description": f"Skill {name}",
            **({"metadata": {"grasp": grasp}} if grasp else {}),
        }
    )
    return Skill(
        frontmatter=fm,
        body=f"body of {name}",
        path=path or Path(f"/skills/{name}/SKILL.md"),
    )


class TestSkillRegistry:
    def test_empty(self) -> None:
        reg = SkillRegistry()
        assert len(reg) == 0
        assert reg.all == []
        assert reg.visible == []
        assert "anything" not in reg
        assert reg.get_optional("nope") is None
        with pytest.raises(SkillNotFoundError):
            reg.get("nope")

    def test_register_and_get(self) -> None:
        reg = SkillRegistry([_make_skill("alpha")])
        assert "alpha" in reg
        assert reg.get("alpha").name == "alpha"
        assert reg.get_optional("alpha") is not None
        assert len(reg) == 1

    def test_iteration_order_preserved(self) -> None:
        reg = SkillRegistry([_make_skill(n) for n in ["c", "a", "b"]])
        assert [s.name for s in reg] == ["c", "a", "b"]

    def test_visible_excludes_disabled(self) -> None:
        reg = SkillRegistry(
            [
                _make_skill("alpha"),
                _make_skill("hidden", disabled=True),
                _make_skill("beta"),
            ]
        )
        assert [s.name for s in reg.all] == ["alpha", "hidden", "beta"]
        assert [s.name for s in reg.visible] == ["alpha", "beta"]

    def test_duplicate_name_warns_and_overrides(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = _make_skill("dup", path=Path("/a/dup/SKILL.md"))
        second = _make_skill("dup", path=Path("/b/dup/SKILL.md"))
        with caplog.at_level("WARNING"):
            reg = SkillRegistry([first, second])
        assert len(reg) == 1
        assert reg.get("dup").path == Path("/b/dup/SKILL.md")
        assert any("registered twice" in rec.message for rec in caplog.records)

    def test_register_same_path_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = Path("/a/dup/SKILL.md")
        reg = SkillRegistry()
        with caplog.at_level("WARNING"):
            reg.register(_make_skill("dup", path=path))
            reg.register(_make_skill("dup", path=path))
        assert not any("registered twice" in rec.message for rec in caplog.records)
