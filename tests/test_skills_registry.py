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


# ---------- refresh semantics ----------


def _skill(name: str, path: Path) -> Skill:
    return Skill(
        frontmatter=SkillFrontmatter(name=name, description=f"skill {name}"),
        body=f"body of {name}",
        path=path,
    )


def _write_skill_dir(parent: Path, name: str) -> Path:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: skill {name}\n---\nbody of {name}\n",
        encoding="utf-8",
    )
    return skill_dir


class TestSkillsRefresh:
    def test_refresh_keeps_programmatic_skills(self, tmp_path: Path) -> None:
        source = tmp_path / "skills"
        _write_skill_dir(source, "disk-skill")

        registry = SkillRegistry()
        registry.add_source(source)
        registry.register(_skill("code-skill", tmp_path / "virtual" / "SKILL.md"))

        registry.refresh()

        assert "code-skill" in registry
        assert "disk-skill" in registry

    def test_refresh_survives_vanished_source(self, tmp_path: Path) -> None:
        source = tmp_path / "skills"
        _write_skill_dir(source, "disk-skill")

        registry = SkillRegistry()
        registry.add_source(source)
        registry.register(_skill("code-skill", tmp_path / "virtual" / "SKILL.md"))

        # The source directory disappears between sessions.
        import shutil

        shutil.rmtree(source)

        registry.refresh()  # no raise

        assert "code-skill" in registry
        assert "disk-skill" not in registry
