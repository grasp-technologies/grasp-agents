"""
Regression tests for the P2 skills fixes
(consolidated audit 2026-06-11, §3 item 24; item 23 resolved as won't-fix —
memory is the agent's own trusted notes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grasp_agents.skills.registry import SkillRegistry
from grasp_agents.skills.types import Skill, SkillFrontmatter

pytestmark = pytest.mark.anyio


# ---------- Item 24: skills refresh semantics ----------


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
