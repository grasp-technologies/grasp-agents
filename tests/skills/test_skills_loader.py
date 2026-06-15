"""Tests for the SKILL.md loader and frontmatter schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from grasp_agents.skills import (
    Skill,
    SkillFormatError,
    SkillFrontmatter,
    discover_skills,
    load_skill_md,
    parse_skill_md,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------- SkillFrontmatter validation ----------


class TestSkillFrontmatter:
    def test_minimal_valid(self) -> None:
        fm = SkillFrontmatter.model_validate(
            {"name": "pdf-tools", "description": "Process PDF documents."}
        )
        assert fm.name == "pdf-tools"
        assert fm.description == "Process PDF documents."
        assert fm.metadata == {}
        assert fm.allowed_tools is None

    def test_all_optional_fields(self) -> None:
        fm = SkillFrontmatter.model_validate(
            {
                "name": "pdf-tools",
                "description": "Process PDFs.",
                "license": "Apache-2.0",
                "compatibility": "Requires uv and Python 3.11+",
                "metadata": {"author": "me", "grasp": {"inject_body": True}},
                "allowed-tools": "Bash(git:*) Read",
            }
        )
        assert fm.license == "Apache-2.0"
        assert fm.compatibility.startswith("Requires uv")  # type: ignore[union-attr]
        assert fm.metadata["author"] == "me"
        assert fm.allowed_tools == "Bash(git:*) Read"
        assert fm.inject_body is True

    @pytest.mark.parametrize(
        "name",
        [
            "pdf-processing",
            "data-analysis",
            "code-review",
            "a",
            "skill1",
            "a-1-b",
            "a" * 64,
        ],
    )
    def test_valid_names(self, name: str) -> None:
        fm = SkillFrontmatter.model_validate({"name": name, "description": "x"})
        assert fm.name == name

    @pytest.mark.parametrize(
        "name",
        [
            "PDF-processing",  # uppercase
            "-pdf",  # leading hyphen
            "pdf-",  # trailing hyphen
            "pdf--processing",  # consecutive hyphens
            "pdf processing",  # space
            "pdf_processing",  # underscore
            "",  # empty
            "a" * 65,  # too long
        ],
    )
    def test_invalid_names(self, name: str) -> None:
        with pytest.raises(ValidationError):
            SkillFrontmatter.model_validate({"name": name, "description": "x"})

    def test_description_required(self) -> None:
        with pytest.raises(ValidationError):
            SkillFrontmatter.model_validate({"name": "x", "description": ""})

    def test_description_max_length(self) -> None:
        SkillFrontmatter.model_validate({"name": "x", "description": "d" * 1024})
        with pytest.raises(ValidationError):
            SkillFrontmatter.model_validate({"name": "x", "description": "d" * 1025})

    def test_compatibility_max_length(self) -> None:
        with pytest.raises(ValidationError):
            SkillFrontmatter.model_validate(
                {"name": "x", "description": "d", "compatibility": "y" * 501}
            )

    def test_unknown_keys_preserved(self) -> None:
        # Forward-compat: extra keys round-trip via Pydantic's extra=allow.
        fm = SkillFrontmatter.model_validate(
            {"name": "x", "description": "d", "future_field": "value"}
        )
        dumped = fm.model_dump()
        assert dumped["future_field"] == "value"

    def test_grasp_metadata_extensions(self) -> None:
        fm = SkillFrontmatter.model_validate(
            {
                "name": "x",
                "description": "d",
                "metadata": {
                    "grasp": {"disable_model_invocation": True, "inject_body": False}
                },
            }
        )
        assert fm.disable_model_invocation is True
        assert fm.inject_body is False

    def test_grasp_metadata_missing(self) -> None:
        fm = SkillFrontmatter.model_validate({"name": "x", "description": "d"})
        assert fm.disable_model_invocation is False
        assert fm.inject_body is False

    def test_grasp_metadata_non_dict(self) -> None:
        # If someone sets metadata.grasp to a non-dict, fall back to defaults.
        fm = SkillFrontmatter.model_validate(
            {"name": "x", "description": "d", "metadata": {"grasp": "not-a-dict"}}
        )
        assert fm.disable_model_invocation is False
        assert fm.inject_body is False


# ---------- parse_skill_md ----------


class TestParseSkillMd:
    def test_valid(self) -> None:
        text = "---\nname: hello\ndescription: A greeting skill.\n---\n# Hello\nBody.\n"
        fm, body = parse_skill_md(text)
        assert fm.name == "hello"
        assert "Body." in body
        assert body.startswith("# Hello")

    def test_no_frontmatter(self) -> None:
        with pytest.raises(SkillFormatError):
            parse_skill_md("Just markdown, no frontmatter.")

    def test_unterminated_frontmatter(self) -> None:
        with pytest.raises(SkillFormatError):
            parse_skill_md("---\nname: hello\ndescription: x\n# Body without close")

    def test_invalid_yaml(self) -> None:
        with pytest.raises(SkillFormatError):
            parse_skill_md("---\n: : :\n---\nbody")

    def test_yaml_not_mapping(self) -> None:
        with pytest.raises(SkillFormatError):
            parse_skill_md("---\n- one\n- two\n---\nbody")

    def test_validation_error_wrapped(self) -> None:
        with pytest.raises(SkillFormatError):
            parse_skill_md("---\nname: BAD\ndescription: x\n---\nbody")

    def test_crlf_line_endings(self) -> None:
        text = "---\r\nname: hello\r\ndescription: x\r\n---\r\nBody\r\n"
        fm, body = parse_skill_md(text)
        assert fm.name == "hello"
        assert "Body" in body

    def test_empty_body(self) -> None:
        fm, body = parse_skill_md("---\nname: hello\ndescription: d\n---\n")
        assert fm.name == "hello"
        assert not body


# ---------- load_skill_md (path-based) ----------


def _write_skill(root: Path, name: str, description: str, body: str = "Body.") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}\n",
        encoding="utf-8",
    )
    return skill_md


class TestLoadSkillMd:
    def test_load_ok(self, tmp_path: Path) -> None:
        path = _write_skill(tmp_path, "hello", "A greeting skill.")
        skill = load_skill_md(path)
        assert isinstance(skill, Skill)
        assert skill.name == "hello"
        assert "Body." in skill.body

    def test_parent_dir_name_mismatch(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "wrong-dir"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: real-name\ndescription: x\n---\n",
            encoding="utf-8",
        )
        with pytest.raises(SkillFormatError, match="parent directory"):
            load_skill_md(skill_md)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SkillFormatError):
            load_skill_md(tmp_path / "nope" / "SKILL.md")


# ---------- discover_skills (walk semantics) ----------


class TestDiscoverSkills:
    def test_explicit_skill_md_file(self, tmp_path: Path) -> None:
        path = _write_skill(tmp_path, "hello", "x")
        skills = discover_skills(path)
        assert len(skills) == 1
        assert skills[0].name == "hello"

    def test_single_skill_dir(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "hello", "x")
        skills = discover_skills(tmp_path / "hello")
        assert [s.name for s in skills] == ["hello"]

    def test_parent_dir_walks_one_level(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "alpha", "x")
        _write_skill(tmp_path, "beta", "y")
        # Decoy: subdir without SKILL.md
        (tmp_path / "decoy").mkdir()
        (tmp_path / "decoy" / "README.md").write_text("nope", encoding="utf-8")
        skills = discover_skills(tmp_path)
        assert sorted(s.name for s in skills) == ["alpha", "beta"]

    def test_failed_skill_skipped_in_walk(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_skill(tmp_path, "good", "x")
        # Bad: parent-dir-name mismatch.
        bad_dir = tmp_path / "actual-dir"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text(
            "---\nname: different-name\ndescription: x\n---\n",
            encoding="utf-8",
        )
        with caplog.at_level("ERROR"):
            skills = discover_skills(tmp_path)
        assert [s.name for s in skills] == ["good"]
        assert any("Failed to load skill" in rec.message for rec in caplog.records)

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        with pytest.raises(SkillFormatError):
            discover_skills(tmp_path / "does-not-exist")

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "hello", "x")
        skills = discover_skills(str(tmp_path / "hello"))
        assert [s.name for s in skills] == ["hello"]
