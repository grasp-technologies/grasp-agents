"""Tests for the memdir loader and frontmatter schema."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from grasp_agents.memory import (
    INDEX_FILE_NAME,
    MAX_INDEX_BYTES,
    MAX_INDEX_LINES,
    MEMORY_TYPES,
    MemoryFormatError,
    MemoryFrontmatter,
    load_memory_entry,
    parse_memory_md,
    scan_memdir,
)
from grasp_agents.memory.loader import truncate_index

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _topic(
    name: str,
    description: str,
    type_: str | None = None,
    body: str = "x",
) -> str:
    type_line = f"\ntype: {type_}" if type_ else ""
    return f"---\nname: {name}\ndescription: {description}{type_line}\n---\n{body}\n"


# ---------- MemoryFrontmatter ----------


class TestMemoryFrontmatter:
    def test_minimal_valid(self) -> None:
        fm = MemoryFrontmatter.model_validate(
            {"name": "user_profile", "description": "Who the user is."}
        )
        assert fm.name == "user_profile"
        assert fm.description == "Who the user is."
        assert fm.memory_type is None
        assert fm.metadata == {}

    def test_typed_user(self) -> None:
        fm = MemoryFrontmatter.model_validate(
            {"name": "user_pref", "description": "Style.", "type": "user"}
        )
        assert fm.memory_type == "user"

    @pytest.mark.parametrize("t", MEMORY_TYPES)
    def test_all_known_types(self, t: str) -> None:
        fm = MemoryFrontmatter.model_validate(
            {"name": "x", "description": "y", "type": t}
        )
        assert fm.memory_type == t

    def test_unknown_type_graceful(self) -> None:
        fm = MemoryFrontmatter.model_validate(
            {"name": "x", "description": "y", "type": "future_kind"}
        )
        # Raw value preserved, but memory_type degrades to None.
        assert fm.raw_type == "future_kind"
        assert fm.memory_type is None

    def test_name_regex_allows_hyphen_and_underscore(self) -> None:
        fm = MemoryFrontmatter.model_validate(
            {"name": "user-pref_v2", "description": "ok"}
        )
        assert fm.name == "user-pref_v2"

    @pytest.mark.parametrize(
        "bad",
        [
            "User",                # uppercase
            "-leading",            # leading hyphen
            "trailing-",           # trailing hyphen
            "double--hyphen",      # consecutive separators
            "double__under",       # consecutive separators
            "spaced name",         # space
            "",                    # empty
        ],
    )
    def test_name_regex_rejects(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            MemoryFrontmatter.model_validate(
                {"name": bad, "description": "ok"}
            )

    def test_name_max_length(self) -> None:
        with pytest.raises(ValidationError):
            MemoryFrontmatter.model_validate(
                {"name": "a" * 65, "description": "ok"}
            )

    def test_description_required(self) -> None:
        with pytest.raises(ValidationError):
            MemoryFrontmatter.model_validate({"name": "x"})

    def test_description_max_length(self) -> None:
        with pytest.raises(ValidationError):
            MemoryFrontmatter.model_validate(
                {"name": "x", "description": "a" * 2049}
            )

    def test_extra_fields_preserved(self) -> None:
        fm = MemoryFrontmatter.model_validate(
            {
                "name": "x",
                "description": "y",
                "metadata": {"app": "mentor"},
                "future_field": "ignored",
            }
        )
        assert fm.metadata == {"app": "mentor"}
        # extra="allow" preserves unknown fields silently
        assert fm.model_dump().get("future_field") == "ignored"


# ---------- parse_memory_md ----------


class TestParseMemoryMd:
    def test_minimal(self) -> None:
        text = "---\nname: x\ndescription: y\n---\nbody\n"
        fm, body = parse_memory_md(text)
        assert fm.name == "x"
        assert body == "body"

    def test_missing_frontmatter(self) -> None:
        with pytest.raises(MemoryFormatError):
            parse_memory_md("just text, no fm")

    def test_invalid_yaml(self) -> None:
        text = "---\nname: x\ndescription: [unterminated\n---\nbody\n"
        with pytest.raises(MemoryFormatError):
            parse_memory_md(text)

    def test_non_mapping_yaml(self) -> None:
        text = "---\n- list\n- entry\n---\nbody\n"
        with pytest.raises(MemoryFormatError):
            parse_memory_md(text)

    def test_validation_failure_propagates(self) -> None:
        text = "---\nname: BadName\ndescription: y\n---\nbody\n"
        with pytest.raises(MemoryFormatError):
            parse_memory_md(text)

    def test_body_strip(self) -> None:
        text = "---\nname: x\ndescription: y\n---\n\n\nactual body\n\n"
        _, body = parse_memory_md(text)
        assert body == "actual body"


# ---------- load_memory_entry ----------


class TestLoadMemoryEntry:
    def test_success(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "user_profile.md", _topic("user_profile", "x", body="B"))
        entry = load_memory_entry(f)
        assert entry.name == "user_profile"
        assert entry.body == "B"
        assert entry.mtime_ms > 0

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(MemoryFormatError):
            load_memory_entry(tmp_path / "nope.md")

    def test_typed_entry(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "x.md", _topic("x", "y", type_="feedback"))
        assert load_memory_entry(f).memory_type == "feedback"

    def test_unknown_type_loads(self, tmp_path: Path) -> None:
        f = _write(tmp_path / "x.md", _topic("x", "y", type_="future"))
        assert load_memory_entry(f).memory_type is None


# ---------- truncate_index ----------


class TestTruncateIndex:
    def test_under_caps_unchanged(self) -> None:
        text = "line\n" * 5
        out, truncated = truncate_index(text)
        assert out == text
        assert truncated is False

    def test_line_cap(self) -> None:
        text = "x\n" * (MAX_INDEX_LINES + 50)
        out, truncated = truncate_index(text)
        assert truncated
        assert out.count("\n") == MAX_INDEX_LINES

    def test_byte_cap(self) -> None:
        # Single very long line that beats the byte cap (one line, no '\n').
        text = "a" * (MAX_INDEX_BYTES + 100)
        out, truncated = truncate_index(text)
        assert truncated
        assert len(out.encode("utf-8")) <= MAX_INDEX_BYTES


# ---------- scan_memdir ----------


class TestScanMemdir:
    def test_empty_or_missing_dir(self, tmp_path: Path) -> None:
        index, mtime, _, entries = scan_memdir(tmp_path / "missing")
        assert index is None
        assert mtime is None
        assert entries == []

    def test_index_only(self, tmp_path: Path) -> None:
        _write(tmp_path / INDEX_FILE_NAME, "# index body\n")
        index, mtime, _, entries = scan_memdir(tmp_path)
        assert index is not None
        assert "index body" in index
        assert mtime is not None
        assert entries == []

    def test_topic_only(self, tmp_path: Path) -> None:
        _write(tmp_path / "alpha.md", _topic("alpha", "Alpha"))
        index, _, _, entries = scan_memdir(tmp_path)
        assert index is None
        assert len(entries) == 1
        assert entries[0].name == "alpha"

    def test_mixed(self, tmp_path: Path) -> None:
        _write(tmp_path / INDEX_FILE_NAME, "# idx\n")
        _write(tmp_path / "alpha.md", _topic("alpha", "Alpha"))
        _write(tmp_path / "beta.md", _topic("beta", "Beta"))
        index, _, _, entries = scan_memdir(tmp_path)
        assert index is not None
        assert {e.name for e in entries} == {"alpha", "beta"}

    def test_index_excluded_from_entries(self, tmp_path: Path) -> None:
        _write(tmp_path / INDEX_FILE_NAME, "# idx\n")
        _, _, _, entries = scan_memdir(tmp_path)
        assert all(e.path.name != INDEX_FILE_NAME for e in entries)

    def test_hidden_files_skipped(self, tmp_path: Path) -> None:
        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        _write(hidden / "x.md", _topic("x", "y"))
        _, _, _, entries = scan_memdir(tmp_path)
        assert entries == []

    def test_sort_newest_first(self, tmp_path: Path) -> None:
        f1 = _write(tmp_path / "alpha.md", _topic("alpha", "A"))
        time.sleep(0.01)
        f2 = _write(tmp_path / "beta.md", _topic("beta", "B"))
        # Force ordering: f1 older, f2 newer
        old = time.time() - 100
        os.utime(f1, (old, old))
        _, _, _, entries = scan_memdir(tmp_path)
        assert [e.name for e in entries] == ["beta", "alpha"]
        del f2

    def test_max_files_cap(self, tmp_path: Path) -> None:
        for i in range(10):
            _write(tmp_path / f"m{i}.md", _topic(f"m{i}", "x"))
        _, _, _, entries = scan_memdir(tmp_path, max_files=3)
        assert len(entries) == 3

    def test_malformed_skipped(self, tmp_path: Path) -> None:
        _write(tmp_path / "good.md", _topic("good", "g"))
        _write(tmp_path / "bad.md", "no frontmatter")
        _, _, _, entries = scan_memdir(tmp_path)
        assert [e.name for e in entries] == ["good"]

    def test_recursive_walk(self, tmp_path: Path) -> None:
        sub = tmp_path / "team"
        sub.mkdir()
        _write(sub / "shared.md", _topic("shared", "s"))
        _, _, _, entries = scan_memdir(tmp_path)
        assert [e.name for e in entries] == ["shared"]
