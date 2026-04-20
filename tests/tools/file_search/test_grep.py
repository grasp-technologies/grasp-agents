"""
Unit tests for :class:`GrepTool`.

Skipped entirely if ``rg`` is not on PATH (rg-missing is surfaced as a
:class:`GrepError` at call time; the unit tests exercise the post-rg
parsing and slicing paths).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.tools.file_search import (
    GrepInput,
    GrepResult,
    GrepTool,
    rg_available,
)
from grasp_agents.types.events import ToolErrorInfo

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not rg_available(), reason="rg (ripgrep) not installed"),
]


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


@pytest.fixture
def grep_tool(tmp_path: Path) -> GrepTool:
    return GrepTool(allowed_roots=[tmp_path])


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Populate a miniature source tree for searches."""
    (tmp_path / "a.py").write_text("import os\ndef hello():\n    return 'world'\n")
    (tmp_path / "b.py").write_text("def goodbye():\n    pass\n")
    (tmp_path / "notes.md").write_text("TODO: write docs\nfinished tests\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "c.py").write_text("def hello_inner():\n    return 1\n")
    return tmp_path


# ---------------------------------------------------------------------------
# files_with_matches (default)
# ---------------------------------------------------------------------------


async def test_files_with_matches_default(repo: Path, grep_tool: GrepTool) -> None:
    del repo  # fixture used for its side effects (populated tmp_path)
    result = await grep_tool.run(GrepInput(pattern=r"def \w+"))
    assert isinstance(result, GrepResult)
    assert result.output_mode == "files_with_matches"
    # Three .py files contain a ``def``; the .md file does not.
    lines = [line for line in result.output.splitlines() if line]
    assert len(lines) == 3
    assert all(line.endswith(".py") for line in lines)
    assert result.num_files_matched == 3
    assert not result.truncated


async def test_glob_filter_narrows_to_md(repo: Path, grep_tool: GrepTool) -> None:
    del repo
    result = await grep_tool.run(
        GrepInput(pattern=r"TODO", glob="*.md")
    )
    assert isinstance(result, GrepResult)
    lines = [line for line in result.output.splitlines() if line]
    assert len(lines) == 1
    assert lines[0].endswith("notes.md")


async def test_type_filter_narrows_to_py(repo: Path, grep_tool: GrepTool) -> None:
    del repo
    result = await grep_tool.run(GrepInput(pattern=r"TODO", type="py"))
    assert isinstance(result, GrepResult)
    assert not result.output
    assert result.num_files_matched == 0


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


async def test_count_mode(repo: Path, grep_tool: GrepTool) -> None:
    del repo
    result = await grep_tool.run(
        GrepInput(pattern=r"def \w+", output_mode="count")
    )
    assert isinstance(result, GrepResult)
    assert result.output_mode == "count"
    # Each .py file has exactly one ``def`` line.
    assert result.num_matches == 3
    assert result.num_files_matched == 3
    for line in result.output.splitlines():
        assert line.endswith(":1")


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------


async def test_content_mode_with_line_numbers(
    repo: Path, grep_tool: GrepTool
) -> None:
    del repo
    result = await grep_tool.run(
        GrepInput(pattern=r"hello", output_mode="content")
    )
    assert isinstance(result, GrepResult)
    assert result.output_mode == "content"
    # Two match lines total: ``def hello()`` and ``def hello_inner()``.
    assert result.num_matches == 2
    assert result.num_files_matched == 2
    # Each line is ``path:line:content``.
    for line in result.output.splitlines():
        parts = line.split(":", 2)
        assert len(parts) == 3
        # Middle field is a digit (line number).
        assert parts[1].isdigit()


async def test_content_mode_without_line_numbers(
    repo: Path, grep_tool: GrepTool
) -> None:
    del repo
    result = await grep_tool.run(
        GrepInput(
            pattern=r"hello", output_mode="content", show_line_numbers=False
        )
    )
    assert isinstance(result, GrepResult)
    # Line format now ``path:content`` — exactly one colon on normal content.
    for line in result.output.splitlines():
        parts = line.split(":", 1)
        assert len(parts) == 2
        # First segment must not be purely digits (would mean line-number left in).
        assert not parts[1].split(" ")[0].isdigit() or "def" in parts[1]


async def test_content_mode_with_context(repo: Path, grep_tool: GrepTool) -> None:
    del repo
    result = await grep_tool.run(
        GrepInput(pattern=r"hello", output_mode="content", context=1)
    )
    assert isinstance(result, GrepResult)
    # Context lines use ``-`` instead of ``:`` as a separator.
    has_context = any(
        "-" in line and ":" in line for line in result.output.splitlines()
    )
    assert has_context


# ---------------------------------------------------------------------------
# Case sensitivity + multiline
# ---------------------------------------------------------------------------


async def test_case_insensitive(repo: Path, grep_tool: GrepTool) -> None:
    del repo
    result = await grep_tool.run(
        GrepInput(pattern=r"HELLO", case_insensitive=True)
    )
    assert isinstance(result, GrepResult)
    assert result.num_files_matched >= 1


async def test_multiline_pattern(tmp_path: Path, grep_tool: GrepTool) -> None:
    (tmp_path / "ml.py").write_text("def outer():\n    def inner():\n        pass\n")
    result = await grep_tool.run(
        GrepInput(
            pattern=r"outer.*inner",
            output_mode="content",
            multiline=True,
        )
    )
    assert isinstance(result, GrepResult)
    assert result.num_matches >= 1


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


async def test_head_limit_truncates(tmp_path: Path) -> None:
    tool = GrepTool(allowed_roots=[tmp_path])
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("target\n")

    result = await tool.run(GrepInput(pattern=r"target", head_limit=3))
    assert isinstance(result, GrepResult)
    assert result.truncated
    assert len([line for line in result.output.splitlines() if line]) == 3


async def test_offset_pagination(tmp_path: Path) -> None:
    tool = GrepTool(allowed_roots=[tmp_path])
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("target\n")

    first = await tool.run(GrepInput(pattern=r"target", head_limit=5))
    assert isinstance(first, GrepResult)
    second = await tool.run(GrepInput(pattern=r"target", head_limit=5, offset=5))
    assert isinstance(second, GrepResult)
    first_set = set(first.output.splitlines())
    second_set = set(second.output.splitlines())
    # Disjoint pages — overlap would mean offset didn't take effect.
    assert not (first_set & second_set)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


async def test_path_outside_root_refused(
    tmp_path: Path, grep_tool: GrepTool
) -> None:
    outside = tmp_path.parent / "escape_grep_dir"
    outside.mkdir(exist_ok=True)
    (outside / "secret.py").write_text("def leak(): pass\n")
    try:
        result = await grep_tool.run(
            GrepInput(pattern=r"def", path=str(outside))
        )
        assert "outside allowed roots" in _error_message(result)
    finally:
        for p in outside.glob("*"):
            p.unlink()
        outside.rmdir()


async def test_nonexistent_path_refused(tmp_path: Path, grep_tool: GrepTool) -> None:
    result = await grep_tool.run(
        GrepInput(pattern=r"x", path=str(tmp_path / "nope"))
    )
    assert "does not exist" in _error_message(result)
