"""
Unit tests for :class:`GlobTool`.

Uses the public ``.run(...)`` API — errors come back as ``ToolErrorInfo``
rather than raising, matching how the agent loop consumes tool calls.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from grasp_agents.tools.file_search import GlobInput, GlobResult, GlobTool
from grasp_agents.types.events import ToolErrorInfo

pytestmark = pytest.mark.asyncio


def _error_message(result: Any) -> str:
    assert isinstance(result, ToolErrorInfo), (
        f"Expected a ToolErrorInfo, got {type(result).__name__}: {result!r}"
    )
    return result.error


@pytest.fixture
def glob_tool(tmp_path: Path) -> GlobTool:
    return GlobTool(allowed_roots=[tmp_path])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_top_level_glob(tmp_path: Path, glob_tool: GlobTool) -> None:
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.txt").write_text("")

    result = await glob_tool.run(GlobInput(pattern="*.py"))
    assert isinstance(result, GlobResult)
    names = sorted(Path(p).name for p in result.files)
    assert names == ["a.py", "b.py"]
    assert result.num_files == 2
    assert not result.truncated


async def test_recursive_glob(tmp_path: Path, glob_tool: GlobTool) -> None:
    (tmp_path / "a.py").write_text("")
    sub = tmp_path / "sub" / "deeper"
    sub.mkdir(parents=True)
    (sub / "b.py").write_text("")

    result = await glob_tool.run(GlobInput(pattern="**/*.py"))
    assert isinstance(result, GlobResult)
    names = sorted(Path(p).name for p in result.files)
    assert names == ["a.py", "b.py"]


async def test_results_sorted_by_mtime_desc(
    tmp_path: Path, glob_tool: GlobTool
) -> None:
    older = tmp_path / "older.py"
    newer = tmp_path / "newer.py"
    older.write_text("")
    # Backdate older's mtime to make the ordering deterministic even on
    # filesystems with coarse mtime resolution.
    past = time.time() - 3600
    os.utime(older, (past, past))
    newer.write_text("")

    result = await glob_tool.run(GlobInput(pattern="*.py"))
    assert isinstance(result, GlobResult)
    assert [Path(p).name for p in result.files] == ["newer.py", "older.py"]


async def test_skips_hidden_by_default(tmp_path: Path, glob_tool: GlobTool) -> None:
    (tmp_path / "visible.py").write_text("")
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "inner.py").write_text("")
    (tmp_path / ".secret.py").write_text("")

    result = await glob_tool.run(GlobInput(pattern="**/*.py"))
    assert isinstance(result, GlobResult)
    names = [Path(p).name for p in result.files]
    assert "visible.py" in names
    assert ".secret.py" not in names
    assert "inner.py" not in names


async def test_include_hidden_traverses_dotdirs(tmp_path: Path) -> None:
    tool = GlobTool(allowed_roots=[tmp_path], include_hidden=True)
    (tmp_path / "visible.py").write_text("")
    hidden_dir = tmp_path / ".config"
    hidden_dir.mkdir()
    (hidden_dir / "inner.py").write_text("")

    result = await tool.run(GlobInput(pattern="**/*.py"))
    assert isinstance(result, GlobResult)
    names = [Path(p).name for p in result.files]
    assert "visible.py" in names
    assert "inner.py" in names


async def test_skips_cache_dirs(tmp_path: Path, glob_tool: GlobTool) -> None:
    (tmp_path / "visible.py").write_text("")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "junk.py").write_text("")

    result = await glob_tool.run(GlobInput(pattern="**/*.py"))
    assert isinstance(result, GlobResult)
    names = [Path(p).name for p in result.files]
    assert names == ["visible.py"]


async def test_truncation_flag(tmp_path: Path) -> None:
    tool = GlobTool(allowed_roots=[tmp_path], head_limit=3)
    for i in range(10):
        (tmp_path / f"f{i}.py").write_text("")
    result = await tool.run(GlobInput(pattern="*.py"))
    assert isinstance(result, GlobResult)
    assert result.truncated
    assert result.num_files == 3


async def test_explicit_path_subdir(tmp_path: Path, glob_tool: GlobTool) -> None:
    sub = tmp_path / "only-here"
    sub.mkdir()
    (tmp_path / "outside.py").write_text("")
    (sub / "inside.py").write_text("")

    result = await glob_tool.run(GlobInput(pattern="*.py", path=str(sub)))
    assert isinstance(result, GlobResult)
    names = [Path(p).name for p in result.files]
    assert names == ["inside.py"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


async def test_path_outside_root_refused(tmp_path: Path, glob_tool: GlobTool) -> None:
    outside = tmp_path.parent / "escape_glob_dir"
    outside.mkdir(exist_ok=True)
    try:
        result = await glob_tool.run(GlobInput(pattern="*.py", path=str(outside)))
        assert "outside allowed roots" in _error_message(result)
    finally:
        # Leave no trace; the dir may contain files from parallel runs.
        for p in outside.glob("*"):
            p.unlink()
        outside.rmdir()


async def test_path_must_be_directory(tmp_path: Path, glob_tool: GlobTool) -> None:
    f = tmp_path / "file.txt"
    f.write_text("")
    result = await glob_tool.run(GlobInput(pattern="*.py", path=str(f)))
    assert "must be a directory" in _error_message(result)


async def test_nonexistent_path_refused(tmp_path: Path, glob_tool: GlobTool) -> None:
    result = await glob_tool.run(
        GlobInput(pattern="*.py", path=str(tmp_path / "nope"))
    )
    assert "does not exist" in _error_message(result)
