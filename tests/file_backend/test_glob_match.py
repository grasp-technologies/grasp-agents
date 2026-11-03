"""
Glob matching shared by the ``Glob`` fallback and :func:`grep_texts`.

The hostile-pattern tests run the match in a daemon thread and fail if it is
still running after a deadline, so an exponential matcher fails the test
instead of hanging the suite. ``LocalFileBackend.find_files`` uses the same
matcher as :func:`glob_filter_entries` but walks in an executor thread, which
the interpreter would wait for at exit, so no hostile pattern goes through it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.file_backend import FileEntry, LocalFileBackend
from grasp_agents.file_backend.glob_match import (
    MAX_BRACE_EXPANSIONS,
    expand_braces,
    match_segments,
)
from grasp_agents.file_backend.local import glob_filter_entries
from grasp_agents.file_backend.text_grep import grep_texts

if TYPE_CHECKING:
    from collections.abc import Callable

DEEP = "a/b/c/d/e/f/g/h/i/j/k/l.md"
MANY_DOUBLE_STARS = "**/" * 40 + "zz"


def _within[T](seconds: float, fn: Callable[[], T]) -> T:
    box: dict[str, Any] = {}

    def target() -> None:
        box["result"] = fn()

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still matching after {seconds}s"
    return box["result"]


# ---- hostile patterns ---------------------------------------------------------


def test_grep_glob_with_many_double_stars_is_fast() -> None:
    root = Path("/r")

    result = _within(
        2,
        lambda: grep_texts(
            [(root / DEEP, "x")], "x", root=root, glob=MANY_DOUBLE_STARS
        ),
    )

    assert result.files == []


def test_glob_filter_with_many_double_stars_is_fast() -> None:
    root = Path("/r")
    entries = [FileEntry(name="l.md", path=root / DEEP, is_dir=False, mtime=1.0)]

    matched, _ = _within(
        2,
        lambda: glob_filter_entries(
            entries, root, MANY_DOUBLE_STARS, include_hidden=False, head_limit=10
        ),
    )

    assert matched == []


def test_grep_refuses_a_glob_with_too_many_brace_alternatives() -> None:
    root = Path("/r")

    with pytest.raises(ValueError, match="alternatives"):
        grep_texts([(root / "a.md", "x")], "x", root=root, glob="{a,b}" * 8)


# ---- match_segments ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**", "", True),
        ("**", "a/b", True),
        ("**/*.md", "a.md", True),
        ("**/*.md", "x/y/a.md", True),
        ("**/*.md", "x/a.py", False),
        ("src/**/test_*.py", "src/test_a.py", True),
        ("src/**/test_*.py", "src/a/b/test_a.py", True),
        ("src/**/test_*.py", "lib/test_a.py", False),
        ("**/**/a", "x/a", True),
        ("a/**", "a", True),
        ("a/*", "a", False),
        ("*/b", "a/b", True),
        ("*/b", "a/c/b", False),
    ],
)
def test_match_segments(pattern: str, path: str, expected: bool) -> None:
    path_parts = path.split("/") if path else []

    assert match_segments(pattern.split("/"), path_parts) is expected


def test_match_segments_on_empty_inputs() -> None:
    assert match_segments([], [])
    assert not match_segments([], ["a"])
    assert not match_segments(["a"], [])


# ---- expand_braces ----------------------------------------------------------------


def test_expand_braces() -> None:
    assert expand_braces("*.md") == ["*.md"]
    assert expand_braces("*.{md,txt}") == ["*.md", "*.txt"]
    assert expand_braces("{a,b}{c,d}") == ["ac", "ad", "bc", "bd"]
    assert sorted(set(expand_braces("{a,{b,c}}"))) == ["a", "b", "c"]


def test_expand_braces_refuses_past_its_limit() -> None:
    assert len(expand_braces("{a,b}" * 7)) == MAX_BRACE_EXPANSIONS
    with pytest.raises(ValueError, match="alternatives"):
        expand_braces("{a,b}" * 8)
    with pytest.raises(ValueError, match="alternatives"):
        expand_braces("{a,b,c}", limit=2)


# ---- glob_filter_entries and LocalFileBackend.find_files --------------------------


def _entry(path: str, mtime: float) -> FileEntry:
    return FileEntry(name=Path(path).name, path=Path(path), is_dir=False, mtime=mtime)


def test_glob_filter_keeps_the_newest_before_truncating() -> None:
    entries = [_entry(f"/r/f{i}.md", mtime=float(i)) for i in range(10)]

    matched, truncated = glob_filter_entries(
        entries, Path("/r"), "*.md", include_hidden=False, head_limit=3
    )

    assert [e.name for e in matched] == ["f9.md", "f8.md", "f7.md"]
    assert truncated


def test_glob_filter_skips_build_and_cache_dirs_below_the_root() -> None:
    entries = [
        _entry("/r/app.js", 1),
        _entry("/r/build.js", 2),
        _entry("/r/node_modules/pkg/index.js", 3),
        _entry("/r/__pycache__/m.js", 4),
        _entry("/r/src/dist/bundle.js", 5),
    ]

    matched, truncated = glob_filter_entries(
        entries, Path("/r"), "**/*.js", include_hidden=False, head_limit=10
    )
    inside, _ = glob_filter_entries(
        entries, Path("/r/node_modules"), "**/*.js", include_hidden=False, head_limit=10
    )

    assert [e.name for e in matched] == ["build.js", "app.js"]
    assert not truncated
    assert [e.name for e in inside] == ["index.js"]


@pytest.mark.asyncio
async def test_local_find_files_keeps_the_newest(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    for i in range(10):
        f = root / f"f{i}.md"
        f.write_text("x")
        os.utime(f, (1_000 + i, 1_000 + i))

    matched, truncated = await LocalFileBackend(allowed_roots=[root]).find_files(
        root, "*.md", head_limit=3
    )

    assert [e.name for e in matched] == ["f9.md", "f8.md", "f7.md"]
    assert truncated
