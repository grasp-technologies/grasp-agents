"""
Tests for :func:`grep_texts`, the pure-Python grep shared by backends that
hold file text themselves (in-memory, database-backed).

The parity tests run ripgrep through :class:`LocalFileBackend` over the same
files and require identical results, so a backend built on ``grep_texts``
renders exactly what the ``Grep`` tool already shows for the local backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.file_backend.text_grep import grep_texts
from grasp_agents.tools.file_search import rg_available

if TYPE_CHECKING:
    from pathlib import Path

FILES: dict[str, str] = {
    "notes/alpha.md": "intro\nTODO first\nmiddle\nmore\nTODO second\nend\n",
    "notes/beta.md": "nothing here\nstill nothing\n",
    "code/main.py": "def main():\n    # todo lower\n    return 1\n",
    "code/util.py": "import os\n\ndef helper():\n    return os.sep\n",
    "readme.txt": "TODO in text\n",
    "notes/empty.md": "",
    "notes/blank.md": "\n",
}


def _write_tree(root: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        pairs.append((p, text))
    return pairs


PARITY_CASES: list[dict[str, Any]] = [
    {"pattern": "TODO", "output_mode": "files_with_matches"},
    {"pattern": "TODO", "output_mode": "count"},
    {"pattern": "TODO", "output_mode": "content"},
    {"pattern": "TODO", "output_mode": "content", "show_line_numbers": False},
    {"pattern": "todo", "output_mode": "content", "case_insensitive": True},
    {"pattern": "TODO", "output_mode": "content", "context": 1},
    {"pattern": "TODO", "output_mode": "content", "before_context": 2},
    {"pattern": "TODO", "output_mode": "content", "after_context": 1},
    {"pattern": "def \\w+", "output_mode": "content", "glob": "*.py"},
    {"pattern": "return", "output_mode": "count", "file_type": "py"},
    {"pattern": "TODO", "output_mode": "files_with_matches", "file_type": "md"},
    {"pattern": "^\\s+return", "output_mode": "content"},
    {"pattern": "^", "output_mode": "files_with_matches"},
    {"pattern": "^$", "output_mode": "count"},
]


@pytest.mark.skipif(not rg_available(), reason="rg (ripgrep) not installed")
@pytest.mark.asyncio
@pytest.mark.parametrize("case", PARITY_CASES)
async def test_matches_ripgrep(tmp_path: Path, case: dict[str, Any]) -> None:
    root = tmp_path.resolve()
    pairs = _write_tree(root)
    kwargs: dict[str, Any] = {
        "glob": None,
        "file_type": None,
        "case_insensitive": False,
        "multiline": False,
        "output_mode": "files_with_matches",
        "show_line_numbers": True,
        "before_context": None,
        "after_context": None,
        "context": None,
        **{k: v for k, v in case.items() if k != "pattern"},
    }

    expected = await LocalFileBackend(allowed_roots=[root]).grep(
        root, case["pattern"], **kwargs
    )
    actual = grep_texts(pairs, case["pattern"], root=root, **kwargs)

    assert actual.files == expected.files
    assert actual.counts == expected.counts
    assert actual.lines == expected.lines
    assert actual.num_matches == expected.num_matches
    assert actual.num_files_matched == expected.num_files_matched


def test_root_scopes_the_search(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / rel, text) for rel, text in FILES.items()]

    result = grep_texts(pairs, "TODO", root=root / "notes")

    assert result.files == [root / "notes/alpha.md"]


def test_root_may_be_a_single_file(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    target = root / "notes/alpha.md"
    pairs = [(target, FILES["notes/alpha.md"])]

    result = grep_texts(pairs, "TODO", root=target, output_mode="count")

    assert result.counts == [(target, 2)]


def test_hidden_files_are_skipped_unless_asked(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / ".hidden/a.md", "TODO\n"), (root / "b.md", "TODO\n")]

    default = grep_texts(pairs, "TODO", root=root)
    with_hidden = grep_texts(pairs, "TODO", root=root, include_hidden=True)

    assert default.files == [root / "b.md"]
    assert with_hidden.files == [root / ".hidden/a.md", root / "b.md"]


def test_an_empty_file_has_no_lines(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / "empty.md", ""), (root / "blank.md", "\n")]

    anything = grep_texts(pairs, "^", root=root)
    blank_lines = grep_texts(pairs, "^$", root=root, output_mode="count")

    assert anything.files == [root / "blank.md"]
    assert blank_lines.counts == [(root / "blank.md", 1)]


def test_text_with_nul_bytes_is_treated_as_binary(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / "bin.dat", "TODO\x00\x01"), (root / "a.md", "TODO\n")]

    result = grep_texts(pairs, "TODO", root=root)

    assert result.files == [root / "a.md"]


def test_slashed_glob_is_relative_to_root(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / rel, text) for rel, text in FILES.items()]
    pairs.append((root / "notes/deep/gamma.md", "TODO deep\n"))

    result = grep_texts(pairs, "TODO", root=root, glob="notes/*.md")

    assert result.files == [root / "notes/alpha.md"]


def test_negated_glob_excludes(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / rel, text) for rel, text in FILES.items()]

    result = grep_texts(pairs, "TODO", root=root, glob="!*.md")

    assert result.files == [root / "readme.txt"]


def test_brace_glob(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / rel, text) for rel, text in FILES.items()]

    result = grep_texts(pairs, "TODO", root=root, glob="*.{md,txt}")

    assert result.files == [root / "notes/alpha.md", root / "readme.txt"]


def test_multiline_match_spans_lines(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    pairs = [(root / "a.md", "one\ntwo\nthree\n")]

    result = grep_texts(
        pairs, "one.two", root=root, multiline=True, output_mode="content"
    )

    assert result.lines == [f"{root / 'a.md'}:1:one\ntwo"]
    assert result.num_matches == 1


def test_unknown_file_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unrecognized file type"):
        grep_texts([], "x", root=tmp_path, file_type="nonsense")


def test_invalid_regex_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid regex"):
        grep_texts([(tmp_path / "a", "x")], "(", root=tmp_path)
