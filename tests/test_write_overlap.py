"""
Write-overlap detection: the agent loop serializes a foreground tool batch
when two calls declare overlapping write targets (else runs them concurrently).
"""

from __future__ import annotations

from typing import Any

from grasp_agents.types.tool import _keys_overlap, batch_has_concurrency_conflict


class _WriteFake:
    def __init__(self, *paths: str) -> None:
        self._paths = list(paths)

    def concurrency_conflict_keys(self, inp: Any) -> list[str] | None:
        return self._paths


class _ReadFake:
    def concurrency_conflict_keys(self, inp: Any) -> list[str] | None:
        return None


def test_paths_overlap_same_file() -> None:
    assert _keys_overlap("/x/a.txt", "/x/a.txt")


def test_paths_overlap_ancestor_either_order() -> None:
    assert _keys_overlap("/x", "/x/a.txt")
    assert _keys_overlap("/x/a.txt", "/x")


def test_paths_no_overlap_for_siblings() -> None:
    assert not _keys_overlap("/x/a.txt", "/x/b.txt")


def test_paths_overlap_normalizes_dot_segments() -> None:
    assert _keys_overlap("/x/./a.txt", "/x/a.txt")


def test_batch_overlap_on_same_path() -> None:
    calls = [(_WriteFake("/x/a.txt"), None), (_WriteFake("/x/a.txt"), None)]
    assert batch_has_concurrency_conflict(calls)


def test_batch_no_overlap_distinct_paths() -> None:
    calls = [(_WriteFake("/x/a.txt"), None), (_WriteFake("/x/b.txt"), None)]
    assert not batch_has_concurrency_conflict(calls)


def test_batch_readonly_calls_never_overlap() -> None:
    calls = [(_ReadFake(), None), (_ReadFake(), None), (_WriteFake("/x/a.txt"), None)]
    assert not batch_has_concurrency_conflict(calls)
