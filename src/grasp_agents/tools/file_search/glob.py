"""
``Glob`` — fast path-pattern matching under the toolkit's allowed roots.

Accepts standard glob patterns (``*.py``, ``**/*.tsx``, ``src/**/test_*.py``)
and returns matching file paths sorted by mtime (newest first) — useful for
"what changed recently" queries.

Guards:

1. ``allowed_roots`` — the search root must be under one of the configured
   roots; resolved symlinks that escape the sandbox are rejected.
2. Result cap — large result sets are truncated to ``head_limit`` (default
   250) and ``truncated=True`` is returned so the model can narrow the
   pattern instead of parsing an unbounded list.
3. Hidden directories (``.git``, ``.venv``, ``node_modules``, ...) are
   skipped by default; opt in with ``include_hidden=True`` on the
   :class:`GlobTool` constructor (not per-call — a search-scope decision,
   not a per-query one).
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ...types.tool import BaseTool, ToolProgressCallback
from ..file_edit.paths import PathAccessError, resolve_safe

if TYPE_CHECKING:
    from pathlib import Path

    from ...run_context import RunContext

# Default cap on the returned ``files`` list. 250 balances "enough to see
# the shape of the repo" against "context bloat". Callers raise via the
# constructor when a larger budget is needed.
DEFAULT_HEAD_LIMIT = 250

# Directory names always skipped on traversal. These are the common ones
# that contain huge numbers of files the model does not want to see.
_ALWAYS_SKIP_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


class GlobInput(BaseModel):
    """Input schema for the ``Glob`` tool."""

    pattern: str = Field(
        description=(
            "Glob pattern to match file paths against. Supports ``*``, "
            "``?``, ``[abc]``, and ``**`` (recursive). Examples: "
            "``*.py``, ``**/*.tsx``, ``src/**/test_*.py``."
        ),
        min_length=1,
    )
    path: str | None = Field(
        default=None,
        description=(
            "Directory to search within. Must resolve under one of the "
            "toolkit's allowed roots. Defaults to the first allowed root."
        ),
    )


class GlobResult(BaseModel):
    """Output schema for the ``Glob`` tool."""

    files: list[str]
    num_files: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class _MatchedFile:
    path: str
    mtime: float


def _walk_and_match(
    search_root: Path,
    pattern: str,
    *,
    include_hidden: bool,
    head_limit: int,
) -> tuple[list[_MatchedFile], bool]:
    """
    Walk ``search_root`` and return files whose *relative* path matches
    ``pattern``. Stops collecting once more than ``head_limit`` entries are
    found so the post-sort slice is accurate.

    Returned flag is True if the walker stopped early due to the cap.
    """
    matched: list[_MatchedFile] = []
    truncated = False

    # Over-collect by one so we can reliably detect truncation post-sort.
    # The caller sorts by mtime and keeps the top ``head_limit``; without
    # the overshoot we'd have to walk the whole tree to know "was there a
    # 251st match?".
    collect_budget = head_limit + 1

    for dirpath, dirnames, filenames in os.walk(search_root):
        # Prune in-place so ``os.walk`` skips the subtree.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _ALWAYS_SKIP_DIRS
            and (include_hidden or not d.startswith("."))
        ]

        for fname in filenames:
            if not include_hidden and fname.startswith("."):
                continue
            # String concatenation (not Path / Path) is deliberate here:
            # os.walk already yields strings, and a full walk creates
            # tens of thousands of entries. Path() allocation on each
            # iteration is measurable. We build Path objects only on
            # demand at the stat / return boundary.
            abs_path = os.path.join(dirpath, fname)  # noqa: PTH118
            rel_path = os.path.relpath(abs_path, search_root)
            if _matches(pattern, rel_path):
                try:
                    mtime = os.stat(abs_path).st_mtime  # noqa: PTH116
                except OSError:
                    # Path disappeared between walk and stat (race with
                    # a concurrent delete). Skip silently.
                    continue
                matched.append(_MatchedFile(path=abs_path, mtime=mtime))
                if len(matched) >= collect_budget:
                    truncated = True
                    return matched, truncated

    return matched, truncated


def _matches(pattern: str, rel_path: str) -> bool:
    """
    Match ``rel_path`` against ``pattern`` with ``**`` semantics.

    ``fnmatch`` does not understand ``**``; rewrite the pattern segment-
    wise so ``src/**/test_*.py`` matches arbitrary-depth subpaths.
    """
    # Normalize separators — pattern authors on Windows might use ``/``.
    norm = rel_path.replace(os.sep, "/")

    if "**" not in pattern:
        # Plain fnmatch — but with the whole relative path, so ``*.py``
        # only matches top-level files (same as shell ``*.py``). Use
        # ``**/*.py`` to match recursively.
        return fnmatch.fnmatchcase(norm, pattern)

    # Segment-wise match with ``**`` = "zero or more path segments".
    return _match_segments(pattern.split("/"), norm.split("/"))


def _match_segments(pattern_parts: list[str], path_parts: list[str]) -> bool:
    """Recursive glob matcher with ``**`` support."""
    if not pattern_parts:
        return not path_parts
    head, *tail = pattern_parts
    if head == "**":
        # ``**`` matches zero or more segments — try each split point.
        if not tail:
            return True  # trailing ``**`` matches everything remaining
        for i in range(len(path_parts) + 1):
            if _match_segments(tail, path_parts[i:]):
                return True
        return False
    if not path_parts:
        return False
    if fnmatch.fnmatchcase(path_parts[0], head):
        return _match_segments(tail, path_parts[1:])
    return False


class GlobTool(BaseTool[GlobInput, GlobResult, Any]):
    """
    Match file paths against a glob pattern, sorted by mtime.

    Attach via :class:`FileSearchToolkit` or instantiate directly with
    ``allowed_roots``.
    """

    name = "Glob"
    description = (
        "Find files by glob pattern, sorted by modification time (newest "
        "first). Supports ``*``, ``?``, ``[abc]``, and ``**`` for "
        "recursive match. Examples: ``*.py`` (top-level Python), "
        "``**/*.tsx`` (all TSX files), ``src/**/test_*.py``. Hidden files "
        "and common build/cache dirs are skipped. Results are capped at "
        "the tool's ``head_limit`` — narrow the pattern if truncated."
    )

    def __init__(
        self,
        *,
        allowed_roots: list[Path],
        head_limit: int = DEFAULT_HEAD_LIMIT,
        include_hidden: bool = False,
        timeout: float | None = None,
    ) -> None:
        super().__init__(timeout=timeout)
        self._allowed_roots = allowed_roots
        self._head_limit = head_limit
        self._include_hidden = include_hidden

    async def _run(
        self,
        inp: GlobInput,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> GlobResult:
        del ctx, exec_id, progress_callback

        # Resolve search root: explicit ``path`` or default to the first
        # allowed root. ``must_exist=True`` because walking a non-existent
        # dir would silently return nothing.
        raw_root = inp.path if inp.path is not None else str(self._allowed_roots[0])
        try:
            search_root = resolve_safe(
                raw_root, self._allowed_roots, must_exist=True
            )
        except PathAccessError as exc:
            raise ValueError(str(exc)) from exc

        if not search_root.is_dir():
            raise ValueError(f"Glob search path must be a directory: {search_root}")

        matched, truncated = await asyncio.to_thread(
            _walk_and_match,
            search_root,
            inp.pattern,
            include_hidden=self._include_hidden,
            head_limit=self._head_limit,
        )

        matched.sort(key=lambda m: m.mtime, reverse=True)
        kept = matched[: self._head_limit]
        # If the walk stopped early (overshoot tripped) OR we collected
        # exactly head_limit+1 here, the caller is seeing a truncated view.
        is_truncated = truncated or len(matched) > self._head_limit

        return GlobResult(
            files=[m.path for m in kept],
            num_files=len(kept),
            truncated=is_truncated,
        )
