"""
Segment-wise glob matching shared by the ``Glob`` fallback
(:func:`~grasp_agents.file_backend.local.glob_filter_entries`) and
:func:`~grasp_agents.file_backend.text_grep.grep_texts`.

A pattern is matched one ``/``-separated segment per path component with
:func:`fnmatch.fnmatchcase`; a ``**`` segment spans zero or more components.
:func:`match_segments` collapses runs of ``**`` and fills a table over
(segment, component) positions, so one match costs at most
``segments x components`` ``fnmatch`` calls however many ``**`` the pattern
holds. :func:`expand_braces` turns ``{a,b}`` alternatives into separate
patterns and refuses more than :data:`MAX_BRACE_EXPANSIONS` of them, so a
caller expands a glob once per search rather than once per file.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

MAX_BRACE_EXPANSIONS = 128
"""Most alternatives a glob's ``{a,b}`` groups may expand to."""

_BRACE = re.compile(r"\{([^{}]*)\}")


def match_segments(pattern: Sequence[str], path: Sequence[str]) -> bool:
    """
    Return True if the path components match the pattern segments.

    Each segment matches one component with :func:`fnmatch.fnmatchcase`,
    except ``**``, which matches any number of components, including none.
    """
    segments: list[str] = []
    for segment in pattern:
        if segment == "**" and segments and segments[-1] == "**":
            continue
        segments.append(segment)

    n = len(path)
    # ``rest[j]``: the segments after the current one match ``path[j:]``.
    rest = [False] * n + [True]
    for segment in reversed(segments):
        row = [False] * (n + 1)
        if segment == "**":
            spans = False
            for j in range(n, -1, -1):
                spans = spans or rest[j]
                row[j] = spans
        else:
            for j in range(n):
                row[j] = rest[j + 1] and fnmatch.fnmatchcase(path[j], segment)
        rest = row
    return rest[0]


def expand_braces(pattern: str, *, limit: int = MAX_BRACE_EXPANSIONS) -> list[str]:
    """
    Expand ``{a,b}`` alternatives into separate patterns, in order.

    Nested groups expand from the innermost out. Braces are not escapable.

    Raises:
        ValueError: The pattern expands to more than ``limit`` alternatives.

    """
    done: list[str] = []
    pending = [pattern]
    while pending:
        current = pending.pop()
        group = _BRACE.search(current)
        if group is None:
            done.append(current)
            continue
        options = group.group(1).split(",")
        if len(done) + len(pending) + len(options) > limit:
            raise ValueError(
                f"Glob {pattern!r} expands to more than {limit} alternatives; "
                "use fewer brace groups."
            )
        head, tail = current[: group.start()], current[group.end() :]
        pending.extend(head + option + tail for option in reversed(options))
    return done


__all__ = ["MAX_BRACE_EXPANSIONS", "expand_braces", "match_segments"]
