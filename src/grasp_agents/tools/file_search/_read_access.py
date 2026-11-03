"""
Read-access filtering for search results.

``Grep`` and ``Glob`` validate only the root they search; the backend then
walks everything under it, including paths its policy refuses to read
(``deny_read`` carve-outs outside ``allow_read``, credential files). These
helpers drop such paths from the results, so a search reveals no more than
``Read`` would. They work on any :class:`FileBackend`, through its
:meth:`~FileBackend.validate_path`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grasp_agents.file_backend.paths import PathAccessError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from grasp_agents.file_backend.base import FileBackend


async def readable_paths(
    backend: FileBackend,
    paths: Iterable[Path],
    *,
    dotfile_overrides: set[Path] | None,
) -> set[Path]:
    """Return the members of ``paths`` that ``backend`` lets the agent read."""
    readable: set[Path] = set()
    for path in set(paths):
        try:
            await backend.validate_path(
                path,
                must_exist=False,
                access="read",
                dotfile_overrides=dotfile_overrides,
            )
        except PathAccessError:
            continue
        readable.add(path)
    return readable


def filter_content_lines(
    lines: Sequence[str], *, files: Iterable[Path], readable: set[Path]
) -> tuple[list[str], int, int]:
    """
    Keep the ``content``-mode lines that come from a readable file.

    A line starts with its file's path, then ``:`` for a match or ``-`` for
    a context line. Paths may contain those characters too, so each line is
    attributed to ``files``, the files the same search matched. A line is
    kept only when it starts with a readable file and with no unreadable
    one: when one path is a prefix of another, dropping is the safe reading.
    A line that starts with none of ``files`` is dropped as well.

    Returns ``(kept_lines, num_matches, num_files_matched)``, counting only
    match lines.
    """
    is_readable = {str(path): path in readable for path in files}
    longest = max(map(len, is_readable), default=0)
    kept: list[str] = []
    num_matches = 0
    matched_files: set[str] = set()
    for line in lines:
        owners = [
            line[:index]
            for index, char in enumerate(line[: longest + 1])
            if char in ":-" and line[:index] in is_readable
        ]
        if not owners or not all(is_readable[owner] for owner in owners):
            continue
        kept.append(line)
        owner = max(owners, key=len)
        if line[len(owner)] == ":":
            num_matches += 1
            matched_files.add(owner)
    return kept, num_matches, len(matched_files)
