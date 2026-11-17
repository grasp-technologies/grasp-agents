"""
Read-access filtering for search results.

``Grep`` and ``Glob`` validate only the root they search; the backend then
walks everything under it, including paths its policy refuses to read
(``deny_read`` carve-outs outside ``allow_read``, credential files).
:func:`readable_paths` picks out the results ``Read`` would allow, so a
search reveals no more than ``Read`` would. It works on any
:class:`FileBackend`, through its :meth:`~FileBackend.validate_path`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from grasp_agents.file_backend.paths import PathAccessError

if TYPE_CHECKING:
    from collections.abc import Iterable
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
