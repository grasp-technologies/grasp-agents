from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .types import (
    MemoryEntry,
    MemoryFormatError,
    MemoryFrontmatter,
)

logger = logging.getLogger(__name__)

INDEX_FILE_NAME = "MEMORY.md"

# CC's caps on the always-loaded MEMORY.md index.
MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000

# Silent cap from CC's memoryScan.
MAX_MEMORY_FILES = 200

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z",
    re.DOTALL,
)


def parse_memory_md(
    text: str, *, path: Path | None = None
) -> tuple[MemoryFrontmatter, str]:
    """
    Parse a topic memory file source into ``(frontmatter, body)``.

    Raises :class:`MemoryFormatError` on missing frontmatter, invalid YAML,
    or schema-violating frontmatter.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise MemoryFormatError(
            path,
            "Topic memory file must begin with a YAML frontmatter block "
            "delimited by '---'",
        )

    raw_yaml, body = match.group(1), match.group(2)

    try:
        loaded: Any = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise MemoryFormatError(path, f"Invalid YAML frontmatter: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise MemoryFormatError(
            path, "Frontmatter must be a YAML mapping (key/value pairs)"
        )

    try:
        frontmatter = MemoryFrontmatter.model_validate(loaded)
    except ValidationError as exc:
        raise MemoryFormatError(
            path, f"Frontmatter validation failed: {exc}"
        ) from exc

    return frontmatter, body.strip("\n")


def load_memory_entry(path: Path) -> MemoryEntry:
    """Load a single topic memory ``.md`` file."""
    if not path.is_file():
        raise MemoryFormatError(path, "Memory file not found or not a regular file")

    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_memory_md(text, path=path)
    mtime_ms = int(path.stat().st_mtime * 1000)
    return MemoryEntry(
        frontmatter=frontmatter, body=body, path=path, mtime_ms=mtime_ms
    )


def truncate_index(content: str) -> tuple[str, bool]:
    """
    Apply CC-style line + byte caps to ``MEMORY.md`` content.

    Truncates by line first; if still over the byte cap, truncates at the
    last newline before the byte limit. Returns ``(content, was_truncated)``.
    """
    lines = content.splitlines(keepends=True)
    truncated = False
    if len(lines) > MAX_INDEX_LINES:
        lines = lines[:MAX_INDEX_LINES]
        truncated = True
    text = "".join(lines)
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_INDEX_BYTES:
        cut = encoded[:MAX_INDEX_BYTES]
        last_nl = cut.rfind(b"\n")
        if last_nl > 0:
            cut = cut[:last_nl]
        text = cut.decode("utf-8", errors="ignore")
        truncated = True
    return text, truncated


def scan_memdir(
    root: Path | str,
    *,
    max_files: int = MAX_MEMORY_FILES,
) -> tuple[str | None, int | None, list[MemoryEntry]]:
    """
    Walk a memdir root and return ``(index_text, index_mtime_ms, entries)``.

    - ``index_text``: ``MEMORY.md`` content (already line/byte-capped) or ``None``.
    - ``index_mtime_ms``: filesystem mtime of ``MEMORY.md`` or ``None``.
    - ``entries``: every successfully-parsed topic ``.md`` file under ``root``,
      sorted by mtime (newest first), capped at ``max_files``. Topic files
      that fail to parse are logged and skipped.

    ``MEMORY.md`` is excluded from ``entries``. Hidden files (``.``-prefixed)
    and non-``.md`` files are skipped.
    """
    path = Path(root).expanduser()
    if not path.is_dir():
        return None, None, []

    index_text: str | None = None
    index_mtime_ms: int | None = None
    index_path = path / INDEX_FILE_NAME
    if index_path.is_file():
        raw = index_path.read_text(encoding="utf-8")
        capped, _ = truncate_index(raw)
        index_text = capped
        index_mtime_ms = int(index_path.stat().st_mtime * 1000)

    entries: list[MemoryEntry] = []
    for child in sorted(path.rglob("*.md")):
        if child.name == INDEX_FILE_NAME:
            continue
        if any(part.startswith(".") for part in child.relative_to(path).parts):
            continue
        try:
            entries.append(load_memory_entry(child))
        except MemoryFormatError:
            logger.exception("Failed to load memory at %s", child)

    entries.sort(key=lambda e: e.mtime_ms, reverse=True)
    if len(entries) > max_files:
        entries = entries[:max_files]
    return index_text, index_mtime_ms, entries
