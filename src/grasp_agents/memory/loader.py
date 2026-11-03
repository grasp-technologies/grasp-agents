from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError
from yaml.composer import ComposerError

from .types import (
    INDEX_FILE_NAME,
    MAX_FRONTMATTER_CHARS,
    MAX_INDEX_BYTES,
    MAX_INDEX_LINES,
    MAX_MEMORY_FILES,
    MemoryEntry,
    MemoryFormatError,
    MemoryFrontmatter,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z",
    re.DOTALL,
)


class _FrontmatterLoader(yaml.SafeLoader):
    """
    ``SafeLoader`` that refuses anchors, and with them aliases.

    Memory files may come from untrusted writers, and a few hundred bytes of
    nested aliases expand to gigabytes once anything walks the result. An
    alias can only name an anchor defined before it, so refusing each anchor
    as soon as its node is composed also refuses every alias.
    """

    def compose_node(self, parent: yaml.Node | None, index: int) -> yaml.Node | None:
        node = super().compose_node(parent, index)
        if self.anchors:
            raise ComposerError(
                problem="anchors and aliases are not allowed",
                problem_mark=node.start_mark if node is not None else None,
            )
        return node


def _describe_yaml_error(exc: yaml.YAMLError) -> str:
    """Summarize a YAML error without the input snippet PyYAML attaches."""
    if not isinstance(exc, yaml.MarkedYAMLError):
        return str(exc).splitlines()[0]
    problem = exc.problem or exc.context or "malformed YAML"
    mark = exc.problem_mark or exc.context_mark
    if mark is None:
        return problem
    # Marks count lines from the one after the opening ``---``.
    return f"{problem} (line {mark.line + 2}, column {mark.column + 1})"


def _describe_validation_error(exc: ValidationError) -> str:
    """List each failing field and why, without echoing its value."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'frontmatter'}: "
        f"{error['msg']}"
        for error in exc.errors(
            include_url=False, include_context=False, include_input=False
        )
    )


def strip_frontmatter(text: str) -> str:
    """Drop a leading ``--- ... ---`` YAML block; pass-through if absent."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return text
    return match.group(2).lstrip("\n")


def parse_memory_md(
    text: str, *, path: Path | None = None
) -> tuple[MemoryFrontmatter, str]:
    """
    Parse a topic memory file source into ``(frontmatter, body)``.

    The text may come from an untrusted writer: frontmatter longer than
    :data:`MAX_FRONTMATTER_CHARS`, YAML anchors and aliases, and nesting
    too deep to parse are all refused, and error messages never quote the
    offending values.

    Raises :class:`MemoryFormatError` on missing or oversized frontmatter,
    invalid or refused YAML, or schema-violating frontmatter.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise MemoryFormatError(
            path,
            "Topic memory file must begin with a YAML frontmatter block "
            "delimited by '---'",
        )

    raw_yaml, body = match.group(1), match.group(2)
    if len(raw_yaml) > MAX_FRONTMATTER_CHARS:
        raise MemoryFormatError(
            path, f"Frontmatter exceeds {MAX_FRONTMATTER_CHARS} characters"
        )

    # The errors below drop their cause: rendering it would echo the input.
    try:
        loaded: Any = _FrontmatterLoader(raw_yaml).get_single_data()
    except yaml.YAMLError as exc:
        raise MemoryFormatError(
            path, f"Invalid YAML frontmatter: {_describe_yaml_error(exc)}"
        ) from None
    except RecursionError:
        raise MemoryFormatError(path, "Frontmatter is nested too deeply") from None

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
            path, f"Frontmatter validation failed: {_describe_validation_error(exc)}"
        ) from None

    return frontmatter, body.strip("\n")


def load_memory_entry(path: Path) -> MemoryEntry:
    """Load a single topic memory ``.md`` file."""
    if not path.is_file():
        raise MemoryFormatError(path, "Memory file not found or not a regular file")

    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_memory_md(text, path=path)
    mtime_ms = int(path.stat().st_mtime * 1000)
    return MemoryEntry(frontmatter=frontmatter, body=body, path=path, mtime_ms=mtime_ms)


def truncate_index(content: str) -> tuple[str, bool]:
    """
    Apply line + byte caps to ``MEMORY.md`` content.

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
    root: Path,
    *,
    max_files: int = MAX_MEMORY_FILES,
) -> tuple[str | None, int | None, bool, list[MemoryEntry]]:
    """
    Walk a memdir root and return ``(index_text, index_mtime_ms,
    index_truncated, entries)``.

    - ``index_text``: ``MEMORY.md`` content (already line/byte-capped) or ``None``.
    - ``index_mtime_ms``: filesystem mtime of ``MEMORY.md`` or ``None``.
    - ``index_truncated``: ``True`` when the caps cut content.
    - ``entries``: every successfully-parsed topic ``.md`` file under ``root``,
      sorted by mtime (newest first), capped at ``max_files``. Topic files
      that fail to parse are logged and skipped.

    ``MEMORY.md`` is excluded from ``entries``. Hidden files (``.``-prefixed)
    and non-``.md`` files are skipped.
    """
    path = root.expanduser()
    if not path.is_dir():
        return None, None, False, []

    index_text: str | None = None
    index_mtime_ms: int | None = None
    index_truncated = False
    index_path = path / INDEX_FILE_NAME
    if index_path.is_file():
        raw = index_path.read_text(encoding="utf-8")
        index_text, index_truncated = truncate_index(raw)
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

    return index_text, index_mtime_ms, index_truncated, entries
