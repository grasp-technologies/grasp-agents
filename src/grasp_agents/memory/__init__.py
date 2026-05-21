"""
Cross-session memory — Claude-Code-style memdir loader and provider ABC.

A *memdir* is a directory of markdown files: an always-loaded ``MEMORY.md``
index plus topic ``.md`` files with YAML frontmatter (``name`` +
``description`` required, optional ``type`` from
``{user, feedback, project, reference}``). Loaded once per session as a
frozen :class:`MemorySnapshot`.

The ``memory`` system-prompt section auto-attaches to every :class:`LLMAgent`
at construction and reads ``ctx.memory`` at compute time, so populating that
field with a :class:`MemoryProvider` is the only wiring step.

This package also re-exports :class:`Memory` and :class:`DummyMemory` — the
working-memory ABC consumed by :class:`Processor` (distinct concern, kept
here for backwards compatibility with the previous ``grasp_agents.memory``
module).

See ``docs/roadmap/13-memory-system.md``.
"""

from __future__ import annotations

from .base import DummyMemory, Memory
from .injection import (
    MEMORY_SECTION_NAME,
    memory_system_prompt_section,
    render_memory_block,
)
from .loader import (
    INDEX_FILE_NAME,
    MAX_INDEX_BYTES,
    MAX_INDEX_LINES,
    MAX_MEMORY_FILES,
    load_memory_entry,
    parse_memory_md,
    scan_memdir,
    truncate_index,
)
from .provider import (
    DEFAULT_STALE_AFTER,
    FileMemoryProvider,
    InMemoryMemoryProvider,
    MemoryProvider,
    MemorySnapshot,
    default_memdir_path,
)
from .types import (
    MEMORY_TYPES,
    MemoryEntry,
    MemoryError,  # noqa: A004
    MemoryFormatError,
    MemoryFrontmatter,
    MemoryNotFoundError,
    MemoryType,
)

__all__ = [
    "DEFAULT_STALE_AFTER",
    "INDEX_FILE_NAME",
    "MAX_INDEX_BYTES",
    "MAX_INDEX_LINES",
    "MAX_MEMORY_FILES",
    "MEMORY_SECTION_NAME",
    "MEMORY_TYPES",
    "DummyMemory",
    "FileMemoryProvider",
    "InMemoryMemoryProvider",
    "Memory",
    "MemoryEntry",
    "MemoryError",
    "MemoryFormatError",
    "MemoryFrontmatter",
    "MemoryNotFoundError",
    "MemoryProvider",
    "MemorySnapshot",
    "MemoryType",
    "default_memdir_path",
    "load_memory_entry",
    "memory_system_prompt_section",
    "parse_memory_md",
    "render_memory_block",
    "scan_memdir",
    "truncate_index",
]
