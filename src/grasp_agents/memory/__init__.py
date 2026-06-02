"""
Cross-session memory — Claude-Code-style memdir loader and provider.

A *memdir* is a directory of markdown files: an always-loaded ``MEMORY.md``
index plus topic ``.md`` files with YAML frontmatter (``name`` +
``description`` required, optional ``type`` from
``{user, feedback, project, reference}``). Loaded once per session as a
frozen :class:`MemorySnapshot`.

Memory **authoring** goes through the generic file-edit tools rooted at
the memdir — there are no specialized save/load/delete tools. The
``memory`` system-prompt section teaches the agent the format and
taxonomy; the file tools handle the actual reads / writes.

The ``memory`` system-prompt section auto-attaches to every :class:`LLMAgent`
when ``enable_memory=True`` and reads ``ctx.memory`` at compute time, so
populating that field with a :class:`MemoryProvider` and pointing
``ctx.file_backend`` at the surrounding backend is the only wiring step.

Distinct from :class:`LLMAgentTranscript` (the per-run message history owned
by an agent). "Transcript" = within-run conversation; "memory" =
cross-session knowledge.

See ``docs/roadmap/13-memory-system.md``.
"""

from __future__ import annotations

from .default_path import (
    GRASP_HOME_DIR_NAME,
    GRASP_MEMORY_ENV,
    MEMDIR_DIR_NAME,
    PROJECTS_DIR_NAME,
    default_memdir_path,
)
from .injection import (
    MEMORY_SECTION_NAME,
    RELEVANT_MEMORIES_ATTACHMENT_NAME,
    make_memory_section,
    memory_system_prompt_section,
    relevant_memories_attachment,
    render_memory_index,
    render_memory_instructions,
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
    InMemoryMemoryProvider,
    MemoryProvider,
    MemorySelector,
    MemorySnapshot,
)
from .selectors import (
    DEFAULT_MAX_SELECT,
    DEFAULT_MAX_TOKENS,
    SELECT_MEMORIES_SYSTEM_PROMPT,
    extract_latest_user_text,
    format_manifest,
    make_llm_relevance_selector,
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
    "DEFAULT_MAX_SELECT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_STALE_AFTER",
    "GRASP_HOME_DIR_NAME",
    "GRASP_MEMORY_ENV",
    "INDEX_FILE_NAME",
    "MAX_INDEX_BYTES",
    "MAX_INDEX_LINES",
    "MAX_MEMORY_FILES",
    "MEMDIR_DIR_NAME",
    "MEMORY_SECTION_NAME",
    "MEMORY_TYPES",
    "PROJECTS_DIR_NAME",
    "RELEVANT_MEMORIES_ATTACHMENT_NAME",
    "SELECT_MEMORIES_SYSTEM_PROMPT",
    "InMemoryMemoryProvider",
    "MemoryEntry",
    "MemoryError",
    "MemoryFormatError",
    "MemoryFrontmatter",
    "MemoryNotFoundError",
    "MemoryProvider",
    "MemorySelector",
    "MemorySnapshot",
    "MemoryType",
    "default_memdir_path",
    "extract_latest_user_text",
    "format_manifest",
    "load_memory_entry",
    "make_llm_relevance_selector",
    "make_memory_section",
    "memory_system_prompt_section",
    "parse_memory_md",
    "relevant_memories_attachment",
    "render_memory_index",
    "render_memory_instructions",
    "scan_memdir",
    "truncate_index",
]
