"""
Cross-session memory — Claude-Code-style memdir loader and provider ABC.

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
populating that field with a :class:`MemoryProvider` is the only wiring
step.

Distinct from :class:`LLMAgentTranscript` (the per-run message history owned
by an agent). "Transcript" = within-run conversation; "memory" =
cross-session knowledge.

See ``docs/roadmap/13-memory-system.md``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .injection import (
    MEMORY_RELEVANCE_ATTACHMENT_NAME,
    MEMORY_SECTION_NAME,
    make_memory_section,
    memory_relevance_attachment,
    memory_system_prompt_section,
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
    FileMemoryProvider,
    InMemoryMemoryProvider,
    MemoryProvider,
    MemorySelector,
    MemorySnapshot,
    default_memdir_path,
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

if TYPE_CHECKING:
    from .mcp_provider import MCPMemoryProvider


def __getattr__(name: str) -> Any:
    if name == "MCPMemoryProvider":
        try:
            module = importlib.import_module(".mcp_provider", __name__)
        except ImportError as exc:
            raise AttributeError(
                "MCPMemoryProvider requires the 'mcp' optional dependency; "
                "install with: pip install grasp-agents[mcp]"
            ) from exc
        attr = module.MCPMemoryProvider
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_MAX_SELECT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_STALE_AFTER",
    "INDEX_FILE_NAME",
    "MAX_INDEX_BYTES",
    "MAX_INDEX_LINES",
    "MAX_MEMORY_FILES",
    "MEMORY_RELEVANCE_ATTACHMENT_NAME",
    "MEMORY_SECTION_NAME",
    "MEMORY_TYPES",
    "SELECT_MEMORIES_SYSTEM_PROMPT",
    "FileMemoryProvider",
    "InMemoryMemoryProvider",
    "MCPMemoryProvider",
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
    "memory_relevance_attachment",
    "memory_system_prompt_section",
    "parse_memory_md",
    "render_memory_index",
    "render_memory_instructions",
    "scan_memdir",
    "truncate_index",
]
