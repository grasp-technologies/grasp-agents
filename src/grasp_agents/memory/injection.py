from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agent.prompt_builder import SystemPromptSection

if TYPE_CHECKING:
    from ..run_context import RunContext
    from .provider import MemorySnapshot

MEMORY_SECTION_NAME = "memory"


def render_memory_block(snapshot: MemorySnapshot) -> str | None:
    """
    Render the memory section content. Returns ``None`` when nothing to inject.

    Currently injects only the always-loaded ``MEMORY.md`` index (Claude Code's
    pattern). Topic files are addressable through the snapshot for direct
    lookup but are not eagerly added to the system prompt — adding a relevance
    selector is deferred.
    """
    if snapshot.index is None or not snapshot.index.strip():
        return None

    parts: list[str] = ["# memory"]
    if snapshot.index_freshness_warning:
        parts.extend(["", snapshot.index_freshness_warning])
    parts.extend(["", snapshot.index.strip()])
    return "\n".join(parts)


async def _compute_memory_section(
    *, ctx: RunContext[Any] | None = None, exec_id: str | None = None
) -> str | None:
    if ctx is None or ctx.memory is None:
        return None
    snapshot = await ctx.memory.load(session_id=exec_id or "", ctx=ctx)
    return render_memory_block(snapshot)


memory_system_prompt_section = SystemPromptSection(
    name=MEMORY_SECTION_NAME,
    compute=_compute_memory_section,
)
