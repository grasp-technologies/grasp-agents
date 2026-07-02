"""
``SystemPromptSection`` factory for MCP server-supplied instructions.

Each :class:`MCPClient` may report an ``instructions`` string when it
connects (per the MCP ``InitializeResult`` spec). This module builds a
single section that concatenates the instructions of every connected
client. The section carries a ``cache_control`` checkpoint: it is rendered
last, so the marker caches the whole system-prompt prefix up to and
including it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from grasp_agents.context.prompt_builder import SystemPromptSection
from grasp_agents.context.untrusted_content import wrap_untrusted
from grasp_agents.types.content import CacheControl

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from grasp_agents.session_context import SessionContext

    from .client import MCPClient


MCP_INSTRUCTIONS_SECTION_NAME = "mcp_instructions"

_BLOCK_HEADING = "## MCP server instructions"


def make_mcp_instructions_section(
    clients: Sequence[MCPClient] | Callable[[], Sequence[MCPClient]],
    *,
    section_name: str = MCP_INSTRUCTIONS_SECTION_NAME,
) -> SystemPromptSection:
    """
    Build a section that renders every connected client's instructions.

    ``clients`` may be a static :class:`Sequence` (snapshot at section-
    construction time) or a zero-arg callable that returns the current
    sequence — useful when clients connect / disconnect across the agent's
    lifetime, so the section reflects whatever is wired *now*.
    Disconnected clients (no instructions, or never connected) are skipped.
    The section returns ``None`` when no client supplies instructions, so the
    block is omitted from the prompt entirely.
    """

    async def compute(  # noqa: RUF029
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        **_: Any,
    ) -> str | None:
        del ctx, exec_id
        current = clients() if callable(clients) else clients
        blocks: list[str] = []
        for client in current:
            text = client.instructions
            if not text:
                continue
            # Server-supplied text is third-party content: fence it so a
            # malicious/compromised server's instructions read as data, not
            # as trusted system-prompt directives.
            fenced = wrap_untrusted(text.strip(), source=f"mcp:{client.name}")
            blocks.append(f"### {client.name}\n\n{fenced}")
        if not blocks:
            return None
        return "\n\n".join([_BLOCK_HEADING, *blocks])

    return SystemPromptSection(
        name=section_name,
        compute=compute,
        cache_control=CacheControl(),
    )
