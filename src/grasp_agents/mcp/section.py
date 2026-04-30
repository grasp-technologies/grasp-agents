"""
``SystemPromptSection`` factory for MCP server-supplied instructions.

Each :class:`MCPClient` may report an ``instructions`` string when it
connects (per the MCP ``InitializeResult`` spec). This module builds a
single section that concatenates the instructions of every connected
client. The section is marked ``cache_break=True`` because servers
connect / disconnect across turns and the text is not cache-stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..agent.prompt_builder import SystemPromptSection

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..run_context import RunContext
    from .client import MCPClient


MCP_INSTRUCTIONS_SECTION_NAME = "mcp_instructions"

_BLOCK_HEADING = "## MCP server instructions"


def make_mcp_instructions_section(
    clients: Sequence[MCPClient],
    *,
    section_name: str = MCP_INSTRUCTIONS_SECTION_NAME,
) -> SystemPromptSection:
    """
    Build a section that renders every connected client's instructions.

    Pass the same ``MCPClient`` instances you attached to the agent's tools.
    Disconnected clients (no instructions, or never connected) are skipped.
    The section returns ``None`` when no client supplies instructions, so the
    block is omitted from the prompt entirely.
    """

    async def compute(  # noqa: RUF029
        *, ctx: RunContext[Any] | None = None, exec_id: str | None = None
    ) -> str | None:
        del ctx, exec_id
        blocks: list[str] = []
        for client in clients:
            text = client.instructions
            if not text:
                continue
            blocks.append(f"### {client.name}\n\n{text.strip()}")
        if not blocks:
            return None
        return "\n\n".join([_BLOCK_HEADING, *blocks])

    return SystemPromptSection(
        name=section_name,
        compute=compute,
        cache_break=True,
    )
