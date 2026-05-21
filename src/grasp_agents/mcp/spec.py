"""Setup-time spec for an MCP client attached to an agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .client import MCPClient


@dataclass(frozen=True)
class MCPClientSpec:
    """
    Wrap an :class:`MCPClient` with optional tool-name filters.

    Used at :class:`LLMAgent` construction (``mcp_clients=[...]``) or with
    :meth:`LLMAgent.add_mcp_client` (passing the spec fields as kwargs).
    Bare ``MCPClient`` instances are also accepted by the ctor — they expand
    to ``MCPClientSpec(client)`` (all tools, no filter).

    ``include`` and ``exclude`` are tool-name filters. Both can be ``None``
    (no filter on that axis). ``include={"a", "b"}`` exposes only ``a`` and
    ``b``; ``exclude={"c"}`` exposes everything except ``c``; both set
    applies the intersection. Setting ``include=set()`` (empty) blocks all
    tools but still surfaces the server's ``instructions`` to the prompt.
    """

    client: MCPClient
    include: Iterable[str] | None = None
    exclude: Iterable[str] | None = None
