from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import AnyUrl, BaseModel

from grasp_agents.types.tool import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from grasp_agents.run_context import RunContext

try:
    from mcp import ClientSession
    from mcp.types import PaginatedRequestParams, TextResourceContents
except ImportError as _err:
    msg = (
        "MCP support requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
    raise ImportError(msg) from _err

logger = logging.getLogger(__name__)


class ListResourcesInput(BaseModel):
    cursor: str | None = None
    """Optional pagination cursor from a previous list_resources call."""


class ReadResourceInput(BaseModel):
    uri: AnyUrl
    """URI of the resource to read. Use list_resources to discover available URIs."""


class MCPListResourcesTool(BaseTool[ListResourcesInput, str, None]):
    """Lists available resources and resource templates from an MCP server."""

    _copy_shared_attrs = frozenset({"_session"})

    def __init__(self, *, session: ClientSession, server_name: str) -> None:
        super().__init__(
            name=f"{server_name}_list_resources",
            description=(
                f"List available resources from the '{server_name}' server. "
                "Returns resource names, URIs, descriptions, and MIME types. "
                "Also lists resource templates (parameterized URI patterns)."
            ),
        )
        self._session = session

    async def _run(
        self,
        inp: ListResourcesInput,
        *,
        ctx: RunContext[None] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> str:
        del ctx, exec_id, progress_callback
        params = PaginatedRequestParams(cursor=inp.cursor) if inp.cursor else None
        resources_result = await self._session.list_resources(params=params)
        templates_result = await self._session.list_resource_templates()

        lines: list[str] = []

        if resources_result.resources:
            lines.append("Resources:")
            for r in resources_result.resources:
                mime = f" ({r.mimeType})" if r.mimeType else ""
                desc = f" — {r.description}" if r.description else ""
                lines.append(f"  - {r.uri}{mime}{desc}")
            if resources_result.nextCursor:
                lines.append(
                    f"  (more available, pass cursor={resources_result.nextCursor!r})"
                )

        if templates_result.resourceTemplates:
            if lines:
                lines.append("")
            lines.append("Resource Templates:")
            for t in templates_result.resourceTemplates:
                mime = f" ({t.mimeType})" if t.mimeType else ""
                desc = f" — {t.description}" if t.description else ""
                lines.append(f"  - {t.uriTemplate}{mime}{desc}")

        if not lines:
            return "No resources available."

        return "\n".join(lines)


class MCPReadResourceTool(BaseTool[ReadResourceInput, str, None]):
    """Reads a resource from an MCP server by URI."""

    _copy_shared_attrs = frozenset({"_session"})

    def __init__(
        self,
        *,
        session: ClientSession,
        server_name: str,
    ) -> None:
        super().__init__(
            name=f"{server_name}_read_resource",
            description=(
                f"Read a resource from the '{server_name}' server by URI. "
                f"Use {server_name}_list_resources first to discover available URIs "
                "and resource templates."
            ),
        )
        self._session = session

    async def _run(
        self,
        inp: ReadResourceInput,
        *,
        ctx: RunContext[None] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
    ) -> str:
        del ctx, exec_id, progress_callback
        result = await self._session.read_resource(inp.uri)

        parts: list[str] = []
        for content in result.contents:
            if isinstance(content, TextResourceContents):
                parts.append(content.text)
            else:
                mime = content.mimeType or "application/octet-stream"
                size = len(content.blob)
                parts.append(f"[Binary content: {mime}, ~{size} bytes base64-encoded]")

        return "\n".join(parts) if parts else "(empty resource)"
