import json
import logging
from datetime import timedelta
from functools import cached_property
from typing import Any

from pydantic import BaseModel

from grasp_agents.run_context import RunContext
from grasp_agents.types.content import InputImage, InputText
from grasp_agents.types.items import ToolOutputPart
from grasp_agents.types.tool import BaseTool, ToolProgressCallback

from .json_schema import json_schema_to_pydantic

try:
    from mcp import ClientSession
    from mcp.types import CallToolResult as McpToolResult
    from mcp.types import (
        EmbeddedResource,
        ImageContent,
        ResourceLink,
        TextContent,
    )
    from mcp.types import Tool as McpToolDef
except ImportError as _err:
    msg = (
        "MCP support requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
    raise ImportError(msg) from _err

logger = logging.getLogger(__name__)


class MCPTool(BaseTool[BaseModel, McpToolResult, None]):
    """
    A tool backed by an MCP server.

    Created by :class:`MCPClient` during tool discovery.

    When the server defines an ``outputSchema``, the structured result
    is validated and available via :pyattr:`last_structured_result`.
    """

    _copy_shared_attrs = frozenset({"_session"})

    def __init__(
        self,
        *,
        session: ClientSession,
        tool_def: McpToolDef,
        timeout: float | None = 30.0,
    ) -> None:
        super().__init__(
            name=tool_def.name,
            description=tool_def.description or "",
            timeout=timeout,
        )
        self._session = session
        self._tool_def = tool_def

        self._in_type = json_schema_to_pydantic(
            tool_def.inputSchema, f"{tool_def.name}_input"
        )
        self._out_type = McpToolResult

    @cached_property
    def input_json_schema(self) -> str:
        return json.dumps(self._tool_def.inputSchema)

    @cached_property
    def struct_output_schema(self) -> type[BaseModel] | None:
        return (
            json_schema_to_pydantic(self._tool_def.outputSchema, f"{self.name}_output")
            if self._tool_def.outputSchema is not None
            else None
        )

    async def _run(
        self,
        inp: BaseModel,
        *,
        ctx: RunContext[None] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        meta: dict[str, Any] | None = None,
    ) -> McpToolResult:
        del ctx, exec_id
        timeout_delta = (
            timedelta(seconds=self.timeout) if self.timeout is not None else None
        )
        return await self._session.call_tool(
            name=self.name,
            arguments=inp.model_dump(),
            progress_callback=progress_callback,
            meta=meta,
            read_timeout_seconds=timeout_delta,
        )


def mcp_tool_result_to_llm_input_parts(
    result: McpToolResult,
    tool_name: str | None = None,
    struct_output_schema: BaseModel | None = None,
) -> list[ToolOutputPart]:
    """Convert MCP content blocks to grasp-agents ToolOutputPart items."""
    parts: list[ToolOutputPart] = []

    if result.isError:
        error_text_parts = [
            p.text for p in (result.content or []) if isinstance(p, TextContent)
        ]
        error_text = (
            "\n".join(error_text_parts) if error_text_parts else "Unknown MCP error"
        )
        msg = f"[MCP Error] {tool_name}: {error_text}"
        logger.warning("MCP tool '%s' failed: %s", tool_name, error_text)

        return [InputText(text=msg)]

    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(InputText(text=block.text))

        elif isinstance(block, ImageContent):
            data_url = f"data:{block.mimeType};base64,{block.data}"
            parts.append(InputImage(image_url=data_url))

        elif isinstance(block, EmbeddedResource):
            raise NotImplementedError(
                "EmbeddedResource content blocks are not yet supported in MCP tools"
            )

        elif isinstance(block, ResourceLink):
            raise NotImplementedError(
                "ResourceLink content blocks are not yet supported in MCP tools"
            )

    if result.structuredContent and struct_output_schema is not None:
        struct_output = struct_output_schema.model_validate(result.structuredContent)
        parts.append(InputText(text=json.dumps(struct_output, indent=2)))

    return parts
