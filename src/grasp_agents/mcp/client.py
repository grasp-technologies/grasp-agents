from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Self

from .resource import MCPListResourcesTool, MCPReadResourceTool
from .tool import MCPTool

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import (
        GetPromptResult,
        ListPromptsResult,
        ServerCapabilities,
    )
except ImportError as _err:
    msg = "MCP support requires the 'mcp' package. Install with: pip install grasp-agents[mcp]"
    raise ImportError(msg) from _err

_logger = logging.getLogger(__name__)


@dataclass
class MCPServerStdio:
    """Stdio-based MCP server configuration."""

    command: str
    args: list[str] = field(default_factory=list[str])
    env: dict[str, str] | None = None


@dataclass
class MCPServerSSE:
    """SSE-based MCP server configuration."""

    url: str


MCPServerConfig = MCPServerStdio | MCPServerSSE


class MCPClient:
    """
    Connects to an MCP server and exposes its tools as BaseTool objects.

    Resource browsing tools are auto-generated if the server supports resources.
    Prompts are available via :meth:`list_prompts` and :meth:`get_prompt`.

    Usage::

        async with MCPClient("my-server", server=MCPServerStdio(command="python", args=["server.py"])) as client:
            tools = client.tools()
            agent = LLMAgent(..., tools=tools)
    """

    def __init__(
        self,
        name: str,
        *,
        server: MCPServerConfig,
        tool_timeout: float | None = 30.0,
    ) -> None:
        self._name = name
        self._server = server
        self._tool_timeout = tool_timeout
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._tools: (
            list[MCPTool | MCPListResourcesTool | MCPReadResourceTool] | None
        ) = None
        self._capabilities: ServerCapabilities | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def server_capabilities(self) -> ServerCapabilities | None:
        """Server capabilities discovered during connection."""
        return self._capabilities

    async def connect(self) -> None:
        """Connect to the MCP server and discover tools."""
        if self._session is not None:
            return

        self._exit_stack = AsyncExitStack()

        if isinstance(self._server, MCPServerStdio):
            server_params = StdioServerParameters(
                command=self._server.command,
                args=self._server.args,
                env=self._server.env,
            )
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
        else:
            try:
                from mcp.client.sse import sse_client
            except ImportError as err:
                msg = "SSE transport requires the 'mcp' package with httpx-sse support."
                raise ImportError(msg) from err
            transport = await self._exit_stack.enter_async_context(
                sse_client(self._server.url)
            )

        read, write = transport
        self._session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        init_result = await self._session.initialize()
        self._capabilities = init_result.capabilities

        tools: list[MCPTool | MCPListResourcesTool | MCPReadResourceTool] = []

        response = await self._session.list_tools()
        tools.extend(
            MCPTool(session=self._session, tool_def=t, timeout=self._tool_timeout)
            for t in response.tools
        )

        if self._capabilities.resources:
            tools.extend(
                [
                    MCPListResourcesTool(session=self._session, server_name=self._name),
                    MCPReadResourceTool(session=self._session, server_name=self._name),
                ]
            )

        self._tools = tools
        _logger.info(
            "MCPClient '%s': connected, discovered %d tools (resources: %s)",
            self._name,
            len(self._tools),
            "yes" if self._capabilities.resources else "no",
        )

    async def close(self) -> None:
        """Close the connection and terminate the server process."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self._session = None
        self._tools = None
        self._capabilities = None

    def tools(self) -> list[MCPTool | MCPListResourcesTool | MCPReadResourceTool]:
        """Return discovered tools. Must call connect() first."""
        if self._tools is None:
            msg = "Not connected. Call connect() or use as async context manager first."
            raise RuntimeError(msg)
        return self._tools

    # --- Prompts (developer-facing API) ---

    def _require_session(self) -> ClientSession:
        if self._session is None:
            msg = "Not connected. Call connect() or use as async context manager first."
            raise RuntimeError(msg)
        return self._session

    async def list_prompts(self) -> ListPromptsResult:
        """List available prompts from the MCP server."""
        return await self._require_session().list_prompts()

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> GetPromptResult:
        """Fetch a prompt by name with optional arguments."""
        return await self._require_session().get_prompt(name, arguments)

    async def __aenter__(self) -> Self:
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
