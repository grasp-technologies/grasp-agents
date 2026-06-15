from __future__ import annotations

import logging
import time
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
    msg = (
        "MCP support requires the 'mcp' package. "
        "Install with: pip install grasp-agents[mcp]"
    )
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

        server = MCPServerStdio(command="python", args=["server.py"])
        async with MCPClient("my-server", server=server) as client:
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
        self._instructions: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def server_capabilities(self) -> ServerCapabilities | None:
        """Server capabilities discovered during connection."""
        return self._capabilities

    @property
    def instructions(self) -> str | None:
        """Server-supplied instructions text (per MCP ``InitializeResult``)."""
        return self._instructions

    @property
    def session(self) -> ClientSession:
        """
        Return the underlying :class:`ClientSession`.

        Raises :class:`RuntimeError` if the client is not connected. Lets
        adapters that wrap an MCP client (file backend, memory provider,
        resource index) share the same session without reaching into
        private attributes.
        """
        return self._require_session()

    async def connect(self) -> None:
        """
        Connect to the MCP server and discover tools.

        Exception-safe: a failed handshake or discovery tears down the
        transport (terminating a stdio server subprocess) and leaves the
        client cleanly disconnected, so a retrying ``connect()`` starts
        over instead of finding a half-connected, wedged client.
        """
        if self._tools is not None:
            return

        t0 = time.monotonic()
        exit_stack = AsyncExitStack()
        try:
            if isinstance(self._server, MCPServerStdio):
                server_params = StdioServerParameters(
                    command=self._server.command,
                    args=self._server.args,
                    env=self._server.env,
                )
                transport = await exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            else:
                try:
                    # Deferred: SSE transport needs the optional httpx-sse extra.
                    from mcp.client.sse import sse_client  # noqa: PLC0415
                except ImportError as err:
                    msg = (
                        "SSE transport requires the 'mcp' package with "
                        "httpx-sse support."
                    )
                    raise ImportError(msg) from err
                transport = await exit_stack.enter_async_context(
                    sse_client(self._server.url)
                )

            read, write = transport
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            init_result = await session.initialize()
            capabilities = init_result.capabilities

            tools: list[MCPTool | MCPListResourcesTool | MCPReadResourceTool] = []

            response = await session.list_tools()
            for t in response.tools:
                try:
                    tools.append(
                        MCPTool(session=session, tool_def=t, timeout=self._tool_timeout)
                    )
                except Exception:
                    # One tool with a pathological schema must not take down the
                    # whole server connection — skip it and keep the rest.
                    _logger.exception(
                        "Skipping MCP tool %r (server %r): failed to build its "
                        "input schema",
                        t.name,
                        self.name,
                    )

            if capabilities.resources:
                tools.extend(
                    [
                        MCPListResourcesTool(session=session, server_name=self._name),
                        MCPReadResourceTool(session=session, server_name=self._name),
                    ]
                )
        except BaseException:
            await exit_stack.aclose()
            raise

        # Commit only after full success — a retry of a failed connect must
        # find a cleanly disconnected client.
        self._exit_stack = exit_stack
        self._session = session
        self._capabilities = capabilities
        self._instructions = init_result.instructions
        self._tools = tools
        _logger.info(
            "MCPClient '%s': connected in %.2fs, discovered %d tools (resources: %s)",
            self._name,
            time.monotonic() - t0,
            len(self._tools),
            "yes" if capabilities.resources else "no",
        )

    async def close(self) -> None:
        """Close the connection and terminate the server process."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            _logger.info("MCPClient '%s': disconnected", self._name)
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
