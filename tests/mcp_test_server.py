"""Simple MCP server for testing. Run via stdio transport."""

import asyncio

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

mcp = FastMCP("test-server")


class MathResult(BaseModel):
    result: float
    operation: str


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back."""
    return message


@mcp.tool()
def multiply(a: float, b: float) -> MathResult:
    """Multiply two numbers with structured output."""
    return MathResult(result=a * b, operation="multiply")


@mcp.tool()
def no_schema_tool(x: int):
    """Tool with no return annotation — no output schema."""
    return f"got {x}"


@mcp.tool()
def failing_tool() -> str:
    """A tool that always fails."""
    msg = "Intentional failure"
    raise RuntimeError(msg)


@mcp.tool()
async def slow_tool() -> str:
    """A tool that takes a long time."""
    await asyncio.sleep(10)
    return "done"


# ---------- Resources ----------


@mcp.resource("docs://readme", description="Project readme", mime_type="text/plain")
def readme_resource() -> str:
    """Return the project readme."""
    return "# Test Project\n\nThis is a test project for MCP."


@mcp.resource(
    "docs://items/{item_id}",
    description="Item details by ID",
    mime_type="application/json",
)
def item_resource(item_id: str) -> str:
    """Return item details."""
    return f'{{"id": "{item_id}", "name": "Item {item_id}"}}'


# ---------- Prompts ----------


@mcp.prompt(name="greet", description="Generate a greeting message")
def greet_prompt(name: str) -> str:
    """Generate a greeting."""
    return f"Please greet {name} warmly."


@mcp.prompt(name="summarize", description="Summarize text")
def summarize_prompt(text: str, style: str = "brief") -> str:
    """Generate a summarization prompt."""
    return f"Summarize the following text in a {style} style:\n\n{text}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
