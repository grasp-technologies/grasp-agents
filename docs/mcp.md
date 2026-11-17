# MCP client

grasp-agents includes a Model Context Protocol client. A connected `MCPClient`
exposes a server's tools as ordinary `BaseTool`s, adds resource-browsing tools
when the server supports resources, renders the server's instructions into the
system prompt, and lets your code fetch the server's prompts. An MCP server can
also act as the session's file backend (`MCPFileBackend`), so the file tools
and memory operate on files the server owns.

Install the `mcp` extra: `pip install "grasp_agents[mcp]"`.

## Key types

| Name | Module | Role |
|---|---|---|
| `MCPClient` | `mcp/client.py` | Connection to one server; tool discovery; prompts |
| `MCPServerStdio(command, args=[], env=None)` | `mcp/client.py` | Launch a server subprocess over stdio |
| `MCPServerSSE(url)` | `mcp/client.py` | Connect to a server over SSE |
| `MCPClientSpec(client, include=None, exclude=None)` | `mcp/spec.py` | Per-agent tool filter |
| `MCPTool` | `mcp/tool.py` | One server tool as a `BaseTool` |
| `MCPListResourcesTool`, `MCPReadResourceTool` | `mcp/resource.py` | `<server>_list_resources`, `<server>_read_resource` |
| `make_mcp_instructions_section` | `mcp/section.py` | The `mcp_instructions` system-prompt section |
| `MCPFileBackend` | `file_backend/mcp.py` | `FileBackend` over an MCP server |

`MCPClient`, `MCPServerStdio` and `MCPServerSSE` are root exports when the
extra is installed. Everything else imports from `grasp_agents.mcp` or
`grasp_agents.file_backend`.

## Connecting and attaching

Connect the client before constructing the agent, and keep it connected for the
agent's lifetime. The agent does not close it; use the client as an async
context manager or call `close()` yourself.

```python
from grasp_agents import LLMAgent, MCPClient, MCPServerStdio
from grasp_agents.mcp import MCPClientSpec

server = MCPServerStdio(command="python", args=["server.py"])
async with MCPClient("notes", server=server, tool_timeout=30.0) as notes:
    agent = LLMAgent[str, str, None](
        name="assistant",
        llm=llm,
        mcp_clients=[MCPClientSpec(notes, exclude={"delete_note"})],
    )
    out = await agent.run("Summarize my notes from today")
```

`connect()` initializes the session, lists the tools and, if the server
advertises resources, adds the two resource tools. It is exception-safe: a
failed handshake tears the transport down (and terminates a stdio subprocess)
so a retry starts clean. A tool whose input schema cannot be converted is
skipped with a logged error; the rest of the server still connects.

`mcp_clients=` accepts `MCPClient`s and `MCPClientSpec`s. A spec filters tools
by name: `include` is an allowlist, `exclude` a blocklist, both together apply
the intersection. `include=set()` exposes no tools but still renders the
server's instructions. Two specs may partition one client's tools between
agents; the client's instructions are rendered once.

MCP tools are auto-attached: if one has the same name as an explicitly passed
tool (or an earlier auto-attached one), it is skipped with a warning instead of
raising. Tool names are the server's names, unprefixed.

## Tools and results

`MCPTool` sends the validated arguments with `exclude_none=True` (omitted
optionals are not sent as `null`) and a per-call read timeout
(`tool_timeout`, default 30 s). Its input type is a pydantic model generated
from the server's JSON schema (`mcp/json_schema.py`).

The tool returns the server's `CallToolResult`. With no output converter it
reaches the model as that result serialized to JSON (content blocks,
`structuredContent`, `isError`). Register `@agent.add_tool_output_converter(name)`
to render it differently, e.g. text blocks only or images as image parts.

All MCP tools and both resource tools have `untrusted_output=True`: their
results are fenced in `<untrusted_content source="...">` tags and the agent
gets the `untrusted_content` system-prompt section (see [Security](security.md)).

## Resources

`<server>_list_resources(cursor=)` lists resources (paginated) and resource
templates with URI, MIME type and description. `<server>_read_resource(uri)`
returns text contents; a binary content block becomes a one-line placeholder
naming its MIME type and size. Both are generated per client, named after the
client's `name`.

## Prompts

Prompts are a developer-facing API, not tools:

```python
listing = await notes.list_prompts()
prompt = await notes.get_prompt("weekly_review", {"week": "45"})
```

Render the returned messages into your agent input yourself.

## Server instructions

A server may return `instructions` in its initialize result. When an agent has
MCP clients, it registers the `mcp_instructions` section
(`make_mcp_instructions_section`), which renders each client's instructions
under a `## MCP server instructions` heading, each block wrapped in
`<untrusted_content source="mcp:<name>">`. The text comes from a third party,
so it is read as data, not as system directives. The section carries a
`CacheControl` breakpoint. Override it by registering a section with the same
name.

## MCP file backend

`MCPFileBackend(client, allowed_roots=[...], resource_uri_scheme="file://",
write_tool_name="write_file", delete_tool_name="delete_file")` routes the file
tools and memory through a server:

- Reads use `resources/read`; `stat`, `exists`, `list_dir`, glob and grep use
  a cached index built from one `resources/list` per session.
- Writes and deletes call the server's `write_file(path, content, mode)` and
  `delete_file(path)` tools and then refresh the index.
- `validate_path` only checks membership in `allowed_roots` in the server's
  address space. Path containment and sensitive-path policy are the server's
  job.

```python
from pathlib import Path
from grasp_agents import SessionContext
from grasp_agents.file_backend import MCPFileBackend
from grasp_agents.memory import MemoryProvider

backend = MCPFileBackend(client, allowed_roots=[Path("/memdir")])
ctx = SessionContext(state=None, file_backend=backend, memory=MemoryProvider(root="/memdir"))
```

`src/grasp_agents/examples/mcp_memory_server.py` is a reference server
exposing exactly this surface. See [Files](files.md).

## Contracts

1. A client must be connected before the agent is built; `tools()` raises
   otherwise (`MCPClient.tools`).
2. MCP-sourced tools never shadow explicit tools (`LLMAgent._merge_auto_attached`).
3. MCP tool results and server instructions are always fenced as untrusted
   (`MCPTool.__init__`, `mcp/section.py`).
4. The agent never closes its MCP clients; their lifetime is the caller's.

## Limitations

- Transports: stdio and SSE only. SSE needs `mcp` with `httpx-sse` support.
- No OAuth or other interactive authentication flows; pass credentials through
  the server's environment or URL.
- No sampling, elicitation or roots support; the client does not serve
  requests from the server.
- The tool list is discovered once, at `connect()`; tool-list change
  notifications are not followed.
- `MCPFileBackend` is text-only (`read_bytes` encodes the text read) and has
  no directories of its own: `mkdir` is a no-op.

## See also

- [Security](security.md), [Files](files.md), [Memory](memory.md),
  [Hooks](hooks.md).
- Code: `mcp/`, `file_backend/mcp.py`.
- Example: `src/grasp_agents/examples/notebooks/mcp_memory.ipynb`.
