# Security model

This page describes what the framework treats as trusted, how it marks content
that is not, and which layers actually enforce limits. The short version:
prompt-level fencing tells the model what is data; approvals and the sandbox
are what stop an action. Fencing alone does not make an agent safe to point at
hostile content.

## Trust boundaries

| Content | Trust | How it reaches the model |
|---|---|---|
| `sys_prompt` and system-prompt sections you register | trusted | system message |
| user input (`chat_inputs`, `in_args`), input attachments | trusted | user message |
| skills (catalog, injected bodies, `load_skill` results) | trusted | system message / tool result, unfenced |
| memory (index section, relevant-memories attachment) | trusted | system message / user message, unfenced |
| results of tools with `untrusted_output=False` (default), incl. sub-agents via `.as_tool()` / `AgentTool` | trusted | tool result, unfenced |
| results of tools with `untrusted_output=True` | untrusted | tool result inside `<untrusted_content>` |
| MCP server instructions | untrusted | system message, inside `<untrusted_content>` |
| provider-native web search results | not fenced | inside the assistant's own message |

## The untrusted-content boundary

A tool whose result carries outside content sets `untrusted_output = True`
(class attribute, `BaseTool(untrusted_output=...)`, or
`@function_tool(untrusted_output=True)`). Built-in tools that do: `Read`,
`Grep`, `NotebookRead`, `Bash`, `BashSession`, `RunPython`, `RunCell`,
`KillTask`, every `MCPTool` and the MCP resource tools.

For those tools the agent loop (`AgentLoop._convert_tool_output`) wraps the
result, after any custom output converter, with
`context/untrusted_content.py` → `wrap_untrusted`:

```
<untrusted_content source="Read">
...file text...
</untrusted_content>
```

- Any `<untrusted_content>` / `</untrusted_content>` tag inside the payload is
  neutralized (angle brackets escaped), so content cannot close the fence
  early. Other text, code and markup pass through verbatim.
- `source` is the tool name with `<`, `>`, `"` and newlines stripped (MCP tool
  names come from the server).
- Image and file parts pass through; text parts are fenced positionally.

When an agent has MCP clients or any `untrusted_output` tool at construction,
it adds the `untrusted_content` system-prompt section
(`make_untrusted_content_section`): a fixed instruction to treat tagged text as
data, never as instructions. The text is constant, so the section is
cache-stable. MCP server instructions are fenced the same way inside the
`mcp_instructions` section (`mcp/section.py`). `unwrap_untrusted` peels the
fence back off for display only; the model always gets the fenced form.

**Sub-agents are trusted by default.** A worker that reads web pages and
returns a summary hands that summary back unfenced. If a sub-agent processes
untrusted content, set `untrusted_output = True` on its tool before building
the parent agent.

## What is trusted, and what that means

System-prompt sections, skills and memory are injected without fencing: they
carry the same authority as your own instructions. Consequences:

- Whoever can write a skill directory or the memdir can instruct every session
  that loads it. Skills are read from the host filesystem at registry build
  time; keep those directories under your release control.
- Memory is written by the agent itself through the file tools. An agent that
  reads untrusted content and has memory enabled can be persuaded to persist an
  injected instruction. Restrict who writes the memdir, use one memdir per
  tenant, and gate writes with approvals when this matters
  (see [Memory](memory.md#trust)).

## Secrets

**Environment scrub.** Subprocesses started by the terminal and code tools
inherit the host environment minus variables matching
`sandbox/policy.py` → `DEFAULT_ENV_SCRUB` (case-insensitive `fnmatch`
patterns: `*_API_KEY`, `*_TOKEN`, `*SECRET*`, `*PASSWORD*`, `*_KEY`,
`DATABASE_URL`, `SSH_AUTH_SOCK`, `KUBECONFIG`, and others). Variables you pass
explicitly (`env=`, `SandboxPolicy.env`) are never scrubbed. Override with
`local_environment(env_scrub=[...])` or disable with `env_scrub=()`; pass
`inherit_host_env=False` for a minimal environment.

**File reads.** `Read` and `Grep` (content mode) run results through a
`SecretRedactor` before they reach the model. The default
`DefaultSecretRedactor` (`tools/file_edit/redact.py`) replaces well-formed
secret shapes (AWS keys, GitHub and provider API keys, JWTs, private-key
blocks) with `<REDACTED:KIND>`. It is tuned for precision, so unusual secret
formats pass. Pass `FileToolkit(redactor=...)` or `ReadTool(redactor=...)` to
plug in a stronger one, or `NullRedactor()` to opt out. The local file backend
also denies credential dotfiles (`.env`, `~/.ssh`, ...) by default
(`LocalFileBackend(include_dotfiles=True)`, `SandboxPolicy.include_dotfile_denylist`);
see [Files](files.md).

**Logs.** `grasp_logging.SecretRedactingFilter` masks API keys, tokens, bearer
headers, `NAME=secret` assignments and PEM blocks in rendered log messages.
`enable_verbose_stdout_logging()` and `setup_logging()` attach it; an embedding
host adds it to its own handlers. `GRASP_LOG_REDACT_SECRETS=0` disables it, and
is read once at import, so a command run by the agent cannot turn redaction off
mid-session. Request and response bodies are logged only as short snippets
unless their `GRASP_LOG_*` gate is set (see [Observability](observability.md)).

**Credentials in config.** `CloudLLM.api_provider` and `platform_config` are
excluded from `repr`, so they do not leak through printed configs or
tracebacks.

## Enforcement layers

Fencing and redaction reduce what the model is likely to do and see. Two layers
enforce:

- **Approvals and before-tool hooks** decide per tool call whether it runs:
  allow, reject with a message, or abort the batch. Stacked hooks merge
  most-restrictive-wins. See [Agent loop](agent-loop.md) and [Hooks](hooks.md).
- **The sandbox** (`SandboxPolicy`) confines what file tools and subprocesses
  can reach: allowed roots, read/write carve-outs, network egress, environment.
  See [Sandbox](sandbox.md).

`confinement="none"` (the `local_environment` default) is **not a sandbox**.
Subprocesses run as your user with full host access; only the file-tool plane
checks paths, and the environment scrub is the one barrier between a command
and your host's other secrets. Use Seatbelt / `srt` or E2B when the agent runs
code it wrote from untrusted input.

## Contracts

1. Output of every `untrusted_output` tool is fenced, including after a custom
   output converter (`AgentLoop._convert_tool_output`).
2. Embedded fence tags in a payload are always neutralized (`wrap_untrusted`).
3. MCP tools, MCP resource tools and MCP instructions are always untrusted.
4. Explicit `env` values are never scrubbed; inherited ones always are, unless
   `env_scrub` is overridden.
5. Log redaction cannot be disabled after import.

## Limitations

- Fencing is advisory: it relies on the model following the
  `untrusted_content` instruction.
- Provider-native web search results are not fenced; their text arrives in the
  assistant's own message.
- The `untrusted_content` section is decided at construction; an untrusted
  tool added to an agent later is fenced, but the model is not told what the
  fence means unless you add the section.
- Memory and skills have no integrity checks or signing.

## See also

- [Sandbox](sandbox.md), [Agent loop](agent-loop.md), [Files](files.md),
  [Memory](memory.md), [Skills](skills.md), [MCP](mcp.md).
- Code: `context/untrusted_content.py`, `tools/base.py` (`untrusted_output`),
  `sandbox/policy.py`, `tools/file_edit/redact.py`, `grasp_logging.py`.
