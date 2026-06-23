# Grasp Agents

<br/>
<picture>
  <source srcset="https://raw.githubusercontent.com/grasp-technologies/grasp-agents/master/.assets/grasp-dark.svg" media="(prefers-color-scheme: dark)">
  <img src="https://raw.githubusercontent.com/grasp-technologies/grasp-agents/master/.assets/grasp.svg" alt="Grasp Agents"/>
</picture>
<br/>
<br/>

[![PyPI version](https://badge.fury.io/py/grasp_agents.svg)](https://badge.fury.io/py/grasp-agents)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](https://mit-license.org/)
[![PyPI downloads](https://img.shields.io/pypi/dm/grasp-agents?style=flat-square)](https://pypi.org/project/grasp-agents/)
[![GitHub Stars](https://img.shields.io/github/stars/grasp-technologies/grasp-agents?style=social)](https://github.com/grasp-technologies/grasp-agents/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/grasp-technologies/grasp-agents?style=social)](https://github.com/grasp-technologies/grasp-agents/network/members)

## Overview

**Grasp Agents** is a lightweight, modular Python framework for building LLM
agents and structured workflows. It focuses on clean and flexible architecture that allows for rapid experimentation, while maintaining resilience and durability necessary for production use. You build the agent and choose how to run it:

- **Embedded** in an application, as the orchestration layer behind its AI
  features; or
- **Standalone**, as a personal agent with its own cross-session memory, skills, sandboxing. A simple multi-agent Terminal UI (using [Textual](https://github.com/textualize/textual)) is included.

The core is small. Tools, sandboxing, memory, skills, durable persistence, MCP,
and the terminal UI are additive layers you opt into.

## Features

- **Strict static typing end to end.** Agents and pipeline steps are
  `Processor[InT, OutT, CtxT]`, generic in their input, output, and shared
  context types; tool inputs/outputs and agent outputs are Pydantic models.
- **Native providers, no lowest-common-denominator.** [OpenResponses](https://www.openresponses.org/) as an internal LLM abstraction layer, with first-class adapters for OpenAI Chat Completions, Responses, Anthropic, Gemini, and LiteLLM. Each adapter exposes provider-specific settings directly (e.g. reasoning, caching, server-side Web search) using the exact same schemas shipped with the provider SDK. `FallbackLLM` cascades across any of them.
First-class integrations.
- **Three composition patterns, one framework.** A single agent driving tools;
  typed processor pipelines (sequential / looped / parallel fan-out); and
  dynamic multi-agent graphs (`Runner` with per-payload routing). Any of them
  can be wrapped as a tool for an agent via `.as_tool()`.
- **Opt-in runtime.** File, shell, code-execution, and notebook tools; a
  two-plane sandbox; cross-session memory; skills; and an MCP client are all
  additive. Nothing touches the host filesystem or network unless you ask for it.
- **Durable by construction.** Agents, workflows, and multi-agent runs
  checkpoint as they go and resume after a crash without re-running completed
  work. Checkpoints are produced at each level of nested composition: e.g. an interrupted subagent in a multi-agent setup resumes without being re-run by its parent. Long-running tool calls are backgrounded as tasks, and their intermediate results are persisted and reported to the parent when it is resumed.
- **Streaming-first.** `.run_stream()` yields typed events (text deltas,
  thinking, tool calls, routing decisions); `.run()` is built on top.

## Capabilities

**Composition & orchestration**

- `LLMAgent` — an LLM with tools and an agentic loop (PRE-ACT/ACT/JUDGE/OBSERVE
  phases, `max_turns`, per-run and per-tool timeouts, forced final answer).
- `SequentialWorkflow`, `LoopedWorkflow`, `ParallelProcessor` — typed,
  composable processing chains and concurrent fan-out.
- `Runner` — multi-agent orchestration over an in-process event bus, with fully
  dynamic per-payload routing (`select_recipients_impl` / `add_recipient_selector`).
- `.as_tool()` — turn any agent or workflow into a tool used by another agent.

**Models**

- Agent-first OpenResponses abstraction layer.
- `FallbackLLM` model cascade, `RetryPolicy` (per-error-type backoff + jitter),
  token/cost/capability metadata, and request rate limiting.

**Tools**

- `@function_tool` — turn an async function into a typed tool (schema inferred
  from the signature and docstring); or subclass `BaseTool` directly.
- File tools (`Read` / `ReadImage` / `Write` / `Edit` / `Delete` / `Glob` / `Grep`) over a
  pluggable backend (local filesystem or MCP), with fuzzy-match edits and
  read-before-write mtime guards.
- Terminal (`Bash`, persistent `BashSession`), REPL code execution (`RunPython`),
  and notebook tools (`RunCell` / `NotebookRead` / `NotebookEdit`).
- MCP: tools, resources, and prompts.

**Sandboxing**

- An opt-in two-plane `SandboxPolicy` (filesystem roots + OS/network). Backends:
  local (no isolation), macOS Seatbelt with [srt](https://github.com/anthropic-experimental/sandbox-runtime) delegation, and remote [E2B](https://github.com/e2b-dev/e2b). The
  file, code, and notebook tools run under whichever backend you bind. Easily extendable to other file and execution backends.

**Memory & skills**

- Cross-session memory: a markdown memory directory with an always-loaded index,
  topic files with frontmatter, and per-turn relevance selection. Authored
  through the generic file tools — no dedicated memory tools (Claude Code inspired).
- Skills: a `SKILL.md` catalog injected into the system prompt, loaded on demand,
  and slash-invocable with arguments.

**Durability**

- `CheckpointStore` (filesystem and in-memory, extendable) with append-only message logs.
  Agents, workflows, and runner teams resume after a crash without re-running
  completed steps. A background-task manager runs long work out of the loop and
  reports or resumes it on restart.

**Output & observability**

- Typed streaming events for text, thinking, tool calls, and routing.
- Rendering/debugging: a Rich `EventConsole` (always available) and an opt-in Textual TUI (experimental but actively developed) with separate subagent views.
- OTel tracing with the backend of your choice (the repo includes a Phoenix / OpenLLMetry exporter).

**Customization**

- Decorator hooks for system/input prompt formatting, output parsing, context engineering (e.g. transcript pruning, compaction), and before/after LLM and tool
  call interventions. Subclassing as an alternative to hooks for more invasive changes.

## What it's for

- **Embedding LLM features in a product.** Typed agents and pipelines as the
  orchestration layer behind an application's AI features. State flows through
  `RunContext`; the application stays the source of truth and rebuilds context
  on resume.

  Orchestration/durability example: [`orchestration_durability_demo.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/orchestration_durability_demo.ipynb)

- **Long-running personal & research agents.** A single agent with tools, subagents,
  cross-session memory, and skills that persists and resumes across sessions —
  driven interactively from the terminal UI or run headless. The runtime ships with the framework: a markdown memory directory, a skills
  catalog, file / shell / code / notebook tools, an opt-in sandbox that confines
  a native process (so local GPU/MPS stays reachable, unlike a VM or container),
  durable checkpoint-and-resume, and background tasks.

  Auto-research example where an agent experiments in a real notebook
  inside an `srt`-sandboxed workspace and tracks results against a hidden holdout: [`autoresearch.py`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/tui/autoresearch.py)
  

  ```bash
  python -m grasp_agents.examples.tui.autoresearch            # interactive TUI
  python -m grasp_agents.examples.tui.autoresearch --headless
  ```

## Installation

The base install includes the OpenAI and LiteLLM providers, the file / shell /
code tools, memory, skills, durability, and the Rich console. Native Anthropic
and Gemini, MCP, the local notebook kernel, the E2B sandbox, the Textual UI, and
tracing are optional extras.

```bash
uv add grasp_agents          # or: pip install grasp_agents
```

| Install | Adds |
|---|---|
| `grasp_agents` | Core: OpenAI + LiteLLM, file/shell/code tools, memory, skills, durability, console |
| `grasp_agents[anthropic]` | Native Anthropic provider |
| `grasp_agents[gemini]` | Native Gemini provider |
| `grasp_agents[all-llm-providers]` | Anthropic + Gemini + Bedrock/Vertex auth deps |
| `grasp_agents[bedrock]` | Claude on AWS Bedrock (adds `boto3` for SigV4) |
| `grasp_agents[vertex]` | Claude + Gemini on Google Vertex AI (adds `google-auth`) |
| `grasp_agents[mcp]` | MCP client |
| `grasp_agents[notebook]` | Local Jupyter kernel for `RunPython` / `RunCell` |
| `grasp_agents[notebook-edit]` | `NotebookRead` / `NotebookEdit` (no kernel) |
| `grasp_agents[e2b]` | Remote E2B sandbox backend |
| `grasp_agents[tui]` | Textual terminal UI |
| `grasp_agents[phoenix]` | Phoenix / OpenLLMetry tracing |

API keys are read from the environment. The same models are also reachable
through AWS Bedrock, Google Vertex AI, and Azure OpenAI — see
[Cloud-hosted models](#cloud-hosted-models).

## Minimal example

A typed single agent with one tool, streamed to the console:

```python
import asyncio
from dotenv import load_dotenv
from grasp_agents import (
    LLMAgent,
    ProcPacketOutEvent,
    function_tool,
    render_events,
)
from grasp_agents.llm_providers.litellm import LiteLLM

load_dotenv()

@function_tool
async def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"It's 18°C and clear in {city}."

# LLMAgent[InT, OutT, CtxT]: this one takes a str, returns a str, no shared state.
agent = LLMAgent[str, str, None](
    name="assistant",
    llm=LiteLLM(model_name="claude-sonnet-4-6"),
    tools=[get_weather],
    sys_prompt="Be helpful.",
    stream_llm=True,
)

async def main() -> None:
    # render_events renders the run live (system prompt, tool calls, deltas).
    # ProcPacketOutEvent carries the typed final output (here, str).
    async for event in render_events(agent.run_stream("What's the weather in Paris?")):
        if isinstance(event, ProcPacketOutEvent):
            print("\nResult:", event.data.payloads[0])

asyncio.run(main())
```

```bash
uv run quickstart.py
```

`.run()` returns the same typed `Packet` without streaming:

```python
out = await agent.run("What's the weather in Paris?")
print(out.payloads[0])
```

## Cloud-hosted models

The native providers reach the same models through the major cloud platforms,
selected with a `platform=` argument; platform-specific client args go in a
typed `platform_config`. The request/response surface is identical, so tools,
streaming, structured outputs, and `FallbackLLM` all work unchanged.

```python
from grasp_agents.llm_providers.anthropic import AnthropicLLM
from grasp_agents.llm_providers.openai_completions import OpenAILLM
from grasp_agents.llm_providers.gemini import GeminiLLM

# Claude on AWS Bedrock (AWS cred chain / profile / bearer token)
bedrock = AnthropicLLM(
    model_name="anthropic.claude-sonnet-4-5-20250929-v1:0",
    platform="bedrock",                  # or "bedrock_mantle" for newest models
    platform_config={"aws_region": "us-east-1"},
)

# Claude on Google Vertex AI
claude_vertex = AnthropicLLM(
    model_name="claude-sonnet-4-5@20250929",
    platform="vertex",
    platform_config={"project_id": "my-project", "region": "us-east5"},
)

# Azure OpenAI — model_name is the deployment name (key or Entra ID auth)
azure = OpenAILLM(
    model_name="my-gpt-deployment",
    platform="azure",
    platform_config={
        "azure_endpoint": "https://my-resource.openai.azure.com",
        "api_version": "2024-10-21",
    },
)

# Gemini on Google Vertex AI
gemini_vertex = GeminiLLM(
    model_name="gemini-2.5-flash",
    platform="vertex",
    platform_config={"project": "my-project", "location": "us-central1"},
)
```

Anything not set in `platform_config` falls back to each platform's own
credential chain and environment variables. The shared `http_client` /
`default_headers` args and a per-provider `extra_*_client_params` escape hatch
are available on every provider.

### Server-side conversation state (OpenAI Responses)

By default the loop is stateless toward the provider: it sends the full
transcript each turn. To instead chain turns server-side with the Responses API
(`previous_response_id`), wire it with the LLM hooks — the framework leaves the
strategy to you. `OpenAIResponsesLLM` slices the input to only the new items
whenever `previous_response_id` (or `conversation`) is set.

```python
last: dict[str, str] = {}  # keyed by exec_id (one run); or use current_run_context().state

@agent.add_after_llm_hook
async def _capture(response, *, exec_id, turn):
    last[exec_id] = response.id

@agent.add_before_llm_hook
async def _chain(*, exec_id, turn, extra_llm_settings):
    if rid := last.get(exec_id):
        extra_llm_settings["previous_response_id"] = rid
```

Pair it with `llm_settings={"store": True}` (the provider-side persistence
toggle); `store=False` keeps responses unstored.

## More examples

Runnable notebooks in
[`src/grasp_agents/examples/notebooks/`](https://github.com/grasp-technologies/grasp-agents/tree/master/src/grasp_agents/examples/notebooks):

- [`basics.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/basics.ipynb)
  — agent basics: typed agents, validated/structured outputs, multimodal input,
  the tool loop, streaming, parallel runs, sequential workflows, agents-as-tools.
- [`advanced_patterns.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/advanced_patterns.ipynb)
  — provider-specific features (thinking, web search, grounding), the full hook
  system, and forced ReAct.
- [`orchestration_durability.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/orchestration_durability.ipynb)
  — composition, multi-agent `Runner` routing, and crash/resume durability.
- [`code_interpreter.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/code_interpreter.ipynb)
  — `RunPython` in a persistent, srt-confined kernel.
- [`memory_skills.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/memory_skills.ipynb)
  — cross-session memory and skills end to end.
- [`mcp_memory.ipynb`](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/mcp_memory.ipynb)
  — the same memory surface over an MCP file backend.

## Project structure

```
src/grasp_agents/
  agent/            LLMAgent, AgentLoop, AgentContext, transcript, background tasks
  processors/       Processor, ParallelProcessor
  workflow/         SequentialWorkflow, LoopedWorkflow, WorkflowProcessor
  runner/           Runner + EventBus (multi-agent orchestration)
  context/          PromptBuilder, prompt sections (env info, untrusted-content fencing)
  llm/              LLM / CloudLLM / FallbackLLM / RetryPolicy / model metadata
  llm_providers/    OpenAI Responses, OpenAI Completions, Anthropic, Gemini, LiteLLM
  tools/            BaseTool, @function_tool, .as_tool(), Bash, RunPython, file tools
  file_backend/     FileBackend ABC + local / MCP backends
  sandbox/          SandboxPolicy, ExecBackend, environments (local / Seatbelt / E2B)
  memory/           cross-session memory provider + relevance selection
  skills/           skills registry + injection + load_skill
  durability/       CheckpointStore, resume, background-task records
  mcp/              MCP client (tools + resources + prompts)
  types/            pure data: content, events, items, response, packet, errors
  telemetry/        @traced + Phoenix / OpenLLMetry exporters
  ui/               EventConsole (Rich) + Textual app ([tui] extra)
  run_context.py    RunContext[CtxT]
  hooks.py          hook Protocols
```

## License

MIT — see [LICENSE.md](LICENSE.md).
