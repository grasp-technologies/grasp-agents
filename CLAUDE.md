# CLAUDE.md — grasp-agents

## What this project is

grasp-agents is a lightweight, type-safe Python framework for building LLM-powered applications: both purely agentic and structured. Its design must enable both embedding inside larger production systems as well as usage as a personal assistant agent. The primary production consumer is **grasp-core** (at `../grasp-core`), which uses it to power an edtech platform: course generation, lesson writing, interactive tutoring, exercise grading, and resource recommendation.

Key principles:
- Transparent, clean, easily extensible architecture
- Complex abstractions only when they leverage model capabilities in a maximally efficient way
- Provider-agnostic (OpenAI, Anthropic, Gemini, any LiteLLM-supported provider)
- Type safety via Python generics and Pydantic

## Build & dev commands

Package manager is **uv**. Build system is **hatchling**.

```bash
uv sync                  # install deps + lockfile
uv build                 # build sdist + wheel
uv run pytest            # run tests (in src/grasp_agents/examples/)
uv run ruff check .      # lint (ruff selects ALL, see ruff.toml for ignores)
uv run ruff format .     # format (line-length 88, LF endings)
uv run pyright           # type check (strict mode, Python 3.12)
```

**Integration tests** (real API/network calls — E2B, OpenAI, Anthropic, …) are
gated behind the `integration` marker and deselected by default
(`addopts = "-m 'not integration'"` in `pyproject.toml`). Run them by overriding
the marker:

```bash
uv run pytest -m integration                      # all integration tests
uv run pytest tests/sandbox/test_e2b.py -m integration   # one file
```

Keys come from `.env` (gitignored), loaded by `tests/conftest.py` via
`load_dotenv()` — `E2B_API_KEY`, `OPENAI_API_KEY`, etc.

**Always invoke tooling via `uv run` — never `.venv/bin/…` directly.**

- **Tests**: `uv run pytest …` (incl. `-m integration`). This form is exempt
  from the command sandbox (`excludedCommands` in `.claude/settings.local.json`),
  so it runs unsandboxed and its network + Jupyter sockets work. Run integration
  tests **foreground**; do **not** pipe through `tee`/`tail` or background with
  output redirection — the output buffers and you lose progress.
- **Lint / type-check**: `uv run --no-sync ruff …` / `uv run --no-sync pyright …`.
  The `--no-sync` is **required**: a bare `uv run ruff` panics under the sandbox
  — `uv`'s sync step builds an HTTP client that probes macOS system-proxy via a
  `configd` mach service the seatbelt blocks (`SCDynamicStore` → NULL → panic);
  `--offline` / clearing proxy env don't help, but `--no-sync` skips the sync and
  runs the tool in the already-synced venv.

A sandboxed *test* invocation (`.venv/bin/pytest`, or any non-exempt form) fails
spuriously — SOCKS-proxy / `Operation not permitted` errors that are sandbox
artifacts, **not** real failures.

Pre-commit hooks: cspell (spell check), large file check, uv sync, uv pip compile.

Publishing: tag with `v*` triggers GitHub Actions → PyPI trusted publishing.

## Code conventions

- **Python >=3.12.12**
- **Pyright strict mode** — all public APIs must be well-typed
- **Ruff** with `select = ["ALL"]` and explicit ignores in `ruff.toml`. Key: no docstring enforcement (D1xx ignored), annotations not enforced (ANN ignored), `print()` allowed (T201 ignored)
- **Line length**: 88
- **Line endings**: LF
- Pydantic `BaseModel` for all data types that cross API boundaries (tool inputs, agent outputs, state)
- Async throughout — all agent/tool `run()` methods are `async`
- Commit messages: short imperative descriptions (e.g., "update LiteLLM", "version bump", "fix copilot comments")

## Comments and docstrings guidance**

* Comments and docstrings are **for framework users, not implementers.** Default to none; add one only when the *why* is non-obvious or there's a contract a caller must honor.
* Never explain design history (that belongs in commits / memory, not in source). Only explain the current design and follow the rule above when doing so.
* Never refer to our plans, roadmap, or your memory.
* Never refer to other frameworks (e.g. "similar to OpenClaw's tool call events"), unless the code is copied verbatim (to avoid breaking any license rules). Keep such notes in your memory.
* Do not state variable types when they are clear from type annotations.
* You are free to edit existing comments/docstrings that violate these rules without asking.
  
## Architecture overview

### Core hierarchy

```
Processor[InT, OutT, CtxT]          # generic computation unit with routing
  └─ LLMAgent[InT, OutT, CtxT]      # agent with LLM + tools + agentic loop (AgentLoop)
  └─ ParallelProcessor[...]         # fan-out over the same processor
  └─ WorkflowProcessor[...]         # Sequential/Looped workflows
  └─ Runner[...]                    # "agent team" orchestration with dynamic routing
```

`Processor` is the base class for all computation units. Routing between processors (used by `Runner` in multi-agent setups) is dynamic: any `Processor` can override `select_recipients_impl(output, ctx, exec_id)` to pick downstream recipients per-payload based on content, state, or LLM judgment.

### Key components

| Component | Location | Purpose |
|-----------|----------|---------|
| `LLMAgent` | `agent/llm_agent.py` | Core agent: LLM + tools + `AgentLoop` |
| `AgentLoop` | `agent/agent_loop.py` | The agentic loop with RL-style PRE-ACT/ACT/JUDGE/OBSERVE phases; enforces `max_turns`, injects dangling-tool-call cancellations on exhaustion, integrates `BackgroundTaskManager` |
| `LLMAgentTranscript` | `agent/llm_agent_transcript.py` | Conversation history (per-run messages). **Renamed from `LLMAgentMemory`**; `agent.transcript`, not `agent.memory`. Note `RunContext.memory` now means the cross-session memdir — a different concept. |
| `AgentContext` | `agent/agent_context.py` | Per-agent, per-loop mutable state (transcript, tools, file-edit state, bg tasks, kernel/session holders), passed to every tool call — the seam that keeps tools stateless. |
| `PromptBuilder` | `agent/prompt_builder.py` | System/input prompt construction; uses the `InputRenderable` protocol |
| `@function_tool` | `tools/function_tool.py` | Decorator turning async Python functions into typed `BaseTool`s |
| `AgentTool` / `ProcessorTool` | `tools/agent_tool.py`, `tools/processor_tool.py` | `.as_tool()` wrappers — any agent/processor becomes a tool. Stateless: ctx is bound per call, so a single wrapper is safe to share across agents / LRU-cached instances. |
| `BaseTool` + event/item/content types | `types/tool.py`, `types/events.py`, `types/items.py`, `types/content.py`, `types/response.py` | Typed streaming events and message parts |
| Hook `Protocol`s | `types/hooks.py` | `SystemPromptBuilder`, `BeforeLlmHook`, `AfterToolHook`, etc. — consolidated in one module |
| `RunContext[CtxT]` | `run_context.py` | Runtime state container (user state, usage tracker, printer) |
| `Packet[T]` | `packet.py` | Multi-payload container with per-payload routing |
| `Runner` + `EventBus` | `runner/runner.py`, `runner/event_bus.py` | Multi-agent orchestration with packet routing and event bubbling |
| `SequentialWorkflow`, `LoopedWorkflow`, `WorkflowProcessor` | `workflow/` | Linear and cyclic processor chains |
| `Processor`, `ParallelProcessor` | `processors/` | Base `Processor` and concurrent fan-out processor |
| `CheckpointStore`, `AgentCheckpoint` etc., resume helpers, `TaskRecord` | `durability/` | Session persistence + CC-style resume + `BackgroundTaskManager` |
| `LLM`, `CloudLLM`, `FallbackLLM`, `RetryPolicy`, `model_info.py` | `llm/` | Base LLM abstractions, multi-model fallback cascade, resilience, token/capability metadata |
| MCP client (tools, resources, prompts) | `mcp/` | Model Context Protocol client |
| Sandbox + exec + background tasks | `sandbox/` (`policy.py`, `exec_backend.py`, `environment.py`, `kernel.py`, `local/`, `e2b/`), `agent/background_tasks.py` | Opt-in two-plane sandbox: local (`confinement="none"`) / Seatbelt+srt / E2B. `SandboxPolicy` (file + OS planes), `ExecBackend`, `KernelCapable`/`SessionCapable`/`SnapshotCapable` seams (now ABCs), durable `BackgroundTaskManager` |
| Terminal / code-exec / notebook tools | `tools/bash.py`, `bash_session.py`, `code_interpreter.py` (`RunPython`), `notebook_exec.py` (`RunCell`), `file_edit/notebook.py` (`NotebookRead`/`NotebookEdit`) | `Bash` (fresh) + `BashSession` (persistent shell); `RunPython` REPL; cell-level notebook read/edit/execute behind `KernelCapable` (local `jupyter_client` / E2B) |
| File tools + pluggable backends | `tools/file_edit/`, `tools/file_search/`, `tools/file_backend/` (`base.py`, `local.py`, `mcp.py`), `tools/file_toolkit.py` | `Read`/`Write`/`Edit`/`Delete`/`Glob`/`Grep` over `ctx.file_backend` (local fs or MCP); 9-strategy fuzzy edit + mtime read-before-write |
| Terminal UI | `ui/` (`console.py`, `_event_render.py`, `app.py`, `notebook.py`) | Shared event-render layer; Rich `EventConsole` (always available) + opt-in Textual app behind the `[tui]` extra (zero impact on a non-TUI install) |
| Tracing `@traced` + Phoenix/Traceloop | `telemetry/` (`decorators.py`, `phoenix.py`, `exporters.py`, `setup.py`) | Optional observability; span-decorator on processors/methods |

### LLM providers

All providers live under `llm_providers/` and inherit from `CloudLLM` (`llm/cloud_llm.py`). Each ships its own converters (provider input/output ↔ our `Response` item types) — no shared intermediate format:

| Class | Location | Notes |
|-------|----------|-------|
| `OpenAIResponsesLLM` | `llm_providers/openai_responses/responses_llm.py` | OpenAI Responses API (first-class) |
| `OpenAILLM` | `llm_providers/openai_completions/completions_llm.py` | OpenAI Chat Completions; also used for Gemini/OpenRouter OpenAI-compat endpoints |
| `AnthropicLLM` | `llm_providers/anthropic/anthropic_llm.py` | Anthropic Messages API (native) |
| `GeminiLLM` | `llm_providers/gemini/gemini_llm.py` | Google GenAI SDK (native) |
| `LiteLLM` | `llm_providers/litellm/lite_llm.py` | Long-tail (100+) providers via `litellm` |

`FallbackLLM` (`llm/fallback_llm.py`) composes a cascade of any of the above for model-level fallback.

### Customization via decorators

Agents are customized through decorator hooks on an `LLMAgent` instance. Subclassing is also supported (and useful for more complex processors/agents):

```python
@agent.add_system_prompt_builder    # dynamic system prompt from context
@agent.add_input_content_builder    # format input from typed args + state
@agent.add_output_parser            # parse LLM text into typed output
@agent.add_final_answer_extractor   # choose when a turn's output is the final answer
@agent.add_transcript_builder       # custom transcript initialization
@agent.add_before_llm_hook          # inspect/modify settings before each LLM call
@agent.add_after_llm_hook           # observe completed LLM responses
@agent.add_before_tool_hook         # pre-tool hook (per tool call)
@agent.add_after_tool_hook          # post-tool hook
@agent.add_tool_input_converter(tool_name=...)   # custom typed-args → tool input
@agent.add_tool_output_converter(tool_name=...)  # custom tool result → message
```

Hook function signatures are defined as `Protocol`s in `types/hooks.py`. Dynamic multi-agent routing is done by overriding `select_recipients_impl` on the processor (not a decorator).

### Agents as tools

Any processor/workflow can be converted to a tool via `.as_tool()`, enabling manager/worker architectures where the manager agent calls sub-workflows as tools.

## Directory structure

```
src/
  grasp_agents/               # single top-level package
    agent/                    # LLMAgent, AgentLoop, AgentContext, LLMAgentTranscript,
                              # PromptBuilder, loop_state (NextStep ADT), approval store,
                              # BackgroundTaskManager
    tools/                    # @function_tool, AgentTool/ProcessorTool (.as_tool()),
                              # Bash/BashSession, RunPython (code_interpreter), RunCell +
                              # notebook tools, task tools; file_edit/ + file_search/ +
                              # file_backend/ (Local/MCP) + FileToolkit
    sandbox/                  # SandboxPolicy (file+OS planes), ExecBackend, environments,
                              # KernelCapable/SessionCapable/SnapshotCapable (ABCs);
                              # local/ (none/Seatbelt/srt) + e2b/
    llm/                      # LLM / CloudLLM / FallbackLLM / RetryPolicy / model_info
    llm_providers/            # one subdir per provider, each with own converters:
      openai_responses/       #   OpenAIResponsesLLM
      openai_completions/     #   OpenAILLM (Chat Completions)
      anthropic/              #   AnthropicLLM (native Messages API)
      gemini/                 #   GeminiLLM (native google-genai)
      litellm/                #   LiteLLM (long-tail providers)
    processors/               # Processor, ParallelProcessor
    workflow/                 # SequentialWorkflow, LoopedWorkflow, WorkflowProcessor
    runner/                   # Runner + EventBus (multi-agent orchestration)
    durability/               # CheckpointStore (ABC), resume, append-log, TaskRecord
    skills/                   # skills registry + injection + load_skill/list_skills tools
    memory/                   # MemoryProvider + memdir + relevance selectors + injection
    types/                    # content, events, items, hooks (Protocols), tool, response
    mcp/                      # MCP client (tools + resources + prompts)
    telemetry/                # @traced decorators, Phoenix + Traceloop exporters
    ui/                       # EventConsole + _event_render (shared) + Textual app ([tui] extra)
    rate_limiting/            # Rate limiter for LLM calls
    data_retrieval/           # AsyncHTTPXRetriever + caching/batching primitives for tool kits
    utils/                    # Helpers (streaming merging, etc.)
    kits/                     # Domain-specific tool kits built on grasp_agents
      research/               #   research kits: arXiv, S2, HF Papers, Tavily, OpenAlex, vault
    examples/
      notebooks/              #   Jupyter demos
    run_context.py            # RunContext[CtxT]
    packet.py                 # Packet[T]
    env_section.py            # env-info / current-time prompt sections
    usage_tracker.py          # token/cost accounting
    printer.py, grasp_logging.py
```

## How grasp-core uses this library

grasp-core's `libs/grasp_data/src/grasp_data/course_gen/` is the primary consumer. Key patterns:

- **Syllabus generation**: Multi-agent pipeline (planner → reviewer → editor) with interactive learner input via tools. Planner uses `Runner` with `select_recipients_impl` for conditional routing.
- **Lesson generation**: `ParallelProcessor` runs web/YouTube search agents across lessons concurrently, then a writer agent generates content. Uses `stream_concurrent()` to merge parallel streams.
- **Mentor/tutoring**: Long-running agent with persistent memory across chat turns. Memory manually serialized to/from DB as JSON. LRU cache for agent instances.
- **Tools mutate state**: Tools like `AskLearnerTool`, `MakeDraftSyllabusTool` modify `RunContext.state` directly and return status strings.
- **Context management**: Done via hooks/callbacks — `cleanup_stale_tool_calls()`, `select_top_k_resources()`, learner info summarization. Task-specific, not generic.

## Important design decisions

- **Hooks over subclassing (primary), but subclassing is kept**: The primary customization path for `LLMAgent` is decorator hooks (`@agent.add_*`). Subclassing is supported and useful for more complex processors/agents — the forwarding layer (`_register_overridden_implementations`) stays.
- **State mutations via context**: `RunContext[CtxT].state` carries user-defined state through the entire pipeline. Tools and hooks mutate it directly.
- **No universal context management, no universal compaction**: Context window management and summarization are task-specific and handled via callbacks. The framework provides the hook points, not the strategies.
- **DB/application is the source of truth for *business* state**: The framework persists *conversation history* (via `durability/`), not business artifacts; `CtxT` is rebuilt on resume via `@agent.add_state_builder`. The agent's *working filesystem* within its sandbox `allowed_roots`, by contrast, is *session* state — snapshotted opt-in via an optional `SnapshotCapable` backend capability (the framework delegates to the MCP server / sandbox provider / local shadow-git and stores only a ref; it never claims the user's real VCS-managed tree). See `docs/roadmap/15-sandbox-and-terminal.md`.
- **Streaming first**: All agents support `.run_stream()` yielding typed events; non-streaming `.run()` is built on top.
- **Provider-agnostic**: Five peer LLM providers under `llm_providers/` (OpenAI Responses, OpenAI Chat Completions, Anthropic, Gemini, LiteLLM), each with its own converters. Model-specific features (reasoning_effort, thinking, web_search) exposed via provider-specific settings classes (`OpenAIResponsesLLMSettings`, `AnthropicLLMSettings`, etc.).
- **Sandbox-ready**: The framework does not assume unrestricted host access. File / terminal / code-execution / notebook tools are **shipped**, additive and opt-in, and run under a two-plane `SandboxPolicy` (local `confinement="none"` / Seatbelt+srt / E2B). See `docs/roadmap/14-file-edit-tools.md`, `15-sandbox-and-terminal.md`, `24-notebook-tools.md`.

## Telemetry

Optional observability via Arize Phoenix (local Docker):

```bash
cd phoenix && docker compose up -d
# Set PHOENIX_COLLECTOR_HTTP_ENDPOINT=http://localhost:6006/v1/traces
```

Initialize in code: `init_tracing(project_name=...)` (`telemetry/setup.py`) to install a TracerProvider, then `init_phoenix(...)` (`telemetry/phoenix.py`) to attach the Phoenix exporter. Spans are emitted by `@traced` decorators (`telemetry/decorators.py`) applied to processor/method entrypoints.

## Roadmap

A **rough** roadmap for future development can be found in `docs/roadmap/.
