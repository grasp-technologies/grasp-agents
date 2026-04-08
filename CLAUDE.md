# CLAUDE.md — grasp-agents

## What this project is

grasp-agents is a lightweight, type-safe Python framework for building structured LLM-powered workflows with agentic capabilities. It is designed to be embedded inside applications (not used as a standalone agent). The primary production consumer is **grasp-core** (at `../grasp-core`), which uses it to power an edtech platform: course generation, lesson writing, interactive tutoring, exercise grading, and resource recommendation.

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
uv run pyright           # type check (strict mode, Python 3.11)
```

Pre-commit hooks: cspell (spell check), large file check, uv sync, uv pip compile.

Publishing: tag with `v*` triggers GitHub Actions → PyPI trusted publishing.

## Code conventions

- **Python >=3.11.4**
- **Pyright strict mode** — all public APIs must be well-typed
- **Ruff** with `select = ["ALL"]` and explicit ignores in `ruff.toml`. Key: no docstring enforcement (D1xx ignored), annotations not enforced (ANN ignored), `print()` allowed (T201 ignored)
- **Line length**: 88
- **Line endings**: LF
- Generic types use the pattern `ClassName[InT, OutT, CtxT]` where `InT` = input type, `OutT` = output type, `CtxT` = context state type
- Pydantic `BaseModel` for all data types that cross API boundaries (tool inputs, agent outputs, state)
- Async throughout — all agent/tool `run()` methods are `async`
- Commit messages: short imperative descriptions (e.g., "update LiteLLM", "version bump", "fix copilot comments")

## Architecture overview

### Core hierarchy

```
BaseProcessor[InT, OutT, CtxT]      # abstract base for all computation units
  └─ Processor[InT, OutT, CtxT]     # generic processor with routing
       └─ LLMAgent[InT, OutT, CtxT] # agent with LLM + tools + agentic loop
```

### Key components

| Component | Location | Purpose |
|-----------|----------|---------|
| `LLMAgent` | `llm_agent.py` | Core agent: LLM + tools + ReAct loop |
| `LLMPolicyExecutor` | `llm_policy_executor.py` | Implements the agentic loop (generate → tool calls → observe → repeat) |
| `BaseTool` | `typing/tool.py` | Abstract tool with typed Pydantic input/output |
| `LLMAgentMemory` | `llm_agent_memory.py` | Conversation history (messages) |
| `RunContext[CtxT]` | `run_context.py` | Runtime state container (user state, usage tracker, printer) |
| `Packet[T]` | `packet.py` | Multi-payload container with per-payload routing |
| `Runner` | `runner.py` | Multi-agent orchestration with packet routing |
| `SequentialWorkflow` | `workflow/sequential_workflow.py` | Linear chain of processors |
| `LoopedWorkflow` | `workflow/looped_workflow.py` | Cyclic processor chain with termination |
| `ParallelProcessor` | `processors/parallel_processor.py` | Runs a processor on multiple inputs concurrently |
| `PromptBuilder` | `prompt_builder.py` | System/input prompt construction |
| Event types | `typing/events.py` | Typed streaming events (CompletionChunk, ToolCall, GenMessage, etc.) |

### LLM providers

| Class | Location | Providers |
|-------|----------|-----------|
| `OpenAILLM` | `openai/` | OpenAI, Gemini (via openai compat), OpenRouter |
| `LiteLLM` | `litellm/` | 100+ providers via litellm (Anthropic, Bedrock, Vertex, etc.) |

### Customization via decorators

Agents are customized through decorator hooks, not subclassing:

```python
@agent.add_system_prompt_builder    # dynamic system prompt from context
@agent.add_input_content_builder    # format input from typed args + state
@agent.add_output_parser            # parse LLM text into typed output
@agent.add_recipient_selector       # route output to specific agents
@agent.add_before_llm_hook     # modify settings before each LLM call
@agent.add_tool_output_converter    # custom tool result → message conversion
@agent.add_memory_builder           # custom memory initialization
```

### Agents as tools

Any processor/workflow can be converted to a tool via `.as_tool()`, enabling manager/worker architectures where the manager agent calls sub-workflows as tools.

## Directory structure

```
src/
  grasp_agents/           # main package
    openai/               # OpenAI LLM provider
    litellm/              # LiteLLM provider
    processors/           # BaseProcessor, Processor, ParallelProcessor
    workflow/             # SequentialWorkflow, LoopedWorkflow
    typing/               # Message, Event, Tool, Content, Completion types
    tracing_decorators/   # @workflow, @task, @tool span decorators
    telemetry/            # Phoenix + Traceloop integration
    rate_limiting/        # Rate limiter for LLM calls
    utils/                # Helpers (streaming, etc.)
    examples/             # Demo notebooks and scripts
      demo/               # pip/poetry/uv example projects
      notebooks/          # Jupyter demos (agents_demo.ipynb, litellm_tests.ipynb)
  grasp_agents_kits/      # Domain-specific kits built on grasp_agents
    education/            # Edtech-specific: planner, resources, context presets
```

## How grasp-core uses this library

grasp-core's `libs/grasp_data/src/grasp_data/course_gen/` is the primary consumer. Key patterns:

- **Syllabus generation**: Multi-agent pipeline (planner → reviewer → editor) with interactive learner input via tools. Planner uses `Runner` with `select_recipients_impl` for conditional routing.
- **Lesson generation**: `ParallelProcessor` runs web/YouTube search agents across lessons concurrently, then a writer agent generates content. Uses `stream_concurrent()` to merge parallel streams.
- **Mentor/tutoring**: Long-running agent with persistent memory across chat turns. Memory manually serialized to/from DB as JSON. LRU cache for agent instances.
- **Tools mutate state**: Tools like `AskLearnerTool`, `MakeDraftSyllabusTool` modify `RunContext.state` directly and return status strings.
- **Context management**: Done via hooks/callbacks — `cleanup_stale_tool_calls()`, `select_top_k_resources()`, learner info summarization. Task-specific, not generic.

## Important design decisions

- **Hooks over subclassing**: Agent behavior is customized via decorator hooks (`@agent.add_*`), not by subclassing `LLMAgent`. This keeps the agent class generic.
- **State mutations via context**: `RunContext[CtxT].state` carries user-defined state through the entire pipeline. Tools and hooks mutate it directly.
- **No universal context management**: Context window management is task-specific and handled via callbacks. The framework provides the hook points, not the strategies.
- **Streaming first**: All agents support `.run_stream()` yielding typed events. Non-streaming `.run()` is built on top of streaming.
- **Provider-agnostic**: No dependency on a specific LLM provider. Model-specific features (reasoning_effort, thinking, web_search) exposed via provider-specific settings.

## Telemetry

Optional observability via Arize Phoenix (local Docker):

```bash
cd phoenix && docker compose up -d
# Set PHOENIX_COLLECTOR_HTTP_ENDPOINT=http://localhost:6006/v1/traces
```

Initialize in code: `init_traceloop()` for spans, `init_phoenix()` for the Phoenix backend.

## Roadmap

A **rough** roadmap for future development can be found in `docs/roadmap/.
