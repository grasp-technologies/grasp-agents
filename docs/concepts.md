# Concepts

grasp-agents is a typed Python framework for LLM agents and structured LLM
workflows. You build the agent once and choose how to run it:

- **Embedded**, as the AI layer of an application: typed agents and pipelines
  behind product features, with the application owning business state.
- **Standalone**, as a long-running personal or research agent: tools,
  sub-agents, sandboxed execution, memory, skills, and a terminal UI, resumable
  across sessions.

The agent loop and the LLM layer are the core. Everything heavier (file, shell,
code and notebook tools, sandboxing, memory, skills, MCP, durable
checkpoints, compaction, evals, tracing, the TUI) is a layer you opt into. A bare
`LLMAgent` touches neither the host filesystem nor the network beyond its LLM
calls.

## Principles

1. **Opt-in layers.** Every capability beyond the loop is off until you wire it
   (`enable_memory=`, `enable_skills=`, `checkpoint_store=`, `enable_compaction=`,
   an execution environment, …). Defaults never compact, persist, execute code
   or read files on your behalf.
2. **Typed end to end.** Every unit is a `Processor[InT, OutT, CtxT]`, generic
   in its input, output and shared-context types. Tool inputs, tool outputs and
   agent outputs are Pydantic models, and the code is checked with pyright in strict mode.
3. **Hooks first, subclassing second.** Agents are customized with
   `@agent.add_*` decorators. Observer and decision hooks stack, while
   transform hooks hold a single slot. Overriding the `*_impl` methods in a subclass
   remains available for invasive changes. See [Hooks](hooks.md).
4. **Mechanism on primitives, policy in apps.** The framework ships seams and
   reference strategies. Thresholds, UX and business rules belong to the
   application.
5. **Provider-native, provider-agnostic.** Each provider has its own converters
   to and from one internal item and event schema. Provider settings use the vendor
   SDK's own schemas, with no lowest-common-denominator API. See [LLM layer](llm.md).
6. **Streaming first.** `run_stream()` yields typed events, and `run()` is built on
   top of it.
7. **One session per `SessionContext`, bound at construction.** Context flows
   top-down through composition. Execution resources (shells, kernels,
   background tasks) are session-scoped and released only by `aclose()`. See
   [Session](session.md).
8. **The transcript is an immutable log; the model sees a projection of it.**
   Context management is opt-in and only ever changes the view. See
   [Context window](context-window.md).
9. **Durable by construction, honest about guarantees.** With a checkpoint
   store, agents checkpoint at every closed boundary and resume after a crash.
   The crash contract is at-least-once, and every guarantee holds on the file and
   in-memory stores. Transactional stores strengthen it. See
   [Durability](durability.md).
10. **Applications own business state.** `ctx.state` is the session's working
    copy. Durable external writes go through deferred effects, which commit with
    the checkpoint.
11. **One writer per piece of state.** The context window manager is the only
    writer of the log, one session writer owns session-scoped persistence, and
    there is one live writer per checkpoint key.
12. **Nothing touches the host without explicit wiring.** Once it is wired, one
    `SandboxPolicy` is enforced on two planes. `confinement="none"` is a
    trusted-host convenience, not a sandbox. See [Sandbox](sandbox.md).
13. **Failures surface.** Store failures propagate instead of degrading
    silently. Tool errors are returned to the model as structured results, and LLM
    errors are typed and carry a recovery hint.

## Shapes of use

| Shape | Building blocks | Page |
|---|---|---|
| One agent driving tools | `LLMAgent`, `BaseTool`, `@function_tool`, built-in tool sets | [Agent loop](agent-loop.md) |
| Typed pipelines | `SequentialWorkflow`, `LoopedWorkflow`, `ParallelProcessor` | [Orchestration](orchestration.md) |
| Manager and workers | `.as_tool()` on any agent or processor | [Orchestration](orchestration.md) |
| Dynamic multi-agent graphs | `Runner` + per-payload routing (`add_recipient_selector`) | [Orchestration](orchestration.md) |
| Resident peer agents (experimental) | `grasp_agents.agent_team` | [Orchestration](orchestration.md) |

Any of these can be wrapped as a tool for another agent, and any can run
inside a `with SessionContext(...)` block that supplies shared state, stores,
the execution environment and usage tracking.

```python
from pydantic import BaseModel

from grasp_agents import LLMAgent
from grasp_agents.llm_providers import AnthropicLLM


class Answer(BaseModel):
    text: str
    confidence: float


agent = LLMAgent[str, Answer, None](
    name="qa",
    llm=AnthropicLLM(model_name="claude-haiku-5-5"),
    sys_prompt="Answer concisely.",
)

result = await agent.run("What is a monad?")
print(result.payloads[0].confidence)
```

## Layers and packages

| Layer | Packages / modules | Install extra |
|---|---|---|
| Pure data (events, items, content, packets, errors) | `types/` | — |
| Contracts and primitives | `hooks.py`, `selector.py`, `session_context.py`, `printer.py`, `grasp_logging.py` | — |
| LLM layer | `llm/`, `llm_providers/`, `rate_limiting/` | `anthropic`, `gemini`, `bedrock`, `vertex`, `all-llm-providers` |
| Agent core | `agent/`, `processors/`, `context/` | — |
| Composition | `workflow/`, `runner/`, `runtime/`, `agent_team/` (experimental) | — |
| Tools | `tools/` (function tools, agent-as-tool, shell, code, notebooks, files) | `code-exec`, `notebook-edit`, `notebook` |
| Storage and execution | `file_backend/`, `sandbox/` (local, Seatbelt, srt, E2B) | `e2b` |
| Capabilities | `memory/`, `skills/`, `mcp/` | `mcp` |
| Durability | `durability/`, `inbox.py`, `mailbox.py` | — |
| Quality and operations | `evals/` (+ `grasp-evals` CLI), `telemetry/`, `usage_tracker.py` | `phoenix` |
| UI | `ui/` (Rich `EventConsole`; Textual app) | `tui` |

**Import direction.**
- `types/` imports nothing outside itself.
- Shared abstractions used by several packages (for example `file_backend/`)
  live below their consumers, never inside one of them.
- `agent/` is the composition root: it wires tools, memory, skills, the sandbox and
  durability together.
- Feature packages should depend on `agent/` only for types (under
  `TYPE_CHECKING`). The exceptions are the adapters whose job is to bridge into
  an agent (agent-as-tool, task tools, MCP tools).
- Within a package, use relative imports (`from .x import …`). Across packages,
  use absolute ones. Never use `from ..x`.

**Public API surface.** The root `grasp_agents` package exports the headline
API (`LLMAgent`, `Processor`, the workflows, `Runner`, `SessionContext`,
`BaseTool`, `function_tool`, message types, stores, …). Everything else is
imported from its subpackage, such as `grasp_agents.llm_providers`,
`grasp_agents.context`, `grasp_agents.durability` or `grasp_agents.types.events`.
Each subpackage curates its own `__all__`.

## What is distinctive

- **Typed composition.** Agents, workflows, parallel fan-out and graphs are the
  same generic `Processor` type. Wrapping one as a tool keeps its types.
- **Side effects commit with checkpoints.** Tools defer durable writes with
  `agent_ctx.defer_effect`. The writes run when the conversation state is saved,
  atomically on a transactional store, and rollback never replays them. See
  [Durability](durability.md).
- **Linear rollback that understands the session.** Rolling back to a human
  turn also voids mailbox messages consumed after it, re-injects background
  task outcomes, and can restore the workspace snapshot. See
  [Rollback](rollback.md).
- **In-process background work.** Any tool or sub-agent can move to the
  background and push its result back as a note, with no server and no polling.
  See [Background tasks](background-tasks.md).
- **Notebook-grade execution.** Persistent Python kernels and cell-level
  notebook tools, local or remote. See [Notebooks](notebooks.md).
- **One sandbox policy, two enforcement planes.** Path checks in the file
  tools and OS confinement (Seatbelt / srt) are generated from the same
  `SandboxPolicy`.
- **Open observability.** OpenTelemetry spans, an optional Phoenix exporter,
  and an evaluation toolkit in the same package. See
  [Observability](observability.md) and [Evals](evals.md).

## What it deliberately does not do

- **Branching or forking sessions.** Rollback is linear, to human-turn
  boundaries. Use a graph runtime if you need time travel across branches.
- **Hosted, provider-coupled features as framework mechanisms.** Server-side
  state such as `previous_response_id` is available per call through the
  provider settings, but the loop stays stateless toward the provider by default.
- **Channels, gateways, cron daemons, UI products.** These belong to the host
  application. The framework offers primitives such as resident agents and a
  wakeup tool, and the host drives them.
- **Compaction or memory by default.** Both are available and nothing happens
  until you enable them.

## See also

- [Docs index](README.md)
- The repository [README](../README.md) for installation and quick examples.
