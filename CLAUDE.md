# CLAUDE.md — grasp-agents

## What this project is

grasp-agents is a typed Python framework for LLM agents and structured LLM
workflows. The same code runs **embedded** as the AI layer of an application
(typed agents and pipelines, with the app owning business state) or
**standalone** as a long-running personal or research agent (tools, sandboxed
execution, memory, skills, a terminal UI, resumable sessions).

Key principles. The full list is in [docs/concepts.md](docs/concepts.md).
- **Opt-in layers.** The loop and the LLM layer are the core. Tools, sandbox,
  memory, skills, MCP, durability, compaction, evals, tracing and the TUI are
  off until wired.
- **Typed end to end.** `Processor[InT, OutT, CtxT]` generics, Pydantic at
  every API boundary, pyright strict.
- **Mechanism on primitives, policy in apps.** Ship seams and reference
  strategies, never hard-coded thresholds or business rules.
- **Provider-native, provider-agnostic.** Each provider has its own converters, and its settings
  use the vendor SDK's own schemas.
- **A new abstraction must earn its place.** It must use a model capability
  well or remove real duplication. Otherwise, write a recipe on existing hooks.

## Where things are documented

- `docs/`: one reference page per subsystem (current design, contracts,
  limitations). **Read the relevant page before changing a subsystem, and update
  it in the same change.** The index is `docs/README.md`.
- `.claude/skills/`: workflows for working on this repo (checks, tests,
  tools, providers, durability changes, releases, public writing) and for
  using `grasp_agents.evals` (`evals`).
- `src/grasp_agents/examples/`: runnable notebooks and TUI demos.

## Build & dev commands

Package manager is **uv**, build system is **hatchling**. Run tooling via
`uv run` (add `--no-sync` when the environment can't reach PyPI).

```bash
uv sync                                        # install deps + lockfile
uv run pytest                                  # unit tests (integration deselected)
uv run pytest -m integration tests/<path>      # live API / network tests (keys in env)
uv run ruff check <files>                      # lint (autofixes: fix = true)
uv run ruff check --no-fix .                   # whole-tree lint without rewriting files
uv run ruff format <files>                     # format only files you touched
uv run pyright src/grasp_agents                # type check (strict, Python 3.12)
```

- `ruff.toml` sets `fix = true` and `select = ["ALL"]`, so `ruff check .` rewrites
  the whole tree. Lint and format only the files you changed, and use
  `--no-fix` for broad runs. `F401` is unfixable: remove unused imports by hand.
- Integration tests (real provider, E2B or Phoenix calls) carry
  `@pytest.mark.integration` and are deselected by default. API keys come from
  the environment or a local `.env` (gitignored, loaded by `tests/conftest.py`).
  Use the cheapest model tier for live tests.
- There is no CI for lint, types or tests; run them locally before pushing.
  Pre-commit hooks run cspell (US spelling; project words in `cspell.json`), a
  large-file check, `uv sync` and `uv pip compile`.
- Publishing: a `v*` tag triggers PyPI trusted publishing (`workflow.yml`). A
  `legacy-v*` tag publishes the frozen `grasp_agents_legacy` distribution from
  `legacy/`.
- Never use `git stash` or `git reset --hard` to test something; work on the
  tree as it is.

See the `dev-checks`, `writing-tests` and `release` skills for details.

## Test layout & conventions

`tests/` mirrors `src/grasp_agents/`, with one subdir per subpackage (`agent`,
`agent_team`, `context`, `durability`, `evals`, `llm`, `mcp`, `memory`,
`orchestration`, `processors`, `runtime`, `sandbox`, `skills`, `telemetry`,
`tools`, `tui`) plus the provider dirs and `integration/`. Put a new test in
the matching subdir, name files by subsystem, and keep history out of test
docstrings.

- **Shared code split.** Importable helpers (queue-driven `MockLLM`, `Response`
  builders, sample tools) live in `tests/_helpers.py`. Pytest fixtures live in
  `conftest.py`. Import absolutely (`from tests._helpers import …`); ruff `TID252`
  bans `from ..x`.
- **Async.** Use `@pytest.mark.asyncio` (`asyncio_mode = "strict"`); no `anyio`. In
  modules that mix sync and async tests, decorate async tests individually.
- **Real-API tests** are marked `@pytest.mark.integration`.

## Code conventions

- Python ≥ 3.12, PEP 695 type parameters, pyright strict: every public API is fully typed.
- Ruff with `select = ["ALL"]` and explicit ignores in `ruff.toml` (no docstring
  enforcement, annotations not enforced, `print()` allowed). Suppress a rule with
  `# ruff: ignore[<rule-name>]` rather than a bare `# noqa`.
- Line length 88, LF endings, US spelling.
- **Imports.** Within a package use `from .sibling import …`; across packages
  use `from grasp_agents.x import …`; never `from ..x`.
- **Package layering.** Dependencies point downward:
  - `types/` is a pure-data leaf that imports nothing outside itself. New
    message, event, error and content types go there.
  - Cross-cutting contracts (`hooks.py`, `selector.py`) and primitives
    (`session_context.py`, `printer.py`, `grasp_logging.py`) are top-level
    modules.
  - A shared abstraction used by several packages is its own lower package
    (e.g. `file_backend/`), never nested in one consumer.
  - `agent/` is the composition root. Feature packages (`tools`, `memory`,
    `skills`, `sandbox`, `mcp`, …) import `agent/` only under `TYPE_CHECKING`.
    The exceptions are agent adapters (agent-as-tool, task tools, MCP tools).
    Don't add new sideways runtime imports.
- **Public API.** The root `__init__.py` exports only the headline API (about 50
  names). Everything else is imported from its subpackage, and each subpackage
  curates `__all__`.
- Pydantic `BaseModel` for every data type crossing an API boundary (tool
  inputs, agent outputs, state).
- Async throughout: every agent and tool `run()` is `async`.
- **Failures surface.** Never wrap a checkpoint-store `save` in try/except. Let
  store failures propagate.
- Section-divider comments in long modules are a kept convention.
- Commit messages: short imperative descriptions ("update LiteLLM", "fix
  rollback of parked agents").

## Comments, docstrings and public text

- Comments and docstrings are **for framework users, not implementers**. The default is
  none. Add one only when the *why* is non-obvious or there's a contract a caller
  must honor.
- Explain only the current design: no design history, plans, roadmap
  references, ticket ids or names of private projects.
- Never refer to other agent frameworks, unless code is copied verbatim (then
  attribute it as the license requires).
- Don't restate types that annotations already show.
- A tool's class docstring is developer-facing. The agent sees only
  `description=` and `Field(description=...)`. `@function_tool` is the one
  place where the docstring becomes the agent-facing description.
- The same rules apply to PR descriptions and release notes (this is an
  open-source repo). See the `public-writing` skill.
- You may fix existing comments or docstrings that break these rules without
  asking.

## Architecture overview

### Core hierarchy

```
Processor[InT, OutT, CtxT]           # generic computation unit with routing
  ├─ LLMAgent[InT, OutT, CtxT]       # LLM + tools + agentic loop (AgentLoop)
  ├─ ParallelProcessor[...]          # fan-out over replicas of one processor
  └─ WorkflowProcessor[...]          # SequentialWorkflow / LoopedWorkflow
Runner                               # multi-agent graph over an EventBus (not a Processor)
AgentTeam (experimental)             # resident peer agents with mailboxes (not a Processor)
```

Routing is dynamic: a processor picks downstream recipients per payload via
`@proc.add_recipient_selector` or by overriding
`select_recipients_impl(output, *, exec_id)`.

### Key components

| Component | Location | Purpose |
|---|---|---|
| `LLMAgent` | `agent/llm_agent.py` | Agent = LLM + tools + loop. Owns the session-scoped lifecycle (shells, kernels and background tasks survive runs; released by `aclose()` / `async with`), checkpoint save/load and step rollback |
| `AgentLoop` | `agent/agent_loop.py`, `agent/loop_state.py` | Per-run executor: PRE-ACT/ACT/JUDGE phases, pure `decide_next_step` → `NextStep*` ADT, `max_turns` / `run_timeout` / per-tool timeouts, forced final answer, dangling-call closure, parking on background tasks or inbox |
| `ContextWindowManager` | `agent/context_window.py` | Sole writer of the transcript log; per-turn view projection, folds, token budget, compaction. Reached as `agent_ctx.cw`; `agent.transcript` is a read-only view; `agent.reset_transcript()` / `replace_transcript()` for callers |
| `AgentContext` | `agent/agent_context.py` | Per-agent session state (cw, tools, file-edit state, background tasks, inbox, deferred-effects outbox, kernel/shell holders). Passed to each tool call as a per-call copy (`for_tool_call`), which keeps tools stateless. Tools defer durable writes with `defer_effect` |
| `BackgroundTaskManager` | `agent/background_tasks.py`, `agent/task_progress.py` | Backgrounded tools and sub-agents, push-delivered task notes, log and result files, spill of large outputs, durable `TaskRecord`s |
| Tool decisions & approvals | `agent/tool_decision.py`, `agent/approval_store.py`, `agent/approval_callback.py` | Allow / reject-with-content / raise per call; ONCE/SESSION/ALWAYS approvals |
| Prompt + context strategies | `context/` (`prompt_builder.py`, `compaction.py`, `projection.py`, `system_reminder.py`, `env_section.py`, `untrusted_content.py`) | System-prompt sections, input attachments, opt-in compaction strategies (`Compaction`, `SummarizingCompactor`, `CollapseToolOutputsProjector`), untrusted-content fencing |
| `SessionContext[CtxT]` | `session_context.py` | Session-scoped DI container: user state, stores, environment, usage tracker, printer, session writer, fs-snapshot policy. Bound at construction (explicit → ambient `with` → process default); `current_session_context()` resolves it |
| `BaseTool`, `@function_tool` | `tools/base.py`, `tools/function_tool.py` | Tool ABC (timeouts, background axes, spill, untrusted output, conflict keys); async function → typed tool |
| `AgentTool` / `ProcessorTool` | `tools/agent_tool.py`, `tools/processor_tool.py` | `.as_tool()`: any agent or processor becomes a tool; stateless, ctx bound per call |
| Shell / code / notebook tools | `tools/bash.py`, `bash_session.py`, `code_interpreter.py` (`RunPython`), `notebook_exec.py` (`RunCell`), `file_edit/notebook.py` | `bash_tools()` = `[Bash, KillTask]`; persistent `BashSession` (experimental); kernels behind `KernelCapable` |
| File tools + backends | `tools/file_edit/`, `tools/file_search/`, `tools/file_toolkit.py`; `file_backend/` | `Read/Write/Edit/Delete/Glob/Grep/ReadImage` over `FileBackend` (local, MCP, E2B); fuzzy edits; read-before-write |
| Sandbox | `sandbox/` (`policy.py`, `exec_backend.py`, `environment.py`, `kernel.py`, `local/`, `e2b/`) | One `SandboxPolicy`, two planes. ABCs: `ExecBackend`, `ExecSession`, `KernelSession`, `ExecutionEnvironment`. Capability Protocols: `KernelCapable`, `SessionCapable`, `SnapshotCapable` (E2B only) |
| Durability | `durability/`, `inbox.py`, `mailbox.py` | `CheckpointStore` ABC (file, in-memory; optional `transaction()`), append-only message log, `AgentCheckpoint` / `SessionCheckpoint` / `TaskRecord`, schema versions, resume, store keys, durable mailboxes |
| `LLM`, `CloudLLM`, `FallbackLLM`, `RetryPolicy` | `llm/` | Base LLM abstractions, platform/api-provider resolution, retries-first fallback cascade, model info, token counting, thought signatures |
| Providers | `llm_providers/` | `OpenAIResponsesLLM`, `OpenAILLM` (Chat Completions), `AnthropicLLM`, `GeminiLLM`, `LiteLLM`; each with its own converters; import from `grasp_agents.llm_providers` |
| Composition | `processors/`, `workflow/`, `runner/`, `runtime/`, `agent_team/` | `Processor`, `ParallelProcessor`, workflows, `Runner` + `EventBus` (on the `runtime` actor driver), experimental `AgentTeam` |
| Capabilities | `memory/`, `skills/`, `mcp/` | Memdir + relevance selection; `SKILL.md` skills; MCP client (tools, resources, prompts) |
| Data types | `types/` | Pure data: items, events, content, response, packet, folds, errors, recovery hints |
| Hook Protocols | `hooks.py`, `selector.py` | Signatures for every `add_*` hook |
| Evals | `evals/` | Datasets, scorers, judges, runner, compare/pairwise, Phoenix; `grasp-evals` CLI |
| Telemetry / logging | `telemetry/`, `grasp_logging.py`, `usage_tracker.py` | `@traced` OTel spans, Phoenix exporter; logging conventions; token/cost accounting |
| UI | `ui/` | Rich `EventConsole` (always available) + Textual app (`[tui]` extra) |

`run_context.py` and the root `RunContext` name are deprecated aliases of
`SessionContext`. Don't use them in new code.

### Customization

The primary path is decorator hooks on an `LLMAgent` instance. Subclassing with
`*_impl` overrides is supported for invasive changes.

```python
@agent.add_initial_context_builder   # system prompt + leading messages (ephemeral, single slot)
@agent.add_input_content_builder     # typed args → input content (single slot)
@agent.add_output_parser             # LLM text → typed output (single slot)
@agent.add_final_answer_extractor    # decide when a turn's output is final (single slot)
@agent.add_view_projector            # per-turn model-facing view of the log (stacks)
@agent.add_compactor                 # record a summary fold under pressure (single slot)
@agent.add_before_llm_hook           # adjust settings before each LLM call (stacks)
@agent.add_after_llm_hook            # observe responses (stacks)
@agent.add_before_tool_hook          # per-call decision; most restrictive wins (stacks)
@agent.add_after_tool_hook           # observe results (stacks)
@agent.add_tool_input_converter(tool_name=...)
@agent.add_tool_output_converter(tool_name=...)
agent.add_compaction()               # opt-in default compaction bundle
agent.add_system_prompt_section(...) / agent.add_input_attachment(...)
```

[docs/hooks.md](docs/hooks.md) is the authoritative list, with signatures
(from `hooks.py`) and stacking rules.

## Load-bearing invariants (details in docs/)

- **ctx flows top-down.** A processor binds its `SessionContext` once, at
  construction. Containers cascade theirs onto children via `on_adopted`.
  **Never pass `ctx=` to a subprocessor.** Multi-tenant servers must construct
  agents per request (or inside `with SessionContext(...)`). See `docs/session.md`.
- **A run is not a resource boundary.** Shells, kernels and background tasks are
  released only by `aclose()`.
- **Tools are stateless.** All per-agent state arrives as `agent_ctx` per call.
- **The log is append-only, with one writer.** Write through `agent_ctx.cw` or
  `agent.reset_transcript()` / `replace_transcript()`. Never mutate messages in
  place. Compaction changes only the view. See `docs/context-window.md`.
- **The initial context is ephemeral.** The system prompt and leading messages are rebuilt
  every step and never persisted.
- **Checkpoints are written only at closed boundaries.** Input is durable before
  the model responds to it. A reload happens only on a cold start. Bump
  `CURRENT_SCHEMA_VERSION` with a `SCHEMA_VERSION_SUMMARIES` entry for any
  persisted-shape change. See `docs/durability.md`.
- **Deferred effects are at-least-once.** They are atomic with the checkpoint
  only on a transactional store, so effects must tolerate running twice.
- **One session writer per session, one live writer per checkpoint key.**
- **Execution is granted only through `ctx.environment`.** `confinement="none"`
  (the default) is not a sandbox. See `docs/sandbox.md`.

## Telemetry & logging

- **Tracing.** `init_tracing(project_name=...)` installs a TracerProvider, and
  `init_phoenix(...)` attaches the Phoenix exporter. Set
  `TELEMETRY_COLLECTOR_HTTP_ENDPOINT` (for example
  `http://localhost:6006/v1/traces`; run a local Phoenix with
  `cd phoenix && docker compose up -d`). Spans come from `@traced` on
  processor and method entry points.
- **Logging.** stdlib `logging` under the `grasp_agents.*` namespace. The library
  attaches no handlers (the host owns output, or calls
  `enable_verbose_stdout_logging()`). The levels:
  - INFO is a one-line operational heartbeat with timings.
  - DEBUG carries per-turn detail and content snippets.
  - WARNING is for retries, timeouts and recoverable degradation.
- **Never log a request, response or tool body ungated.** Build it with
  `body_for_log(text, full=LOG_*)` inside
  `if logger.isEnabledFor(logging.DEBUG):`. Full bodies are logged only when the
  matching `GRASP_LOG_{LLM,TOOL}_{INPUT,OUTPUT}` gate is set.
- **Don't redact or thread IDs at call sites.** `SecretRedactingFilter` and
  `install_log_correlation` / `log_context(...)` handle both.
