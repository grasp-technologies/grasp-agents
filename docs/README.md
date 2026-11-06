# grasp-agents docs

Reference pages on the current design of each subsystem: what it is for, how
it works, and the contracts a contributor must not break. For installation and
quick examples, see the repository [README](../README.md). For runnable
walkthroughs, see `src/grasp_agents/examples/`.

Start with [Concepts](concepts.md).

## Core

| Page | Covers |
|---|---|
| [Concepts](concepts.md) | Principles, shapes of use, layers and import direction, what is in and out of scope |
| [Agent loop](agent-loop.md) | `LLMAgent` / `AgentLoop`, turn budgets, timeouts, tool execution, tool decisions and approvals |
| [Hooks](hooks.md) | Every customization point; which hooks stack and which hold a single slot; subclassing |
| [Session](session.md) | `SessionContext` binding, context flow, session-scoped resources, `AgentContext` |
| [Context window](context-window.md) | Transcript log and view, initial context, projectors, folds, compaction, tool-output spill |
| [Orchestration](orchestration.md) | `.as_tool()`, parallel fan-out, workflows, `Runner`, experimental agent teams |
| [LLM layer](llm.md) | Providers, platforms, settings, streaming contract, retries, fallback, errors, caching |

## Runtime

| Page | Covers |
|---|---|
| [Background tasks](background-tasks.md) | Backgrounded tools and sub-agents, delivery, logs, kill and cancel |
| [Durability](durability.md) | Checkpoint stores, keys, write order, resume, schema versions, deferred effects |
| [Rollback](rollback.md) | Step rollback, the session writer, filesystem snapshots |
| [Sandbox](sandbox.md) | `SandboxPolicy`, enforcement planes, environments, shell tools |
| [Notebooks](notebooks.md) | `RunPython`, notebook tools, kernels |
| [Files](files.md) | File tools and `FileBackend` implementations |

## Capabilities

| Page | Covers |
|---|---|
| [MCP](mcp.md) | MCP client: tools, resources, prompts |
| [Skills](skills.md) | `SKILL.md` format, loading and invocation |
| [Memory](memory.md) | Cross-session memory directory and relevance selection |
| [Security](security.md) | Trust model, untrusted-content fencing, secret handling |
| [Evals](evals.md) | `grasp_agents.evals` and the `grasp-evals` CLI |
| [Observability](observability.md) | Tracing, logging, usage and cost tracking |

The terminal UI is documented next to its code in
[`src/grasp_agents/ui/README.md`](../src/grasp_agents/ui/README.md).

## Conventions for these pages

- Pages describe the code as it is. When code and a page disagree, the code
  wins; fix the page in the same change.
- Code is referenced by module path and symbol (for example
  `agent/context_window.py` → `ContextWindowManager`), never by line number.
- Each page ends with **Contracts** (the rules that must hold) and
  **Limitations** (what is deliberately not supported).
