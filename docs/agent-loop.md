# Agent loop

`LLMAgent` is the framework's tool-using agent: an LLM, a set of tools, and an `AgentLoop` that runs generate → judge → tools → repeat until the agent produces a final answer. This page covers one run's lifecycle, the loop's decision rules, turn and time budgets, how tool calls execute, and how tool calls are approved or rejected. Read it when you need to know exactly when a run stops, what the model sees when it is cut off, or how to gate dangerous tools.

## Key types

| Name | Module | Role |
|---|---|---|
| `LLMAgent` | `agent/llm_agent.py` | Owns the session (`AgentContext`), prompt builder, hooks and checkpointing; drives one `AgentLoop` per run. |
| `AgentLoop` | `agent/agent_loop.py` | Per-run executor: phases, tool dispatch, forced final answer, dangling-call closure. Holds no resources. |
| `NextStep*`, `decide_next_step` | `agent/loop_state.py` | The six-variant next-step ADT and the pure function that picks one. |
| `ToolCallDecision` (`AllowTool`, `RejectToolContent`, `RaiseToolException`) | `agent/tool_decision.py` | What a before-tool hook returns per call. |
| `ApprovalStore`, `LocalApprovalStore`, `build_store_approval` | `agent/approval_store.py` | Asynchronous approvals with scoped allowlists. |
| `build_callback_approval`, `ApprovalCallback` | `agent/approval_callback.py` | Inline yes/no approval. |
| `BaseTool.timeout`, `BaseTool.concurrency_conflict_keys` | `tools/base.py` | Per-call deadline and write-overlap exclusivity. |

The `agent/` names are exported from `grasp_agents.agent`, except `NextStepResidentAnswer` / `NextStepForceResidentAnswer` (import them from `grasp_agents.agent.loop_state`). `LLMAgent` is also a root export.

## Run lifecycle

`await agent.run(chat_inputs, in_args=..., step=...)` (or `run_stream`) does, in order:

1. `ctx.load_checkpoint()` — session-scoped restore, once per `SessionContext` (see [Durability](durability.md)).
2. `agent.load_checkpoint()` — reloads this agent's transcript from the store on a cold start only.
3. Entry validation: a run with no input is allowed only when there is something to resume, a resident inbox, or a finished background task to deliver. A run with no input while [parked after a rollback](rollback.md) is refused.
4. Compose the ephemeral initial context (system prompt + sections). It is never written to the transcript (see [Context window](context-window.md)).
5. For a new step: archive a rollback boundary, reset `turn` to 0, append the input message.
6. Run `AgentLoop.execute_stream`, then parse the final answer with the output parser.

An interrupted run (`CancelledError`, consumer abort) or a failed one is *settled*: only the tool round in flight is dropped; completed rounds stay. `max_retries` re-enters the run and continues the settled delivery instead of re-sending the input.

## Phases of one turn

Each loop iteration is one turn:

- **PRE-ACT.** If the transcript does not owe a response, wait: a lone agent waits for the next background-task completion (bounded by the run deadline); a resident waits for inbox mail or a completion. Both re-emit background progress every 0.25 s while parked. Then deliver one inbox message (residents), bubble background events and deliver finished task notes. Run the compactor if one is registered; after a fold, deliver the restoration message (attachments marked `restore_after_compaction`, running background tasks). Checkpoint `AFTER_INPUT` if anything was appended or folded. Emit `TurnStartEvent`.
- **ACT.** Run before-LLM hooks, pick `tool_choice`, project the view and call the LLM. Streamed items are committed to the transcript only after the response validates. Run after-LLM hooks; emit `GenerationEndEvent`.
- **JUDGE.** `decide_next_step(...)` classifies the response.
- **Dispatch.** Run the handler for the chosen `NextStep`.

## `NextStep` and its precedence

`decide_next_step` is pure (no I/O). For a **lone** agent the precedence is:

1. **Stop** (`NextStepStop`) — a final answer was extracted, unless an answer-blocking background task is still tracked *and* the run is within its turn budget and deadline (then the loop continues and waits).
2. **TIMEOUT** (`NextStepForceFinalAnswer(StopReason.TIMEOUT)`) — `run_timeout` has passed.
3. **MAX_TURNS** (`NextStepForceFinalAnswer(StopReason.MAX_TURNS)`) — `turn >= max_turns`.
4. **Run tools** (`NextStepRunTools`) — the response has tool calls.
5. **Continue** (`NextStepContinue`) — reasoning turns under `force_react_mode`, or a final answer held back by blocking background work.

For a **resident** agent (an inbox is attached by a multi-agent host via `attach_inbox()`; see [Orchestration](orchestration.md)) the run never stops on an answer:

- Over its per-message budget: `NextStepResidentAnswer` if the response has an answer, else `NextStepForceResidentAnswer`.
- Otherwise: tool calls → `NextStepRunTools`; an answer → `NextStepResidentAnswer` (checkpoint `AFTER_RESIDENT_ANSWER`, then park for the next message); else `NextStepContinue`.

A resident never waits on answer-blocking background work and ignores `run_timeout`; it ends only when its task is cancelled from outside.

## Turn budgets

| Agent | Budget | Notes |
|---|---|---|
| Lone | `max_turns` (default 100) per run | `max_turns=N` allows N tool rounds; the generation at turn N must answer, or its calls are closed and a final answer is forced. |
| Resident | `max_turns` per inbox message | The count restarts when a message is delivered and when a run (re)starts. |

- A new step resets `turn` to 0.
- A resume keeps the restored `turn`: an interrupted run continues with what is left of its budget.
- A **delivery run** — `run()` with no input on a session whose last run finished, made to deliver finished background tasks — gets the full budget again (see [Background tasks](background-tasks.md)).

## Time budgets

- `LLMAgent(run_timeout=...)` (seconds) sets a deadline at the start of `execute_stream`. It is checked at JUDGE, bounds the lone agent's idle wait for background tasks, and bounds before-tool hooks: when the deadline passes while a hook is waiting (e.g. for an approval), the whole batch is rejected with `"[Tool call cancelled: run deadline exceeded]"` and the next JUDGE forces a `TIMEOUT` answer.
- `BaseTool(timeout=...)` bounds the whole call, not the gap between streamed events. A timeout becomes a tool result `ToolErrorInfo(timed_out=True, error="Timed out after Ns")`; it never raises out of the loop.

## Forced final answer and closure wording

When the loop forces an answer it first closes unanswered calls, then appends a `<system-reminder>`-wrapped user message and makes one more LLM call:

| Reason | Notice appended | Unanswered call result |
|---|---|---|
| `MAX_TURNS` | `Exceeded the maximum number of turns: provide a final answer now!` | `[Tool call cancelled: agent reached turn limit]` |
| `TIMEOUT` | `Ran out of time for this run: provide a final answer now!` | `[Tool call cancelled: agent ran out of time]` |
| Stop with pending calls | — | `Final answer recorded.` for the `final_answer` call, else `[Tool call cancelled: agent stopped with a final answer]` |

Closures are emitted as `ToolOutputItemEvent`s but are not passed to after-tool hooks. Forcing an answer never cancels background tasks; their notes land at a later turn boundary. The run's `StopReason` is recorded on the `AFTER_FINAL_ANSWER` checkpoint.

## Final answer: tool or extractor

- Default: a response with no tool calls is the final answer (`output_text`). When the response was truncated and the output schema is a string, `[Response truncated: <reason>]` is appended.
- `final_answer_as_tool_call=True` adds a `final_answer` tool whose input schema is the agent's output type (must be a `BaseModel`). Its arguments are the answer, and the forced call uses `tool_choice` naming it.
- `@agent.add_final_answer_extractor` replaces the default (single slot). Return `None` to keep looping.

## Tool execution

```python
from grasp_agents import LLMAgent, function_tool
from grasp_agents.llm_providers.openai_responses import OpenAIResponsesLLM

@function_tool(timeout=30.0)
async def fetch(url: str) -> str:
    """Fetch a URL."""
    ...

agent = LLMAgent[str, str, None](
    name="researcher",
    llm=OpenAIResponsesLLM(model_name="gpt-4.1-mini"),
    tools=[fetch],
    max_turns=20,
    run_timeout=300.0,
)
```

- All calls of one response form a batch. Foreground calls run **concurrently** through `run_stream`, so nested sub-agent events bubble live.
- If any two calls declare overlapping `concurrency_conflict_keys` (the same key, or one path nested under another; `"/"` claims everything), the batch runs **serially**. `Write`/`Edit`/`Delete`/`NotebookEdit` claim their path; `Bash`, `BashSession`, `RunPython` and `RunCell` claim `"/"`.
- Calls of tools with `auto_background_at` set race their own deadline concurrently with the foreground batch (see [Background tasks](background-tasks.md)).
- Invalid arguments, input-converter errors and tool exceptions become tool results the model can react to; they never crash the run. Calls whose arguments already failed validation at the LLM layer get a synthesized error result and are not dispatched.
- Results are converted (custom `add_tool_output_converter`, else the default), spilled when over `max_inline_result_chars`, and fenced as untrusted when the tool sets `untrusted_output=True`.

## Tool decisions and approvals

A before-tool hook receives the whole batch and returns `Mapping[call_id, ToolCallDecision] | None`; missing ids default to `AllowTool()`.

| Decision | Effect |
|---|---|
| `AllowTool()` | Run normally. |
| `RejectToolContent(content=...)` | Skip the call; `content` becomes its tool result. |
| `RaiseToolException(exception=...)` | Abort the whole batch; the exception propagates out of the run. |

Before-tool hooks stack. Their decisions merge per call with the most restrictive winning (raise > reject > allow).

**Callback gate** — an inline yes/no, awaited once per gated call:

```python
from grasp_agents.agent import build_callback_approval

async def approve(call, *, ctx, exec_id) -> bool:
    return input(f"Run {call.name}({call.arguments})? [y/N] ") == "y"

agent.add_before_tool_hook(build_callback_approval(approve, tool_names={"Bash"}))
```

**Store gate** — for approvals resolved later from another code path (a UI, an HTTP handler):

```python
from pathlib import Path
from grasp_agents import SessionContext
from grasp_agents.agent import (
    ApprovalAllow, ApprovalScope, LocalApprovalStore, build_store_approval,
)

store = LocalApprovalStore(persist_path=Path(".grasp/approvals.json"))
ctx = SessionContext(approval_store=store, session_key="user-42")
agent.add_before_tool_hook(build_store_approval(tool_names={"Bash", "Write"}, timeout=600))

# Elsewhere, e.g. in a UI handler:
for pending in await store.list_pending("user-42"):
    await store.resolve("user-42", pending.call_id, ApprovalAllow(scope=ApprovalScope.SESSION))
```

- The hook reads `ctx.approval_store`; with none set it allows everything.
- All gated calls of a batch are submitted up front (so a UI can show them together), then awaited in order.
- Scopes: `ONCE` remembers nothing; `SESSION` allowlists the `approval_key` for this `session_key`; `ALWAYS` allowlists it globally. The key defaults to the tool name; pass `approval_key_fn` for per-argument keys.
- `timeout` (seconds per call) auto-denies; `ApprovalDeny(reason=...)` becomes the tool result.
- With `persist_path`, `LocalApprovalStore` writes both the `ALWAYS` and `SESSION` allowlists to that JSON file, so they survive a restart. A malformed file is ignored on load.
- `ApprovalStore` is a `Protocol`; implement it to back pending requests and allowlists with your own storage.

## Contracts

1. Every tool call in the transcript is answered before the next input message. The loop closes unanswered calls on MAX_TURNS, TIMEOUT and stop (`AgentLoop._close_dangling_tool_calls`, `_close_stop_tool_calls`) and validates pairing before every generation (`LLMAgentTranscript.validate_tool_call_pairing`).
2. `decide_next_step` stays pure and keeps the precedence above (`agent/loop_state.py`).
3. Tool names are unique per agent and must not equal the agent's name (`ValueError` at construction, `llm_agent._validate_tool_names`). With `final_answer_as_tool_call=True` no tool may be named `final_answer` (`AgentLoop.__init__`). Auto-attached memory/skills/MCP tools that clash with explicit tools are skipped with a warning.
4. Transcript writes for a streamed response happen only after the response validates; item events are emitted after the write (`AgentLoop._try_query_llm`).
5. Running out of turns or time never cancels background work (`AgentLoop._handle_force_final_answer`).
6. Before-tool hooks are bounded by the run deadline (`AgentLoop._handle_run_tools`).

## Limitations

- `run_timeout` is checked between steps: it does not interrupt an in-flight LLM call or a foreground tool call. Give long tools their own `timeout`.
- Pending approvals live in memory: the awaiting run must stay alive, and pending requests die with the process. Only the allowlists persist.
- A batch either runs fully concurrently or fully serially; there is no partial ordering and no "first completed" mode.
- On the forced final-answer call `tool_choice` is left to the before-LLM hooks (unless `final_answer_as_tool_call` is set); a response that calls a tool there raises `AgentFinalAnswerError`.

## See also

[Concepts](concepts.md) · [Hooks](hooks.md) · [Session](session.md) · [Context window](context-window.md) · [Background tasks](background-tasks.md) · [Durability](durability.md) · [Security](security.md) · code: `agent/agent_loop.py`, `agent/loop_state.py`, `agent/llm_agent.py`, `agent/approval_store.py`
