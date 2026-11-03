# Context window

An agent's conversation is stored as an immutable **log** (the transcript). What the model sees each turn is a **view** derived from it: an ephemeral initial context (system prompt) followed by the log, optionally reshaped by view projectors and summary folds. This page covers that split, the single writer that owns the log, the hooks that shape the view, the built-in compaction strategies, and how large tool outputs are kept out of the context. Use it when a long-running agent approaches its model's window, or when you need to edit an agent's history safely.

**Context management is opt-in and view-only.** Nothing compacts unless you register a projector or compactor, and nothing ever edits the log after the fact. The only defaults that shape context are per-tool output caps applied before a result enters the log: `Bash` inlines at most 8,000 chars (see [Tool-output spill](#tool-output-spill)), and `Read` truncates a window at 100,000.

## Key types

| Name | Module | Role |
|---|---|---|
| `ContextWindowManager` | `agent/context_window.py` | Sole writer of the log; builds the view; token accounting; folds; compaction. `agent.agent_ctx.cw`. |
| `LLMAgentTranscript` | `agent/llm_agent_transcript.py` | Read-only `Sequence[InputItem]` view of the log (`agent.transcript`). |
| `ViewProjector`, `Compactor`, `InitialContextBuilder` | `hooks.py` | Hook protocols. |
| `FoldSpec` | `types/folds.py` | A summarized log span `[start, end)`. |
| `ContextBudget`, `CollapseToolOutputsProjector`, `SummarizingCompactor`, `LLMSummarizer`, `Compaction` | `context/compaction.py` | Built-in budget and strategies (exported from `grasp_agents.context`). |
| `repair_tool_call_pairing`, `apply_folds` | `context/projection.py` | View-time repair and fold application. |
| `spill_if_large`, `spill_parts_if_large` | `agent/task_progress.py` | Tool-output spill to `.grasp/tasks/<call_id>.result`. |

## The log and its writer

- `agent.transcript` is a read-only `Sequence`: index, slice, iterate, `len`. `transcript.messages` returns a tuple snapshot.
- Every write goes through the `ContextWindowManager`: `add_messages`, `truncate_transcript`, `replace_transcript`, `clear_transcript`. Truncation and replacement repair the state derived from the log in the same step: folds past the cut are dropped and the token anchor is reset.
- From application code use the agent-level methods, which also drop pending effects of calls no longer in the log, drop rollback boundaries, and make the next checkpoint rewrite the log in full:

```python
agent.reset_transcript()                 # start a fresh conversation
agent.replace_transcript(seed_messages)  # seed history from outside the store
```

- `LLMAgent(reset_transcript_on_run=True)` clears the log at the start of every new step (not on resume).

## The initial context

The system message (`sys_prompt` plus registered `SystemPromptSection`s) and any leading messages form the **initial context**. It is composed at every run entry, prepended to the view on every turn, and never written to the log. A resumed session therefore always runs with the current system prompt, and the log stays pure conversation.

- `agent.add_system_prompt_section(section)` adds a section; a section with the same name replaces the earlier one. Built-in sections (environment info, memory, skills, MCP instructions, background tasks, untrusted-content guidance) register the same way.
- `@agent.add_initial_context_builder` (single slot) transforms the composed initial context: augment it, replace it with your own system message, or reorder it.
- `await agent.preview_initial_context()` returns what a run entered now would compose.
- Per-turn context that belongs on the *input* (current time, relevant memories) goes through `agent.add_input_attachment(...)` instead. See [Hooks](hooks.md).

## View projection

Each turn the view is built as `initial_context + projectors(apply_folds(log))`:

1. Folds replace their spans with one summary message.
2. Registered view projectors run as a pipeline in registration order (a subclass's `project_view_impl` runs first), each receiving the previous output.
3. The result is repaired once with `repair_tool_call_pairing`: orphaned tool results are dropped, and calls whose result was removed get the stub `[Tool result omitted from this view by context management.]`.
4. The view is sent and discarded; it is never written back.

```python
@agent.add_view_projector
async def hide_old_screenshots(messages, *, exec_id, input_tokens):
    return [m for m in messages if not is_stale_screenshot(m)]
```

`input_tokens` is the agent's current estimate of the view's size, for size-reactive projectors.

## Folds and compactors

A `FoldSpec(start, end, summary)` summarizes log messages `[start, end)`. In the view it becomes one user message wrapped as `<system-reminder subject="your previous conversation, summarized to compact the input context">`. Folds are persisted with the agent checkpoint, so a summary is not recomputed on resume; a fold reaching past the end of the log is dropped.

A **compactor** (`@agent.add_compactor`, single slot) is called once per turn boundary, before generation, with the projected token count and the existing folds. It returns a new `FoldSpec` or `None`. Unlike a projector it may call an LLM and produces persisted state. Each recorded fold emits a `CompactionEvent` (`folded_turns`, `preserved_turns`, `context_tokens`, `context_window`, `summary`).

**Overflow net.** If the provider rejects the view with `LlmContextWindowError`, the loop asks the compactor for a fold with `force=True` and retries the call once. With no compactor, or nothing left to fold, the error propagates.

## Budget and token accounting

`ContextBudget(model, max_input_tokens, buffer_tokens)` sets a soft limit: `max_input_tokens - buffer_tokens`. `ContextBudget.for_model(model)` / `from_capabilities(model, caps)` fill the window from model metadata (128,000 when unknown) and the buffer from the model's max output tokens (8,192 when unknown). Strategies registered without a budget receive the agent's model-derived one automatically; `agent.context_window` reports the window a registered strategy targets.

The view's size is anchored on the last response's reported `input_tokens`, plus a count of the messages appended since. After a fold, rollback or resume the whole view is recounted.

## Built-in strategies

**`CollapseToolOutputsProjector`** keeps the last `keep_recent_turns` turns (default 8) verbatim and cuts older tool outputs to head + tail (`head_chars=3000`, `tail_chars=1000`).

- Budget-gated by default: it does nothing while the view fits, which keeps the prompt-cache prefix intact.
- `proactive=True` collapses every turn, keeping the view minimal at the cost of cache reuse.
- It never touches recent turns, so it cannot hide the result the model is acting on now.

**`SummarizingCompactor(summarizer=...)`** folds the oldest unfolded span when the projected view exceeds the budget.

- It keeps the last `keep_recent_turns` turns (fewer if they exceed `keep_recent_tokens`).
- It cuts only on turn boundaries, so a fold never splits a call from its result or reasoning.
- It bounds each span by the summarizer's own window (an `LLMSummarizer`'s model) or `max_summary_input_tokens`. A large backlog folds one chunk per turn.

`LLMSummarizer(llm)` renders the span as a plain-text transcript and asks for a second-person briefing. Any async `Summarizer` (a processor, a workflow) works too.

**`Compaction`** bundles both behind one budget. `agent.add_compaction()` registers the default bundle built from the agent's own LLM:

```python
from grasp_agents.context import Compaction, LLMSummarizer

agent.add_compaction()  # collapse projector + summarizing compactor, budget from the model

# Or configure it:
agent.add_compaction(Compaction(summarizer=LLMSummarizer(cheap_llm), keep_recent_turns=4))
```

The compactor's gate measures the view *after* projection, so summarization fires only when collapsing old tool outputs is not enough.

## Tool-output spill

A tool's `max_inline_result_chars` caps how much of each result enters the log. A larger result is written in full to `<first allowed root>/.grasp/tasks/<call_id>.result`. The model gets a head + tail excerpt that names the file, so `Read` / `Grep` can recover the rest.

- Set it on any tool: `BaseTool(max_inline_result_chars=...)`, `@function_tool(max_inline_result_chars=...)`, or `processor.as_tool(..., max_inline_result_chars=...)`. Default `None` inlines everything; `Bash` defaults to 8,000.
- Spilling needs `ctx.file_backend` with at least one allowed root. With no backend, or if the write fails, the result is inlined whole.
- Error results and the output of a custom `add_tool_output_converter` are never spilled.
- For a multi-part result the cap bounds all text parts together. Each text part longer than its share is spilled to its own file (`<call_id>.result`, `<call_id>-2.result`, …); images and files stay inline.
- Spilling happens before untrusted-content fencing. Background-task results spill the same way (see [Background tasks](background-tasks.md)).

## Contracts

1. The log only grows or is truncated from the end; it is never compacted or edited in place (`ContextWindowManager`). Replace messages, never mutate a persisted message: in-place edits are not re-persisted (`AgentCheckpointPersistMixin._serialize_agent_checkpoint`).
2. All log writes go through `ContextWindowManager` or `LLMAgent.reset_transcript` / `replace_transcript` (`LLMAgentTranscript` exposes no mutators).
3. Truncate and replace repair folds and the token anchor in the same step (`truncate_transcript`, `replace_transcript`).
4. The initial context is ephemeral: composed at every run entry, never persisted (`LLMAgent._process_stream_body`).
5. Projectors must be deterministic over `(log, folds)`: the view is re-derived every turn, on resume and after rollback, and nothing but folds persists (`hooks.ViewProjector`).
6. A fold never splits a turn (`compaction._safe_fold_end`); folds past a truncation are dropped (`truncate_transcript`).
7. Compaction is off by default; the overflow net applies only when a compactor is registered (`ContextWindowManager.maybe_compact`, `AgentLoop.query_llm`).
8. Spilling never hides output without a recovery path (`spill_if_large`).

## Limitations

- No strategy drops old messages outright; the built-ins collapse tool outputs and summarize.
- Folds cannot be expanded back to the original messages in the view (the log keeps them).
- The initial context is rebuilt per run, not per turn: a section reading live state updates at the next run.
- The compactor is called only at turn boundaries and on the overflow path, never mid-generation.

## See also

[Agent loop](agent-loop.md) · [Hooks](hooks.md) · [Durability](durability.md) · [Rollback](rollback.md) · [Background tasks](background-tasks.md) · code: `agent/context_window.py`, `context/compaction.py`, `context/projection.py`, `agent/task_progress.py`
