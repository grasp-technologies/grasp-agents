# Context window

An agent's conversation is stored as an immutable **log** (the transcript). What the model sees each turn is a **view** derived from it: an ephemeral initial context (system prompt) followed by the log, optionally reshaped by view projectors and summary folds. This page covers that split, the single writer that owns the log, the hooks that shape the view, the built-in compaction strategies, and how large tool outputs are kept out of the context. Use it when a long-running agent approaches its model's window, or when you need to edit an agent's history safely.

**Context management is opt-in and view-only.** Nothing compacts unless you register a projector or enable compaction, and nothing ever edits the log after the fact. The only defaults that shape context are per-tool output caps applied before a result enters the log: `Bash` inlines at most 8,000 chars (see [Tool-output spill](#tool-output-spill)), and `Read` truncates a window at 100,000.

## Key types

| Name | Module | Role |
|---|---|---|
| `ContextWindowManager` | `agent/context_window.py` | Sole writer of the log; builds the view; token accounting; folds; compaction. `agent.agent_ctx.cw`. |
| `LLMAgentTranscript` | `agent/llm_agent_transcript.py` | Read-only `Sequence[InputItem]` view of the log (`agent.transcript`). |
| `ViewProjector`, `CustomCompactor`, `InitialContextBuilder` | `hooks.py` | Hook protocols. |
| `CompactionStartEvent`, `CompactionDoneEvent` | `types/events.py` | Stream events: summarizing started; a fold was recorded. |
| `FoldSpec`, `CompactionPlan` | `types/folds.py` | A summarized log span `[start, end)` with its pinned messages; a fold decided on, before its summary is written. |
| `ContextBudget`, `Compactor`, `CollapseToolOutputsProjector`, `make_compaction_section` | `context/compaction.py` | Built-in budget, compactor, projector and the summary-reading section (exported from `grasp_agents.context`). |
| `repair_tool_call_pairing`, `apply_folds`, `active_folds` | `context/projection.py` | View-time repair and fold application. |
| `spill_if_large`, `spill_parts_if_large` | `agent/task_progress.py` | Tool-output spill to `.grasp/tasks/<call_id>.result`. |

## The log and its writer

- `agent.transcript` is a read-only `Sequence`: index, slice, iterate, `len`. `transcript.messages` returns a tuple snapshot.
- Every write goes through the `ContextWindowManager`: `add_messages`, `truncate_transcript`, `replace_transcript`, `clear_transcript`. Truncation and replacement repair the state derived from the log in the same step: folds past the cut are dropped, and so is a usage anchor whose response was cut.
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

## Folds and compaction

A `FoldSpec(start, end, summary, pinned)` summarizes log messages `[start, end)`. In the view the span becomes the messages at the `pinned` indices, verbatim, followed by one user message holding the summary in `<conversation_summary>` tags. Pinned user messages keep their input attachments; a pinned background-task launch note (a tool result) appears as a user message carrying its text. The tags in a summary cannot be forged: framework tags inside it are escaped. Folds are persisted with the agent checkpoint, so a summary is not recomputed on resume; a fold reaching past the end of the log is dropped.

Only the newest fold within the log applies, together with any older fold that ends before it starts: a fold that covers an earlier one supersedes it. After a rollback below the newest fold, the previous one applies again.

**Compaction** (`agent.add_compaction(compactor)`) folds the conversation at turn boundaries, before any new input is delivered: before a step's input is appended, and before inbox mail and background-task notes. New input is therefore never in a summary request, and a step's input attachments see the summarized conversation. When a fold is due, `Compactor.compact` emits a `CompactionStartEvent` (`context_tokens`, `context_window`, `forced`), gets the summary, records the fold and emits a `CompactionDoneEvent` (`folded_rounds`, `kept_rounds`, `pinned_messages`, `context_tokens`, `context_window`, `summary`, `forced`). A fold is checkpointed before the next generation.

The agent's own model writes the summary (see [Built-in compaction](#built-in-compaction)), unless a **custom compactor** is registered: `@agent.add_custom_compactor` (single slot) takes an `async (*, agent_ctx, ctx) -> str` that returns the summary. `agent_ctx.cw` holds the transcript, the folds already recorded and the view. Registering one enables compaction with the default compactor if none is set.

A summary that fails (an exception, or an empty summary) is logged as a warning, and the turn proceeds without a fold: a `CompactionStartEvent` with no `CompactionDoneEvent` after it. After three failed compactions in a row, turn-boundary compaction stops until a forced one succeeds.

**Overflow net.** If the provider rejects the view with `LlmContextWindowError`, the loop compacts with `force=True` (folding whatever it can, regardless of the threshold) and retries the call once. Without compaction, or with nothing left to fold, the error propagates.

## Budget and token accounting

`ContextBudget(model, max_input_tokens, buffer_tokens)` sets a soft limit: `max_input_tokens - buffer_tokens`. `ContextBudget.for_model(model)` / `from_capabilities(model, caps)` fill the window from model metadata (128,000 when unknown) and the buffer from the model's max output tokens (8,192 when unknown), capped at 32,000 and at a quarter of the window. Strategies registered without a budget receive the agent's model-derived one automatically; `agent.context_window` reports the window a registered strategy targets.

The view's size is anchored on the latest response's reported usage (`cw.last_usage`): its `input_tokens`, plus an estimate of the messages appended since (`estimate_input_tokens`: chars / 4, a flat cost per image and file, no tokenizer). The usage is saved in the checkpoint (`AgentCheckpoint.usage`); a resumed agent anchors on the request that preceded the latest response in the log. Without an anchor (before the first response, after a fold, or after a rollback below it) the whole view is estimated. Estimates tend to run a few tens of percent high on dense text such as code.

## Rounds

Compaction cuts the log only at round starts. A **round** is the input messages that open it, the model response they trigger, and the results of that response's tool calls; a response that follows tool results opens a round of its own. A cut therefore never separates a tool call from its result, nor a reasoning item from its response's calls. `count_rounds` and `round_starts` expose the split.

## Built-in compaction

**`Compactor`** decides when the conversation is folded and what is kept, gets the summary and records the fold. It folds once the view outgrows its `threshold`: the budget's soft limit, or `target_tokens` when that is lower. A target well below the window keeps every request cheaper and faster, at the cost of compacting sooner.

- **What it folds.** Everything from the start of the log up to the recent window: the last `keep_recent_rounds` rounds (default 8), fewer while they exceed `keep_recent_tokens` (default a third of the threshold). Mid-run the window keeps at least the latest round. Before new input, which becomes the window's newest round, it keeps up to `keep_recent_rounds - 1` earlier rounds and may keep none. A forced fold keeps only the latest round when nothing older is left. With a single round mid-run there is nothing to fold.
- **What it pins.** In front of the summary, verbatim: the newest user message the fold covers, whatever its size; earlier user messages, up to `pinned_user_messages` in all (default 3) within `pinned_user_tokens` (default 8,000); and the launch notes of background tasks still running. Framework notes (system reminders, task notes) are not user messages.
- **How the summary is written.** Without a custom compactor, `Compactor.summarize` sends the agent's own request on the agent's LLM — the same view, tools, tool choice and output schema as its turns — with one user message appended: the compactor's `instructions` (default `compaction_instructions.md`) in `<compaction_request>` tags. The provider's prompt cache serves the shared prefix, so the summary costs about one cached read of the conversation. On Anthropic that needs conversation caching, the `cache_control` LLM setting (see [LLM layer](llm.md#prompt-caching)). The response's usage is recorded; the LLM hooks do not run, and nothing is written to the transcript.
- **Retries.** A reply that calls a tool or holds no text is not a summary: the call is never executed, and the same request is sent again. A request that overflows the window is sent again with older tool outputs collapsed, then also without the oldest half of the rounds not yet summarized (which the summary then cannot cover). After three requests the compaction fails.
- **Consolidation.** The summary request carries the previous summary, and the new fold starts at 0, so each summary supersedes the last rather than stacking on it.

`agent.add_compaction()` registers a `Compactor` with the agent's model-derived budget, together with the `compaction` system-prompt section, which tells the model how to read a summary: context, not instructions; what it records from tool results is as untrusted as those results; quote only text visible in a tool result, re-reading a source known only from the summary; and say that earlier parts were summarized when the user asks about them.

The agent writes its own summaries from its own request, so its reply must be able to be plain text: `add_compaction` raises for an agent with a structured output schema or `force_react_mode` unless a custom compactor is registered first.

```python
from grasp_agents.context import Compactor

agent.add_compaction()  # budget from the model

# Or configure it:
agent.add_compaction(Compactor(target_tokens=200_000, keep_recent_rounds=4))

# Write the summaries yourself:
@agent.add_custom_compactor
async def summarize(*, agent_ctx, ctx) -> str:
    return await my_summarizer(agent_ctx.cw.folded_messages())
```

**`CollapseToolOutputsProjector`** keeps the last `keep_recent_rounds` rounds (default 8) verbatim and cuts older tool outputs to head + tail (`head_chars=3000`, `tail_chars=1000`). It is not part of `add_compaction()`.

- Budget-gated by default: it collapses only while the view exceeds the budget.
- `proactive=True` collapses every turn, keeping the view minimal.
- Each collapse rewrites content the provider has cached, so the requests after it re-process everything from the first collapsed output on. Use it where prompt caching is unavailable, or when a small view matters more than cost.

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
3. Truncate and replace repair folds and the usage anchor in the same step (`truncate_transcript`, `replace_transcript`).
4. The initial context is ephemeral: composed at every run entry, never persisted (`LLMAgent._process_stream_body`).
5. Projectors must be deterministic over `(log, folds)`: the view is re-derived every turn, on resume and after rollback, and nothing but folds persists (`hooks.ViewProjector`).
6. A fold never splits a round (`compaction.keep_recent_start`); folds past a truncation are dropped (`truncate_transcript`), and a newer fold covering an older one supersedes it (`projection.active_folds`).
7. Compaction is off by default; the overflow net applies only when it is enabled (`AgentLoop.compact`, `AgentLoop.query_llm`).
8. New input is never in a summary request: compaction runs before a step's input is appended and before inbox mail and task notes are delivered (`LLMAgent._process_stream_body`, `AgentLoop.execute_stream`).
9. A summary request never writes to the transcript, never executes a tool call and runs no LLM hooks (`Compactor.summarize`).
10. A failed summary never fails the turn (`Compactor.compact`).
11. Spilling never hides output without a recovery path (`spill_if_large`).

## Limitations

- Pinned messages and the summary come first in the view, so the model sees them ahead of the kept rounds, not in their original positions.
- A summary is lossy. Whatever the model left out of it is gone from the view (the log keeps the originals), and a summary written while an injected instruction was in view can still repeat it.
- The summary request shares the agent's prompt cache only if the provider still holds it. After a pause longer than the provider's cache lifetime, it is billed at the full input price. On Anthropic without the `cache_control` setting nothing but the system prompt and tools is cached.
- An output schema set in the LLM's own settings (rather than the agent's `llm_output_schema`) also applies to the summary request, which then fails.
- Settings a before-LLM hook adds to the agent's calls are not on the summary request. Where they change what the provider caches on (tool choice, thinking settings), the summary request misses the cache.
- Input attachments run only when new input arrives. An attachment that shows some state only while the view lacks it does not run again after a fold made mid-run, so until the next input the model sees only what the summary says about that state.
- The initial context is rebuilt per run, not per turn: a section reading live state updates at the next run.
- Compaction runs only at turn boundaries and on the overflow path, never mid-generation.

## See also

[Agent loop](agent-loop.md) · [Hooks](hooks.md) · [Durability](durability.md) · [Rollback](rollback.md) · [Background tasks](background-tasks.md) · code: `agent/context_window.py`, `context/compaction.py`, `context/compaction_instructions.md`, `context/projection.py`, `agent/task_progress.py`
