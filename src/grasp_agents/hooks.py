"""
Protocol definitions for all user-facing hooks in grasp-agents.

Hooks are registered on LLMAgent via decorators (@agent.add_*) or subclass
overrides (*_impl methods). The Protocols here define the callable signatures.

Agent Loop Hooks (registered on AgentLoop via LLMAgent):
    ViewProjector  — derives the per-turn model-facing view from the log
    CustomCompactor — writes compaction summaries in place of the agent's model
    BeforeLlmHook  — fires before each LLM call of an agent turn
    AfterLlmHook   — fires after each LLM response of an agent turn
    FinalAnswerExtractor  — determines when the agent should stop
    BeforeToolHook      — fires before tool execution
    AfterToolHook       — fires after tool execution
    ToolOutputConverter — per-tool output → input parts conversion
    ToolInputConverter  — per-tool input preprocessing

Prompt Hooks (registered on PromptBuilder via LLMAgent):
    InputContentBuilder  — formats input arguments to Content
    InitialContextBuilder — transforms the ephemeral initial context

Agent Hooks (handled directly by LLMAgent):
    OutputParser   — parses LLM text into typed output

Processor Hooks:
    RecipientSelector      — routes output to downstream agents
    WorkflowLoopTerminator — determines when a looped workflow exits
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from grasp_agents.agent.tool_decision import ToolCallDecision
from grasp_agents.selector import Selector
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import Content
from grasp_agents.types.io import ProcName
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    ToolOutputPart,
)
from grasp_agents.types.packet import Packet
from grasp_agents.types.response import Response

if TYPE_CHECKING:
    from grasp_agents.agent.agent_context import AgentContext

__all__ = [
    "AfterLlmHook",
    "AfterToolHook",
    "BeforeLlmHook",
    "BeforeToolHook",
    "CustomCompactor",
    "FinalAnswerExtractor",
    "InitialContextBuilder",
    "InputContentBuilder",
    "OutputParser",
    "RecipientSelector",
    "Selector",
    "ToolInputConverter",
    "ToolOutputConverter",
    "UsageHook",
    "ViewProjector",
    "WorkflowLoopTerminator",
]


# --- Agent Loop Hooks ---


class BeforeLlmHook(Protocol):
    async def __call__(
        self,
        *,
        exec_id: str,
        turn: int,
        extra_llm_settings: dict[str, Any],
    ) -> None: ...


class AfterLlmHook(Protocol):
    async def __call__(
        self,
        response: Response,
        *,
        exec_id: str,
        turn: int,
    ) -> None: ...


class UsageHook(Protocol):
    """
    Sees each response the agent pays for: its turns' responses and the
    summary requests of compaction, which no LLM hook sees. For usage
    accounting.
    """

    async def __call__(self, response: Response, *, exec_id: str) -> None: ...


class FinalAnswerExtractor(Protocol):
    def __call__(
        self,
        *,
        exec_id: str,
        **kwargs: Any,
    ) -> str | None: ...


class BeforeToolHook[CtxT](Protocol):
    # Keeps ``ctx``: built by standalone factories (e.g.
    # ``build_callback_approval`` / ``build_store_approval``) that have no
    # bound processor to read ``self.ctx`` from.
    async def __call__(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: SessionContext[CtxT],
        exec_id: str,
    ) -> Mapping[str, ToolCallDecision] | None: ...


class AfterToolHook(Protocol):
    async def __call__(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[FunctionToolOutputItem | InputMessageItem],
        exec_id: str,
    ) -> None: ...


class ToolOutputConverter(Protocol):
    async def __call__(
        self,
        tool_output: Any,
        *,
        exec_id: str | None,
    ) -> str | list[ToolOutputPart]: ...


class ToolInputConverter(Protocol):
    async def __call__(
        self,
        llm_args: BaseModel,
        *,
        exec_id: str | None,
    ) -> BaseModel: ...


class ViewProjector(Protocol):
    """
    Derive the per-turn model-facing view from the transcript log.

    The transcript is the immutable append-only log; the projector returns the
    (possibly pruned / collapsed / summarized) message list the LLM should see
    this turn, without mutating the log — so step rollback stays valid across
    compaction. Multiple
    projectors compose as a pipeline (registration order), each receiving the
    previous one's output; with none registered the view is the log itself. The
    returned view is repaired for ``tool_call``
    / ``tool_result`` pairing (``context.projection.repair_tool_call_pairing``)
    before the provider call, then discarded — it is never persisted, so a
    projection must be deterministic enough to re-derive on the next turn and on
    resume. ``input_tokens`` is the current (folded) log's estimated cost: the last
    response's reported input tokens plus an estimate of the messages appended
    since, or an estimate of the whole view when there is no reported count (after
    a fold or rollback) — the budget signal for size-reactive projectors.
    """

    async def __call__(
        self,
        messages: list[InputItem],
        *,
        exec_id: str,
        input_tokens: int,
    ) -> Sequence[InputItem]: ...


class CustomCompactor[CtxT](Protocol):
    """
    Write the summary a compaction folds the conversation into, in place of
    the agent's own model.

    Called when compaction (``agent.add_compaction()``) folds: the summary
    replaces the conversation up to the recent window, behind the pinned
    messages. ``agent_ctx.cw`` holds the conversation: the transcript, the
    folds already recorded and the view the model sees. The compactor decides
    where the fold ends, so summarize the whole conversation (for example
    ``agent_ctx.cw.folded_messages()``). An empty summary or an exception
    counts as a failed compaction: the turn proceeds without a fold, or, on the
    context-window-error path, the original error is raised. Single-slot
    (``@agent.add_custom_compactor``).
    """

    async def __call__(
        self, *, agent_ctx: "AgentContext", ctx: SessionContext[CtxT]
    ) -> str: ...


# --- Prompt Hooks ---


class InputContentBuilder[InT](Protocol):
    def __call__(self, in_args: InT, *, exec_id: str) -> Content: ...


class InitialContextBuilder(Protocol):
    """
    Transform the ephemeral initial context prepended to the model-facing view.

    Receives the default initial context — the system-prompt message composed
    from ``sys_prompt`` + sections (carrying each section's ``cache_control``) —
    and returns the final list. Augment it (``[*messages, reminder]``), replace
    it (``[my_system_message]``), or reorder. The result is recomposed fresh
    each step and prepended to the view; it is NOT stored in the transcript log,
    so it always reflects current state and the log stays pure conversation.
    With no builder the default is
    used unchanged. For a dynamic system prompt, return a single system message;
    for capability sections (skills / memory / env / MCP) keep ``messages``.
    """

    async def __call__(
        self, messages: list[InputItem], *, exec_id: str
    ) -> Sequence[InputItem]: ...


# --- Agent Hooks ---


class OutputParser[InT, OutT](Protocol):
    def __call__(
        self,
        final_answer: str,
        *,
        in_args: InT | None = None,
        exec_id: str,
    ) -> OutT: ...


# --- Processor Hooks ---


class RecipientSelector[OutT](Protocol):
    def __call__(self, output: OutT, *, exec_id: str) -> Sequence[ProcName]: ...


class WorkflowLoopTerminator[OutT](Protocol):
    def __call__(
        self,
        out_packet: Packet[OutT],
        **kwargs: Any,
    ) -> bool: ...


# --- Catalog Selectors ---
# ``Selector`` lives in ``types.selector`` to avoid a circular import:
# ``session_context`` imports ``memory/provider`` which references ``Selector``;
# this module imports ``session_context`` itself, so the Protocol can't live
# here. Re-exported above.
