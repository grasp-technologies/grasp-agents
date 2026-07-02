"""
Protocol definitions for all user-facing hooks in grasp-agents.

Hooks are registered on LLMAgent via decorators (@agent.add_*) or subclass
overrides (*_impl methods). The Protocols here define the callable signatures.

Agent Loop Hooks (registered on AgentLoop via LLMAgent):
    ViewProjector  — derives the per-turn model-facing view from the log
    BeforeLlmHook  — fires before each LLM call
    AfterLlmHook   — fires after each LLM response
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
from typing import Any, Protocol

from pydantic import BaseModel

from grasp_agents.agent.tool_decision import ToolCallDecision
from grasp_agents.selector import Selector
from grasp_agents.session_context import SessionContext
from grasp_agents.types.content import Content
from grasp_agents.types.folds import FoldSpec
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

__all__ = [
    "AfterLlmHook",
    "AfterToolHook",
    "BeforeLlmHook",
    "BeforeToolHook",
    "Compactor",
    "FinalAnswerExtractor",
    "InitialContextBuilder",
    "InputContentBuilder",
    "OutputParser",
    "RecipientSelector",
    "Selector",
    "ToolInputConverter",
    "ToolOutputConverter",
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
    resume. ``input_tokens`` is the provider-reported size of the last view (0
    before the first response) — the budget signal for size-reactive projectors.
    """

    async def __call__(
        self,
        messages: list[InputItem],
        *,
        exec_id: str,
        input_tokens: int,
    ) -> Sequence[InputItem]: ...


class Compactor(Protocol):
    """
    Summarize an old span of the transcript log under context-window pressure.

    Called once per turn with ``input_tokens`` (what the last view actually
    cost) and the ``folds`` already recorded. Returns a new :class:`FoldSpec`
    (a summarized span, applied to the view by ``apply_folds`` and persisted) or
    ``None`` to leave the log as is. Single-slot (``@agent.add_compactor``).
    Unlike a :class:`ViewProjector` it may call an LLM and produce persisted
    state, so it runs at the turn boundary, not on every view. ``force`` is set
    on the context-window-error recovery path: fold regardless of the budget.
    """

    async def __call__(
        self,
        messages: Sequence[InputItem],
        *,
        input_tokens: int,
        folds: Sequence[FoldSpec],
        exec_id: str,
        force: bool = False,
    ) -> FoldSpec | None: ...


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
