"""
Protocol definitions for all user-facing hooks in grasp-agents.

Hooks are registered on LLMAgent via decorators (@agent.add_*) or subclass
overrides (*_impl methods). The Protocols here define the callable signatures.

Agent Loop Hooks (registered on AgentLoop via LLMAgent):
    BeforeLlmHook  — fires before each LLM call
    AfterLlmHook   — fires after each LLM response
    FinalAnswerExtractor  — determines when the agent should stop
    BeforeToolHook      — fires before tool execution
    AfterToolHook       — fires after tool execution
    ToolOutputConverter — per-tool output → input parts conversion
    ToolInputConverter  — per-tool input preprocessing

Prompt Hooks (registered on PromptBuilder via LLMAgent):
    SystemPromptBuilder  — builds system prompt from context
    InputContentBuilder  — formats input arguments to Content

Agent Hooks (handled directly by LLMAgent):
    TranscriptBuilder  — custom transcript initialization on fresh init
    StateBuilder   — rebuild CtxT / RunContext.state on resume-from-checkpoint
    OutputParser   — parses LLM text into typed output

Processor Hooks:
    RecipientSelector      — routes output to downstream agents
    WorkflowLoopTerminator — determines when a looped workflow exits
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from grasp_agents.agent.tool_decision import ToolCallDecision
from grasp_agents.durability.checkpoints import AgentCheckpoint
from grasp_agents.packet import Packet
from grasp_agents.run_context import RunContext
from grasp_agents.types.content import Content, InputText
from grasp_agents.types.io import LLMPrompt, ProcName
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    ToolOutputPart,
)
from grasp_agents.types.response import Response
from grasp_agents.types.selector import Selector

__all__ = [
    "AfterLlmHook",
    "AfterToolHook",
    "BeforeLlmHook",
    "BeforeToolHook",
    "FinalAnswerExtractor",
    "InputContentBuilder",
    "OutputParser",
    "RecipientSelector",
    "Selector",
    "StateBuilder",
    "SystemPromptBuilder",
    "ToolInputConverter",
    "ToolOutputConverter",
    "TranscriptBuilder",
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
        ctx: RunContext[CtxT],
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


# --- Prompt Hooks ---


class SystemPromptBuilder(Protocol):
    def __call__(self, *, exec_id: str) -> str | Sequence[InputText] | None: ...


class InputContentBuilder[InT](Protocol):
    def __call__(self, in_args: InT, *, exec_id: str) -> Content: ...


# --- Agent Hooks ---


class TranscriptBuilder[InT](Protocol):
    """
    Seed the agent's transcript (conversation history) on fresh init.

    Receives the freshly-built system prompt as ``instructions`` — the same
    ``str | Sequence[InputText]`` that :meth:`LLMAgentTranscript.reset`
    accepts, so forwarding it (``agent.transcript.reset(instructions)``)
    preserves each part's :class:`CacheControl`. The hook populates
    ``agent.transcript`` (e.g. a custom system message or a primed
    multi-turn history). Distinct from cross-session
    :class:`RunContext.memory`. On resume-from-checkpoint this does NOT
    fire — use :class:`StateBuilder` there instead.
    """

    def __call__(
        self,
        *,
        instructions: LLMPrompt | Sequence[InputText] | None = None,
        in_args: InT | None = None,
        exec_id: str,
    ) -> None: ...


class StateBuilder(Protocol):
    """
    Rebuild business state (typically ``ctx.state``) from external sources
    after loading a checkpoint.

    Fires exactly once per resume — i.e. when ``_load_checkpoint`` returned
    a non-``None`` checkpoint — after conversation messages have been
    restored into memory and before the agent's next turn. Does not fire
    on fresh init; use :class:`TranscriptBuilder` there instead.
    """

    async def __call__(
        self,
        *,
        checkpoint: AgentCheckpoint,
        exec_id: str,
    ) -> None: ...


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
    def __call__(
        self, output: OutT, *, exec_id: str
    ) -> Sequence[ProcName]: ...


class WorkflowLoopTerminator[OutT](Protocol):
    def __call__(
        self,
        out_packet: Packet[OutT],
        **kwargs: Any,
    ) -> bool: ...


# --- Catalog Selectors ---
# ``Selector`` lives in ``types.selector`` to avoid a circular import:
# ``run_context`` imports ``memory/provider`` which references ``Selector``;
# this module imports ``run_context`` itself, so the Protocol can't live
# here. Re-exported above.
