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
    MemoryBuilder  — custom memory initialization
    OutputParser   — parses LLM text into typed output

Processor Hooks:
    RecipientSelector      — routes output to downstream agents
    WorkflowLoopTerminator — determines when a looped workflow exits
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from grasp_agents.agent.tool_decision import ToolCallDecision
from grasp_agents.packet import Packet
from grasp_agents.run_context import CtxT, RunContext
from grasp_agents.types.content import Content
from grasp_agents.types.io import LLMPrompt, ProcName
from grasp_agents.types.items import (
    FunctionToolCallItem,
    FunctionToolOutputItem,
    InputMessageItem,
    ToolOutputPart,
)
from grasp_agents.types.response import Response

_InT_contra = TypeVar("_InT_contra", contravariant=True)
_OutT_co = TypeVar("_OutT_co", covariant=True)
_OutT_contra = TypeVar("_OutT_contra", contravariant=True)


# --- Agent Loop Hooks ---


class BeforeLlmHook(Protocol[CtxT]):
    async def __call__(
        self,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        num_turns: int,
        extra_llm_settings: dict[str, Any],
    ) -> None: ...


class AfterLlmHook(Protocol[CtxT]):
    async def __call__(
        self,
        response: Response,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        num_turns: int,
    ) -> None: ...


class FinalAnswerExtractor(Protocol[CtxT]):
    def __call__(
        self,
        *,
        ctx: RunContext[CtxT],
        exec_id: str,
        **kwargs: Any,
    ) -> str | None: ...


class BeforeToolHook(Protocol[CtxT]):
    async def __call__(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> Mapping[str, ToolCallDecision] | None: ...


class AfterToolHook(Protocol[CtxT]):
    async def __call__(
        self,
        *,
        tool_calls: Sequence[FunctionToolCallItem],
        tool_messages: Sequence[FunctionToolOutputItem | InputMessageItem],
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> None: ...


class ToolOutputConverter(Protocol[CtxT]):
    async def __call__(
        self,
        tool_output: Any,
        *,
        ctx: RunContext[CtxT],
        exec_id: str | None,
    ) -> str | list[ToolOutputPart]: ...


class ToolInputConverter(Protocol[CtxT]):
    async def __call__(
        self,
        llm_args: BaseModel,
        *,
        ctx: RunContext[CtxT],
        exec_id: str | None,
    ) -> BaseModel: ...


# --- Prompt Hooks ---


class SystemPromptBuilder(Protocol[CtxT]):
    def __call__(self, *, ctx: RunContext[CtxT], exec_id: str) -> str | None: ...


class InputContentBuilder(Protocol[_InT_contra, CtxT]):
    def __call__(
        self, in_args: _InT_contra, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Content: ...


# --- Agent Hooks ---


class MemoryBuilder(Protocol[_InT_contra]):
    def __call__(
        self,
        *,
        instructions: LLMPrompt | None = None,
        in_args: _InT_contra | None = None,
        ctx: RunContext[Any],
        exec_id: str,
    ) -> None: ...


class OutputParser(Protocol[_InT_contra, _OutT_co, CtxT]):
    def __call__(
        self,
        final_answer: str,
        *,
        in_args: _InT_contra | None = None,
        ctx: RunContext[CtxT],
        exec_id: str,
    ) -> _OutT_co: ...


# --- Processor Hooks ---


class RecipientSelector(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self, output: _OutT_contra, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]: ...


class WorkflowLoopTerminator(Protocol[_OutT_contra, CtxT]):
    def __call__(
        self,
        out_packet: Packet[_OutT_contra],
        *,
        ctx: RunContext[CtxT],
        **kwargs: Any,
    ) -> bool: ...
