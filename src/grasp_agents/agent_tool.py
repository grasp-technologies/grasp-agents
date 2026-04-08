from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel

from .llm import LLM
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .types.events import Event, ProcPacketOutEvent, ToolOutputEvent
from .types.tool import BaseTool, ToolProgressCallback

if TYPE_CHECKING:
    from .llm_agent import LLMAgent


@runtime_checkable
class AgentPromptBuilder(Protocol):
    """
    Builds a prompt string for an AgentTool child agent.

    Receives the task prompt from ``AgentToolInput.prompt``, the
    parent agent's :class:`LLMAgentMemory`, and the current
    :class:`RunContext`.

    May be sync or async — the framework awaits when the return value
    is awaitable.
    """

    def __call__(
        self, prompt: str, memory: LLMAgentMemory, ctx: RunContext[Any]
    ) -> str | Awaitable[str]: ...


async def _resolve_builder(
    builder: AgentPromptBuilder,
    prompt: str,
    memory: LLMAgentMemory,
    ctx: RunContext[Any],
) -> str:
    result = builder(prompt, memory, ctx)
    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
        return await result  # type: ignore[misc]
    return result  # type: ignore[return-value]


class AgentToolInput(BaseModel):
    prompt: str


class AgentTool(BaseTool[AgentToolInput, str, CtxT]):
    """
    A tool that dynamically spawns a fresh LLMAgent per invocation.

    Configured at setup time with LLM, tools, system prompt, etc.
    At runtime, the parent agent calls it with a prompt and gets
    the child's final answer as a string.

    Optional ``sys_prompt_builder`` / ``in_prompt_builder`` callables
    allow dynamic prompt construction from the task prompt, parent
    message history, and :class:`RunContext`.  When provided,
    ``sys_prompt_builder`` overrides the static ``sys_prompt`` and
    ``in_prompt_builder`` transforms the raw prompt into the child's
    user message.
    """

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        *,
        name: str,
        description: str,
        llm: LLM,
        tools: list[BaseTool[Any, Any, CtxT]] | None = None,
        sys_prompt: str | None = None,
        sys_prompt_builder: AgentPromptBuilder | None = None,
        in_prompt_builder: AgentPromptBuilder | None = None,
        max_turns: int = 30,
        background: bool = False,
        timeout: float | None = None,
        inherit_tools: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            background=background,
            timeout=timeout,
        )
        self._llm = llm
        self._own_tools = tools or []
        self._sys_prompt = sys_prompt
        self._sys_prompt_builder = sys_prompt_builder
        self._in_prompt_builder = in_prompt_builder
        self._max_turns = max_turns
        self.inherit_tools = inherit_tools
        self._parent_tools: list[BaseTool[Any, Any, CtxT]] = []
        self._parent_memory: LLMAgentMemory | None = None

        self._in_type = AgentToolInput
        self._out_type: type[str] = str

    @property
    def resumable(self) -> bool:
        return True

    def set_parent_tools(self, tools: list[BaseTool[Any, Any, CtxT]]) -> None:
        """Called by parent LLMAgent to provide sibling tools."""
        self._parent_tools = tools

    def set_parent_memory(self, memory: LLMAgentMemory) -> None:
        """Called by parent LLMAgent to provide access to its memory."""
        self._parent_memory = memory

    def _resolve_tools(self) -> list[BaseTool[Any, Any, CtxT]] | None:
        """Build the child's tool list, excluding AgentTool instances."""
        seen: set[str] = set()
        result: list[BaseTool[Any, Any, CtxT]] = []

        # Explicit tools first (higher priority)
        for t in self._own_tools:
            result.append(t)
            seen.add(t.name)

        # Inherited tools (excluding AgentTools to prevent recursion)
        if self.inherit_tools:
            for t in self._parent_tools:
                if not isinstance(t, AgentTool) and t.name not in seen:
                    result.append(t)

        return result or None

    async def _build_prompts(
        self, prompt: str, ctx: RunContext[Any] | None = None
    ) -> tuple[str | None, str]:
        """Resolve sys_prompt and user message via builders or defaults."""
        memory = self._parent_memory or LLMAgentMemory()
        sys_prompt = self._sys_prompt
        user_msg = prompt

        if ctx is not None:
            if self._sys_prompt_builder is not None:
                sys_prompt = await _resolve_builder(
                    self._sys_prompt_builder, prompt, memory, ctx
                )

            if self._in_prompt_builder is not None:
                user_msg = await _resolve_builder(
                    self._in_prompt_builder, prompt, memory, ctx
                )

        return sys_prompt, user_msg

    async def _prepare_child(
        self,
        inp: AgentToolInput | None,
        *,
        ctx: RunContext[CtxT] | None,
        exec_id: str | None,
        session_id: str | None,
    ) -> tuple[LLMAgent[AgentToolInput, str, CtxT], str | None]:
        """Build the child agent with resolved prompts."""
        from .llm_agent import LLMAgent as _LLMAgent

        if inp is not None:
            sys_prompt, in_prompt = await self._build_prompts(inp.prompt, ctx)
        else:
            sys_prompt, in_prompt = None, None

        child_name = f"{self.name}:{(exec_id or 'x')[:8]}"

        agent = _LLMAgent[AgentToolInput, str, CtxT](
            name=child_name,
            llm=self._llm,
            tools=self._resolve_tools(),
            sys_prompt=sys_prompt,
            max_turns=self._max_turns,
            session_id=session_id,
        )

        return agent, in_prompt

    async def _run(
        self,
        inp: AgentToolInput,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> str:
        agent, in_prompt = await self._prepare_child(
            inp, ctx=ctx, exec_id=exec_id, session_id=session_id
        )
        result = await agent.run(chat_inputs=in_prompt, ctx=ctx, exec_id=exec_id)

        return result.payloads[0]

    async def _run_stream(
        self,
        inp: AgentToolInput,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        agent, in_prompt = await self._prepare_child(
            inp=inp, ctx=ctx, exec_id=exec_id, session_id=session_id
        )
        async for event in self._yield_child_events(
            agent, chat_inputs=in_prompt, ctx=ctx, exec_id=exec_id
        ):
            yield event

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        agent, _ = await self._prepare_child(
            inp=None, ctx=ctx, exec_id=exec_id, session_id=session_id
        )
        async for event in self._yield_child_events(
            agent, chat_inputs=None, ctx=ctx, exec_id=exec_id
        ):
            yield event

    async def _yield_child_events(
        self,
        agent: LLMAgent[AgentToolInput, str, CtxT],
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        chat_inputs: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        async for event in agent.run_stream(
            chat_inputs=chat_inputs, ctx=ctx, exec_id=exec_id
        ):
            if isinstance(event, ProcPacketOutEvent) and event.source == agent.name:
                yield ToolOutputEvent(
                    data=event.data.payloads[0], source=agent.name, exec_id=exec_id
                )
            else:
                yield event
