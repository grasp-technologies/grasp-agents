from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.run_context import CtxT, RunContext
from grasp_agents.types.events import Event, ProcPacketOutEvent, ToolOutputEvent
from grasp_agents.types.tool import BaseTool, ToolProgressCallback
from grasp_agents.utils.io import get_prompt

from ..agent.llm_agent_transcript import LLMAgentTranscript

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable
    from pathlib import Path

    from grasp_agents.llm.llm import LLM

    from ..agent.agent_context import AgentContext
    from ..agent.llm_agent import LLMAgent
    from ..agent.prompt_builder import InputAttachment, SystemPromptSection


@runtime_checkable
class AgentToolPromptBuilder(Protocol):
    """
    Builds a prompt string for an AgentTool child agent.

    Receives the task prompt from ``AgentToolInput.prompt``, the
    parent agent's :class:`LLMAgentTranscript`, and the current
    :class:`RunContext`.

    May be sync or async — the framework awaits when the return value
    is awaitable.
    """

    def __call__(
        self, prompt: str, transcript: LLMAgentTranscript, ctx: RunContext[Any]
    ) -> str | Awaitable[str]: ...


async def _resolve_builder(
    builder: AgentToolPromptBuilder,
    prompt: str,
    transcript: LLMAgentTranscript,
    ctx: RunContext[Any],
) -> str:
    result = builder(prompt, transcript, ctx)
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

    def __init__(
        self,
        *,
        name: str,
        description: str,
        llm: LLM,
        tools: list[BaseTool[Any, Any, CtxT]] | None = None,
        sys_prompt: str | None = None,
        sys_prompt_builder: AgentToolPromptBuilder | None = None,
        in_prompt_builder: AgentToolPromptBuilder | None = None,
        auto_background_at: float | None = None,
        blocks_final_answer: bool = True,
        max_inline_result_chars: int | None = None,
        timeout: float | None = None,
        inherit_tools: bool = False,
        sys_prompt_path: str | Path | None = None,
        force_react_mode: bool = False,
        max_turns: int = 30,
        max_retries: int = 0,
        env_info: bool | SystemPromptSection = True,
        stream_llm: bool = False,
        enable_memory: bool = False,
        enable_skills: bool = False,
        time_aware: bool | InputAttachment = False,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            auto_background_at=auto_background_at,
            blocks_final_answer=blocks_final_answer,
            max_inline_result_chars=max_inline_result_chars,
            timeout=timeout,
        )
        self._llm = llm

        self._sys_prompt = get_prompt(
            prompt_text=sys_prompt, prompt_path=sys_prompt_path
        )
        self._sys_prompt_builder = sys_prompt_builder
        self._in_prompt_builder = in_prompt_builder
        self.inherit_tools = inherit_tools
        self._own_tools = tools or []

        self._force_react_mode = force_react_mode
        self._max_turns = max_turns
        self._max_retries = max_retries
        self._env_info = env_info
        self._enable_memory = enable_memory
        self._enable_skills = enable_skills
        self._time_aware = time_aware

        self.stream_llm = stream_llm

        self._in_type = AgentToolInput
        self._out_type = str

    @property
    def resumable(self) -> bool:
        return True

    @property
    def checkpoint_kind(self) -> CheckpointKind | None:
        return CheckpointKind.AGENT

    def _resolve_tools(
        self, parent_tools: list[BaseTool[Any, Any, CtxT]]
    ) -> list[BaseTool[Any, Any, CtxT]] | None:
        """Build the child's tool list, excluding AgentTool instances."""
        seen: set[str] = set()
        result: list[BaseTool[Any, Any, CtxT]] = []

        # Explicit tools first (higher priority)
        for t in self._own_tools:
            result.append(t)
            seen.add(t.name)

        # Inherited tools (excluding AgentTools to prevent recursion)
        if self.inherit_tools:
            for t in parent_tools:
                if not isinstance(t, AgentTool) and t.name not in seen:
                    result.append(t)

        return result or None

    async def _build_prompts(
        self,
        prompt: str,
        ctx: RunContext[Any] | None,
        parent_transcript: LLMAgentTranscript | None,
    ) -> tuple[str | None, str]:
        """Resolve sys_prompt and user message via builders or defaults."""
        transcript = parent_transcript or LLMAgentTranscript()
        sys_prompt = self._sys_prompt
        in_prompt = prompt

        if ctx is not None:
            if self._sys_prompt_builder is not None:
                sys_prompt = await _resolve_builder(
                    self._sys_prompt_builder, prompt, transcript, ctx
                )

            if self._in_prompt_builder is not None:
                in_prompt = await _resolve_builder(
                    self._in_prompt_builder, prompt, transcript, ctx
                )

        return sys_prompt, in_prompt

    async def _prepare_child(
        self,
        inp: AgentToolInput | None,
        *,
        ctx: RunContext[CtxT] | None,
        exec_id: str | None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> tuple[LLMAgent[AgentToolInput, str, CtxT], str | None]:
        """
        Build the child agent with resolved prompts.

        The parent agent's transcript and sibling tools are read from
        ``agent_ctx`` (the calling loop's agent-scope state) — used for
        ``inherit_tools`` and for prompt builders that reference the parent's
        message history.
        """
        del exec_id
        from ..agent.llm_agent import LLMAgent as _LLMAgent  # noqa: PLC0415

        parent_transcript = agent_ctx.transcript if agent_ctx is not None else None
        parent_tools = list(agent_ctx.tools.values()) if agent_ctx is not None else []

        if inp is not None:
            sys_prompt, in_prompt = await self._build_prompts(
                inp.prompt, ctx, parent_transcript
            )
        else:
            sys_prompt, in_prompt = None, None

        agent = _LLMAgent[AgentToolInput, str, CtxT](
            name=self.name,
            ctx=ctx,
            llm=self._llm,
            tools=self._resolve_tools(parent_tools),
            sys_prompt=sys_prompt,
            max_turns=self._max_turns,
            max_retries=self._max_retries,
            force_react_mode=self._force_react_mode,
            env_info=self._env_info,
            enable_memory=self._enable_memory,
            enable_skills=self._enable_skills,
            time_aware=self._time_aware,
            stream_llm=self.stream_llm,
        )
        # Adopt the tool's tracing settings (excludes + enabled, themselves
        # inherited from the host) and stamp the per-call path lineage.
        agent.on_adopted(parent=self, path=path)

        return agent, in_prompt

    async def _run(
        self,
        inp: AgentToolInput,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> str:
        del progress_callback
        agent, in_prompt = await self._prepare_child(
            inp, ctx=ctx, exec_id=exec_id, path=path, agent_ctx=agent_ctx
        )
        result = await agent.run(chat_inputs=in_prompt, exec_id=exec_id)

        return result.payloads[0]

    async def _run_stream(
        self,
        inp: AgentToolInput,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        progress_callback: ToolProgressCallback | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> AsyncIterator[Event[Any]]:
        del progress_callback
        agent, in_prompt = await self._prepare_child(
            inp=inp, ctx=ctx, exec_id=exec_id, path=path, agent_ctx=agent_ctx
        )
        async for event in self._yield_child_events(
            agent, chat_inputs=in_prompt, exec_id=exec_id
        ):
            yield event

    async def resume_stream(
        self,
        *,
        ctx: RunContext[CtxT] | None = None,
        exec_id: str | None = None,
        path: list[str] | None = None,
        agent_ctx: AgentContext | None = None,
    ) -> AsyncIterator[Event[Any]]:
        agent, _ = await self._prepare_child(
            inp=None, ctx=ctx, exec_id=exec_id, path=path, agent_ctx=agent_ctx
        )
        async for event in self._yield_child_events(
            agent, chat_inputs=None, exec_id=exec_id
        ):
            yield event

    async def _yield_child_events(
        self,
        agent: LLMAgent[AgentToolInput, str, CtxT],
        *,
        exec_id: str | None = None,
        chat_inputs: str | None = None,
    ) -> AsyncIterator[Event[Any]]:
        async for event in agent.run_stream(chat_inputs=chat_inputs, exec_id=exec_id):
            if isinstance(event, ProcPacketOutEvent) and event.source == agent.name:
                yield ToolOutputEvent(
                    data=event.data.payloads[0], source=agent.name, exec_id=exec_id
                )
            else:
                yield event
