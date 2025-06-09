import asyncio
import json
from collections.abc import Coroutine, Sequence
from logging import getLogger
from typing import Any, Generic, Protocol

from pydantic import BaseModel

from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .typing.converters import Converters
from .typing.message import AssistantMessage, Message, Messages, ToolMessage
from .typing.tool import BaseTool, ToolCall, ToolChoice

logger = getLogger(__name__)


class ExitToolCallLoopHandler(Protocol[CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT] | None,
        **kwargs: Any,
    ) -> bool: ...


class ManageMemoryHandler(Protocol[CtxT]):
    def __call__(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None,
        **kwargs: Any,
    ) -> None: ...


class ToolOrchestrator(Generic[CtxT]):
    def __init__(
        self,
        agent_name: str,
        llm: LLM[LLMSettings, Converters],
        tools: list[BaseTool[BaseModel, Any, CtxT]] | None,
        max_turns: int,
        react_mode: bool = False,
    ) -> None:
        self._agent_name = agent_name

        self._llm = llm
        self._llm.tools = tools

        self._max_turns = max_turns
        self._react_mode = react_mode

        self.exit_tool_call_loop_impl: ExitToolCallLoopHandler[CtxT] | None = None
        self.manage_memory_impl: ManageMemoryHandler[CtxT] | None = None

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._llm.tools or {}

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def _exit_tool_call_loop_fn(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        if self.exit_tool_call_loop_impl:
            return self.exit_tool_call_loop_impl(conversation, ctx=ctx, **kwargs)

        assert conversation, "Conversation must not be empty"
        assert isinstance(conversation[-1], AssistantMessage), (
            "Last message in conversation must be an AssistantMessage"
        )

        return not bool(conversation[-1].tool_calls)

    def _manage_memory_fn(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.manage_memory_impl:
            self.manage_memory_impl(memory=memory, ctx=ctx, **kwargs)

    async def generate_once(
        self,
        memory: LLMAgentMemory,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[AssistantMessage]:
        completion_batch = await self.llm.generate_completion_batch(
            memory.message_history, tool_choice=tool_choice
        )
        message_batch = [c.messages[0] for c in completion_batch]
        memory.update(message_batch=message_batch)

        if ctx is not None:
            ctx.completions[self.agent_name].extend(completion_batch)
            self._print_messages_and_track_usage(message_batch, ctx=ctx)

        return message_batch

    async def run_loop(
        self, memory: LLMAgentMemory, ctx: RunContext[CtxT] | None = None
    ) -> None:
        assert memory.message_history.batch_size == 1, (
            "Batch size must be 1 for tool call loop"
        )

        tool_choice: ToolChoice = "none" if self._react_mode else "auto"
        gen_message_batch = await self.generate_once(
            memory, tool_choice=tool_choice, ctx=ctx
        )

        turns = 0

        while True:
            self._manage_memory_fn(memory, ctx=ctx, num_turns=turns)

            conversation = memory.message_history.conversations[0]
            if self._exit_tool_call_loop_fn(conversation, ctx=ctx, num_turns=turns):
                return
            if turns >= self.max_turns:
                logger.info(
                    f"Max turns reached: {self.max_turns}. Stopping tool call loop."
                )
                return

            msg = gen_message_batch[0]
            if msg.tool_calls:
                tool_messages = await self.call_tools(msg.tool_calls, ctx=ctx)
                memory.update(message_list=tool_messages)

            tool_choice = "none" if (self._react_mode and msg.tool_calls) else "auto"
            gen_message_batch = await self.generate_once(
                memory, tool_choice=tool_choice, ctx=ctx
            )

            turns += 1

    async def call_tools(
        self,
        calls: Sequence[ToolCall],
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[ToolMessage]:
        corouts: list[Coroutine[Any, Any, BaseModel]] = []
        for call in calls:
            tool = self.tools[call.tool_name]
            args = json.loads(call.tool_arguments)
            corouts.append(tool(ctx=ctx, **args))

        outs = await asyncio.gather(*corouts)

        tool_messages = [
            ToolMessage.from_tool_output(out, call, model_id=self.agent_name)
            for out, call in zip(outs, calls, strict=False)
        ]

        self._print_messages(tool_messages, ctx=ctx)

        return tool_messages

    def _print_messages(
        self, message_batch: Sequence[Message], ctx: RunContext[CtxT] | None = None
    ) -> None:
        if ctx:
            ctx.printer.print_llm_messages(message_batch, agent_name=self.agent_name)

    def _print_messages_and_track_usage(
        self, message_batch: Sequence[AssistantMessage], ctx: RunContext[CtxT]
    ) -> None:
        self._print_messages(message_batch, ctx=ctx)
        ctx.usage_tracker.update(messages=message_batch, model_name=self.llm.model_name)
