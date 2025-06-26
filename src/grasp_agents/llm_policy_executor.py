import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Sequence
from itertools import starmap
from logging import getLogger
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from pydantic import BaseModel

from .generics_utils import AutoInstanceAttributesMixin
from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory
from .run_context import CtxT, RunContext
from .typing.completion import Completion
from .typing.converters import Converters
from .typing.events import (
    CompletionChunkEvent,
    CompletionEvent,
    Event,
    GenMessageEvent,
    ToolCallEvent,
    ToolMessageEvent,
    UserMessageEvent,
)
from .typing.message import AssistantMessage, Messages, ToolMessage, UserMessage
from .typing.tool import BaseTool, NamedToolChoice, ToolCall, ToolChoice

logger = getLogger(__name__)


FINAL_ANSWER_TOOL_NAME = "final_answer"


_FinalAnswerT = TypeVar("_FinalAnswerT")


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


class LLMPolicyExecutor(AutoInstanceAttributesMixin, Generic[_FinalAnswerT, CtxT]):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_final_answer_type",
    }

    def __init__(
        self,
        agent_name: str,
        llm: LLM[LLMSettings, Converters],
        tools: list[BaseTool[BaseModel, Any, CtxT]] | None,
        max_turns: int,
        react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
    ) -> None:
        self._final_answer_type: type[_FinalAnswerT]
        super().__init__()

        self._agent_name = agent_name

        _tools: list[BaseTool[BaseModel, Any, CtxT]] | None = tools
        self._final_answer_tool_name: str | None = None
        if tools and final_answer_as_tool_call:
            final_answer_tool = self.get_final_answer_tool()
            self._final_answer_tool_name = final_answer_tool.name
            _tools = tools + [final_answer_tool]

        self._llm = llm
        self._llm.tools = _tools

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

    def _exit_tool_call_loop(
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

    def _manage_memory(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.manage_memory_impl:
            self.manage_memory_impl(memory=memory, ctx=ctx, **kwargs)

    async def generate_messages(
        self,
        memory: LLMAgentMemory,
        run_id: str,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[AssistantMessage]:
        completion = await self.llm.generate_completion(
            memory.message_history, tool_choice=tool_choice
        )
        memory.update(completion.messages)

        if ctx is not None:
            ctx.completions[self.agent_name].append(completion)
            self._track_usage(self.agent_name, completion, ctx=ctx)
            self._print_completion(completion, run_id=run_id, ctx=ctx)

        return completion.messages

    async def generate_messages_stream(
        self,
        memory: LLMAgentMemory,
        run_id: str,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[CompletionChunkEvent | CompletionEvent | GenMessageEvent]:
        message_hist = memory.message_history

        completion: Completion | None = None
        async for event in await self.llm.generate_completion_stream(
            message_hist, tool_choice=tool_choice
        ):
            yield event
            if isinstance(event, CompletionEvent):
                completion = event.data
        if completion is None:
            raise RuntimeError("No completion generated during stream.")

        memory.update(completion.messages)

        for message in completion.messages:
            yield GenMessageEvent(name=self.agent_name, data=message)

        if ctx is not None:
            self._track_usage(self.agent_name, completion, ctx=ctx)
            ctx.completions[self.agent_name].append(completion)

    async def call_tools(
        self,
        calls: Sequence[ToolCall],
        memory: LLMAgentMemory,
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[ToolMessage]:
        corouts: list[Coroutine[Any, Any, BaseModel]] = []
        for call in calls:
            tool = self.tools[call.tool_name]
            args = json.loads(call.tool_arguments)
            corouts.append(tool(ctx=ctx, **args))

        outs = await asyncio.gather(*corouts)
        tool_messages = list(
            starmap(ToolMessage.from_tool_output, zip(outs, calls, strict=True))
        )
        memory.update(tool_messages)

        if ctx is not None:
            ctx.printer.print_llm_messages(
                tool_messages, agent_name=self.agent_name, run_id=run_id
            )

        return tool_messages

    async def call_tools_stream(
        self,
        calls: Sequence[ToolCall],
        memory: LLMAgentMemory,
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[ToolMessageEvent]:
        tool_messages = await self.call_tools(
            calls, memory=memory, run_id=run_id, ctx=ctx
        )
        for tool_message, call in zip(tool_messages, calls, strict=True):
            yield ToolMessageEvent(name=call.tool_name, data=tool_message)

    def _extract_final_answer_from_tool_calls(
        self, gen_message: AssistantMessage, memory: LLMAgentMemory
    ) -> AssistantMessage | None:
        final_answer_message: AssistantMessage | None = None
        for tool_call in gen_message.tool_calls or []:
            if tool_call.tool_name == self._final_answer_tool_name:
                final_answer_message = AssistantMessage(
                    name=self.agent_name, content=tool_call.tool_arguments
                )
                gen_message.tool_calls = None
                memory.update([final_answer_message])
                return final_answer_message

        return final_answer_message

    async def _generate_final_answer(
        self, memory: LLMAgentMemory, run_id: str, ctx: RunContext[CtxT] | None = None
    ) -> AssistantMessage:
        assert self._final_answer_tool_name is not None

        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!"
        )
        memory.update([user_message])
        if ctx is not None:
            ctx.printer.print_llm_messages(
                [user_message], agent_name=self.agent_name, run_id=run_id
            )

        tool_choice = NamedToolChoice(name=self._final_answer_tool_name)
        gen_message = (
            await self.generate_messages(
                memory, tool_choice=tool_choice, run_id=run_id, ctx=ctx
            )
        )[0]

        final_answer_message = self._extract_final_answer_from_tool_calls(
            gen_message, memory=memory
        )
        if final_answer_message is None:
            raise RuntimeError(
                "Final answer tool call did not return a final answer message."
            )

        return final_answer_message

    async def _generate_final_answer_stream(
        self, memory: LLMAgentMemory, run_id: str, ctx: RunContext[CtxT] | None = None
    ) -> AsyncIterator[Event[Any]]:
        assert self._final_answer_tool_name is not None

        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!",
        )
        memory.update([user_message])
        yield UserMessageEvent(name=self.agent_name, data=user_message)

        tool_choice = NamedToolChoice(name=self._final_answer_tool_name)
        event: Event[Any] | None = None
        async for event in self.generate_messages_stream(
            memory, tool_choice=tool_choice, run_id=run_id, ctx=ctx
        ):
            yield event

        assert isinstance(event, GenMessageEvent)
        gen_message = event.data
        final_answer_message = self._extract_final_answer_from_tool_calls(
            gen_message, memory=memory
        )
        if final_answer_message is None:
            raise RuntimeError(
                "Final answer tool call did not return a final answer message."
            )
        yield GenMessageEvent(name=self.agent_name, data=final_answer_message)

    async def execute(
        self, memory: LLMAgentMemory, run_id: str, ctx: RunContext[CtxT] | None = None
    ) -> AssistantMessage | Sequence[AssistantMessage]:
        # 1. Generate the first message:
        #    In ReAct mode, we generate the first message without tool calls
        #    to force the agent to plan its actions in a separate message.
        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = "none" if self._react_mode else "auto"
        gen_messages = await self.generate_messages(
            memory, tool_choice=tool_choice, run_id=run_id, ctx=ctx
        )
        if not self.tools:
            return gen_messages

        if len(gen_messages) > 1:
            raise ValueError("n_choices must be 1 when executing the tool call loop.")
        gen_message = gen_messages[0]
        turns = 0

        while True:
            # 2. Check if we should exit the tool call loop

            # When final_answer_tool_name is None, we use exit_tool_call_loop_impl
            # to determine whether to exit the loop.
            if self._final_answer_tool_name is None and self._exit_tool_call_loop(
                memory.message_history, ctx=ctx, num_turns=turns
            ):
                return gen_message

            # When final_answer_tool_name is set, we check if the last message contains
            # a tool call to the final answer tool. If it does, we exit the loop.
            if self._final_answer_tool_name is not None:
                final_answer = self._extract_final_answer_from_tool_calls(
                    gen_message, memory=memory
                )
                if final_answer is not None:
                    return final_answer

            # Exit if the maximum number of turns is reached
            if turns >= self.max_turns:
                # When final_answer_tool_name is set, we force the agent to provide
                # a final answer by generating a message with a final answer
                # tool call.
                # Otherwise, we simply return the last generated message.
                if self._final_answer_tool_name is not None:
                    final_answer = await self._generate_final_answer(
                        memory, run_id=run_id, ctx=ctx
                    )
                else:
                    final_answer = gen_message
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return final_answer

            # 3. Call tools if there are any tool calls in the generated message.

            if gen_message.tool_calls:
                await self.call_tools(
                    gen_message.tool_calls, memory=memory, run_id=run_id, ctx=ctx
                )

            # Apply the memory management function if provided.
            self._manage_memory(memory, ctx=ctx, num_turns=turns)

            # 4. Generate the next message based on the updated memory.
            #    In ReAct mode, we set tool_choice to "none" if we just called tools,
            #    so the next message will be an observation/planning message with
            #    no immediate tool calls.
            #    If we are not in ReAct mode, we set tool_choice to "auto" to allow
            #    the LLM to choose freely whether to call tools.

            if self._react_mode and gen_message.tool_calls:
                tool_choice = "none"
            elif gen_message.tool_calls:
                tool_choice = "auto"
            else:
                tool_choice = "required"

            gen_message = (
                await self.generate_messages(
                    memory, tool_choice=tool_choice, run_id=run_id, ctx=ctx
                )
            )[0]

            turns += 1

    async def execute_stream(
        self, memory: LLMAgentMemory, run_id: str, ctx: RunContext[CtxT] | None = None
    ) -> AsyncIterator[Event[Any]]:
        tool_choice: ToolChoice = "none" if self._react_mode else "auto"
        gen_message: AssistantMessage | None = None
        async for event in self.generate_messages_stream(
            memory, tool_choice=tool_choice, run_id=run_id, ctx=ctx
        ):
            yield event
            if isinstance(event, GenMessageEvent):
                gen_message = event.data
        assert isinstance(gen_message, AssistantMessage)

        turns = 0

        while True:
            if self._final_answer_tool_name is None and self._exit_tool_call_loop(
                memory.message_history, ctx=ctx, num_turns=turns
            ):
                return

            if self._final_answer_tool_name is not None:
                final_answer_message = self._extract_final_answer_from_tool_calls(
                    gen_message, memory=memory
                )
                if final_answer_message is not None:
                    yield GenMessageEvent(
                        name=self.agent_name, data=final_answer_message
                    )
                    return

            if turns >= self.max_turns:
                if self._final_answer_tool_name is not None:
                    async for event in self._generate_final_answer_stream(
                        memory, run_id=run_id, ctx=ctx
                    ):
                        yield event
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return

            if gen_message.tool_calls:
                for tool_call in gen_message.tool_calls:
                    yield ToolCallEvent(name=self.agent_name, data=tool_call)

                async for tool_message_event in self.call_tools_stream(
                    gen_message.tool_calls, memory=memory, run_id=run_id, ctx=ctx
                ):
                    yield tool_message_event

            self._manage_memory(memory, ctx=ctx, num_turns=turns)

            if self._react_mode and gen_message.tool_calls:
                tool_choice = "none"
            elif gen_message.tool_calls:
                tool_choice = "auto"
            else:
                tool_choice = "required"
            async for event in self.generate_messages_stream(
                memory, tool_choice=tool_choice, run_id=run_id, ctx=ctx
            ):
                yield event
                if isinstance(event, GenMessageEvent):
                    gen_message = event.data

            turns += 1

    def _track_usage(
        self,
        agent_name: str,
        completion: Completion,
        ctx: RunContext[CtxT],
    ) -> None:
        ctx.usage_tracker.update(
            agent_name=agent_name,
            completions=[completion],
            model_name=self.llm.model_name,
        )

    def get_final_answer_tool(self) -> BaseTool[BaseModel, None, Any]:
        if not issubclass(self._final_answer_type, BaseModel):
            raise TypeError(
                "final_answer_type must be a subclass of BaseModel to create "
                "a final answer tool."
            )

        class FinalAnswerTool(BaseTool[self._final_answer_type, None, Any]):
            name: str = FINAL_ANSWER_TOOL_NAME
            description: str = (
                "You must call this tool to provide the final answer. "
                "DO NOT output your answer before calling the tool. "
            )

            async def run(
                self, inp: _FinalAnswerT, ctx: RunContext[Any] | None = None
            ) -> None:
                return None

        return FinalAnswerTool()

    def _print_completion(
        self, completion: Completion, run_id: str, ctx: RunContext[CtxT]
    ) -> None:
        ctx.printer.print_llm_messages(
            completion.messages,
            usages=[completion.usage],
            agent_name=self.agent_name,
            run_id=run_id,
        )
