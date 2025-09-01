import asyncio
import json
from collections.abc import AsyncIterator, Coroutine, Mapping, Sequence
from itertools import starmap
from logging import getLogger
from typing import Any, Generic, Protocol, final

from pydantic import BaseModel

from grasp_agents.typing.completion_chunk import CompletionChunk

from .errors import AgentFinalAnswerError
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
    LLMStreamingErrorEvent,
    ToolCallEvent,
    ToolMessageEvent,
    UserMessageEvent,
)
from .typing.message import AssistantMessage, Messages, ToolMessage, UserMessage
from .typing.tool import BaseTool, NamedToolChoice, ToolCall, ToolChoice

logger = getLogger(__name__)


class ToolCallLoopTerminator(Protocol[CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> bool: ...


class MemoryManager(Protocol[CtxT]):
    def __call__(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> None: ...


class LLMPolicyExecutor(Generic[CtxT]):
    def __init__(
        self,
        *,
        agent_name: str,
        llm: LLM[LLMSettings, Converters],
        tools: list[BaseTool[BaseModel, Any, CtxT]] | None,
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        max_turns: int,
        react_mode: bool = False,
        final_answer_type: type[BaseModel] = BaseModel,
        final_answer_as_tool_call: bool = False,
    ) -> None:
        super().__init__()

        self._agent_name = agent_name

        self._final_answer_type = final_answer_type
        self._final_answer_as_tool_call = final_answer_as_tool_call
        self._final_answer_tool = self.get_final_answer_tool()

        tools_list: list[BaseTool[BaseModel, Any, CtxT]] | None = tools
        if tools and final_answer_as_tool_call:
            tools_list = tools + [self._final_answer_tool]
        self._tools = {t.name: t for t in tools_list} if tools_list else None

        self._response_schema = response_schema
        self._response_schema_by_xml_tag = response_schema_by_xml_tag

        self._llm = llm

        self._max_turns = max_turns
        self._react_mode = react_mode

        self.tool_call_loop_terminator: ToolCallLoopTerminator[CtxT] | None = None
        self.memory_manager: MemoryManager[CtxT] | None = None

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._llm

    @property
    def response_schema(self) -> Any | None:
        return self._response_schema

    @response_schema.setter
    def response_schema(self, value: Any | None) -> None:
        self._response_schema = value

    @property
    def response_schema_by_xml_tag(self) -> Mapping[str, Any] | None:
        return self._response_schema_by_xml_tag

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._tools or {}

    @property
    def max_turns(self) -> int:
        return self._max_turns

    @final
    def _terminate_tool_call_loop(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> bool:
        if self.tool_call_loop_terminator:
            return self.tool_call_loop_terminator(
                conversation, ctx=ctx, call_id=call_id, **kwargs
            )

        return False

    @final
    def _manage_memory(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> None:
        if self.memory_manager:
            self.memory_manager(memory=memory, ctx=ctx, call_id=call_id, **kwargs)

    async def generate_message(
        self,
        memory: LLMAgentMemory,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AssistantMessage:
        completion = await self.llm.generate_completion(
            memory.message_history,
            response_schema=self.response_schema,
            response_schema_by_xml_tag=self.response_schema_by_xml_tag,
            tools=self.tools,
            tool_choice=tool_choice,
            proc_name=self.agent_name,
            call_id=call_id,
        )
        memory.update(completion.messages)
        self._process_completion(completion, ctx=ctx, call_id=call_id)

        return completion.messages[0]

    async def generate_message_stream(
        self,
        memory: LLMAgentMemory,
        *,
        tool_choice: ToolChoice | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[
        CompletionChunkEvent[CompletionChunk]
        | CompletionEvent
        | GenMessageEvent
        | LLMStreamingErrorEvent
    ]:
        completion: Completion | None = None

        llm_event_stream = self.llm.generate_completion_stream(
            memory.message_history,
            response_schema=self.response_schema,
            response_schema_by_xml_tag=self.response_schema_by_xml_tag,
            tools=self.tools,
            tool_choice=tool_choice,
            proc_name=self.agent_name,
            call_id=call_id,
        )
        llm_event_stream_post = self.llm.postprocess_event_stream(llm_event_stream)  # type: ignore[assignment]

        async for event in llm_event_stream_post:
            if isinstance(event, CompletionEvent):
                completion = event.data
            yield event
        if completion is None:
            return

        yield GenMessageEvent(
            proc_name=self.agent_name, call_id=call_id, data=completion.messages[0]
        )

        memory.update(completion.messages)

        self._process_completion(completion, ctx=ctx, call_id=call_id)

    async def call_tools(
        self,
        calls: Sequence[ToolCall],
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> Sequence[ToolMessage]:
        # TODO: Add image support
        corouts: list[Coroutine[Any, Any, BaseModel]] = []
        for call in calls:
            tool = self.tools[call.tool_name]
            args = json.loads(call.tool_arguments)
            corouts.append(tool(ctx=ctx, call_id=call_id, **args))

        outs = await asyncio.gather(*corouts)
        tool_messages = list(
            starmap(ToolMessage.from_tool_output, zip(outs, calls, strict=True))
        )

        memory.update(tool_messages)

        if ctx.printer:
            ctx.printer.print_messages(
                tool_messages, agent_name=self.agent_name, call_id=call_id
            )

        return tool_messages

    async def call_tools_stream(
        self,
        calls: Sequence[ToolCall],
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[ToolMessageEvent]:
        tool_messages = await self.call_tools(
            calls, memory=memory, ctx=ctx, call_id=call_id
        )
        for tool_message, call in zip(tool_messages, calls, strict=True):
            yield ToolMessageEvent(
                proc_name=call.tool_name, call_id=call_id, data=tool_message
            )

    def _extract_final_answer_from_tool_calls(
        self, memory: LLMAgentMemory
    ) -> AssistantMessage | None:
        last_message = memory.message_history[-1]
        if not isinstance(last_message, AssistantMessage):
            return None

        for tool_call in last_message.tool_calls or []:
            if tool_call.tool_name == self._final_answer_tool.name:
                final_answer_message = AssistantMessage(
                    name=self.agent_name, content=tool_call.tool_arguments
                )
                last_message.tool_calls = None
                memory.update([final_answer_message])

                return final_answer_message

    async def _generate_final_answer(
        self, memory: LLMAgentMemory, ctx: RunContext[CtxT], call_id: str
    ) -> AssistantMessage:
        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!"
        )
        memory.update([user_message])
        if ctx.printer:
            ctx.printer.print_messages(
                [user_message], agent_name=self.agent_name, call_id=call_id
            )

        tool_choice = NamedToolChoice(name=self._final_answer_tool.name)
        await self.generate_message(
            memory, tool_choice=tool_choice, ctx=ctx, call_id=call_id
        )

        final_answer_message = self._extract_final_answer_from_tool_calls(memory=memory)
        if final_answer_message is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, call_id=call_id)

        return final_answer_message

    async def _generate_final_answer_stream(
        self, memory: LLMAgentMemory, ctx: RunContext[CtxT], call_id: str
    ) -> AsyncIterator[Event[Any]]:
        user_message = UserMessage.from_text(
            "Exceeded the maximum number of turns: provide a final answer now!",
        )
        memory.update([user_message])
        yield UserMessageEvent(
            proc_name=self.agent_name, call_id=call_id, data=user_message
        )
        if ctx.printer:
            ctx.printer.print_messages(
                [user_message], agent_name=self.agent_name, call_id=call_id
            )

        tool_choice = NamedToolChoice(name=self._final_answer_tool.name)
        async for event in self.generate_message_stream(
            memory, tool_choice=tool_choice, ctx=ctx, call_id=call_id
        ):
            yield event

        final_answer_message = self._extract_final_answer_from_tool_calls(memory)
        if final_answer_message is None:
            raise AgentFinalAnswerError(proc_name=self.agent_name, call_id=call_id)
        yield GenMessageEvent(
            proc_name=self.agent_name, call_id=call_id, data=final_answer_message
        )

    async def execute(
        self, memory: LLMAgentMemory, ctx: RunContext[CtxT], call_id: str
    ) -> AssistantMessage | Sequence[AssistantMessage]:
        # 1. Generate the first message:
        #    In ReAct mode, we generate the first message without tool calls
        #    to force the agent to plan its actions in a separate message.

        tool_choice: ToolChoice | None = None
        if self.tools:
            tool_choice = "none" if self._react_mode else "auto"
        gen_message = await self.generate_message(
            memory, tool_choice=tool_choice, ctx=ctx, call_id=call_id
        )
        if not self.tools:
            return gen_message

        turns = 0

        while True:
            # 2. Check if we should exit the tool call loop

            # If a final answer is not provided via a tool call, we use
            # _terminate_tool_call_loop to determine whether to exit the loop.
            if not self._final_answer_as_tool_call and self._terminate_tool_call_loop(
                memory.message_history, ctx=ctx, call_id=call_id, num_turns=turns
            ):
                return gen_message

            # If a final answer is provided via a tool call, we check
            # if the last message contains the corresponding tool call.
            # If it does, we exit the loop.
            if self._final_answer_as_tool_call:
                final_answer = self._extract_final_answer_from_tool_calls(memory)
                if final_answer is not None:
                    return final_answer

            # Exit if the maximum number of turns is reached
            if turns >= self.max_turns:
                # If a final answer is provided via a tool call, we force the
                # agent to use the final answer tool.
                # Otherwise, we simply return the last generated message.
                if self._final_answer_as_tool_call:
                    final_answer = await self._generate_final_answer(
                        memory, ctx=ctx, call_id=call_id
                    )
                else:
                    final_answer = gen_message
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return final_answer

            # 3. Call tools.

            if gen_message.tool_calls:
                await self.call_tools(
                    gen_message.tool_calls, memory=memory, ctx=ctx, call_id=call_id
                )

            # Apply memory management (e.g. compacting or pruning memory)
            self._manage_memory(memory, ctx=ctx, call_id=call_id, num_turns=turns)

            # 4. Generate the next message based on the updated memory.
            #    In ReAct mode, we set tool_choice to "none" if we just called tools,
            #    so the next message will be an observation/planning message with
            #    no immediate tool calls.

            if self._react_mode and gen_message.tool_calls:
                tool_choice = "none"
            elif gen_message.tool_calls:
                tool_choice = "auto"
            else:
                tool_choice = "required"

            gen_message = await self.generate_message(
                memory, tool_choice=tool_choice, ctx=ctx, call_id=call_id
            )

            turns += 1

    async def execute_stream(
        self,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[Event[Any]]:
        tool_choice: ToolChoice = "none" if self._react_mode else "auto"
        gen_message: AssistantMessage | None = None
        async for event in self.generate_message_stream(
            memory, tool_choice=tool_choice, ctx=ctx, call_id=call_id
        ):
            if isinstance(event, GenMessageEvent):
                gen_message = event.data
            yield event
        if gen_message is None:
            return

        if not self.tools:
            return

        turns = 0

        while True:
            if not self._final_answer_as_tool_call and self._terminate_tool_call_loop(
                memory.message_history, ctx=ctx, call_id=call_id, num_turns=turns
            ):
                return

            if self._final_answer_as_tool_call:
                final_answer_message = self._extract_final_answer_from_tool_calls(
                    memory
                )
                if final_answer_message is not None:
                    yield GenMessageEvent(
                        proc_name=self.agent_name,
                        call_id=call_id,
                        data=final_answer_message,
                    )
                    return

            if turns >= self.max_turns:
                if self._final_answer_as_tool_call:
                    async for event in self._generate_final_answer_stream(
                        memory, ctx=ctx, call_id=call_id
                    ):
                        yield event
                logger.info(
                    f"Max turns reached: {self.max_turns}. Exiting the tool call loop."
                )
                return

            if gen_message.tool_calls:
                for tool_call in gen_message.tool_calls:
                    yield ToolCallEvent(
                        proc_name=self.agent_name, call_id=call_id, data=tool_call
                    )

                async for event in self.call_tools_stream(
                    gen_message.tool_calls, memory=memory, ctx=ctx, call_id=call_id
                ):
                    yield event

            self._manage_memory(memory, ctx=ctx, call_id=call_id, num_turns=turns)

            if self._react_mode and gen_message.tool_calls:
                tool_choice = "none"
            elif gen_message.tool_calls:
                tool_choice = "auto"
            else:
                tool_choice = "required"

            async for event in self.generate_message_stream(
                memory, tool_choice=tool_choice, ctx=ctx, call_id=call_id
            ):
                yield event
                if isinstance(event, GenMessageEvent):
                    gen_message = event.data

            turns += 1

    def get_final_answer_tool(self) -> BaseTool[BaseModel, None, Any]:
        class FinalAnswerTool(BaseTool[self._final_answer_type, None, Any]):
            name: str = "final_answer"
            description: str = (
                "You must call this tool to provide the final answer. "
                "DO NOT output your answer before calling the tool. "
            )

            async def run(
                self,
                inp: BaseModel,
                *,
                ctx: RunContext[Any] | None = None,
                call_id: str | None = None,
            ) -> None:
                return None

        return FinalAnswerTool()

    def _process_completion(
        self, completion: Completion, *, ctx: RunContext[CtxT], call_id: str
    ) -> None:
        ctx.completions[self.agent_name].append(completion)
        ctx.usage_tracker.update(
            agent_name=self.agent_name,
            completions=[completion],
            model_name=self.llm.model_name,
        )
        if ctx.printer:
            usages = [None] * (len(completion.messages) - 1) + [completion.usage]
            ctx.printer.print_messages(
                completion.messages,
                usages=usages,
                agent_name=self.agent_name,
                call_id=call_id,
            )
