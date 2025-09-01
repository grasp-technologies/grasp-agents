from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar, cast, final

from pydantic import BaseModel

from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory, MemoryPreparator
from .llm_policy_executor import (
    LLMPolicyExecutor,
    MemoryManager,
    ToolCallLoopTerminator,
)
from .processors.parallel_processor import ParallelProcessor
from .prompt_builder import (
    InputContentBuilder,
    PromptBuilder,
    SystemPromptBuilder,
)
from .run_context import CtxT, RunContext
from .typing.content import Content, ImageData
from .typing.converters import Converters
from .typing.events import (
    Event,
    ProcPayloadOutputEvent,
    SystemMessageEvent,
    UserMessageEvent,
)
from .typing.io import InT, LLMPrompt, OutT, ProcName
from .typing.message import Message, Messages, SystemMessage, UserMessage
from .typing.tool import BaseTool
from .utils import get_prompt, validate_obj_from_json_or_py_string

_InT_contra = TypeVar("_InT_contra", contravariant=True)
_OutT_co = TypeVar("_OutT_co", covariant=True)


class OutputParser(Protocol[_InT_contra, _OutT_co, CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        in_args: _InT_contra | None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> _OutT_co: ...


class LLMAgent(
    ParallelProcessor[InT, OutT, LLMAgentMemory, CtxT],
    Generic[InT, OutT, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        *,
        # LLM
        llm: LLM[LLMSettings, Converters],
        # Tools
        tools: list[BaseTool[Any, Any, CtxT]] | None = None,
        # Input prompt template (combines user and received arguments)
        in_prompt: LLMPrompt | None = None,
        in_prompt_path: str | Path | None = None,
        # System prompt template
        sys_prompt: LLMPrompt | None = None,
        sys_prompt_path: str | Path | None = None,
        # LLM response validation
        response_schema: Any | None = None,
        response_schema_by_xml_tag: Mapping[str, Any] | None = None,
        # Agent loop settings
        max_turns: int = 100,
        react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Agent memory management
        reset_memory_on_run: bool = False,
        # Agent run retries
        max_retries: int = 0,
        # Multi-agent routing
        recipients: Sequence[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, recipients=recipients, max_retries=max_retries)

        # Agent memory

        self._memory: LLMAgentMemory = LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run

        self.memory_preparator: MemoryPreparator | None
        if not hasattr(type(self), "memory_preparator"):
            self.memory_preparator = None

        self.output_parser: OutputParser[InT, OutT, CtxT] | None
        if not hasattr(type(self), "output_parser"):
            self.output_parser = None

        # LLM policy executor

        if issubclass(self._out_type, BaseModel):
            final_answer_type = self._out_type
        elif not final_answer_as_tool_call:
            final_answer_type = BaseModel
        else:
            raise TypeError(
                "Final answer type must be a subclass of BaseModel if "
                "final_answer_as_tool_call is True."
            )

        self._used_default_llm_response_schema: bool = False
        if (
            response_schema is None
            and tools is None
            and not hasattr(type(self), "output_parser")
        ):
            response_schema = self.out_type
            self._used_default_llm_response_schema = True

        self._policy_executor: LLMPolicyExecutor[CtxT] = LLMPolicyExecutor[CtxT](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            response_schema=response_schema,
            response_schema_by_xml_tag=response_schema_by_xml_tag,
            max_turns=max_turns,
            react_mode=react_mode,
            final_answer_type=final_answer_type,
            final_answer_as_tool_call=final_answer_as_tool_call,
        )

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)

        self._prompt_builder = PromptBuilder[self.in_type, CtxT](
            agent_name=self._name, sys_prompt=sys_prompt, in_prompt=in_prompt
        )

        self._register_overridden_handlers()

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._policy_executor.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._policy_executor.tools

    @property
    def max_turns(self) -> int:
        return self._policy_executor.max_turns

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    @final
    def _prepare_memory(
        self,
        memory: LLMAgentMemory,
        *,
        in_args: InT | None = None,
        sys_prompt: LLMPrompt | None = None,
        ctx: RunContext[Any],
        call_id: str,
    ) -> None:
        if self.memory_preparator:
            return self.memory_preparator(
                memory=memory,
                in_args=in_args,
                sys_prompt=sys_prompt,
                ctx=ctx,
                call_id=call_id,
            )

    def _memorize_inputs(
        self,
        memory: LLMAgentMemory,
        *,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> tuple[SystemMessage | None, UserMessage | None]:
        formatted_sys_prompt = self._prompt_builder.build_system_prompt(
            ctx=ctx, call_id=call_id
        )

        system_message: SystemMessage | None = None
        if self._reset_memory_on_run or memory.is_empty:
            memory.reset(formatted_sys_prompt)
            if formatted_sys_prompt is not None:
                system_message = cast("SystemMessage", memory.message_history[0])
        else:
            self._prepare_memory(
                memory=memory,
                in_args=in_args,
                sys_prompt=formatted_sys_prompt,
                ctx=ctx,
                call_id=call_id,
            )

        input_message = self._prompt_builder.build_input_message(
            chat_inputs=chat_inputs, in_args=in_args, ctx=ctx, call_id=call_id
        )
        if input_message:
            memory.update([input_message])

        return system_message, input_message

    def parse_output_default(self, conversation: Messages) -> OutT:
        return validate_obj_from_json_or_py_string(
            str(conversation[-1].content or ""),
            schema=self._out_type,
            from_substring=False,
            strip_language_markdown=True,
        )

    def _parse_output(
        self,
        conversation: Messages,
        *,
        in_args: InT | None = None,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> OutT:
        if self.output_parser:
            return self.output_parser(
                conversation=conversation, in_args=in_args, ctx=ctx, call_id=call_id
            )

        return self.parse_output_default(conversation)

    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> OutT:
        system_message, input_message = self._memorize_inputs(
            memory=memory,
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )
        if system_message:
            self._print_messages([system_message], ctx=ctx, call_id=call_id)
        if input_message:
            self._print_messages([input_message], ctx=ctx, call_id=call_id)

        await self._policy_executor.execute(memory, ctx=ctx, call_id=call_id)

        return self._parse_output(
            conversation=memory.message_history,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT],
        call_id: str,
    ) -> AsyncIterator[Event[Any]]:
        system_message, input_message = self._memorize_inputs(
            memory=memory,
            chat_inputs=chat_inputs,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )
        if system_message:
            self._print_messages([system_message], ctx=ctx, call_id=call_id)
            yield SystemMessageEvent(
                data=system_message, proc_name=self.name, call_id=call_id
            )
        if input_message:
            self._print_messages([input_message], ctx=ctx, call_id=call_id)
            yield UserMessageEvent(
                data=input_message, proc_name=self.name, call_id=call_id
            )

        async for event in self._policy_executor.execute_stream(
            memory, ctx=ctx, call_id=call_id
        ):
            yield event

        output = self._parse_output(
            conversation=memory.message_history,
            in_args=in_args,
            ctx=ctx,
            call_id=call_id,
        )
        yield ProcPayloadOutputEvent(data=output, proc_name=self.name, call_id=call_id)

    def _print_messages(
        self, messages: Sequence[Message], ctx: RunContext[CtxT], call_id: str
    ) -> None:
        if ctx.printer:
            ctx.printer.print_messages(messages, agent_name=self.name, call_id=call_id)

    # -- Override these methods in subclasses if needed --

    def _register_overridden_handlers(self) -> None:
        cur_cls = type(self)
        base_cls = LLMAgent[Any, Any, Any]

        # Prompt builder

        if cur_cls.system_prompt_builder is not base_cls.system_prompt_builder:
            self._prompt_builder.system_prompt_builder = self.system_prompt_builder

        if cur_cls.input_content_builder is not base_cls.input_content_builder:
            self._prompt_builder.input_content_builder = self.input_content_builder

        # Policy executor

        if cur_cls.tool_call_loop_terminator is not base_cls.tool_call_loop_terminator:
            self._policy_executor.tool_call_loop_terminator = (
                self.tool_call_loop_terminator
            )

        if cur_cls.memory_manager is not base_cls.memory_manager:
            self._policy_executor.memory_manager = self.memory_manager

    def system_prompt_builder(self, ctx: RunContext[CtxT], call_id: str) -> str | None:
        if self._prompt_builder.system_prompt_builder is not None:
            return self._prompt_builder.system_prompt_builder(ctx=ctx, call_id=call_id)
        raise NotImplementedError("System prompt builder is not implemented.")

    def input_content_builder(
        self, in_args: InT, ctx: RunContext[CtxT], call_id: str
    ) -> Content:
        if self._prompt_builder.input_content_builder is not None:
            return self._prompt_builder.input_content_builder(
                in_args=in_args, ctx=ctx, call_id=call_id
            )
        raise NotImplementedError("Input content builder is not implemented.")

    def tool_call_loop_terminator(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> bool:
        if self._policy_executor.tool_call_loop_terminator is not None:
            return self._policy_executor.tool_call_loop_terminator(
                conversation=conversation, ctx=ctx, call_id=call_id, **kwargs
            )
        raise NotImplementedError("Tool call loop terminator is not implemented.")

    def memory_manager(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT],
        call_id: str,
        **kwargs: Any,
    ) -> None:
        if self._policy_executor.memory_manager is not None:
            return self._policy_executor.memory_manager(
                memory=memory, ctx=ctx, call_id=call_id, **kwargs
            )
        raise NotImplementedError("Memory manager is not implemented.")

    # Decorators for custom implementations as an alternative to overriding methods

    def add_system_prompt_builder(
        self, func: SystemPromptBuilder[CtxT]
    ) -> SystemPromptBuilder[CtxT]:
        self._prompt_builder.system_prompt_builder = func

        return func

    def add_input_content_builder(
        self, func: InputContentBuilder[InT, CtxT]
    ) -> InputContentBuilder[InT, CtxT]:
        self._prompt_builder.input_content_builder = func

        return func

    def add_memory_manager(self, func: MemoryManager[CtxT]) -> MemoryManager[CtxT]:
        self._policy_executor.memory_manager = func

        return func

    def add_tool_call_loop_terminator(
        self, func: ToolCallLoopTerminator[CtxT]
    ) -> ToolCallLoopTerminator[CtxT]:
        self._policy_executor.tool_call_loop_terminator = func

        return func

    def add_output_parser(
        self, func: OutputParser[InT, OutT, CtxT]
    ) -> OutputParser[InT, OutT, CtxT]:
        if self._used_default_llm_response_schema:
            self._policy_executor.response_schema = None
        self.output_parser = func

        return func

    def add_memory_preparator(self, func: MemoryPreparator) -> MemoryPreparator:
        self.memory_preparator = func
        return func
