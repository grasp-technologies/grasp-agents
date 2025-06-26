from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from pydantic import BaseModel

from .comm_processor import CommProcessor
from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory, MakeMemoryHandler
from .llm_policy_executor import (
    ExitToolCallLoopHandler,
    LLMPolicyExecutor,
    ManageMemoryHandler,
)
from .packet_pool import PacketPool
from .prompt_builder import (
    MakeInputContentHandler,
    MakeSystemPromptHandler,
    PromptBuilder,
)
from .run_context import CtxT, RunContext
from .typing.content import Content, ImageData
from .typing.converters import Converters
from .typing.events import Event, ProcOutputEvent, SystemMessageEvent, UserMessageEvent
from .typing.io import InT, LLMPrompt, LLMPromptArgs, OutT_co, ProcName
from .typing.message import Message, Messages, SystemMessage, UserMessage
from .typing.tool import BaseTool
from .utils import get_prompt, validate_obj_from_json_or_py_string

_InT_contra = TypeVar("_InT_contra", contravariant=True)
_OutT_co = TypeVar("_OutT_co", covariant=True)


class ParseOutputHandler(Protocol[_InT_contra, _OutT_co, CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        in_args: _InT_contra | None,
        ctx: RunContext[CtxT] | None,
    ) -> _OutT_co: ...


class LLMAgent(
    CommProcessor[InT, OutT_co, LLMAgentMemory, CtxT],
    Generic[InT, OutT_co, CtxT],
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
        # System args (static args provided via RunContext)
        sys_args_schema: type[LLMPromptArgs] | None = None,
        # User args (static args provided via RunContext)
        usr_args_schema: type[LLMPromptArgs] | None = None,
        # Agent loop settings
        max_turns: int = 100,
        react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Agent memory management
        reset_memory_on_run: bool = False,
        # Retries
        num_par_run_retries: int = 0,
        # Multi-agent routing
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            packet_pool=packet_pool,
            recipients=recipients,
            num_par_run_retries=num_par_run_retries,
        )

        # Agent memory

        self._memory: LLMAgentMemory = LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run

        # LLM policy executor

        self._used_default_llm_response_format: bool = False
        if llm.response_format is None and tools is None:
            llm.response_format = self.out_type
            self._used_default_llm_response_format = True

        self._policy_executor: LLMPolicyExecutor[OutT_co, CtxT] = LLMPolicyExecutor[
            self.out_type, CtxT
        ](
            agent_name=self.name,
            llm=llm,
            tools=tools,
            max_turns=max_turns,
            react_mode=react_mode,
            final_answer_as_tool_call=final_answer_as_tool_call,
        )

        # Prompt builder

        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)
        self._prompt_builder: PromptBuilder[InT, CtxT] = PromptBuilder[
            self.in_type, CtxT
        ](
            agent_name=self._name,
            sys_prompt_template=sys_prompt,
            in_prompt_template=in_prompt,
            sys_args_schema=sys_args_schema,
            usr_args_schema=usr_args_schema,
        )

        # self.no_tqdm = getattr(llm, "no_tqdm", False)

        self._make_memory_impl: MakeMemoryHandler | None = None
        self._parse_output_impl: ParseOutputHandler[InT, OutT_co, CtxT] | None = None
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
    def sys_args_schema(self) -> type[LLMPromptArgs] | None:
        return self._prompt_builder.sys_args_schema

    @property
    def usr_args_schema(self) -> type[LLMPromptArgs] | None:
        return self._prompt_builder.usr_args_schema

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt_template

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt_template

    def _memorize_inputs(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        ctx: RunContext[CtxT] | None = None,
    ) -> tuple[SystemMessage | None, UserMessage | None, LLMAgentMemory]:
        # 1. Get run arguments
        sys_args: LLMPromptArgs | None = None
        usr_args: LLMPromptArgs | None = None
        if ctx is not None:
            run_args = ctx.run_args.get(self.name)
            if run_args is not None:
                sys_args = run_args.sys
                usr_args = run_args.usr

        # 2. Make system prompt (can be None)

        formatted_sys_prompt = self._prompt_builder.make_system_prompt(
            sys_args=sys_args, ctx=ctx
        )

        # 3. Set agent memory

        system_message: SystemMessage | None = None
        if self._reset_memory_on_run or memory.is_empty:
            memory.reset(formatted_sys_prompt)
            if formatted_sys_prompt is not None:
                system_message = memory.message_history[0]  # type: ignore[union-attr]
        elif self._make_memory_impl:
            memory = self._make_memory_impl(
                prev_memory=memory,
                in_args=in_args,
                sys_prompt=formatted_sys_prompt,
                ctx=ctx,
            )

        # 3. Make and add user messages

        user_message = self._prompt_builder.make_user_message(
            chat_inputs=chat_inputs, in_args=in_args, usr_args=usr_args, ctx=ctx
        )
        if user_message:
            memory.update([user_message])

        return system_message, user_message, memory

    def _parse_output(
        self,
        conversation: Messages,
        *,
        in_args: InT | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> OutT_co:
        if self._parse_output_impl:
            return self._parse_output_impl(
                conversation=conversation, in_args=in_args, ctx=ctx
            )

        return validate_obj_from_json_or_py_string(
            str(conversation[-1].content or ""),
            adapter=self._out_type_adapter,
            from_substring=True,
        )

    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        system_message, user_message, memory = self._memorize_inputs(
            chat_inputs=chat_inputs, in_args=in_args, memory=memory, ctx=ctx
        )
        if system_message:
            self._print_messages([system_message], run_id=run_id, ctx=ctx)
        if user_message:
            self._print_messages([user_message], run_id=run_id, ctx=ctx)

        await self._policy_executor.execute(memory, run_id=run_id, ctx=ctx)

        return [
            self._parse_output(
                conversation=memory.message_history, in_args=in_args, ctx=ctx
            )
        ]

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: InT | None = None,
        memory: LLMAgentMemory,
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        system_message, user_message, memory = self._memorize_inputs(
            chat_inputs=chat_inputs, in_args=in_args, memory=memory, ctx=ctx
        )
        if system_message:
            yield SystemMessageEvent(data=system_message)
        if user_message:
            yield UserMessageEvent(data=user_message)

        # 4. Run tool call loop (new messages are added to the message
        #    history inside the loop)
        async for event in self._policy_executor.execute_stream(
            memory, run_id=run_id, ctx=ctx
        ):
            yield event

        output = self._parse_output(
            conversation=memory.message_history, in_args=in_args, ctx=ctx
        )
        yield ProcOutputEvent(data=output, name=self.name)

    def _print_messages(
        self,
        messages: Sequence[Message],
        run_id: str,
        ctx: RunContext[CtxT] | None = None,
    ) -> None:
        if ctx:
            ctx.printer.print_llm_messages(
                messages, agent_name=self.name, run_id=run_id
            )

    # -- Decorators for custom implementations --

    def make_system_prompt(
        self, func: MakeSystemPromptHandler[CtxT]
    ) -> MakeSystemPromptHandler[CtxT]:
        self._prompt_builder.make_system_prompt_impl = func

        return func

    def make_input_content(
        self, func: MakeInputContentHandler[InT, CtxT]
    ) -> MakeInputContentHandler[InT, CtxT]:
        self._prompt_builder.make_input_content_impl = func

        return func

    def parse_output(
        self, func: ParseOutputHandler[InT, OutT_co, CtxT]
    ) -> ParseOutputHandler[InT, OutT_co, CtxT]:
        if self._used_default_llm_response_format:
            self._policy_executor.llm.response_format = None
        self._parse_output_impl = func

        return func

    def make_memory(self, func: MakeMemoryHandler) -> MakeMemoryHandler:
        self._make_memory_impl = func

        return func

    def manage_memory(
        self, func: ManageMemoryHandler[CtxT]
    ) -> ManageMemoryHandler[CtxT]:
        self._policy_executor.manage_memory_impl = func

        return func

    def exit_tool_call_loop(
        self, func: ExitToolCallLoopHandler[CtxT]
    ) -> ExitToolCallLoopHandler[CtxT]:
        self._policy_executor.exit_tool_call_loop_impl = func

        return func

    # -- Override these methods in subclasses if needed --

    def _register_overridden_handlers(self) -> None:
        cur_cls = type(self)
        base_cls = LLMAgent[Any, Any, Any]

        if cur_cls._make_system_prompt is not base_cls._make_system_prompt:  # noqa: SLF001
            self._prompt_builder.make_system_prompt_impl = self._make_system_prompt

        if cur_cls._make_input_content is not base_cls._make_input_content:  # noqa: SLF001
            self._prompt_builder.make_input_content_impl = self._make_input_content

        if cur_cls._make_memory is not base_cls._make_memory:  # noqa: SLF001
            self._make_memory_impl = self._make_memory

        if cur_cls._manage_memory is not base_cls._manage_memory:  # noqa: SLF001
            self._policy_executor.manage_memory_impl = self._manage_memory

        if (
            cur_cls._exit_tool_call_loop is not base_cls._exit_tool_call_loop  # noqa: SLF001
        ):
            self._policy_executor.exit_tool_call_loop_impl = self._exit_tool_call_loop

        if (
            cur_cls._parse_output is not base_cls._parse_output  # noqa: SLF001
            and self._used_default_llm_response_format
        ):
            self._policy_executor.llm.response_format = None

    def _make_system_prompt(
        self, sys_args: LLMPromptArgs | None, *, ctx: RunContext[CtxT] | None = None
    ) -> str:
        raise NotImplementedError(
            "LLMAgent._format_sys_args must be overridden by a subclass "
            "if it's intended to be used as the system arguments formatter."
        )

    def _make_input_content(
        self,
        *,
        in_args: InT | None = None,
        usr_args: LLMPromptArgs | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Content:
        raise NotImplementedError(
            "LLMAgent._format_in_args must be overridden by a subclass"
        )

    def _make_memory(
        self,
        prev_memory: LLMAgentMemory,
        in_args: Sequence[InT] | None = None,
        sys_prompt: LLMPrompt | None = None,
        ctx: RunContext[Any] | None = None,
    ) -> LLMAgentMemory:
        raise NotImplementedError(
            "LLMAgent._make_memory must be overridden by a subclass"
        )

    def _exit_tool_call_loop(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError(
            "LLMAgent._exit_tool_call_loop must be overridden by a subclass"
        )

    def _manage_memory(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "LLMAgent._manage_memory must be overridden by a subclass"
        )
