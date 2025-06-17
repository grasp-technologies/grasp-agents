from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol

from pydantic import BaseModel

from .comm_processor import CommProcessor
from .llm import LLM, LLMSettings
from .llm_agent_memory import LLMAgentMemory, SetMemoryHandler
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
from .typing.io import InT_contra, LLMPrompt, LLMPromptArgs, OutT_co, ProcName
from .typing.message import Message, Messages, SystemMessage, UserMessage
from .typing.tool import BaseTool
from .utils import get_prompt, validate_obj_from_json_or_py_string


class ParseOutputHandler(Protocol[InT_contra, OutT_co, CtxT]):
    def __call__(
        self,
        conversation: Messages,
        *,
        in_args: InT_contra | None,
        batch_idx: int,
        ctx: RunContext[CtxT] | None,
    ) -> OutT_co: ...


class LLMAgent(
    CommProcessor[InT_contra, OutT_co, LLMAgentMemory, CtxT],
    Generic[InT_contra, OutT_co, CtxT],
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
        sys_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # User args (static args provided via RunContext)
        usr_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # Agent loop settings
        max_turns: int = 100,
        react_mode: bool = False,
        final_answer_as_tool_call: bool = False,
        # Agent memory management
        reset_memory_on_run: bool = False,
        # Multi-agent routing
        packet_pool: PacketPool[CtxT] | None = None,
        recipients: list[ProcName] | None = None,
    ) -> None:
        super().__init__(name=name, packet_pool=packet_pool, recipients=recipients)

        # Agent memory

        self._memory: LLMAgentMemory = LLMAgentMemory()
        self._reset_memory_on_run = reset_memory_on_run
        self._set_memory_impl: SetMemoryHandler | None = None

        # LLM policy executor

        self._using_default_llm_response_format: bool = False
        if llm.response_format is None and tools is None:
            llm.response_format = self.out_type
            self._using_default_llm_response_format = True

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
        self._prompt_builder: PromptBuilder[InT_contra, CtxT] = PromptBuilder[
            self.in_type, CtxT
        ](
            agent_name=self._name,
            sys_prompt_template=sys_prompt,
            in_prompt_template=in_prompt,
            sys_args_schema=sys_args_schema,
            usr_args_schema=usr_args_schema,
        )

        self.no_tqdm = getattr(llm, "no_tqdm", False)

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

    def _parse_output(
        self,
        conversation: Messages,
        *,
        in_args: InT_contra | None = None,
        batch_idx: int = 0,
        ctx: RunContext[CtxT] | None = None,
    ) -> OutT_co:
        if self._parse_output_impl:
            if self._using_default_llm_response_format:
                # When using custom output parsing, the required LLM response format
                # can differ from the final agent output type ->
                # set it back to None unless it was specified explicitly at init.
                self._policy_executor.llm.response_format = None
                # self._using_default_llm_response_format = False

            return self._parse_output_impl(
                conversation=conversation, in_args=in_args, batch_idx=batch_idx, ctx=ctx
            )

        return validate_obj_from_json_or_py_string(
            str(conversation[-1].content or ""),
            adapter=self._out_type_adapter,
            from_substring=True,
        )

    def _memorize_inputs(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: Sequence[InT_contra] | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> tuple[SystemMessage | None, Sequence[UserMessage], LLMAgentMemory]:
        # Get run arguments
        sys_args: LLMPromptArgs | None = None
        usr_args: LLMPromptArgs | None = None
        if ctx is not None:
            run_args = ctx.run_args.get(self.name)
            if run_args is not None:
                sys_args = run_args.sys
                usr_args = run_args.usr

        # 1. Make system prompt (can be None)

        formatted_sys_prompt = self._prompt_builder.make_sys_prompt(
            sys_args=sys_args, ctx=ctx
        )

        # 2. Set agent memory

        _memory = self.memory.model_copy(deep=True)
        prev_message_hist_length = len(_memory.message_history)
        if self._reset_memory_on_run or _memory.is_empty:
            _memory.reset(formatted_sys_prompt)
        elif self._set_memory_impl:
            _memory = self._set_memory_impl(
                prev_memory=_memory,
                in_args=in_args,
                sys_prompt=formatted_sys_prompt,
                ctx=ctx,
            )

        # 3. Make and add user messages

        user_message_batch = self._prompt_builder.make_user_messages(
            chat_inputs=chat_inputs, in_args_batch=in_args, usr_args=usr_args, ctx=ctx
        )
        if user_message_batch:
            _memory.update(message_batch=user_message_batch)

        # 4. Extract system message if it was added

        system_message: SystemMessage | None = None
        if (
            len(_memory.message_history) == 1
            and prev_message_hist_length == 0
            and isinstance(_memory.message_history[0][0], SystemMessage)
        ):
            system_message = _memory.message_history[0][0]

        return system_message, user_message_batch, _memory

    def _extract_outputs(
        self,
        memory: LLMAgentMemory,
        in_args: Sequence[InT_contra] | None = None,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        outputs: list[OutT_co] = []
        for i, _conv in enumerate(memory.message_history.conversations):
            if in_args is not None:
                _in_args_single = in_args[min(i, len(in_args) - 1)]
            else:
                _in_args_single = None

            outputs.append(
                self._parse_output(
                    conversation=_conv, in_args=_in_args_single, batch_idx=i, ctx=ctx
                )
            )

        return outputs

    async def _process(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: Sequence[InT_contra] | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> Sequence[OutT_co]:
        system_message, user_message_batch, memory = self._memorize_inputs(
            chat_inputs=chat_inputs, in_args=in_args, ctx=ctx
        )

        if system_message is not None:
            self._print_messages([system_message], ctx=ctx)
        if user_message_batch:
            self._print_messages(user_message_batch, ctx=ctx)

        await self._policy_executor.execute(memory, ctx=ctx)

        if not forgetful:
            self._memory = memory

        return self._extract_outputs(memory=memory, in_args=in_args, ctx=ctx)

    async def _process_stream(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_args: Sequence[InT_contra] | None = None,
        forgetful: bool = False,
        ctx: RunContext[CtxT] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        system_message, user_message_batch, memory = self._memorize_inputs(
            chat_inputs=chat_inputs, in_args=in_args, ctx=ctx
        )

        if system_message is not None:
            yield SystemMessageEvent(data=system_message)
        if user_message_batch:
            for user_message in user_message_batch:
                yield UserMessageEvent(data=user_message)

        # 4. Run tool call loop (new messages are added to the message
        #    history inside the loop)
        async for event in self._policy_executor.execute_stream(memory, ctx=ctx):
            yield event

        if not forgetful:
            self._memory = memory

        outputs = self._extract_outputs(memory=memory, in_args=in_args, ctx=ctx)
        for output in outputs:
            yield ProcOutputEvent(data=output, name=self.name)

    def _print_messages(
        self, messages: Sequence[Message], ctx: RunContext[CtxT] | None = None
    ) -> None:
        if ctx:
            ctx.printer.print_llm_messages(messages, agent_name=self.name)

    # -- Decorators for custom implementations --

    def make_sys_prompt(
        self, func: MakeSystemPromptHandler[CtxT]
    ) -> MakeSystemPromptHandler[CtxT]:
        self._prompt_builder.make_sys_prompt_impl = func

        return func

    def make_in_content(
        self, func: MakeInputContentHandler[InT_contra, CtxT]
    ) -> MakeInputContentHandler[InT_contra, CtxT]:
        self._prompt_builder.make_in_content_impl = func

        return func

    def parse_output(
        self, func: ParseOutputHandler[InT_contra, OutT_co, CtxT]
    ) -> ParseOutputHandler[InT_contra, OutT_co, CtxT]:
        self._parse_output_impl = func

        return func

    def set_memory(self, func: SetMemoryHandler) -> SetMemoryHandler:
        self._set_memory_impl = func

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

        if cur_cls._make_sys_prompt_fn is not base_cls._make_sys_prompt_fn:  # noqa: SLF001
            self._prompt_builder.make_sys_prompt_impl = self._make_sys_prompt_fn

        if cur_cls._make_in_content_fn is not base_cls._make_in_content_fn:  # noqa: SLF001
            self._prompt_builder.make_in_content_impl = self._make_in_content_fn

        if cur_cls._set_memory_fn is not base_cls._set_memory_fn:  # noqa: SLF001
            self._set_memory_impl = self._set_memory_fn

        if cur_cls._manage_memory_fn is not base_cls._manage_memory_fn:  # noqa: SLF001
            self._policy_executor.manage_memory_impl = self._manage_memory_fn

        if (
            cur_cls._exit_tool_call_loop_fn is not base_cls._exit_tool_call_loop_fn  # noqa: SLF001
        ):
            self._policy_executor.exit_tool_call_loop_impl = (
                self._exit_tool_call_loop_fn
            )

        self._parse_output_impl: (
            ParseOutputHandler[InT_contra, OutT_co, CtxT] | None
        ) = None

    def _make_sys_prompt_fn(
        self, sys_args: LLMPromptArgs | None, *, ctx: RunContext[CtxT] | None = None
    ) -> str:
        raise NotImplementedError(
            "LLMAgent._format_sys_args must be overridden by a subclass "
            "if it's intended to be used as the system arguments formatter."
        )

    def _make_in_content_fn(
        self,
        *,
        in_args: InT_contra | None = None,
        usr_args: LLMPromptArgs | None = None,
        batch_idx: int = 0,
        ctx: RunContext[CtxT] | None = None,
    ) -> Content:
        raise NotImplementedError(
            "LLMAgent._format_in_args must be overridden by a subclass"
        )

    def _set_memory_fn(
        self,
        prev_memory: LLMAgentMemory,
        in_args: InT_contra | Sequence[InT_contra] | None = None,
        sys_prompt: LLMPrompt | None = None,
        ctx: RunContext[Any] | None = None,
    ) -> LLMAgentMemory:
        raise NotImplementedError(
            "LLMAgent._set_memory must be overridden by a subclass"
        )

    def _exit_tool_call_loop_fn(
        self,
        conversation: Messages,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError(
            "LLMAgent._exit_tool_call_loop must be overridden by a subclass"
        )

    def _manage_memory_fn(
        self,
        memory: LLMAgentMemory,
        *,
        ctx: RunContext[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "LLMAgent._manage_memory must be overridden by a subclass"
        )
