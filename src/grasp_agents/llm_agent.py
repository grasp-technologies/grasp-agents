from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, Protocol, cast, final

from pydantic import BaseModel

from .agent_message import AgentMessage
from .agent_message_pool import AgentMessagePool
from .comm_agent import CommunicatingAgent
from .llm import LLM, LLMSettings
from .llm_agent_state import (
    LLMAgentState,
    SetAgentState,
    SetAgentStateStrategy,
)
from .prompt_builder import (
    FormatInputArgsHandler,
    FormatSystemArgsHandler,
    PromptBuilder,
)
from .run_context import (
    CtxT,
    InteractionRecord,
    RunContextWrapper,
    SystemRunArgs,
    UserRunArgs,
)
from .tool_orchestrator import (
    ExitToolCallLoopHandler,
    ManageAgentStateHandler,
    ToolOrchestrator,
)
from .typing.content import ImageData
from .typing.converters import Converters
from .typing.io import (
    AgentID,
    AgentState,
    InT,
    LLMFormattedArgs,
    LLMFormattedSystemArgs,
    LLMPrompt,
    LLMPromptArgs,
    OutT,
)
from .typing.message import Conversation, Message, SystemMessage
from .typing.tool import BaseTool
from .utils import get_prompt, validate_obj_from_json_or_py_string


class ParseOutputHandler(Protocol[InT, OutT, CtxT]):
    def __call__(
        self,
        conversation: Conversation,
        *,
        in_args: InT | None,
        batch_idx: int,
        ctx: RunContextWrapper[CtxT] | None,
    ) -> OutT: ...


class LLMAgent(
    CommunicatingAgent[InT, OutT, LLMAgentState, CtxT],
    Generic[InT, OutT, CtxT],
):
    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        agent_id: AgentID,
        *,
        # LLM
        llm: LLM[LLMSettings, Converters],
        # Input prompt template (combines user and received arguments)
        in_prompt: LLMPrompt | None = None,
        in_prompt_path: str | Path | None = None,
        # System prompt template
        sys_prompt: LLMPrompt | None = None,
        sys_prompt_path: str | Path | None = None,
        # System args (static args provided via RunContextWrapper)
        sys_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # User args (static args provided via RunContextWrapper)
        usr_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # Tools
        tools: list[BaseTool[Any, Any, CtxT]] | None = None,
        max_turns: int = 1000,
        react_mode: bool = False,
        # Agent state management
        set_state_strategy: SetAgentStateStrategy = "keep",
        # Multi-agent routing
        message_pool: AgentMessagePool[CtxT] | None = None,
        recipient_ids: list[AgentID] | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id, message_pool=message_pool, recipient_ids=recipient_ids
        )

        # Agent state
        self._state: LLMAgentState = LLMAgentState()
        self.set_state_strategy: SetAgentStateStrategy = set_state_strategy
        self._set_agent_state_impl: SetAgentState | None = None

        # Tool orchestrator

        self._using_default_llm_response_format: bool = False
        if llm.response_format is None and tools is None:
            llm.response_format = self.out_type
            self._using_default_llm_response_format = True

        self._tool_orchestrator: ToolOrchestrator[CtxT] = ToolOrchestrator[CtxT](
            agent_id=self.agent_id,
            llm=llm,
            tools=tools,
            max_turns=max_turns,
            react_mode=react_mode,
        )

        # Prompt builder
        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        in_prompt = get_prompt(prompt_text=in_prompt, prompt_path=in_prompt_path)
        self._prompt_builder: PromptBuilder[InT, CtxT] = PromptBuilder[
            self.in_type, CtxT
        ](
            agent_id=self._agent_id,
            sys_prompt=sys_prompt,
            in_prompt=in_prompt,
            sys_args_schema=sys_args_schema,
            usr_args_schema=usr_args_schema,
        )

        self.no_tqdm = getattr(llm, "no_tqdm", False)

        self._register_overridden_handlers()

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._tool_orchestrator.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, Any, CtxT]]:
        return self._tool_orchestrator.tools

    @property
    def max_turns(self) -> int:
        return self._tool_orchestrator.max_turns

    @property
    def sys_args_schema(self) -> type[LLMPromptArgs]:
        return self._prompt_builder.sys_args_schema

    @property
    def usr_args_schema(self) -> type[LLMPromptArgs]:
        return self._prompt_builder.usr_args_schema

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.sys_prompt

    @property
    def in_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.in_prompt

    def _parse_output(
        self,
        conversation: Conversation,
        *,
        in_args: InT | None = None,
        batch_idx: int = 0,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> OutT:
        if self._parse_output_impl:
            if self._using_default_llm_response_format:
                # When using custom output parsing, the required LLM response format
                # can differ from the final agent output type ->
                # set it back to None unless it was specified explicitly at init.
                self._tool_orchestrator.llm.response_format = None
                self._using_default_llm_response_format = False

            return self._parse_output_impl(
                conversation=conversation,
                in_args=in_args,
                batch_idx=batch_idx,
                ctx=ctx,
            )

        return validate_obj_from_json_or_py_string(
            str(conversation[-1].content or ""),
            adapter=self._out_type_adapter,
            from_substring=True,
        )

    @staticmethod
    def _validate_run_inputs(
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        in_message: AgentMessage[InT, AgentState] | None = None,
        entry_point: bool = False,
    ) -> None:
        multiple_inputs_err_message = (
            "Only one of chat_inputs, in_args, or in_message must be provided."
        )
        if chat_inputs is not None and in_args is not None:
            raise ValueError(multiple_inputs_err_message)
        if chat_inputs is not None and in_message is not None:
            raise ValueError(multiple_inputs_err_message)
        if in_args is not None and in_message is not None:
            raise ValueError(multiple_inputs_err_message)

        if entry_point and in_message is not None:
            raise ValueError(
                "Entry point agent cannot receive messages from other agents."
            )

    @final
    async def run(
        self,
        chat_inputs: LLMPrompt | Sequence[str | ImageData] | None = None,
        *,
        in_message: AgentMessage[InT, AgentState] | None = None,
        in_args: InT | Sequence[InT] | None = None,
        entry_point: bool = False,
        ctx: RunContextWrapper[CtxT] | None = None,
        forbid_state_change: bool = False,
        **gen_kwargs: Any,  # noqa: ARG002
    ) -> AgentMessage[OutT, LLMAgentState]:
        # Get run arguments
        sys_args: SystemRunArgs = LLMPromptArgs()
        usr_args: UserRunArgs = LLMPromptArgs()
        if ctx is not None:
            run_args = ctx.run_args.get(self.agent_id)
            if run_args is not None:
                sys_args = run_args.sys
                usr_args = run_args.usr

        self._validate_run_inputs(
            chat_inputs=chat_inputs,
            in_args=in_args,
            in_message=in_message,
            entry_point=entry_point,
        )

        # 1. Make system prompt (can be None)
        formatted_sys_prompt = self._prompt_builder.make_sys_prompt(
            sys_args=sys_args, ctx=ctx
        )

        # 2. Set agent state

        cur_state = self.state.model_copy(deep=True)
        in_state = in_message.sender_state if in_message else None
        prev_mh_len = len(cur_state.message_history)

        state = LLMAgentState.from_cur_and_in_states(
            cur_state=cur_state,
            in_state=in_state,
            sys_prompt=formatted_sys_prompt,
            strategy=self.set_state_strategy,
            set_agent_state_impl=self._set_agent_state_impl,
            ctx=ctx,
        )

        self._print_sys_msg(state=state, prev_mh_len=prev_mh_len, ctx=ctx)

        # 3. Make and add user messages (can be empty)
        _in_args_batch: Sequence[InT] | None = None
        if in_message is not None:
            _in_args_batch = in_message.payloads
        elif in_args is not None:
            _in_args_batch = in_args if isinstance(in_args, Sequence) else [in_args]  # type: ignore[assignment]

        user_message_batch = self._prompt_builder.make_user_messages(
            chat_inputs=chat_inputs,
            usr_args=usr_args,
            in_args_batch=_in_args_batch,
            entry_point=entry_point,
            ctx=ctx,
        )
        if user_message_batch:
            state.message_history.add_message_batch(user_message_batch)
            self._print_msgs(user_message_batch, ctx=ctx)

        if not self.tools:
            # 4. Generate messages without tools
            await self._tool_orchestrator.generate_once(state=state, ctx=ctx)
        else:
            # 4. Run tool call loop (new messages are added to the message
            #    history inside the loop)
            await self._tool_orchestrator.run_loop(state=state, ctx=ctx)

        # 5. Parse outputs
        batch_size = state.message_history.batch_size
        in_args_batch = in_message.payloads if in_message else batch_size * [None]
        val_output_batch = [
            self._out_type_adapter.validate_python(
                self._parse_output(conversation=conv, in_args=in_args, ctx=ctx)
            )
            for conv, in_args in zip(
                state.message_history.batched_conversations,
                in_args_batch,
                strict=False,
            )
        ]

        # 6. Write interaction history to context

        recipient_ids = self._validate_routing(val_output_batch)

        if ctx:
            interaction_record = InteractionRecord(
                source_id=self.agent_id,
                recipient_ids=recipient_ids,
                chat_inputs=chat_inputs,
                sys_prompt=self.sys_prompt,
                in_prompt=self.in_prompt,
                sys_args=sys_args,
                usr_args=usr_args,
                in_args=(in_message.payloads if in_message is not None else None),
                outputs=val_output_batch,
                state=state,
            )
            ctx.interaction_history.append(
                cast(
                    "InteractionRecord[Any, Any, AgentState]",
                    interaction_record,
                )
            )

        agent_message = AgentMessage(
            payloads=val_output_batch,
            sender_id=self.agent_id,
            sender_state=state,
            recipient_ids=recipient_ids,
        )

        if not forbid_state_change:
            self._state = state

        return agent_message

    def _print_msgs(
        self,
        messages: Sequence[Message],
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> None:
        if ctx:
            ctx.printer.print_llm_messages(messages, agent_id=self.agent_id)

    def _print_sys_msg(
        self,
        state: LLMAgentState,
        prev_mh_len: int,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> None:
        if (
            len(state.message_history) == 1
            and prev_mh_len == 0
            and isinstance(state.message_history[0][0], SystemMessage)
        ):
            self._print_msgs([state.message_history[0][0]], ctx=ctx)

    # -- Handlers for custom implementations --

    def format_sys_args_handler(
        self, func: FormatSystemArgsHandler[CtxT]
    ) -> FormatSystemArgsHandler[CtxT]:
        self._prompt_builder.format_sys_args_impl = func

        return func

    def format_in_args_handler(
        self, func: FormatInputArgsHandler[InT, CtxT]
    ) -> FormatInputArgsHandler[InT, CtxT]:
        self._prompt_builder.format_in_args_impl = func

        return func

    def parse_output_handler(
        self, func: ParseOutputHandler[InT, OutT, CtxT]
    ) -> ParseOutputHandler[InT, OutT, CtxT]:
        self._parse_output_impl = func

        return func

    def set_agent_state_handler(self, func: SetAgentState) -> SetAgentState:
        self._make_custom_agent_state_impl = func

        return func

    def exit_tool_call_loop_handler(
        self, func: ExitToolCallLoopHandler[CtxT]
    ) -> ExitToolCallLoopHandler[CtxT]:
        self._tool_orchestrator.exit_tool_call_loop_impl = func

        return func

    def manage_agent_state_handler(
        self, func: ManageAgentStateHandler[CtxT]
    ) -> ManageAgentStateHandler[CtxT]:
        self._tool_orchestrator.manage_agent_state_impl = func

        return func

    # -- Override these methods in subclasses if needed --

    def _register_overridden_handlers(self) -> None:
        cur_cls = type(self)
        base_cls = LLMAgent[Any, Any, Any]

        if cur_cls._format_sys_args is not base_cls._format_sys_args:  # noqa: SLF001
            self._prompt_builder.format_sys_args_impl = self._format_sys_args

        if cur_cls._format_in_args is not base_cls._format_in_args:  # noqa: SLF001
            self._prompt_builder.format_in_args_impl = self._format_in_args

        if cur_cls._set_agent_state is not base_cls._set_agent_state:  # noqa: SLF001
            self._set_agent_state_impl = self._set_agent_state

        if cur_cls._manage_agent_state is not base_cls._manage_agent_state:  # noqa: SLF001
            self._tool_orchestrator.manage_agent_state_impl = self._manage_agent_state

        if (
            cur_cls._exit_tool_call_loop is not base_cls._exit_tool_call_loop  # noqa: SLF001
        ):
            self._tool_orchestrator.exit_tool_call_loop_impl = self._exit_tool_call_loop

        self._parse_output_impl: ParseOutputHandler[InT, OutT, CtxT] | None = None

    def _format_sys_args(
        self,
        sys_args: LLMPromptArgs,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> LLMFormattedSystemArgs:
        raise NotImplementedError(
            "LLMAgent._format_sys_args must be overridden by a subclass "
            "if it's intended to be used as the system arguments formatter."
        )

    def _format_in_args(
        self,
        *,
        usr_args: LLMPromptArgs,
        in_args: InT,
        batch_idx: int = 0,
        ctx: RunContextWrapper[CtxT] | None = None,
    ) -> LLMFormattedArgs:
        raise NotImplementedError(
            "LLMAgent._format_in_args must be overridden by a subclass"
        )

    def _set_agent_state(
        self,
        cur_state: LLMAgentState,
        *,
        in_state: AgentState | None,
        sys_prompt: LLMPrompt | None,
        ctx: RunContextWrapper[Any] | None,
    ) -> LLMAgentState:
        raise NotImplementedError(
            "LLMAgent._set_agent_state_handler must be overridden by a subclass"
        )

    def _exit_tool_call_loop(
        self,
        conversation: Conversation,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> bool:
        raise NotImplementedError(
            "LLMAgent._tool_call_loop_exit must be overridden by a subclass"
        )

    def _manage_agent_state(
        self,
        state: LLMAgentState,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "LLMAgent._manage_agent_state must be overridden by a subclass"
        )
