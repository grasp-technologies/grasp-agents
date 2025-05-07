from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, cast, final

from pydantic import BaseModel

from .agent_message import AgentMessage
from .agent_message_pool import AgentMessagePool
from .comm_agent import CommunicatingAgent
from .llm import LLM, LLMSettings
from .llm_agent_state import (
    LLMAgentState,
    MakeCustomAgentState,
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
from .tool_orchestrator import ToolCallLoopExitHandler, ToolOrchestrator
from .typing.content import ImageData
from .typing.converters import Converters
from .typing.io import (
    AgentID,
    AgentPayload,
    AgentState,
    InT,
    LLMPrompt,
    LLMPromptArgs,
    OutT,
)
from .typing.message import Conversation, Message, SystemMessage
from .typing.tool import BaseTool
from .utils import get_prompt


class LLMAgent(
    CommunicatingAgent[InT, OutT, LLMAgentState, CtxT],
    Generic[InT, OutT, CtxT],
):
    def __init__(
        self,
        agent_id: AgentID,
        *,
        # LLM
        llm: LLM[LLMSettings, Converters],
        # Input prompt template (combines user and received arguments)
        inp_prompt: LLMPrompt | None = None,
        inp_prompt_path: str | Path | None = None,
        # System prompt template
        sys_prompt: LLMPrompt | None = None,
        sys_prompt_path: str | Path | None = None,
        # System args (static args provided via RunContextWrapper)
        sys_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # User args (static args provided via RunContextWrapper)
        usr_args_schema: type[LLMPromptArgs] = LLMPromptArgs,
        # Received args (args from another agent)
        rcv_args_schema: type[InT] = cast("type[InT]", AgentPayload),
        # Output schema
        out_schema: type[OutT] = cast("type[OutT]", AgentPayload),
        # Tools
        tools: list[BaseTool[BaseModel, BaseModel, CtxT]] | None = None,
        max_turns: int = 1000,
        react_mode: bool = False,
        # Agent state management
        set_state_strategy: SetAgentStateStrategy = "keep",
        # Multi-agent routing
        message_pool: AgentMessagePool[CtxT] | None = None,
        recipient_ids: list[AgentID] | None = None,
        dynamic_routing: bool = False,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            out_schema=out_schema,
            rcv_args_schema=rcv_args_schema,
            message_pool=message_pool,
            recipient_ids=recipient_ids,
            dynamic_routing=dynamic_routing,
        )

        # Agent state
        self._state: LLMAgentState = LLMAgentState()
        self.set_state_strategy: SetAgentStateStrategy = set_state_strategy
        self._make_custom_agent_state_impl: MakeCustomAgentState | None = None

        # Tool orchestrator
        self._tool_orchestrator: ToolOrchestrator[CtxT] = ToolOrchestrator[CtxT](
            agent_id=self.agent_id,
            llm=llm,
            tools=tools,
            max_turns=max_turns,
            react_mode=react_mode,
        )

        # Prompt builder
        sys_prompt = get_prompt(prompt_text=sys_prompt, prompt_path=sys_prompt_path)
        inp_prompt = get_prompt(prompt_text=inp_prompt, prompt_path=inp_prompt_path)
        self._prompt_builder: PromptBuilder[InT, CtxT] = PromptBuilder[InT, CtxT](
            agent_id=self._agent_id,
            sys_prompt=sys_prompt,
            inp_prompt=inp_prompt,
            sys_args_schema=sys_args_schema,
            usr_args_schema=usr_args_schema,
            rcv_args_schema=rcv_args_schema,
        )

        self.no_tqdm = getattr(llm, "no_tqdm", False)

    @property
    def llm(self) -> LLM[LLMSettings, Converters]:
        return self._tool_orchestrator.llm

    @property
    def tools(self) -> dict[str, BaseTool[BaseModel, BaseModel, CtxT]]:
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
    def inp_prompt(self) -> LLMPrompt | None:
        return self._prompt_builder.inp_prompt

    def format_sys_args_handler(
        self, func: FormatSystemArgsHandler[CtxT]
    ) -> FormatSystemArgsHandler[CtxT]:
        self._prompt_builder.format_sys_args_impl = func

        return func

    def format_inp_args_handler(
        self, func: FormatInputArgsHandler[InT, CtxT]
    ) -> FormatInputArgsHandler[InT, CtxT]:
        self._prompt_builder.format_inp_args_impl = func

        return func

    def make_custom_agent_state_handler(
        self, func: MakeCustomAgentState
    ) -> MakeCustomAgentState:
        self._make_custom_agent_state_impl = func

        return func

    def tool_call_loop_exit_handler(
        self, func: ToolCallLoopExitHandler[CtxT]
    ) -> ToolCallLoopExitHandler[CtxT]:
        self._tool_orchestrator.tool_call_loop_exit_impl = func

        return func

    def _parse_output(
        self,
        conversation: Conversation,
        rcv_args: InT | None = None,
        ctx: RunContextWrapper[CtxT] | None = None,
        **kwargs: Any,
    ) -> OutT:
        if self._parse_output_impl:
            return self._parse_output_impl(
                conversation=conversation, rcv_args=rcv_args, ctx=ctx, **kwargs
            )
        try:
            return self._out_schema.model_validate_json(str(conversation[-1].content))
        except Exception:
            return self._out_schema()

    @final
    async def run(
        self,
        inp_items: LLMPrompt | list[str | ImageData] | None = None,
        *,
        ctx: RunContextWrapper[CtxT] | None = None,
        rcv_message: AgentMessage[InT, LLMAgentState] | None = None,
        entry_point: bool = False,
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

        if entry_point:
            assert rcv_message is None, (
                "Entry point run should not have a received message"
            )
        if inp_items:
            assert rcv_message is None, (
                "There must be no received message with user inputs"
            )

        cur_state = self.state.model_copy(deep=True)

        # 1. Make system prompt (can be None)
        formatted_sys_prompt = self._prompt_builder.make_sys_prompt(
            sys_args=sys_args, ctx=ctx
        )

        # 2. Set agent state

        rcv_state = rcv_message.sender_state if rcv_message else None
        prev_mh_len = len(cur_state.message_history)

        state = LLMAgentState.from_cur_and_rcv_states(
            cur_state=cur_state,
            rcv_state=rcv_state,
            sys_prompt=formatted_sys_prompt,
            strategy=self.set_state_strategy,
            make_custom_state_impl=self._make_custom_agent_state_impl,
            ctx=ctx,
        )

        self._print_sys_msg(state=state, prev_mh_len=prev_mh_len, ctx=ctx)

        # 3. Make and add user messages (can be empty)
        user_message_batch = self._prompt_builder.make_user_messages(
            inp_items=inp_items,
            usr_args=usr_args,
            rcv_args_batch=rcv_message.payloads if rcv_message else None,
            entry_point=entry_point,
            ctx=ctx,
        )
        if user_message_batch:
            state.message_history.add_message_batch(user_message_batch)
            self._print_msgs(user_message_batch, ctx=ctx)

        if not self.tools:
            # 4. Generate messages without tools
            await self._tool_orchestrator.generate_once(agent_state=state, ctx=ctx)
        else:
            # 4. Run tool call loop (new messages are added to the message
            #    history inside the loop)
            await self._tool_orchestrator.run_loop(agent_state=state, ctx=ctx)

        # 5. Parse outputs
        batch_size = state.message_history.batch_size
        rcv_args_batch = rcv_message.payloads if rcv_message else batch_size * [None]
        val_output_batch = [
            self.out_schema.model_validate(
                self._parse_output(conversation=conv, rcv_args=rcv_args, ctx=ctx)
            )
            for conv, rcv_args in zip(
                state.message_history.batched_conversations,
                rcv_args_batch,
                strict=False,
            )
        ]

        # 6. Write interaction history to context

        if self.dynamic_routing:
            recipient_ids = self._validate_dynamic_routing(val_output_batch)
        else:
            recipient_ids = self._validate_static_routing(val_output_batch)

        if ctx:
            interaction_record = InteractionRecord(
                source_id=self.agent_id,
                recipient_ids=recipient_ids,
                inp_items=inp_items,
                sys_prompt=self.sys_prompt,
                inp_prompt=self.inp_prompt,
                sys_args=sys_args,
                usr_args=usr_args,
                rcv_args=(rcv_message.payloads if rcv_message is not None else None),
                outputs=val_output_batch,
                state=state,
            )
            ctx.interaction_history.append(
                cast(
                    "InteractionRecord[AgentPayload, AgentPayload, AgentState]",
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

    # def _format_sys_args(
    #     self,
    #     sys_args: LLMPromptArgs,
    #     ctx: RunContextWrapper[CtxT] | None = None,
    # ) -> LLMFormattedSystemArgs:
    #     return self._prompt_builder.format_sys_args(sys_args=sys_args, ctx=ctx)

    # def _format_inp_args(
    #     self,
    #     usr_args: LLMPromptArgs,
    #     rcv_args: InT,
    #     ctx: RunContextWrapper[CtxT] | None = None,
    # ) -> LLMFormattedArgs:
    #     return self._prompt_builder.format_inp_args(
    #         usr_args=usr_args, rcv_args=rcv_args, ctx=ctx
    #     )

    # def _tool_call_loop_exit(
    #     self,
    #     conversation: Conversation,
    #     *,
    #     ctx: RunContextWrapper[CtxT] | None = None,
    #     **kwargs: Any,
    # ) -> bool:
    #     return self._tool_orchestrator.tool_call_loop_exit(
    #         conversation=conversation, ctx=ctx, **kwargs
    #     )
