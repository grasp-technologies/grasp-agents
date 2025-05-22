from copy import deepcopy
from typing import Any, Literal, Optional, Protocol

from pydantic import ConfigDict, Field

from .memory import MessageHistory
from .run_context import RunContextWrapper
from .typing.io import AgentState, LLMPrompt

SetAgentStateStrategy = Literal["keep", "reset", "from_sender", "custom"]


class SetAgentState(Protocol):
    def __call__(
        self,
        cur_state: "LLMAgentState",
        *,
        in_state: AgentState | None,
        sys_prompt: LLMPrompt | None,
        ctx: RunContextWrapper[Any] | None,
    ) -> "LLMAgentState": ...


class LLMAgentState(AgentState):
    message_history: MessageHistory = Field(default_factory=MessageHistory)

    @property
    def batch_size(self) -> int:
        return self.message_history.batch_size

    @classmethod
    def from_cur_and_in_states(
        cls,
        cur_state: "LLMAgentState",
        *,
        in_state: Optional["AgentState"] = None,
        sys_prompt: LLMPrompt | None = None,
        strategy: SetAgentStateStrategy = "from_sender",
        set_agent_state_impl: SetAgentState | None = None,
        ctx: RunContextWrapper[Any] | None = None,
    ) -> "LLMAgentState":
        upd_mh = cur_state.message_history if cur_state else None
        if upd_mh is None or len(upd_mh) == 0:
            upd_mh = MessageHistory(sys_prompt=sys_prompt)

        if strategy == "keep":
            pass

        elif strategy == "reset":
            upd_mh.reset(sys_prompt)

        elif strategy == "from_sender":
            in_mh = (
                in_state.message_history
                if in_state and isinstance(in_state, "LLMAgentState")
                else None
            )
            if in_mh:
                in_mh = deepcopy(in_mh)
            else:
                upd_mh.reset(sys_prompt)

        elif strategy == "custom":
            assert set_agent_state_impl is not None, (
                "Agent state setter implementation is not provided."
            )
            return set_agent_state_impl(
                cur_state=cur_state,
                in_state=in_state,
                sys_prompt=sys_prompt,
                ctx=ctx,
            )

        return cls.model_construct(message_history=upd_mh)

    def __repr__(self) -> str:
        return f"Message History: {len(self.message_history)}"

    model_config = ConfigDict(arbitrary_types_allowed=True)
