from collections.abc import Sequence
from typing import TypeAlias, TypeVar

from pydantic import BaseModel
from pydantic.json_schema import SkipJsonSchema

from .content import ImageData

AgentID: TypeAlias = str


class AgentPayload(BaseModel):
    # TODO: do we need conversation?
    selected_recipient_ids: SkipJsonSchema[Sequence[AgentID] | None] = None


class AgentState(BaseModel):
    pass


InT = TypeVar("InT", bound=AgentPayload, contravariant=True)  # noqa: PLC0105
OutT = TypeVar("OutT", bound=AgentPayload, covariant=True)  # noqa: PLC0105
StateT = TypeVar("StateT", bound=AgentState, covariant=True)  # noqa: PLC0105


class LLMPromptArgs(BaseModel):
    pass


LLMPrompt: TypeAlias = str
LLMFormattedSystemArgs: TypeAlias = dict[str, str]
LLMFormattedArgs: TypeAlias = dict[str, str | ImageData]
