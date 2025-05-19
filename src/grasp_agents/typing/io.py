from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from .content import ImageData

AgentID: TypeAlias = str


class AgentState(BaseModel):
    pass


class LLMPromptArgs(BaseModel):
    pass


InT = TypeVar("InT", contravariant=True)  # noqa: PLC0105
OutT = TypeVar("OutT", covariant=True)  # noqa: PLC0105
StateT = TypeVar("StateT", bound=AgentState, covariant=True)  # noqa: PLC0105

LLMPrompt: TypeAlias = str
LLMFormattedSystemArgs: TypeAlias = Mapping[str, str | int | bool]
LLMFormattedArgs: TypeAlias = Mapping[str, str | int | bool | ImageData]
