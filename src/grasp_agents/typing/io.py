from typing import TypeAlias, TypeVar

from pydantic import BaseModel

ProcName: TypeAlias = str


InT = TypeVar("InT")
OutT_co = TypeVar("OutT_co", covariant=True)


class LLMPromptArgs(BaseModel):
    pass


LLMPrompt: TypeAlias = str
