from typing import TypeAlias, TypeVar

from pydantic import BaseModel

ProcName: TypeAlias = str


InT = TypeVar("InT")
OutT = TypeVar("OutT")


class LLMPromptArgs(BaseModel):
    pass


LLMPrompt: TypeAlias = str
