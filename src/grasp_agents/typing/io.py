from typing import TypeAlias, TypeVar

from pydantic import BaseModel

ProcName: TypeAlias = str


class LLMPromptArgs(BaseModel):
    pass


InT_contra = TypeVar("InT_contra", contravariant=True)
OutT_co = TypeVar("OutT_co", covariant=True)
MemT_co = TypeVar("MemT_co", covariant=True)

LLMPrompt: TypeAlias = str
