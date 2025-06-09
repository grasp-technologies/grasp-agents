from collections.abc import Mapping
from typing import TypeAlias, TypeVar

from pydantic import BaseModel

from .content import ImageData

ProcessorName: TypeAlias = str


class LLMPromptArgs(BaseModel):
    pass


InT_contra = TypeVar("InT_contra", contravariant=True)
OutT_co = TypeVar("OutT_co", covariant=True)
MemT_co = TypeVar("MemT_co", covariant=True)

LLMPrompt: TypeAlias = str
LLMFormattedSystemArgs: TypeAlias = Mapping[str, str | int | bool]
LLMFormattedArgs: TypeAlias = Mapping[str, str | int | bool | ImageData]
