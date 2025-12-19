from collections.abc import Sequence
from uuid import uuid4

from litellm.types.llms.openai import ChatCompletionAnnotation
from pydantic import BaseModel, Field

from ..typing.tool import ToolCall


class Reasoning(BaseModel):
    summaries: list[str]
    encrypted_content: str | None = None


class TextOutput(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    content: str
    annotations: Sequence[ChatCompletionAnnotation] | None = None
    refusal: str | None = None


CompletionItem = ToolCall | Reasoning | TextOutput
