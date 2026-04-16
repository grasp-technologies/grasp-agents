import time
from typing import Any, Literal, TypeAlias, overload
from uuid import uuid4

from litellm.types.utils import ChoiceLogprobs as LiteLLMChoiceLogprobs
from openai.types.chat.chat_completion import ChoiceLogprobs as OpenAIChoiceLogprobs
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt

from .message import AssistantMessage

FinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


class Usage(BaseModel):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt | None = None
    cached_reading_tokens: NonNegativeInt | None = None
    cached_writing_tokens: NonNegativeInt | None = None
    cost: NonNegativeFloat | None = None

    @overload
    @staticmethod
    def _add_opt(
        a: NonNegativeInt | None, b: NonNegativeInt | None
    ) -> NonNegativeInt | None: ...

    @overload
    @staticmethod
    def _add_opt(
        a: NonNegativeFloat | None, b: NonNegativeFloat | None
    ) -> NonNegativeFloat | None: ...

    @staticmethod
    def _add_opt(a: float | None, b: float | None) -> int | float | None:
        if a is not None or b is not None:
            return (a or 0) + (b or 0)
        return None

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self._add_opt(
                self.reasoning_tokens, other.reasoning_tokens
            ),
            cached_reading_tokens=self._add_opt(
                self.cached_reading_tokens, other.cached_reading_tokens
            ),
            cached_writing_tokens=self._add_opt(
                self.cached_writing_tokens, other.cached_writing_tokens
            ),
            cost=self._add_opt(self.cost, other.cost),
        )


class CompletionError(BaseModel):
    message: str
    metadata: dict[str, str | None] | None = None
    code: int


class Completion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str | None
    name: str | None = None
    system_fingerprint: str | None = None
    error: CompletionError | None = None
    usage: Usage | None = None

    # Removed choices to add message directly to Completion
    message: AssistantMessage
    finish_reason: FinishReason | None
    logprobs: OpenAIChoiceLogprobs | LiteLLMChoiceLogprobs | Any | None = None
    # LiteLLM-specific fields
    provider_specific_fields: dict[str, Any] | None = None

    # LiteLLM-specific fields
    response_ms: float | None = None
    hidden_params: dict[str, Any] | None = None
