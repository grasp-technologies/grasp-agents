from abc import ABC
from typing import Literal

from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.completion_usage import CompletionUsage as ChatCompletionUsage
from pydantic import BaseModel

from .message import AssistantMessage


class CompletionChoice(BaseModel):
    message: AssistantMessage
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
        | None
    )
    index: int
    logprobs: ChatCompletionChoiceLogprobs | None = None


class CompletionError(BaseModel):
    message: str
    metadata: dict[str, str | None] | None = None
    code: int


class Completion(BaseModel, ABC):
    id: str
    model_id: str | None = None
    created: int
    choices: list[CompletionChoice]
    usage: ChatCompletionUsage | None = None
    error: CompletionError | None = None

    @property
    def messages(self) -> list[AssistantMessage]:
        return [choice.message for choice in self.choices if choice.message]


class CompletionChunk(BaseModel):
    # TODO: add choices and tool calls
    id: str
    created: int
    delta: str | None = None
    model_id: str | None = None
