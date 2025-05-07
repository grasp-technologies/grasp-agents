from abc import ABC

from pydantic import BaseModel

from .message import AssistantMessage


class CompletionChoice(BaseModel):
    # TODO: add fields
    message: AssistantMessage
    finish_reason: str | None


class CompletionError(BaseModel):
    message: str
    metadata: dict[str, str | None] | None = None
    code: int


class Completion(BaseModel, ABC):
    # TODO: add fields
    choices: list[CompletionChoice]
    model_id: str | None = None
    error: CompletionError | None = None


class CompletionChunk(BaseModel):
    # TODO: add more fields and tool use support (and choices?)
    delta: str | None = None
    model_id: str | None = None
