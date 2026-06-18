from typing import Annotated, Literal

from pydantic import BaseModel, Field


class OpenRouterReasoningSummary(BaseModel):
    type: Literal["reasoning.summary"] = "reasoning.summary"
    summary: str
    index: int | None = None
    id: str | None = None
    format: str | None = None


class OpenRouterReasoningEncrypted(BaseModel):
    type: Literal["reasoning.encrypted"] = "reasoning.encrypted"
    data: str
    index: int | None = None
    id: str | None = None
    format: str | None = None


class OpenRouterReasoningText(BaseModel):
    type: Literal["reasoning.text"] = "reasoning.text"
    text: str | None = None
    signature: str | None = None
    id: str | None = None
    index: int | None = None
    format: str | None = None


OpenRouterReasoningDetails = Annotated[
    OpenRouterReasoningSummary | OpenRouterReasoningEncrypted | OpenRouterReasoningText,
    Field(discriminator="type"),
]
