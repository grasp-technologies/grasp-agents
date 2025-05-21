import json
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt
from pydantic.json import pydantic_encoder

from .content import Content, ImageData
from .tool import ToolCall


class Role(StrEnum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Usage(BaseModel):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt | None = None
    cached_tokens: NonNegativeInt | None = None
    cost: NonNegativeFloat | None = None

    def __add__(self, add_usage: "Usage") -> "Usage":
        input_tokens = self.input_tokens + add_usage.input_tokens
        output_tokens = self.output_tokens + add_usage.output_tokens
        if self.reasoning_tokens is not None or add_usage.reasoning_tokens is not None:
            reasoning_tokens = (self.reasoning_tokens or 0) + (
                add_usage.reasoning_tokens or 0
            )
        else:
            reasoning_tokens = None

        if self.cached_tokens is not None or add_usage.cached_tokens is not None:
            cached_tokens = (self.cached_tokens or 0) + (add_usage.cached_tokens or 0)
        else:
            cached_tokens = None

        cost = (
            (self.cost or 0.0) + add_usage.cost
            if (add_usage.cost is not None)
            else None
        )
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )


class MessageBase(BaseModel):
    message_id: Hashable = Field(default_factory=lambda: str(uuid4())[:8])
    model_id: str | None = None


class AssistantMessage(MessageBase):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: str | None
    usage: Usage | None = None
    tool_calls: Sequence[ToolCall] | None = None
    refusal: str | None = None


class UserMessage(MessageBase):
    role: Literal[Role.USER] = Role.USER
    content: Content

    @classmethod
    def from_text(cls, text: str, model_id: str | None = None) -> "UserMessage":
        return cls(content=Content.from_text(text), model_id=model_id)

    @classmethod
    def from_formatted_prompt(
        cls,
        prompt_template: str,
        prompt_args: Mapping[str, str | int | bool | ImageData] | None = None,
        model_id: str | None = None,
    ) -> "UserMessage":
        content = Content.from_formatted_prompt(
            prompt_template=prompt_template, prompt_args=prompt_args
        )

        return cls(content=content, model_id=model_id)

    @classmethod
    def from_content_parts(
        cls,
        content_parts: Sequence[str | ImageData],
        model_id: str | None = None,
    ) -> "UserMessage":
        content = Content.from_content_parts(content_parts)

        return cls(content=content, model_id=model_id)


class SystemMessage(MessageBase):
    role: Literal[Role.SYSTEM] = Role.SYSTEM
    content: str


class ToolMessage(MessageBase):
    role: Literal[Role.TOOL] = Role.TOOL
    content: str
    tool_call_id: str

    @classmethod
    def from_tool_output(
        cls,
        tool_output: Any,
        tool_call: ToolCall,
        model_id: str | None = None,
        indent: int = 2,
    ) -> "ToolMessage":
        return cls(
            content=json.dumps(tool_output, default=pydantic_encoder, indent=indent),
            tool_call_id=tool_call.id,
            model_id=model_id,
        )


Message = Annotated[
    AssistantMessage | UserMessage | SystemMessage | ToolMessage,
    Field(discriminator="role"),
]

Conversation: TypeAlias = list[Message]
