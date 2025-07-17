import json
from collections.abc import Hashable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Any, Literal, Required, TypeAlias
from uuid import uuid4

from litellm.types.llms.openai import ChatCompletionAnnotation as LiteLLMAnnotation
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from typing_extensions import TypedDict

from .content import Content, ImageData
from .tool import ToolCall


class Role(StrEnum):
    USER = "user"
    SYSTEM = "system"
    DEVELOPER = "developer"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageBase(BaseModel):
    id: Hashable = Field(default_factory=lambda: str(uuid4())[:8])
    name: str | None = None


class ChatCompletionCachedContent(TypedDict):
    type: Literal["ephemeral"]


class ThinkingBlock(TypedDict, total=False):
    type: Required[Literal["thinking"]]
    thinking: str
    signature: str | None
    cache_control: dict[str, Any] | ChatCompletionCachedContent | None


class RedactedThinkingBlock(TypedDict, total=False):
    type: Required[Literal["redacted_thinking"]]
    data: str
    cache_control: dict[str, Any] | ChatCompletionCachedContent | None


class AssistantMessage(MessageBase):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    content: str | None
    tool_calls: Sequence[ToolCall] | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    thinking_blocks: Sequence[ThinkingBlock | RedactedThinkingBlock] | None = None
    annotations: Sequence[LiteLLMAnnotation] | None = None
    provider_specific_fields: dict[str, Any] | None = None


class UserMessage(MessageBase):
    role: Literal[Role.USER] = Role.USER
    content: Content | str

    @classmethod
    def from_text(cls, text: str, name: str | None = None) -> "UserMessage":
        return cls(content=Content.from_text(text), name=name)

    @classmethod
    def from_formatted_prompt(
        cls,
        prompt_template: str,
        name: str | None = None,
        prompt_args: Mapping[str, str | int | bool | ImageData] | None = None,
    ) -> "UserMessage":
        content = Content.from_formatted_prompt(prompt_template, **(prompt_args or {}))

        return cls(content=content, name=name)

    @classmethod
    def from_content_parts(
        cls,
        content_parts: Sequence[str | ImageData],
        name: str | None = None,
    ) -> "UserMessage":
        content = Content.from_content_parts(content_parts)

        return cls(content=content, name=name)


class SystemMessage(MessageBase):
    role: Literal[Role.SYSTEM] = Role.SYSTEM
    content: str


class ToolMessage(MessageBase):
    role: Literal[Role.TOOL] = Role.TOOL
    content: str
    tool_call_id: str

    @classmethod
    def from_tool_output(
        cls, tool_output: Any, tool_call: ToolCall, indent: int = 2
    ) -> "ToolMessage":
        return cls(
            content=json.dumps(tool_output, default=pydantic_encoder, indent=indent),
            tool_call_id=tool_call.id,
            name=tool_call.tool_name,
        )


Message = Annotated[
    AssistantMessage | UserMessage | SystemMessage | ToolMessage,
    Field(discriminator="role"),
]

Messages: TypeAlias = list[Message]
