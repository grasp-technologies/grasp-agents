from __future__ import annotations

import json
from typing import Annotated, Any, Literal, TypeAlias, TypedDict
from uuid import uuid4

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseInputMessageItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_tool_call_output_item import (
    OutputOutputContentList,
)
from openai.types.responses.response_input_message_content_list import (
    ResponseInputMessageContentList,
)
from openai.types.responses.response_output_text import Annotation
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from pydantic import Field, model_validator
from pydantic.json import pydantic_encoder

from .content import (
    InputContent,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    OutputContent,
    OutputRefusalContent,
    OutputTextContent,
    ReasoningSummaryContent,
    ReasoningTextContent,
)

ItemStatus = Literal["in_progress", "completed", "incomplete"]

InputMessageRole = Literal["user", "system", "developer"]
OutputMessageRole = Literal["assistant"]
MessageRole = Literal["user", "system", "developer", "assistant"]

ToolOutputContent: TypeAlias = Annotated[
    InputTextContent | InputImageContent | InputFileContent, Field(discriminator="type")
]


class InputMessageItem(ResponseInputMessageItem):
    """User/system/developer message sent as input. Auto-generates an id."""

    # OpenResponses fields:
    type: Literal["message"] | None = None
    id: str = Field(default_factory=lambda: str(uuid4()))
    status: ItemStatus | None = None
    role: InputMessageRole
    content: ResponseInputMessageContentList = Field(default_factory=list)  # type: ignore

    # grasp-agents fields:

    content_ext: list[InputContent] = Field(
        default_factory=list[InputContent], frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "content_ext" in data and "content" not in data:
            data["content"] = data["content_ext"]
        elif "content" in data and "content_ext" not in data:
            data["content_ext"] = data["content"]
        return data

    @property
    def texts(self) -> list[str]:
        return [
            part.text for part in self.content_ext if isinstance(part, InputTextContent)
        ]

    @property
    def images(self) -> list[InputImageContent]:
        return [
            part for part in self.content_ext if isinstance(part, InputImageContent)
        ]

    @property
    def files(self) -> list[InputFileContent]:
        return [part for part in self.content_ext if isinstance(part, InputFileContent)]

    @classmethod
    def from_text(cls, text: str, role: InputMessageRole = "user") -> InputMessageItem:
        return cls(content_ext=[InputTextContent(text=text)], role=role)


class OutputMessageItem(ResponseOutputMessage):
    """Assistant message produced by the model."""

    # OpenResponses fields:
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: OutputMessageRole = "assistant"
    status: ItemStatus
    content: list[ResponseOutputText | ResponseOutputRefusal] = Field(
        default_factory=list[ResponseOutputText | ResponseOutputRefusal]
    )

    # grasp-agents fields:

    content_ext: list[OutputContent] = Field(
        default_factory=list[OutputContent], frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "content_ext" in data and "content" not in data:
            data["content"] = data["content_ext"]
        elif "content" in data and "content_ext" not in data:
            data["content_ext"] = data["content"]
        return data

    @property
    def text(self) -> str:
        """Concatenated text from all OutputTextContent parts."""
        return "".join(
            part.text
            for part in self.content_ext
            if isinstance(part, OutputTextContent)
        )

    @property
    def refusal(self) -> str | None:
        """Refusal string if any RefusalContent part exists."""
        for part in self.content_ext:
            if isinstance(part, OutputRefusalContent):
                return part.refusal
        return None

    @property
    def annotations(self) -> list[Annotation]:
        """Aggregated annotations from all OutputTextContent parts."""
        annotations: list[Annotation] = []
        for part in self.content_ext:
            if isinstance(part, OutputTextContent):
                annotations.extend(part.annotations)
        return annotations


class FunctionToolCallItem(ResponseFunctionToolCall):
    """A tool call issued by the model (name + JSON arguments)."""

    # OpenResponses fields:
    type: Literal["function_call"] = "function_call"
    id: str | None = Field(default_factory=lambda: str(uuid4()))
    status: ItemStatus | None = None
    call_id: str
    arguments: str
    name: str


class FunctionToolOutputItem(ResponseFunctionToolCallOutputItem):
    """Result of a tool call, sent back as input for the next turn."""

    # OpenResponses fields:
    type: Literal["function_call_output"] = "function_call_output"
    id: str = Field(default_factory=lambda: str(uuid4()))
    call_id: str
    status: ItemStatus | None = None
    output: str | list[OutputOutputContentList] = Field(
        default_factory=list[OutputOutputContentList]
    )

    # grasp-agents fields:

    output_ext: str | list[ToolOutputContent] = Field(
        default_factory=list[ToolOutputContent], frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "output_ext" in data and "output" not in data:
            data["output"] = data["output_ext"]
        elif "output" in data and "output_ext" not in data:
            data["output_ext"] = data["output"]
        return data

    @classmethod
    def from_tool_result(
        cls,
        call_id: str,
        output: Any,
        *,
        indent: int = 2,
    ) -> FunctionToolOutputItem:
        serialized = json.dumps(output, default=pydantic_encoder, indent=indent)
        return cls(call_id=call_id, output_ext=serialized)


class CachedContent(TypedDict):
    """Used by Anthropic and LiteLLM."""

    type: Literal["ephemeral"]


class ReasoningItem(ResponseReasoningItem):
    """Model reasoning/thinking output."""

    # OpenResponses fields:
    type: Literal["reasoning"] = "reasoning"
    id: str = Field(default_factory=lambda: str(uuid4()))
    status: ItemStatus | None = None

    content: list[ResponseReasoningContent] | None = None
    encrypted_content: str | None = None

    summary: list[ResponseReasoningSummary] = Field(
        default_factory=list[ResponseReasoningSummary]
    )

    # grasp-agents fields:
    cache_control: dict[str, object] | CachedContent | None = None
    redacted: bool = False

    content_ext: list[ReasoningTextContent] | None = Field(default=None, frozen=True)
    summary_ext: list[ReasoningSummaryContent] = Field(
        default_factory=list[ReasoningSummaryContent], frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "content_ext" in data and "content" not in data:
            data["content"] = data["content_ext"]
        elif "content" in data and "content_ext" not in data:
            data["content_ext"] = data["content"]
        if "summary_ext" in data and "summary" not in data:
            data["summary"] = data["summary_ext"]
        elif "summary" in data and "summary_ext" not in data:
            data["summary_ext"] = data["summary"]
        return data

    @property
    def content_text(self) -> str | None:
        if self.content_ext is not None:
            return " ".join(c.text for c in self.content_ext)
        return None

    @property
    def summary_text(self) -> str:
        """Concatenated text from all summary blocks."""
        return " ".join(s.text for s in self.summary_ext)


AssistantMessage = OutputMessageItem
ToolMessage = FunctionToolCallItem


class SystemMessage(InputMessageItem):
    role: InputMessageRole = Field(default="system", frozen=True)


class UserMessage(InputMessageItem):
    role: InputMessageRole = Field(default="user", frozen=True)


class DeveloperMessage(InputMessageItem):
    role: InputMessageRole = Field(default="developer", frozen=True)


ToolCallItem = Annotated[FunctionToolCallItem, Field(discriminator="type")]
ToolOutputItem = Annotated[FunctionToolOutputItem, Field(discriminator="type")]

InputItem: TypeAlias = Annotated[
    InputMessageItem
    | OutputMessageItem
    | FunctionToolCallItem
    | FunctionToolOutputItem
    | ReasoningItem,
    Field(discriminator="type"),
]

OutputItem: TypeAlias = Annotated[
    OutputMessageItem | FunctionToolCallItem | ReasoningItem,
    Field(discriminator="type"),
]
