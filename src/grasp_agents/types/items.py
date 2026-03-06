from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias
from uuid import uuid4

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseFunctionWebSearch,
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputMessageItem,
    ResponseInputText,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_function_web_search import (
    Action as WebSearchAction,
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
    Citation,
    InputContentPart,
    InputFile,
    InputImage,
    InputTextContentPart,
    OutputMessageContentPart,
    OutputRefusal,
    OutputTextContentPart,
    ReasoningContentPart,
    ReasoningSummaryPart,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .reasoning import OpenRouterReasoningDetails

ItemStatus = Literal["in_progress", "completed", "incomplete"]

InputMessageRole = Literal["user", "system", "developer"]
OutputMessageRole = Literal["assistant"]
MessageRole = Literal["user", "system", "developer", "assistant"]

ToolOutputPart = Annotated[
    InputTextContentPart | InputImage | InputFile, Field(discriminator="type")
]

ResponseInputPart = ResponseInputText | ResponseInputImage | ResponseInputFile
ResponseOutputPart = ResponseOutputText | ResponseOutputRefusal
ResponseToolOutputPart = ResponseInputText | ResponseInputImage | ResponseInputFile


"""
Responses API reserves a special meaning for output "content" fields: they come from
the raw LLM token stream. Everything that is not "content" is derived artifacts that can
be generated after the fact (e.g. tool calls, reasoning summaries, annotations, etc).
"""


def prefixed_id(prefix: str) -> str:
    """Generate an API-compatible prefixed ID (e.g. msg_, fc_, rs_)."""
    return f"{prefix}_{uuid4().hex[:24]}"


class InputMessageItem(ResponseInputMessageItem):
    """User/system/developer message sent as input. Auto-generates an id."""

    # OpenResponses fields:

    type: Literal["message"] = "message"  # type: ignore[assignment]
    id: str = Field(default_factory=lambda: prefixed_id("msg"))
    status: ItemStatus | None = None
    role: InputMessageRole
    content: list[ResponseInputPart] = Field(default_factory=list[ResponseInputPart])

    # grasp-agents fields:

    content_parts: list[InputContentPart] = Field(
        default_factory=list[InputContentPart], frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "content_parts" in data and "content" not in data:
            data["content"] = data["content_parts"]
        elif "content" in data and "content_parts" not in data:
            data["content_parts"] = data["content"]
        return data

    @property
    def texts(self) -> list[str]:
        return [
            part.text
            for part in self.content_parts
            if isinstance(part, InputTextContentPart)
        ]

    @property
    def text(self) -> str:
        """Concatenated text from all InputTextContentPart parts."""
        return "".join(self.texts)

    @property
    def images(self) -> list[InputImage]:
        return [part for part in self.content_parts if isinstance(part, InputImage)]

    @property
    def files(self) -> list[InputFile]:
        return [part for part in self.content_parts if isinstance(part, InputFile)]

    @classmethod
    def from_text(cls, text: str, role: InputMessageRole = "user") -> InputMessageItem:
        return cls(content_parts=[InputTextContentPart(text=text)], role=role)


class OutputMessageItem(ResponseOutputMessage):
    """Assistant message produced by the model."""

    # OpenResponses fields:
    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: prefixed_id("msg"))
    role: OutputMessageRole = "assistant"
    status: ItemStatus
    content: list[ResponseOutputPart] = Field(default_factory=list[ResponseOutputPart])

    # grasp-agents fields:

    content_parts: list[OutputMessageContentPart] = Field(
        default_factory=list[OutputMessageContentPart], frozen=True
    )

    # Provider-specific opaque data for round-trip fidelity
    # (e.g. Gemini thought_signature on regular text parts)
    provider_specific_fields: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "content_parts" in data and "content" not in data:
            data["content"] = data["content_parts"]
        elif "content" in data and "content_parts" not in data:
            data["content_parts"] = data["content"]
        return data

    @property
    def text(self) -> str:
        """Concatenated text from all OutputTextContentPart parts."""
        return "".join(
            part.text
            for part in self.content_parts
            if isinstance(part, OutputTextContentPart)
        )

    @property
    def refusal(self) -> str | None:
        """Refusal string if any OutputRefusal part exists."""
        for part in self.content_parts:
            if isinstance(part, OutputRefusal):
                return part.refusal
        return None

    @property
    def annotations(self) -> list[Annotation]:
        """Aggregated annotations from all OutputTextContentPart parts."""
        annotations: list[Annotation] = []
        for part in self.content_parts:
            if isinstance(part, OutputTextContentPart):
                annotations.extend(part.annotations)
        return annotations

    @property
    def citations(self) -> list[Citation]:
        """Aggregated citations from all OutputTextContentPart parts."""
        citations: list[Citation] = []
        for part in self.content_parts:
            if isinstance(part, OutputTextContentPart):
                citations.extend(part.citations)
        return citations


class FunctionToolCallItem(ResponseFunctionToolCall):
    """A tool call issued by the model (name + JSON arguments)."""

    # OpenResponses fields:
    type: Literal["function_call"] = "function_call"
    id: str | None = Field(default_factory=lambda: prefixed_id("fc"))
    status: ItemStatus | None = None
    call_id: str
    arguments: str
    name: str

    # Provider-specific opaque data for round-trip fidelity
    # (e.g. Gemini thought_signature on function_call parts)
    provider_specific_fields: dict[str, Any] | None = None


class FunctionToolOutputItem(ResponseFunctionToolCallOutputItem):
    """Result of a tool call, sent back as input for the next turn."""

    # OpenResponses fields:
    type: Literal["function_call_output"] = "function_call_output"
    id: str = Field(default_factory=lambda: prefixed_id("fco"))
    call_id: str
    status: ItemStatus | None = None
    output: str | list[ResponseToolOutputPart] = Field(
        default_factory=list[ResponseToolOutputPart]
    )

    # grasp-agents fields:

    output_parts: str | list[ToolOutputPart] = Field(
        default_factory=list[ToolOutputPart], frozen=True
    )

    @property
    def text(self) -> str:
        if isinstance(self.output_parts, str):
            return self.output_parts
        return "".join(
            part.text
            for part in self.output_parts
            if isinstance(part, InputTextContentPart)
        )

    @property
    def images(self) -> list[InputImage]:
        if isinstance(self.output_parts, str):
            return []
        return [part for part in self.output_parts if isinstance(part, InputImage)]

    @property
    def files(self) -> list[InputFile]:
        if isinstance(self.output_parts, str):
            return []
        return [part for part in self.output_parts if isinstance(part, InputFile)]

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "output_parts" in data and "output" not in data:
            data["output"] = data["output_parts"]
        elif "output" in data and "output_parts" not in data:
            data["output_parts"] = data["output"]
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
        return cls(call_id=call_id, output_parts=serialized)


class ReasoningItem(ResponseReasoningItem):
    """Model reasoning/thinking output."""

    # OpenResponses fields:

    type: Literal["reasoning"] = "reasoning"
    id: str = Field(default_factory=lambda: prefixed_id("rs"))
    status: ItemStatus | None = None
    encrypted_content: str | None = None

    content: list[ResponseReasoningContent] | None = None
    summary: list[ResponseReasoningSummary] = Field(
        default_factory=list[ResponseReasoningSummary]
    )

    # grasp-agents fields:

    cache_control: dict[str, Any] | None = None
    redacted: bool = False

    content_parts: list[ReasoningContentPart] | None = Field(default=None, frozen=True)
    summary_parts: list[ReasoningSummaryPart] = Field(
        default_factory=list[ReasoningSummaryPart], frozen=True
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "content_parts" in data and "content" not in data:
            data["content"] = data["content_parts"]
        elif "content" in data and "content_parts" not in data:
            data["content_parts"] = data["content"]
        if "summary_parts" in data and "summary" not in data:
            data["summary"] = data["summary_parts"]
        elif "summary" in data and "summary_parts" not in data:
            data["summary_parts"] = data["summary"]
        return data

    @property
    def content_text(self) -> str | None:
        if self.content_parts is not None:
            return "".join(c.text for c in self.content_parts)
        return None

    @property
    def summary_text(self) -> str:
        """Concatenated text from all summary blocks."""
        return "".join(s.text for s in self.summary_parts)

    @classmethod
    def from_reasoning_content(
        cls, reasoning_content: str, encrypted_content: str | None = None
    ) -> ReasoningItem:
        return cls(
            summary_parts=[ReasoningSummaryPart(text=reasoning_content)],
            encrypted_content=encrypted_content,
            redacted=False,
            status="completed",
        )

    @classmethod
    def from_thinking_block(cls, block: Mapping[str, Any]) -> ReasoningItem:
        block_type = block.get("type")
        match block_type:
            case "thinking":
                return cls(
                    content_parts=[
                        ReasoningContentPart(text=block.get("thinking", ""))
                    ],
                    encrypted_content=block.get("signature"),
                    cache_control=block.get("cache_control"),
                    redacted=False,
                    status="completed",
                )
            case "redacted_thinking":
                return cls(
                    content_parts=[ReasoningContentPart(text=block.get("data", ""))],
                    encrypted_content=block.get("data"),
                    cache_control=block.get("cache_control"),
                    redacted=True,
                    status="completed",
                )
            case _:
                msg = f"Unknown thinking block type: {block_type}"
                raise ValueError(msg)

    @classmethod
    def from_open_router_reasoning_details(
        cls, detail: OpenRouterReasoningDetails
    ) -> ReasoningItem:
        match detail.type:
            case "reasoning.summary":
                return cls(
                    summary_parts=[ReasoningSummaryPart(text=detail.summary)],
                    redacted=False,
                    status="completed",
                )
            case "reasoning.text":
                # NOTE: always assume summarized reasoning
                return cls(
                    summary_parts=[ReasoningSummaryPart(text=detail.text or "")],
                    encrypted_content=detail.signature,
                    redacted=False,
                    status="completed",
                )
            case "reasoning.encrypted":
                return cls(
                    content_parts=[],
                    encrypted_content=detail.data,
                    redacted=True,
                    status="completed",
                )


class WebSearchCallItem(ResponseFunctionWebSearch):
    """A server-side web search call record."""

    type: Literal["web_search_call"] = "web_search_call"
    id: str = Field(default_factory=lambda: prefixed_id("ws"))
    action: WebSearchAction
    status: Literal["in_progress", "searching", "completed", "failed"] = "completed"

    # Provider-specific opaque data for round-trip fidelity
    # (e.g. Anthropic per-URL encrypted content for web search results)
    provider_specific_fields: dict[str, Any] | None = None


AssistantMessage = OutputMessageItem
ToolMessage = FunctionToolCallItem


class SystemMessage(InputMessageItem):
    role: InputMessageRole = Field(default="system", frozen=True)


class UserMessage(InputMessageItem):
    role: InputMessageRole = Field(default="user", frozen=True)


class DeveloperMessage(InputMessageItem):
    role: InputMessageRole = Field(default="developer", frozen=True)


InputItem: TypeAlias = (
    InputMessageItem
    | OutputMessageItem
    | FunctionToolCallItem
    | FunctionToolOutputItem
    | ReasoningItem
    | WebSearchCallItem
)

OutputItem: TypeAlias = Annotated[
    OutputMessageItem | FunctionToolCallItem | ReasoningItem | WebSearchCallItem,
    Field(discriminator="type"),
]

ToolCallItem = Annotated[FunctionToolCallItem, Field(discriminator="type")]
ToolOutputItem = Annotated[FunctionToolOutputItem, Field(discriminator="type")]
