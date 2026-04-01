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
    InputFile,
    InputImage,
    InputPart,
    InputText,
    OutputMessagePart,
    OutputMessageRefusal,
    OutputMessageText,
    ReasoningSummary,
    ReasoningText,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .reasoning import OpenRouterReasoningDetails

# --- Enum types ---

ItemStatus = Literal["in_progress", "completed", "incomplete"]

InputMessageRole = Literal["user", "system", "developer"]

OutputMessageRole = Literal["assistant"]

MessageRole = Literal["user", "system", "developer", "assistant"]


# --- Union types ---

ToolOutputPart = Annotated[
    InputText | InputImage | InputFile, Field(discriminator="type")
]

ResponseInputPart = ResponseInputText | ResponseInputImage | ResponseInputFile

ResponseOutputPart = ResponseOutputText | ResponseOutputRefusal

ResponseToolOutputPart = ResponseInputText | ResponseInputImage | ResponseInputFile


# --- Item types ---


def prefixed_id(prefix: str) -> str:
    """Generate an API-compatible prefixed ID (e.g. msg_, fc_, rs_)."""
    return f"{prefix}_{uuid4().hex[:24]}"


class InputMessageItem(ResponseInputMessageItem):
    """User/system/developer message sent as input."""

    # OpenResponses fields (Message):

    type: Literal["message"] = "message"  # type: ignore[assignment]
    id: str = Field(default_factory=lambda: prefixed_id("msg"))
    status: ItemStatus | None = None
    role: InputMessageRole
    content: list[ResponseInputPart] = Field(default_factory=list[ResponseInputPart])

    # grasp-agents fields:

    content_parts: list[InputPart] = Field(default_factory=list[InputPart], frozen=True)

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
        return [part.text for part in self.content_parts if isinstance(part, InputText)]

    @property
    def text(self) -> str:
        """Concatenated text from all InputText parts."""
        return "".join(self.texts)

    @property
    def images(self) -> list[InputImage]:
        return [part for part in self.content_parts if isinstance(part, InputImage)]

    @property
    def files(self) -> list[InputFile]:
        return [part for part in self.content_parts if isinstance(part, InputFile)]

    @classmethod
    def from_text(cls, text: str, role: InputMessageRole = "user") -> InputMessageItem:
        return cls(content_parts=[InputText(text=text)], role=role)


class OutputMessageItem(ResponseOutputMessage):
    """Assistant message produced by the model."""

    # OpenResponses fields (Message):

    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: prefixed_id("msg"))
    role: OutputMessageRole = "assistant"
    status: ItemStatus
    content: list[ResponseOutputPart] = Field(default_factory=list[ResponseOutputPart])

    # grasp-agents fields:

    content_parts: list[OutputMessagePart] = Field(
        default_factory=list[OutputMessagePart], frozen=True
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
        """Concatenated text from all OutputText parts."""
        return "".join(
            part.text
            for part in self.content_parts
            if isinstance(part, OutputMessageText)
        )

    @property
    def refusal(self) -> str | None:
        """Refusal string if any OutputMessageRefusal part exists."""
        for part in self.content_parts:
            if isinstance(part, OutputMessageRefusal):
                return part.refusal
        return None

    @property
    def annotations(self) -> list[Annotation]:
        """Aggregated annotations from all OutputMessageText parts."""
        annotations: list[Annotation] = []
        for part in self.content_parts:
            if isinstance(part, OutputMessageText):
                annotations.extend(part.annotations)
        return annotations

    @property
    def citations(self) -> list[Citation]:
        """Aggregated citations from all OutputMessageText parts."""
        citations: list[Citation] = []
        for part in self.content_parts:
            if isinstance(part, OutputMessageText):
                citations.extend(part.citations)
        return citations


class FunctionToolCallItem(ResponseFunctionToolCall):
    """A tool call issued by the model (name + JSON arguments)."""

    # OpenResponses fields (FunctionCall):

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

    # OpenResponses fields (FunctionCallOutput):

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
            part.text for part in self.output_parts if isinstance(part, InputText)
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
        if isinstance(output, list):
            typed: list[InputText | InputImage | InputFile] = []
            for p in output:  # type: ignore[union-attr]
                if isinstance(p, InputText | InputImage | InputFile):
                    typed.append(p)
                else:
                    typed = []
                    break
            if typed:
                return cls(call_id=call_id, output_parts=typed)

        serialized = json.dumps(output, default=pydantic_encoder, indent=indent)

        return cls(call_id=call_id, output_parts=serialized)


class ReasoningItem(ResponseReasoningItem):
    """Model reasoning/thinking output."""

    # OpenResponses fields (ReasoningBody):

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

    content_parts: list[ReasoningText] | None = Field(default=None, frozen=True)
    summary_parts: list[ReasoningSummary] = Field(
        default_factory=list[ReasoningSummary], frozen=True
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
        return "".join(s.text for s in self.summary_parts)

    @classmethod
    def from_reasoning_content(
        cls, reasoning_content: str, encrypted_content: str | None = None
    ) -> ReasoningItem:
        return cls(
            summary_parts=[ReasoningSummary(text=reasoning_content)],
            encrypted_content=encrypted_content,
            redacted=False,
            status="completed",
        )

    @classmethod
    def from_thinking_block(cls, block: Mapping[str, Any]) -> ReasoningItem:
        block_type = block.get("type")
        match block_type:
            # TODO: Do we assign to content_parts or summary_parts?
            case "thinking":
                return cls(
                    content_parts=None,
                    summary_parts=[ReasoningSummary(text=block.get("thinking", ""))],
                    encrypted_content=block.get("signature"),
                    cache_control=block.get("cache_control"),
                    redacted=False,
                    status="completed",
                )
            case "redacted_thinking":
                return cls(
                    content_parts=None,
                    summary_parts=[],
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
                    summary_parts=[ReasoningSummary(text=detail.summary)],
                    redacted=False,
                    status="completed",
                )

            case "reasoning.text":
                # NOTE: always assume summarized reasoning
                return cls(
                    summary_parts=[ReasoningSummary(text=detail.text or "")],
                    encrypted_content=detail.signature,
                    redacted=False,
                    status="completed",
                )

            case "reasoning.encrypted":
                return cls(
                    content_parts=None,
                    encrypted_content=detail.data,
                    redacted=True,
                    status="completed",
                )


class WebSearchCallItem(ResponseFunctionWebSearch):
    """A server-side web search call record."""

    # NOTE: ResponseFunctionWebSearch is Responses API-specific.
    # Not sure if we need to inherit from it

    type: Literal["web_search_call"] = "web_search_call"
    id: str = Field(default_factory=lambda: prefixed_id("ws"))
    action: WebSearchAction
    status: Literal["in_progress", "searching", "completed", "failed"] = "completed"

    # Provider-specific opaque data for round-trip fidelity
    # (e.g. Anthropic per-URL encrypted content for web search results)
    provider_specific_fields: dict[str, Any] | None = None


AssistantMessage = OutputMessageItem


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

ToolMessage = ToolOutputItem
