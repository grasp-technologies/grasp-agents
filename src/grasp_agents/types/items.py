from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import uuid4

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseInputMessageItem,
    ResponseOutputMessage,
    ResponseReasoningItem,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import to_jsonable_python

from .content import (
    Annotation,
    AnnotationUrlCitation,
    CacheControl,
    InputFile,
    InputImage,
    InputPart,
    InputText,
    OutputMessagePart,
    OutputMessageRefusal,
    OutputMessageText,
    ReasoningContent,
    ReasoningSummary,
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
    content: list[InputPart] = Field(default_factory=list[InputPart])  # pyright: ignore[reportIncompatibleVariableOverride] — extended part types

    @property
    def texts(self) -> list[str]:
        return [part.text for part in self.content if isinstance(part, InputText)]

    @property
    def text(self) -> str:
        """Concatenated text from all InputText parts."""
        return "".join(self.texts)

    @property
    def images(self) -> list[InputImage]:
        return [part for part in self.content if isinstance(part, InputImage)]

    @property
    def files(self) -> list[InputFile]:
        return [part for part in self.content if isinstance(part, InputFile)]

    @classmethod
    def from_text(cls, text: str, role: InputMessageRole = "user") -> InputMessageItem:
        return cls(content=[InputText(text=text)], role=role)


class OutputMessageItem(ResponseOutputMessage):
    """Assistant message produced by the model."""

    # OpenResponses fields (Message):

    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: prefixed_id("msg"))
    role: OutputMessageRole = "assistant"
    status: ItemStatus
    phase: Literal["commentary", "final_answer"] | None = None
    content: list[OutputMessagePart] = Field(default_factory=list[OutputMessagePart])  # pyright: ignore[reportIncompatibleVariableOverride] — extended part types

    # grasp-agents fields:

    # Provider-specific opaque data for round-trip fidelity
    # (e.g. Gemini thought_signature on regular text parts)
    provider_specific_fields: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        """Concatenated text from all OutputText parts."""
        return "".join(
            part.text for part in self.content if isinstance(part, OutputMessageText)
        )

    @property
    def refusal(self) -> str | None:
        """Refusal string if any OutputMessageRefusal part exists."""
        for part in self.content:
            if isinstance(part, OutputMessageRefusal):
                return part.refusal
        return None

    @property
    def annotations(self) -> list[Annotation]:
        """Aggregated annotations from all OutputMessageText parts."""
        annotations: list[Annotation] = []
        for part in self.content:
            if isinstance(part, OutputMessageText):
                annotations.extend(part.annotations)
        return annotations

    @property
    def citations(self) -> list[AnnotationUrlCitation]:
        """Aggregated URL citations from all OutputMessageText parts."""
        citations: list[AnnotationUrlCitation] = []
        for part in self.content:
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
    cache_control: CacheControl | None = None


class FunctionToolOutputItem(ResponseFunctionToolCallOutputItem):
    """Result of a tool call, sent back as input for the next turn."""

    # OpenResponses fields (FunctionCallOutput):

    type: Literal["function_call_output"] = "function_call_output"
    id: str = Field(default_factory=lambda: prefixed_id("fco"))
    call_id: str
    status: ItemStatus | None = None  # type: ignore[assignment]
    output: str | list[ToolOutputPart] = Field(  # pyright: ignore[reportIncompatibleVariableOverride] — extended part types
        default_factory=list[ToolOutputPart]
    )

    # grasp-agents fields:

    # Provider-specific opaque data for round-trip fidelity
    provider_specific_fields: dict[str, Any] | None = None
    cache_control: CacheControl | None = None
    # True when this result carries a tool failure (a ``ToolErrorInfo``), so a UI
    # can flag it (e.g. a red border). Display-only; not sent to the provider.
    is_error: bool = False

    @property
    def text(self) -> str:
        if isinstance(self.output, str):
            return self.output
        return "".join(part.text for part in self.output if isinstance(part, InputText))

    @property
    def images(self) -> list[InputImage]:
        if isinstance(self.output, str):
            return []
        return [part for part in self.output if isinstance(part, InputImage)]

    @property
    def files(self) -> list[InputFile]:
        if isinstance(self.output, str):
            return []
        return [part for part in self.output if isinstance(part, InputFile)]

    @classmethod
    def from_tool_result(
        cls,
        call_id: str,
        output: Any,
        *,
        indent: int = 2,
        is_error: bool = False,
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
                return cls(call_id=call_id, output=typed, is_error=is_error)

        if isinstance(output, str):
            serialized = output
        else:
            serialized = json.dumps(to_jsonable_python(output), indent=indent)

        return cls(call_id=call_id, output=serialized, is_error=is_error)


class ReasoningItem(ResponseReasoningItem):
    """Model reasoning/thinking output."""

    # OpenResponses fields (ReasoningBody):

    type: Literal["reasoning"] = "reasoning"
    id: str = Field(default_factory=lambda: prefixed_id("rs"))
    status: ItemStatus | None = None
    encrypted_content: str | None = None

    content: list[ReasoningContent] | None = None  # pyright: ignore[reportIncompatibleVariableOverride] — extended part types
    summary: list[ReasoningSummary] = Field(  # pyright: ignore[reportIncompatibleVariableOverride] — extended part types
        default_factory=list[ReasoningSummary]
    )

    # grasp-agents fields:

    redacted: bool = False

    @property
    def content_text(self) -> str | None:
        if self.content is not None:
            return "".join(c.text for c in self.content)
        return None

    @property
    def summary_text(self) -> str:
        return "".join(s.text for s in self.summary)

    @classmethod
    def from_reasoning_content(
        cls, reasoning_content: str, encrypted_content: str | None = None
    ) -> ReasoningItem:
        return cls(
            summary=[ReasoningSummary(text=reasoning_content)],
            encrypted_content=encrypted_content,
            redacted=False,
            status="completed",
        )

    @classmethod
    def from_thinking_block(cls, block: Mapping[str, Any]) -> ReasoningItem:
        block_type = block.get("type")
        match block_type:
            # TODO: Do we assign to content or summary?
            case "thinking":
                return cls(
                    content=None,
                    summary=[ReasoningSummary(text=block.get("thinking", ""))],
                    encrypted_content=block.get("signature"),
                    redacted=False,
                    status="completed",
                )
            case "redacted_thinking":
                return cls(
                    content=None,
                    summary=[],
                    encrypted_content=block.get("data"),
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
                    summary=[ReasoningSummary(text=detail.summary)],
                    redacted=False,
                    status="completed",
                )

            case "reasoning.text":
                # NOTE: always assume summarized reasoning
                return cls(
                    summary=[ReasoningSummary(text=detail.text or "")],
                    encrypted_content=detail.signature,
                    redacted=False,
                    status="completed",
                )

            case "reasoning.encrypted":
                return cls(
                    content=None,
                    encrypted_content=detail.data,
                    redacted=True,
                    status="completed",
                )


class SearchSource(BaseModel):
    """A source found during a web search."""

    url: str
    title: str = ""
    page_age: str | None = None


class SearchAction(BaseModel):
    """A web search action with queries and resulting sources."""

    type: Literal["search"] = "search"
    queries: list[str] | None = None
    sources: list[SearchSource] | None = None


class OpenPageAction(BaseModel):
    """An action that opens/fetches a specific URL."""

    type: Literal["open_page"] = "open_page"
    url: str | None = None


class FindInPageAction(BaseModel):
    """A 'find in page' action: locate a pattern within an opened page."""

    type: Literal["find_in_page"] = "find_in_page"
    url: str | None = None
    pattern: str | None = None


type WebSearchAction = SearchAction | OpenPageAction | FindInPageAction


class WebSearchCallItem(BaseModel):
    """A server-side web search or web fetch call record."""

    type: Literal["web_search_call"] = "web_search_call"
    id: str = Field(default_factory=lambda: prefixed_id("ws"))
    action: WebSearchAction
    status: Literal["in_progress", "searching", "completed", "failed"] = "completed"

    # Provider-specific opaque data for round-trip fidelity
    # (e.g. Anthropic per-URL encrypted content for web search results)
    provider_specific_fields: dict[str, Any] | None = None
    cache_control: CacheControl | None = None

    @property
    def summary(self) -> str:
        """One-line human description of the search / fetch action."""
        action = self.action
        if isinstance(action, SearchAction):
            queries = ", ".join(action.queries or [])
            return f"search: {queries}" if queries else "search"
        if isinstance(action, OpenPageAction):
            return f"open page: {action.url or ''}"
        line = f"find in page: {action.pattern or ''}"
        return f"{line} @ {action.url}" if action.url else line


# Item types with a dedicated model; UnknownItem refuses these so a malformed
# known item fails validation loudly instead of degrading to a passthrough.
_KNOWN_ITEM_TYPES = frozenset(
    {
        "message",
        "function_call",
        "function_call_output",
        "reasoning",
        "web_search_call",
    }
)


class UnknownItem(BaseModel):
    """
    A response item of a type this framework doesn't model, preserved
    verbatim.

    Round-trips new provider item types — e.g. programmatic-tool-calling
    ``program`` items with their replay ``fingerprint`` — through the
    transcript without dropping fields. Only the OpenAI Responses provider
    sends these back; other providers skip them.
    """

    model_config = ConfigDict(extra="allow")

    type: str

    @model_validator(mode="after")
    def _reject_known_types(self) -> UnknownItem:
        if self.type in _KNOWN_ITEM_TYPES:
            raise ValueError(
                f"item type {self.type!r} must validate as its dedicated model"
            )
        return self


AssistantMessage = OutputMessageItem


class SystemMessage(InputMessageItem):
    role: InputMessageRole = Field(default="system", frozen=True)


class UserMessage(InputMessageItem):
    role: InputMessageRole = Field(default="user", frozen=True)


class DeveloperMessage(InputMessageItem):
    role: InputMessageRole = Field(default="developer", frozen=True)


type InputItem = (
    InputMessageItem
    | OutputMessageItem
    | FunctionToolCallItem
    | FunctionToolOutputItem
    | ReasoningItem
    | WebSearchCallItem
    | UnknownItem
)

# The discriminated union handles the known tags; an unrecognized ``type``
# falls through to the UnknownItem passthrough (which refuses known tags, so
# malformed known items still fail loudly).
type OutputItem = (
    Annotated[
        OutputMessageItem | FunctionToolCallItem | ReasoningItem | WebSearchCallItem,
        Field(discriminator="type"),
    ]
    | UnknownItem
)

ToolCallItem = Annotated[FunctionToolCallItem, Field(discriminator="type")]

ToolOutputItem = Annotated[FunctionToolOutputItem, Field(discriminator="type")]

ToolMessage = ToolOutputItem
