from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from openai.types.responses import (
    Response as _SDKResponse,
)
from openai.types.responses import (
    ResponseError,
    ResponseInputItem,
    ResponseOutputItem,
    ResponseStatus,
    ResponseTextConfig,
    Tool,
)
from openai.types.responses.response import IncompleteDetails, ToolChoice
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)
from openai.types.responses.response_usage import ResponseUsage as _SDKResponseUsage
from openai.types.shared import Metadata, Reasoning, ResponsesModel
from pydantic import Field, model_validator

from .content import OutputTextContent
from .items import FunctionToolCallItem, OutputMessageItem, ReasoningItem


class ResponseUsage(_SDKResponseUsage):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int
    cost: float | None = None

    def __add__(self, other: ResponseUsage) -> ResponseUsage:
        return ResponseUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_tokens_details=self.input_tokens_details,
            output_tokens_details=self.output_tokens_details,
            cost=(self.cost or 0) + (other.cost or 0)
            if self.cost is not None or other.cost is not None
            else None,
        )


class Response(_SDKResponse):
    # OpenResponses fields:

    object: Literal["response"] = "response"
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    completed_at: float | None = None
    safety_identifier: str | None = None

    status: ResponseStatus | None = None
    # one of `completed`, `failed`, `in_progress`, `cancelled`, `queued`, `incomplete`

    incomplete_details: IncompleteDetails | None = None
    error: ResponseError | None = None

    output: list[ResponseOutputItem] = Field(default_factory=list[ResponseOutputItem])
    usage: _SDKResponseUsage | None = None

    model: ResponsesModel
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    reasoning: Reasoning | None = None
    # effort: ["none", "minimal", "low", "medium", "high", "xhigh"] (default=None)
    # summary: ["auto", "concise", "detailed"] (default=None)

    text: ResponseTextConfig | None = None
    # one of ResponseFormatText, ResponseFormatTextJSONSchemaConfig,
    # or ResponseFormatJSONObject (obsolete)

    instructions: str | list[ResponseInputItem] | None = (
        None  # str | None in OpenResponses
    )

    tools: list[Tool] = Field(default_factory=list[Tool])
    tool_choice: ToolChoice = "auto"
    parallel_tool_calls: bool = True
    max_tool_calls: int | None = None

    truncation: Literal["auto", "disabled"] | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    background: bool | None = None
    metadata: Metadata | None = None

    # grasp-agents fields:

    response_ms: float | None = None
    provider_specific_fields: dict[str, Any] | None = None
    hidden_params: dict[str, Any] | None = None
    response_headers: dict[str, Any] | None = None

    output_ext: list[OutputMessageItem | FunctionToolCallItem | ReasoningItem] = Field(
        default_factory=list[OutputMessageItem | FunctionToolCallItem | ReasoningItem],
        frozen=True,
    )
    usage_ext: ResponseUsage | None = Field(default=None, frozen=True)

    # OpenAI-specific fields:

    # conversation: Conversation | None = None
    # prompt: ResponsePrompt | None = None
    # prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    # service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "output_ext" in data and "output" not in data:
            data["output"] = data["output_ext"]
        elif "output" in data and "output_ext" not in data:
            data["output_ext"] = data["output"]
        if "usage_ext" in data and "usage" not in data:
            data["usage"] = data["usage_ext"]
        elif "usage" in data and "usage_ext" not in data:
            data["usage_ext"] = data["usage"]
        return data

    @property
    def output_text(self) -> str:
        """Concatenated output text from all message items."""
        return "".join(
            part.text
            for item in self.output_ext
            if isinstance(item, OutputMessageItem)
            for part in item.content_ext
            if isinstance(part, OutputTextContent)
        )

    @property
    def tool_call_items(self) -> list[FunctionToolCallItem]:
        return [i for i in self.output_ext if isinstance(i, FunctionToolCallItem)]

    @property
    def message_items(self) -> list[OutputMessageItem]:
        return [i for i in self.output_ext if isinstance(i, OutputMessageItem)]

    @property
    def reasoning_items(self) -> list[ReasoningItem]:
        return [i for i in self.output_ext if isinstance(i, ReasoningItem)]

    @property
    def refusal(self) -> str | None:
        """First refusal text found in output messages, if any."""
        for msg in self.message_items:
            r = msg.refusal
            if r is not None:
                return r
        return None
