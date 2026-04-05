from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

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
from openai.types.shared import Metadata, Reasoning
from pydantic import Field, model_validator

from .content import OutputMessageText
from .items import (
    FunctionToolCallItem,
    OutputItem,
    OutputMessageItem,
    ReasoningItem,
    WebSearchCallItem,
    prefixed_id,
)


class ResponseUsage(_SDKResponseUsage):
    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(
        default_factory=lambda: InputTokensDetails(cached_tokens=0)
    )
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=lambda: OutputTokensDetails(reasoning_tokens=0)
    )
    total_tokens: int = 0
    cost: float | None = None

    def __add__(self, other: ResponseUsage) -> ResponseUsage:
        return ResponseUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            input_tokens_details=InputTokensDetails(
                cached_tokens=self.input_tokens_details.cached_tokens
                + other.input_tokens_details.cached_tokens
            ),
            output_tokens_details=OutputTokensDetails(
                reasoning_tokens=self.output_tokens_details.reasoning_tokens
                + other.output_tokens_details.reasoning_tokens
            ),
            cost=(self.cost or 0) + (other.cost or 0)
            if self.cost is not None or other.cost is not None
            else None,
        )


class Response(_SDKResponse):
    # OpenResponses fields:

    object: Literal["response"] = "response"
    id: str = Field(default_factory=lambda: prefixed_id("resp"))
    created_at: float = Field(default_factory=lambda: datetime.now(UTC).timestamp())
    completed_at: float | None = None
    safety_identifier: str | None = None

    status: ResponseStatus | None = None

    incomplete_details: IncompleteDetails | None = None
    error: ResponseError | None = None

    output: list[ResponseOutputItem] = Field(default_factory=list[ResponseOutputItem])
    usage: _SDKResponseUsage | None = None

    # --- Request params ---

    model: str
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
    prompt_cache_key: str | None = None
    background: bool | None = None
    metadata: Metadata | None = None

    previous_response_id: str | None = None

    # ----

    # grasp-agents fields:

    response_ms: float | None = None

    provider_specific_fields: dict[str, Any] | None = None
    hidden_params: dict[str, Any] | None = None
    response_headers: dict[str, Any] | None = None

    output_items: list[OutputItem] = Field(
        default_factory=list[OutputItem], frozen=True
    )

    usage_with_cost: ResponseUsage | None = Field(default=None, frozen=True)

    # OpenAI-specific fields:

    # conversation: Conversation | None = None
    # prompt: ResponsePrompt | None = None
    # prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    # service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "output_items" in data and "output" not in data:
            # Filter out WebSearchCallItem for SDK-compat `output` field
            data["output"] = [
                i for i in data["output_items"] if not isinstance(i, WebSearchCallItem)
            ]
        elif "output" in data and "output_items" not in data:
            data["output_items"] = data["output"]

        if "usage_with_cost" in data and "usage" not in data:
            data["usage"] = data["usage_with_cost"]
        elif "usage" in data and "usage_with_cost" not in data:
            data["usage_with_cost"] = data["usage"]
        return data

    @property
    def output_text(self) -> str:
        return "".join(
            part.text
            for item in self.output_items
            if isinstance(item, OutputMessageItem)
            for part in item.content_parts
            if isinstance(part, OutputMessageText)
        )

    @property
    def reasoning_items(self) -> list[ReasoningItem]:
        return [i for i in self.output_items if isinstance(i, ReasoningItem)]

    @property
    def message_items(self) -> list[OutputMessageItem]:
        return [i for i in self.output_items if isinstance(i, OutputMessageItem)]

    @property
    def tool_call_items(self) -> list[FunctionToolCallItem]:
        return [i for i in self.output_items if isinstance(i, FunctionToolCallItem)]

    @property
    def refusal(self) -> str | None:
        for msg in self.message_items:
            r = msg.refusal
            if r is not None:
                return r
        return None
