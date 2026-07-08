import logging
import os
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._types import omit  # noqa: PLC2701
from openai.lib.streaming.responses._responses import (
    AsyncResponseStreamManager,
)
from openai.types.responses import (
    ParsedResponse,
    Response,
    ResponseIncludable,
    ResponseStreamEvent,
    ResponseTextConfigParam,
)
from openai.types.responses.response_create_params import (
    ContextManagement as ResponsesContextManagement,
)
from openai.types.responses.response_create_params import (
    Conversation as ResponsesConversation,
)
from openai.types.responses.response_create_params import (
    StreamOptions as ResponsesStreamOptionsParam,
)
from openai.types.responses.response_create_params import (
    ToolChoice as ResponseToolChoice,
)
from openai.types.responses.response_input_item_param import (
    ResponseInputItemParam,
)
from openai.types.responses.tool_param import (
    ToolParam as ResponsesToolParam,
)
from openai.types.responses.web_search_tool_param import WebSearchToolParam
from openai.types.shared import Reasoning
from openai.types.shared_params import Metadata
from pydantic import BaseModel, ConfigDict, TypeAdapter, with_config

from grasp_agents.llm.cloud_llm import (
    ApiCallParams,
    APIProvider,
    CloudLLM,
    CloudLLMSettings,
)
from grasp_agents.llm_providers.openai_completions.completions_llm import (
    AzureClientConfig,
)
from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmError
from grasp_agents.types.llm_events import LlmEvent
from grasp_agents.types.response import Response as InternalResponse

from .error_mapping import map_api_error
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool, to_api_tool_choice

logger = logging.getLogger(__name__)

_STREAM_EVENT_ADAPTER: TypeAdapter[LlmEvent] = TypeAdapter(LlmEvent)

# Caller-appended tool-output item types (user / system messages are matched
# by role below). ``function_call_output`` is the only one the framework emits
# (``FunctionToolOutputItem.type``). If other caller-output round-trips are
# ever routed through this provider (computer-use, custom tools, MCP approval),
# add their types here too — otherwise the backward walk stops too early and
# drops a trailing output, which the API rejects.
_NEW_INPUT_ITEM_TYPES = frozenset({"function_call_output"})


def _items_after_last_response(
    api_input: list[ResponseInputItemParam],
) -> list[ResponseInputItemParam]:
    """
    The trailing input items that postdate the model's last output.

    With ``previous_response_id``, the API already holds the prior turns
    server-side, so only the new items may be sent — but *all* of them:
    slicing to a single item drops sibling tool outputs of a parallel
    tool-call batch and the API rejects the request.
    """
    start = len(api_input)
    for i in range(len(api_input) - 1, -1, -1):
        # The union's TypedDicts share no guaranteed "type"/"role" key —
        # probe as a plain mapping.
        item = cast("Mapping[str, Any]", api_input[i])
        item_type = item.get("type", "message")
        is_new_input = item_type in _NEW_INPUT_ITEM_TYPES or (
            item_type == "message"
            and item.get("role") in {"user", "system", "developer"}
        )
        if not is_new_input:
            break
        start = i
    return api_input[start:]


VerbosityLevel = Literal["low", "medium", "high"]
PromptCacheRetention = Literal["in_memory", "24h"]


@with_config(ConfigDict(extra="allow"))
class OpenAIResponsesLLMSettings(CloudLLMSettings, total=False):
    reasoning: Reasoning
    verbosity: VerbosityLevel | None
    parallel_tool_calls: bool
    max_output_tokens: int
    max_tool_calls: int | None
    top_logprobs: int | None

    prompt_cache_key: str
    prompt_cache_retention: PromptCacheRetention | None

    context_management: Iterable[ResponsesContextManagement] | None
    truncation: Literal["auto", "disabled"] | None
    include: list[ResponseIncludable] | None

    web_search: WebSearchToolParam | None

    text: ResponseTextConfigParam
    stream_options: ResponsesStreamOptionsParam | None
    safety_identifier: str
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None
    metadata: Metadata | None
    store: bool | None
    user: str


class ResponsesApiCallParams(ApiCallParams, total=False):
    # Per-call server-side-state pointers, threaded next to api_input/api_tools
    # (not via the settings bag) — mutually exclusive at the API level.
    previous_response_id: str | None
    conversation: ResponsesConversation | None


@dataclass(frozen=True)
class OpenAIResponsesLLM(CloudLLM):
    _settings_type: ClassVar[Any] = OpenAIResponsesLLMSettings

    litellm_provider: str | None = "openai"
    llm_settings: OpenAIResponsesLLMSettings | None = None
    # "openai" routes to api.openai.com (or an OpenAI-compatible endpoint via
    # ``api_provider``); "azure" builds an ``AsyncAzureOpenAI`` client. The
    # Azure Responses API requires ``api_version`` >= "2025-03-01-preview"
    # (or the version-less v1 surface).
    platform: Literal["openai", "azure"] = "openai"
    # Azure client args (see AzureClientConfig). ``model_name`` is the Azure
    # *deployment* name. May carry secrets — kept out of repr.
    platform_config: AzureClientConfig | None = field(default=None, repr=False)
    openai_client_timeout: float = 120.0
    # SDK-level retries default to 0: ``LLM.retry_policy`` is the retry
    # system, and a non-zero value here would multiply with it.
    openai_client_max_retries: int = 0
    extra_openai_client_params: dict[str, Any] | None = None
    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        if self.platform == "azure":
            _client, _api_provider = self._build_azure_client()
            object.__setattr__(self, "api_provider", _api_provider)
            object.__setattr__(self, "client", _client)
            # ``model_name`` is the Azure deployment name — left untouched.
            if self.litellm_provider == "openai":
                object.__setattr__(self, "litellm_provider", "azure")
            return

        _api_provider = self.api_provider or APIProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        _client = AsyncOpenAI(
            base_url=_api_provider.get("base_url"),
            api_key=_api_provider.get("api_key"),
            **self._client_params(),
        )

        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "client", _client)

    def _client_params(self) -> dict[str, Any]:
        # Client args common to the OpenAI and Azure clients; provider-specific
        # client args go through ``extra_openai_client_params``.
        params = deepcopy(self.extra_openai_client_params or {})
        params["timeout"] = self.openai_client_timeout
        params["max_retries"] = self.openai_client_max_retries
        if self.http_client is not None:
            params["http_client"] = self.http_client
        if self.default_headers is not None:
            params.setdefault("default_headers", self.default_headers)
        return params

    def _build_azure_client(self) -> tuple[AsyncAzureOpenAI, APIProvider]:
        config: dict[str, Any] = dict(self.platform_config or {})
        # Anything not supplied here is read from the SDK's Azure env vars.
        azure_kwargs: dict[str, Any] = {**self._client_params(), **config}
        client = AsyncAzureOpenAI(**azure_kwargs)
        api_provider = self.api_provider or APIProvider(
            name="azure", base_url=config.get("azure_endpoint"), api_key=None
        )
        return client, api_provider

    # --- Input preparation ---

    def _make_api_input(
        self,
        input: Sequence[InputItem],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        output_schema: Any | None = None,
        previous_response_id: str | None = None,
        conversation: ResponsesConversation | None = None,
        **extra_llm_settings: Any,
    ) -> ResponsesApiCallParams:
        api_tools: list[ResponsesToolParam] | None = None
        if tools:
            strict = self.apply_tool_call_schema_via_provider
            api_tools = [to_api_tool(tool, strict=strict) for tool in tools.values()]

        api_tool_choice: ResponseToolChoice | None = None
        if tool_choice is not None:
            api_tool_choice = to_api_tool_choice(tool_choice)

        merged: dict[str, Any] = dict(self.llm_settings or {})
        merged.update(extra_llm_settings)

        web_search_tool_param = merged.pop("web_search", None)
        if web_search_tool_param is not None:
            api_tools = api_tools or []
            api_tools.append(web_search_tool_param)

        # Server-side-state pointers are per-call request args, not settings:
        # thread them as first-class params (next to input/tools), so they
        # reach the API call directly rather than via the settings bag.
        api_kwargs: ResponsesApiCallParams = ResponsesApiCallParams(
            api_input=items_to_provider_inputs(input),
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
            previous_response_id=previous_response_id,
            conversation=conversation,
        )
        if output_schema is not None:
            api_kwargs["api_output_schema"] = output_schema
        if merged:
            api_kwargs["extra_settings"] = merged

        return api_kwargs

    # --- Error mapping ---

    def _map_api_error(self, err: Exception) -> LlmError | None:
        return map_api_error(err)

    # --- Provider API layer ---

    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        previous_response_id: str | None = None,
        conversation: ResponsesConversation | None = None,
        **api_llm_settings: Any,
    ) -> ParsedResponse[Any] | Response:
        tools = api_tools or omit
        tool_choice = api_tool_choice or omit
        text_format = api_output_schema or omit

        # With server-held prior turns (a previous response or a conversation),
        # only the items postdating the model's last output may be sent.
        input_items = (
            _items_after_last_response(api_input)
            if (previous_response_id or conversation)
            else api_input
        )

        if self.apply_output_schema_via_provider:
            return await self.client.responses.parse(  # type: ignore[reportUnknownVariableType]
                text_format=text_format,
                model=self.model_name,
                input=input_items,
                tools=tools,
                tool_choice=tool_choice,
                previous_response_id=previous_response_id or omit,
                conversation=conversation or omit,
                **api_llm_settings,
            )
        return await self.client.responses.create(
            model=self.model_name,
            input=input_items,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            previous_response_id=previous_response_id or omit,
            conversation=conversation or omit,
            **api_llm_settings,
        )

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        previous_response_id: str | None = None,
        conversation: ResponsesConversation | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        # With server-held prior turns (a previous response or a conversation),
        # only the items postdating the model's last output may be sent.
        input_items = (
            _items_after_last_response(api_input)
            if (previous_response_id or conversation)
            else api_input
        )

        _api_llm_settings = dict(api_llm_settings)
        if "stream_options" in _api_llm_settings:
            so = dict(_api_llm_settings.get("stream_options") or {})
            so.pop("include_usage", None)
            _api_llm_settings["stream_options"] = so

        async def iterator() -> AsyncIterator[ResponseStreamEvent]:
            stream_manager: AsyncResponseStreamManager[Any] = (
                self.client.responses.stream(
                    model=self.model_name,
                    input=input_items,
                    tool_choice=api_tool_choice or omit,
                    tools=api_tools or omit,
                    text_format=api_output_schema or omit,
                    previous_response_id=previous_response_id or omit,
                    conversation=conversation or omit,
                    **_api_llm_settings,
                )
            )

            async with stream_manager as stream:
                async for response_event in stream:
                    yield response_event

        return iterator()

    # --- Conversion layer ---

    def _convert_api_response(self, raw: Response) -> InternalResponse:
        return InternalResponse.model_validate(
            raw.model_dump(warnings="none", by_alias=True)
        )

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        async for sdk_event in api_stream:
            data = sdk_event.model_dump(warnings="none", by_alias=True)
            try:
                yield _STREAM_EVENT_ADAPTER.validate_python(data)
            except Exception:
                logger.debug(
                    "Skipping unrecognized stream event: %s",
                    data.get("type"),
                )
