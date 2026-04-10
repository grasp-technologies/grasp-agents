import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from openai._types import omit  # noqa: PLC2701
from openai.lib.streaming.responses._responses import (
    AsyncResponseStreamManager,
)
from openai.types.responses import (
    ParsedResponse,
    Response,
    ResponseStreamEvent,
    ResponseTextConfigParam,
)
from openai.types.responses.response_create_params import (
    StreamOptions as ResponsesStreamOptionsParam,
)
from openai.types.responses.response_create_params import (
    ToolChoice as ResponseToolChoice,
)
from openai.types.responses.tool_param import (
    ToolParam as ResponsesToolParam,
)
from openai.types.responses.web_search_tool_param import WebSearchToolParam
from openai.types.shared import Reasoning
from pydantic import BaseModel, TypeAdapter

from grasp_agents.llm.cloud_llm import (
    ApiCallParams,
    CloudLLM,
    CloudLLMSettings,
)
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmError
from grasp_agents.types.llm_events import LlmEvent
from grasp_agents.types.response import Response as InternalResponse
from grasp_agents.types.tool import BaseTool, ToolChoice

from .error_mapping import map_api_error
from .response_to_provider_inputs import items_to_provider_inputs
from .tool_converters import to_api_tool, to_api_tool_choice

logger = logging.getLogger(__name__)

_STREAM_EVENT_ADAPTER: TypeAdapter[LlmEvent] = TypeAdapter(LlmEvent)


class OpenAIResponsesLLMSettings(CloudLLMSettings, total=False):
    reasoning: Reasoning
    parallel_tool_calls: bool
    max_output_tokens: int
    top_logprobs: int | None
    web_search: WebSearchToolParam | None

    text: ResponseTextConfigParam
    stream_options: ResponsesStreamOptionsParam | None
    store: bool | None
    user: str


@dataclass(frozen=True)
class OpenAIResponsesLLM(CloudLLM):
    litellm_provider: str | None = "openai"
    llm_settings: OpenAIResponsesLLMSettings | None = None
    openai_client_timeout: float = 120.0
    openai_client_max_retries: int = 2
    extra_openai_client_params: dict[str, Any] | None = None
    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        _model_name = self.model_name
        _openai_client_params = deepcopy(self.extra_openai_client_params or {})
        _openai_client_params["timeout"] = self.openai_client_timeout
        _openai_client_params["max_retries"] = self.openai_client_max_retries
        if self.http_client is not None:
            _openai_client_params["http_client"] = self.http_client

        _client = AsyncOpenAI(
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            **_openai_client_params,
        )

        object.__setattr__(self, "model_name", _model_name)
        object.__setattr__(self, "client", _client)

    # --- Input preparation ---

    def _make_api_input(
        self,
        input: Sequence[InputItem],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        response_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
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

        api_kwargs: ApiCallParams = ApiCallParams(
            api_input=items_to_provider_inputs(input),
            api_tools=api_tools,
            api_tool_choice=api_tool_choice,
        )
        if response_schema is not None:
            api_kwargs["api_response_schema"] = response_schema
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
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> ParsedResponse[Any] | Response:
        tools = api_tools or omit
        tool_choice = api_tool_choice or omit
        text_format = api_response_schema or omit

        response_id = api_llm_settings.get("previous_response_id")

        if self.apply_response_schema_via_provider:
            return await self.client.responses.parse(  # type: ignore[reportUnknownVariableType]
                text_format=text_format,
                model=self.model_name,
                input=[api_input[-1]] if response_id else api_input,
                tools=tools,
                tool_choice=tool_choice,
                **api_llm_settings,
            )
        return await self.client.responses.create(
            model=self.model_name,
            input=[api_input[-1]] if response_id else api_input,
            stream=False,
            tools=tools,
            tool_choice=tool_choice,
            **api_llm_settings,
        )

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[ResponseStreamEvent]:
        response_id = api_llm_settings.get("previous_response_id")

        _api_llm_settings = dict(api_llm_settings)
        if "stream_options" in _api_llm_settings:
            so = dict(_api_llm_settings.get("stream_options") or {})
            so.pop("include_usage", None)
            _api_llm_settings["stream_options"] = so

        async def iterator() -> AsyncIterator[ResponseStreamEvent]:
            stream_manager: AsyncResponseStreamManager[Any] = (
                self.client.responses.stream(
                    model=self.model_name,
                    input=[api_input[-1]] if response_id else api_input,
                    tool_choice=api_tool_choice or omit,
                    tools=api_tools or omit,
                    text_format=api_response_schema or omit,
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
