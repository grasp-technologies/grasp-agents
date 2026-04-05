import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Required

import httpx
from pydantic import BaseModel
from typing_extensions import TypedDict

from .llm import LLM, LLMSettings
from .rate_limiting.rate_limiter import RateLimiter, limit_rate
from .types.items import InputItem
from .types.llm_errors import LlmError, LlmErrorTuple
from .types.llm_events import LlmEvent
from .types.response import Response
from .types.tool import BaseTool, ToolChoice

logger = logging.getLogger(__name__)


class APIProvider(TypedDict, total=False):
    name: Required[str]
    base_url: Required[str | None]
    api_key: Required[str | None]


class CloudLLMSettings(LLMSettings, total=False):
    extra_headers: dict[str, Any] | None
    extra_body: object | None
    extra_query: dict[str, Any] | None


LLMRateLimiter = RateLimiter[Response | AsyncIterator[LlmEvent]]


class ApiCallParams(TypedDict, total=False):
    api_input: Required[list[Any]]
    api_tools: list[Any] | None
    api_tool_choice: Any
    api_response_schema: type
    extra_settings: dict[str, Any]


@dataclass(frozen=True)
class CloudLLM(LLM):
    llm_settings: CloudLLMSettings | None = None
    api_provider: APIProvider | None = None
    rate_limiter: LLMRateLimiter | None = None
    apply_response_schema_via_provider: bool = False
    apply_tool_call_schema_via_provider: bool = False
    http_client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        if self.rate_limiter is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to "
                f"{self.rate_limiter.rpm} RPM"
            )

        if self.apply_response_schema_via_provider:
            object.__setattr__(self, "apply_tool_call_schema_via_provider", True)

    # --- Provider API layer (abstract) ---

    @abstractmethod
    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any: ...

    @abstractmethod
    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_response_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[Any]: ...

    # --- Conversion layer (abstract) ---

    @abstractmethod
    def _convert_api_response(self, raw: Any) -> Response: ...

    @abstractmethod
    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        yield NotImplemented

    # --- Input preparation ---

    @abstractmethod
    def _make_api_input(
        self,
        input: Sequence[InputItem],  # noqa: A002
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        response_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams: ...

    # --- Error mapping ---

    def _map_api_error(self, err: Exception) -> LlmError | None:  # noqa: ARG002
        """
        Map a provider SDK exception to an LlmError subclass.

        Returns None for unrecognized exceptions (passed through as-is).
        Override in provider subclasses.
        """
        return None

    # --- LLM interface implementation ---

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        if "_get_api_response" in cls.__dict__:
            cls._get_api_response = limit_rate(cls._get_api_response)  # type: ignore[method-assign]

        if "_get_api_stream" in cls.__dict__:
            cls._get_api_stream = limit_rate(cls._get_api_stream)  # type: ignore[method-assign]

    async def _generate_response_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        api_kwargs = self._make_api_input(
            input,
            tools=tools,
            tool_choice=tool_choice,
            response_schema=response_schema,
            **extra_llm_settings,
        )
        extra_settings = api_kwargs.pop("extra_settings", {})
        if not self.apply_response_schema_via_provider:
            api_kwargs.pop("api_response_schema", None)

        try:
            raw = await self._get_api_response(**api_kwargs, **extra_settings)
        except LlmErrorTuple:
            raise
        except Exception as err:
            mapped = self._map_api_error(err)
            if mapped is not None:
                raise mapped from err
            raise

        return self._convert_api_response(raw)

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        response_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        api_kwargs = self._make_api_input(
            input,
            tools=tools,
            tool_choice=tool_choice,
            response_schema=response_schema,
            **extra_llm_settings,
        )

        extra_settings = api_kwargs.pop("extra_settings", {})
        if not self.apply_response_schema_via_provider:
            api_kwargs.pop("api_response_schema", None)

        try:
            api_stream = await self._get_api_stream(**api_kwargs, **extra_settings)
        except LlmErrorTuple:
            raise
        except Exception as err:
            mapped = self._map_api_error(err)
            if mapped is not None:
                raise mapped from err
            raise

        async for event in self._convert_api_stream(api_stream):
            yield event
