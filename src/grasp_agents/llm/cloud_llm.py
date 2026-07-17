import logging
import time
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, ClassVar, NoReturn, Required, TypedDict

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, with_config

from grasp_agents import grasp_logging
from grasp_agents.rate_limiting.rate_limiter import RateLimiter, limit_rate
from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import (
    LlmError,
    LlmErrorTuple,
    LlmInternalServerError,
)
from grasp_agents.types.llm_events import LlmEvent, ResponseCompleted, ResponseFailed
from grasp_agents.types.response import Response
from grasp_agents.usage_tracker import add_cost_to_usage

from .llm import LLM, LLMSettings

logger = logging.getLogger(__name__)


def _format_usage(response: Response) -> str:
    usage = response.usage
    if usage is None:
        return "usage n/a"
    return f"{usage.input_tokens or 0:,} in / {usage.output_tokens or 0:,} out tok"


class APIProvider(TypedDict, total=False):
    name: Required[str]
    base_url: Required[str | None]
    api_key: Required[str | None]


@with_config(ConfigDict(extra="allow"))
class CloudLLMSettings(LLMSettings, total=False):
    extra_headers: dict[str, Any] | None
    extra_body: object | None
    extra_query: dict[str, Any] | None


@cache
def _settings_adapter(settings_type: type) -> TypeAdapter[Any]:
    return TypeAdapter(settings_type)


LLMRateLimiter = RateLimiter[Response | AsyncIterator[LlmEvent]]


class ApiCallParams(TypedDict, total=False):
    api_input: Required[list[Any]]
    api_tools: list[Any] | None
    api_tool_choice: Any
    api_output_schema: type
    extra_settings: dict[str, Any]


@dataclass(frozen=True)
class CloudLLM(LLM):
    # Settings TypedDict that ``llm_settings`` is validated against at
    # construction: declared keys are type-checked; undeclared keys pass
    # through to the provider untouched. ``None`` disables validation.
    _settings_type: ClassVar[Any] = CloudLLMSettings

    llm_settings: CloudLLMSettings | None = None
    # repr=False: carries the resolved API key — must not leak via repr/str
    # (logs, tracebacks, printed configs).
    api_provider: APIProvider | None = field(default=None, repr=False)
    rate_limiter: LLMRateLimiter | None = None
    apply_output_schema_via_provider: bool = False
    apply_tool_call_schema_via_provider: bool = False
    http_client: httpx.AsyncClient | None = None
    default_headers: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.llm_settings is not None and self._settings_type is not None:
            _settings_adapter(self._settings_type).validate_python(self.llm_settings)

        if self.rate_limiter is not None:
            logger.info(
                f"[{self.__class__.__name__}] Set rate limit to "
                f"{self.rate_limiter.rpm} RPM"
            )

        if self.apply_output_schema_via_provider:
            object.__setattr__(self, "apply_tool_call_schema_via_provider", True)

    # --- Provider API layer (abstract) ---

    @abstractmethod
    async def _get_api_response(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
        **api_llm_settings: Any,
    ) -> Any: ...

    @abstractmethod
    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: type | None = None,
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
        output_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams: ...

    # --- Error mapping ---

    def _map_api_error(self, err: Exception) -> LlmError | None:
        """
        Map a provider SDK exception to an LlmError subclass.

        Returns None for unrecognized exceptions (passed through as-is).
        Override in provider subclasses.
        """
        del err
        return None

    def _raise_mapped(self, err: Exception) -> NoReturn:
        mapped = self._map_api_error(err)
        if mapped is not None:
            raise mapped from err
        raise err

    # --- Cost stamping ---

    def _stamp_cost(self, response: Response) -> None:
        """
        Stamp the response's cost with THIS model's pricing identity.

        Called at response-production time, so cost attribution is correct
        however the LLM is composed: under a FallbackLLM, the serving
        member prices its own response — never the ``model_name`` the
        composite reports. Cost is an API-provider concern, which is why
        stamping lives here and not on the base ``LLM`` (a local or mock
        LLM with token usage has no price, and looking one up would warn
        on every call).
        """
        usage = response.usage
        if usage is not None and usage.cost is None and self.model_name:
            add_cost_to_usage(
                usage,
                model_name=self.model_name,
                litellm_provider=self.litellm_provider,
            )

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
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> Response:
        api_kwargs = self._make_api_input(
            input,
            tools=tools,
            tool_choice=tool_choice,
            output_schema=output_schema,
            **extra_llm_settings,
        )
        extra_settings = api_kwargs.pop("extra_settings", {})
        if not self.apply_output_schema_via_provider:
            api_kwargs.pop("api_output_schema", None)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "llm %s: request (%d input items, %d tools): %s",
                self.model_name,
                len(input),
                len(tools) if tools else 0,
                grasp_logging.body_for_log(
                    repr(api_kwargs), full=grasp_logging.LOG_LLM_INPUT
                ),
            )

        t0 = time.monotonic()
        try:
            raw = await self._get_api_response(**api_kwargs, **extra_settings)
            # Conversion is inside the mapped region: a 200 response carrying
            # an error body surfaces here (e.g. ``CompletionError``) and must
            # reach retry/fallback as a typed LlmError, not a bare exception.
            response = self._convert_api_response(raw)
        except LlmErrorTuple:
            raise
        except Exception as err:
            self._raise_mapped(err)

        self._stamp_cost(response)
        logger.info(
            "llm %s → %s in %.2fs",
            self.model_name,
            _format_usage(response),
            time.monotonic() - t0,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "llm %s response: %s",
                self.model_name,
                grasp_logging.body_for_log(
                    response.output_text or "", full=grasp_logging.LOG_LLM_OUTPUT
                ),
            )
        return response

    async def _generate_response_stream_once(
        self,
        input: Sequence[InputItem],  # noqa: A002
        *,
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        output_schema: Any | None = None,
        tool_choice: ToolChoice | None = None,
        **extra_llm_settings: Any,
    ) -> AsyncIterator[LlmEvent]:
        api_kwargs = self._make_api_input(
            input,
            tools=tools,
            tool_choice=tool_choice,
            output_schema=output_schema,
            **extra_llm_settings,
        )

        extra_settings = api_kwargs.pop("extra_settings", {})
        if not self.apply_output_schema_via_provider:
            api_kwargs.pop("api_output_schema", None)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "llm %s: streaming request (%d input items, %d tools): %s",
                self.model_name,
                len(input),
                len(tools) if tools else 0,
                grasp_logging.body_for_log(
                    repr(api_kwargs), full=grasp_logging.LOG_LLM_INPUT
                ),
            )

        t0 = time.monotonic()
        try:
            api_stream = await self._get_api_stream(**api_kwargs, **extra_settings)
        except LlmErrorTuple:
            raise
        except Exception as err:
            self._raise_mapped(err)

        # Provider SDKs typically defer the HTTP request to the first iteration
        # of a lazily-returned stream, so SDK errors can surface here rather
        # than at acquisition — map them too, or the retry/fallback layers
        # (which catch only LlmErrorTuple) never see streaming failures.
        event_stream = self._convert_api_stream(api_stream)
        while True:
            try:
                event = await anext(event_stream)
            except StopAsyncIteration:
                break
            except LlmErrorTuple:
                raise
            except Exception as err:
                self._raise_mapped(err)
            if isinstance(event, ResponseFailed):
                # A terminal failure delivered as a stream event, not an
                # exception — surface it as a typed, retryable error here so
                # the retry/fallback layers engage (and the caller never ends
                # up with no final response at all).
                error = event.response.error
                message = error.message if error else "response failed"
                raise LlmInternalServerError(
                    f"Streamed response failed: {message}",
                    response=httpx.Response(
                        status_code=502,
                        request=httpx.Request("POST", "https://api.openai.com/v1"),
                    ),
                    body=None,
                )
            if isinstance(event, ResponseCompleted):
                self._stamp_cost(event.response)
                logger.info(
                    "llm %s → %s in %.2fs (streamed)",
                    self.model_name,
                    _format_usage(event.response),
                    time.monotonic() - t0,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "llm %s response: %s",
                        self.model_name,
                        grasp_logging.body_for_log(
                            event.response.output_text or "",
                            full=grasp_logging.LOG_LLM_OUTPUT,
                        ),
                    )
            yield event
