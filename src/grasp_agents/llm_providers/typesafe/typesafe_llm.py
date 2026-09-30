"""TypeSafe System One provider (Jev): typed judgments instead of generated text."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, with_config
from typesafe_sdk import AsyncTypeSafeClient, SystemOneResponse
from typesafe_sdk import RetryPolicy as TypeSafeRetryPolicy

from grasp_agents.llm.cloud_llm import ApiCallParams, CloudLLM, CloudLLMSettings
from grasp_agents.llm.model_info import ModelCapabilities
from grasp_agents.tools.base import BaseTool, ToolChoice
from grasp_agents.types.items import InputItem
from grasp_agents.types.llm_errors import LlmError
from grasp_agents.types.llm_events import LlmEvent, ResponseCompleted
from grasp_agents.types.response import Response

from .error_mapping import map_api_error
from .provider_output_to_response import provider_output_to_response
from .questions import QuestionPlan, build_question_plan
from .response_to_provider_inputs import items_to_state

JEV_USD_PER_MILLION_TOKENS = 0.042
# jev-1.13: state and questions together; past it the request is rejected.
JEV_MAX_INPUT_TOKENS = 64_000
TOKENS_PER_MILLION = 1_000_000

_FORWARDED_SETTINGS = ("timeout", "extra_headers", "extra_body")


@with_config(ConfigDict(extra="allow"))
class TypeSafeLLMSettings(CloudLLMSettings, total=False):
    # Framing prefixed to every question: Jev has no request-level instructions.
    instructions: str | None
    timeout: float | None


@dataclass(frozen=True)
class _JevResult:
    response: SystemOneResponse
    plan: QuestionPlan


@dataclass(frozen=True)
class TypeSafeLLM(CloudLLM):
    """
    Jev answers questions about a text; it does not write text.

    The output schema is the question list (see ``questions``) and the single
    user message is the text judged. Sampling settings such as ``temperature``
    have no meaning here and are dropped.
    """

    _settings_type: ClassVar[Any] = TypeSafeLLMSettings

    _native_provider_name: ClassVar[str] = "typesafe"
    _native_api_key_env_vars: ClassVar[tuple[str, ...]] = ("TYPESAFE_API_KEY",)

    llm_settings: TypeSafeLLMSettings | None = None

    usd_per_million_tokens: float = JEV_USD_PER_MILLION_TOKENS
    typesafe_client_timeout: float | None = None
    # Forwarded verbatim to AsyncTypeSafeClient, e.g. an httpx2 ``http_client``.
    extra_typesafe_client_params: dict[str, Any] | None = None

    client: AsyncTypeSafeClient = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.http_client is not None:
            raise ValueError(
                "The TypeSafe SDK needs an httpx2 client: pass it as"
                " extra_typesafe_client_params={'http_client': ...}"
            )

        api_provider = self.api_provider or self._default_api_provider()
        client = AsyncTypeSafeClient(
            api_key=api_provider.get("api_key"),
            base_url=api_provider.get("base_url"),
            # ``LLM.retry_policy`` is the one retry layer.
            retry=TypeSafeRetryPolicy(max_retries=0),
            timeout=self.typesafe_client_timeout,
            headers=self.default_headers,
            **(self.extra_typesafe_client_params or {}),
        )
        object.__setattr__(self, "api_provider", api_provider)
        object.__setattr__(self, "client", client)

    @cached_property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            function_calling=False,
            vision=False,
            output_schema=True,
            prompt_caching=False,
            reasoning=False,
            web_search=False,
            audio_input=False,
            max_input_tokens=JEV_MAX_INPUT_TOKENS,
            max_output_tokens=None,
        )

    # --- Input preparation ---

    def _make_api_input(
        self,
        input: Sequence[InputItem],  # ruff: ignore[builtin-argument-shadowing]
        tools: Mapping[str, BaseTool[BaseModel, Any, Any]] | None = None,
        tool_choice: ToolChoice | None = None,
        output_schema: Any | None = None,
        **extra_llm_settings: Any,
    ) -> ApiCallParams:
        if tools or tool_choice is not None:
            raise ValueError("TypeSafeLLM cannot call tools")
        if output_schema is None:
            raise ValueError(
                "TypeSafeLLM needs an output_schema: its fields are the questions"
            )

        state, framing = items_to_state(input)
        merged: dict[str, Any] = {**(self.llm_settings or {}), **extra_llm_settings}
        instructions = "\n\n".join(
            part for part in (merged.get("instructions"), framing) if part
        )
        plan = build_question_plan(output_schema, instructions or None)

        forwarded = {
            key: merged[key]
            for key in _FORWARDED_SETTINGS
            if merged.get(key) is not None
        }
        return ApiCallParams(
            api_input=[state], extra_settings={"plan": plan, **forwarded}
        )

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
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> _JevResult:
        del api_tools, api_tool_choice, api_output_schema
        return await self._system_one(api_input, api_llm_settings)

    async def _get_api_stream(
        self,
        api_input: list[Any],
        *,
        api_tools: list[Any] | None = None,
        api_tool_choice: Any | None = None,
        api_output_schema: Any | None = None,
        **api_llm_settings: Any,
    ) -> AsyncIterator[_JevResult]:
        del api_tools, api_tool_choice, api_output_schema

        async def single_event() -> AsyncIterator[_JevResult]:
            yield await self._system_one(api_input, api_llm_settings)

        return single_event()

    # Not rate-limited itself: both callers above already are.
    async def _system_one(
        self, api_input: list[Any], api_llm_settings: dict[str, Any]
    ) -> _JevResult:
        plan: QuestionPlan = api_llm_settings["plan"]
        # The SDK's recursive JSON alias resolves to Unknown under strict mode.
        response = await self.client.system_one(  # pyright: ignore[reportUnknownMemberType]
            state=api_input[0],
            questions=plan.questions,
            model=self.model_name,
            timeout=api_llm_settings.get("timeout"),
            extra_headers=api_llm_settings.get("extra_headers"),
            extra_body=api_llm_settings.get("extra_body"),
        )
        return _JevResult(response=response, plan=plan)

    # --- Conversion layer ---

    def _convert_api_response(self, raw: Any) -> Response:
        return provider_output_to_response(raw.response, raw.plan)

    async def _convert_api_stream(
        self, api_stream: AsyncIterator[Any]
    ) -> AsyncIterator[LlmEvent]:
        async for raw in api_stream:
            yield ResponseCompleted(
                sequence_number=0, response=self._convert_api_response(raw)
            )

    # --- Cost stamping ---

    def _stamp_cost(self, response: Response) -> None:
        usage = response.usage
        if usage is not None and usage.cost is None:
            usage.cost = (
                usage.total_tokens / TOKENS_PER_MILLION * self.usd_per_million_tokens
            )
