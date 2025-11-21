import fnmatch
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from ..cloud_llm import APIProvider, CloudLLM, CloudLLMSettings

logger = logging.getLogger(__name__)


def get_openai_compatible_providers() -> list[APIProvider]:
    """Returns a   dictionary of available OpenAI-compatible API providers."""
    return [
        APIProvider(
            name="openai",
            base_url="https://api.openai.com/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            response_schema_support=("*",),
        ),
        APIProvider(
            name="gemini_openai",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("GEMINI_API_KEY"),
            response_schema_support=("*",),
        ),
        # Openrouter does not support structured outputs ATM
        APIProvider(
            name="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            response_schema_support=(),
        ),
    ]


class OpenAILLMSettings(CloudLLMSettings, total=False):
    parallel_tool_calls: bool
    top_logprobs: int | None

    metadata: dict[str, str] | None
    store: bool | None
    user: str
    # TODO: support audio


@dataclass(frozen=True)
class OpenAILLM(CloudLLM):
    openai_client_timeout: float = 120.0
    openai_client_max_retries: int = 2
    extra_openai_client_params: dict[str, Any] | None = None
    client: AsyncOpenAI = field(init=False)

    def __post_init__(self):
        super().__post_init__()

        openai_compatible_providers = get_openai_compatible_providers()

        _api_provider = self.api_provider

        model_name_parts = self.model_name.split("/", 1)
        if _api_provider is not None:
            _model_name = self.model_name
        elif len(model_name_parts) == 2:
            compat_providers_map = {
                provider["name"]: provider for provider in openai_compatible_providers
            }
            _provider_name, _model_name = model_name_parts
            if _provider_name not in compat_providers_map:
                raise ValueError(
                    f"API provider '{_provider_name}' is not a supported OpenAI "
                    f"compatible provider. Supported providers are: "
                    f"{', '.join(compat_providers_map.keys())}"
                )
            _api_provider = compat_providers_map[_provider_name]
        else:
            raise ValueError(
                "Model name must be in the format 'provider/model_name' or "
                "you must provide an 'api_provider' argument."
            )

        response_schema_support: bool = any(
            fnmatch.fnmatch(_model_name, pat)
            for pat in _api_provider.get("response_schema_support") or []
        )
        if self.apply_response_schema_via_provider and not response_schema_support:
            raise ValueError(
                "Native response schema validation is not supported for model "
                f"'{_model_name}' by the API provider '{_api_provider['name']}'. "
                "Please set apply_response_schema_via_provider=False."
            )

        _openai_client_params = deepcopy(self.extra_openai_client_params or {})
        _openai_client_params["timeout"] = self.openai_client_timeout
        _openai_client_params["max_retries"] = self.openai_client_max_retries
        if self.http_client is not None:
            _openai_client_params["http_client"] = self.http_client

        _client = AsyncOpenAI(
            base_url=_api_provider.get("base_url"),
            api_key=_api_provider.get("api_key"),
            **_openai_client_params,
        )

        object.__setattr__(self, "model_name", _model_name)
        object.__setattr__(self, "api_provider", _api_provider)
        object.__setattr__(self, "client", _client)
