import logging
from collections import defaultdict
from collections.abc import Sequence

from litellm import cost_per_token
from pydantic import BaseModel, Field

from .types.response import Response, ResponseUsage

logger = logging.getLogger(__name__)


def add_cost_to_usage(
    usage: ResponseUsage,
    model_name: str,
    litellm_provider: str | None,
) -> None:
    """Stamp ``usage.cost`` from litellm pricing data (no-op when unmapped)."""
    # Prefixed names like "openai/gpt-4o" (OpenRouter-style): use the prefix
    # as the provider hint only when none was given. With an explicit
    # provider, the full (possibly prefixed) name is what litellm expects —
    # e.g. provider "openrouter" + model "openai/gpt-4o".
    if "/" in model_name and litellm_provider is None:
        litellm_provider, model_name = model_name.split("/", 1)

    try:
        prompt_cost, completion_cost = cost_per_token(
            model=model_name,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
            cache_creation_input_tokens=usage.cache_creation_tokens,
            custom_llm_provider=litellm_provider,
        )
        usage.cost = prompt_cost + completion_cost
    except Exception:
        logger.warning(
            "No pricing data for model %s (provider %s); cost not tracked",
            model_name,
            litellm_provider,
        )


class UsageTracker(BaseModel):
    usages: dict[str, ResponseUsage] = Field(default_factory=dict)

    def update(
        self,
        agent_name: str,
        responses: Sequence[Response],
        model_name: str | None = None,
        litellm_provider: str | None = None,
    ) -> None:
        for response in responses:
            if response.usage_with_cost is not None:
                usage = response.usage_with_cost
                if usage.cost is None and model_name is not None:
                    self._add_cost_to_usage(
                        usage=usage,
                        model_name=model_name,
                        litellm_provider=litellm_provider,
                    )
                if agent_name not in self.usages:
                    self.usages[agent_name] = ResponseUsage()
                self.usages[agent_name] += response.usage_with_cost

    @property
    def total_usage(self) -> ResponseUsage:
        return sum((usage for usage in self.usages.values()), ResponseUsage())

    def reset(self) -> None:
        self.usages = defaultdict(ResponseUsage)

    def print_usage(self) -> None:
        usage = self.total_usage

        logger.debug("\n-------------------")

        token_usage_str = (
            f"Total I/O/R/C tokens: {usage.input_tokens}/{usage.output_tokens}"
        )
        token_usage_str += f"/{usage.output_tokens_details.reasoning_tokens}"
        token_usage_str += f"/{usage.input_tokens_details.cached_tokens}"
        logger.debug(
            token_usage_str, extra={"color": "bright_black"}
        )

        if usage.cost is not None:
            logger.debug(
                "Total cost: $%.4f",
                usage.cost,
                extra={"color": "bright_black"},
            )

    def _add_cost_to_usage(
        self,
        usage: ResponseUsage,
        model_name: str,
        litellm_provider: str | None,
    ) -> None:
        add_cost_to_usage(
            usage, model_name=model_name, litellm_provider=litellm_provider
        )
