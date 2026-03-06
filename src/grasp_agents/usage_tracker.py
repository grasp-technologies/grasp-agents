import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from pydantic import BaseModel, Field
from termcolor import colored

from .types.response import Response, ResponseUsage

logger = logging.getLogger(__name__)


COSTS_DICT_PATH: Path = Path(__file__).parent / "costs_dict.yaml"

ModelCostsDict: TypeAlias = dict[str, float]
CostsDict: TypeAlias = dict[str, ModelCostsDict]


class UsageTracker(BaseModel):
    costs_dict_path: str | Path = COSTS_DICT_PATH
    costs_dict: CostsDict | None = None
    usages: dict[str, ResponseUsage] = Field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.costs_dict = self.load_costs_dict()

    def update(
        self,
        agent_name: str,
        responses: Sequence[Response],
        model_name: str | None = None,
    ) -> None:
        if model_name is not None and self.costs_dict is not None:
            model_costs_dict = self.costs_dict.get(model_name.split("/", 1)[-1])
        else:
            model_costs_dict = None

        for response in responses:
            if response.usage_with_cost is not None:
                usage = response.usage_with_cost
                if usage.cost is None and model_costs_dict is not None:
                    self._add_cost_to_usage(
                        usage=usage,
                        model_costs_dict=model_costs_dict,
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
        logger.debug(colored(token_usage_str, "light_grey"))

        if usage.cost is not None:
            logger.debug(colored(f"Total cost: ${usage.cost:.4f}", "light_grey"))

    def load_costs_dict(self) -> CostsDict | None:
        try:
            with Path(self.costs_dict_path).open() as f:
                return yaml.safe_load(f)["costs"]
        except Exception:
            logger.info(f"Failed to load cost dictionary from {self.costs_dict_path}")
            return None

    def _add_cost_to_usage(
        self, usage: ResponseUsage, model_costs_dict: ModelCostsDict
    ) -> None:
        in_rate = model_costs_dict["input"]
        out_rate = model_costs_dict["output"]
        cached_discount = model_costs_dict.get("cached_discount", 1.0)
        input_cost = in_rate * (
            usage.input_tokens - usage.input_tokens_details.cached_tokens
        )
        output_cost = out_rate * usage.output_tokens
        cached_cost: float = (
            cached_discount * in_rate * usage.input_tokens_details.cached_tokens
        )
        usage.cost = (input_cost + output_cost + cached_cost) / 1e6
