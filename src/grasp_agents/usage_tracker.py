import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from pydantic import BaseModel, Field
from termcolor import colored

from .typing.message import AssistantMessage, Message, Usage

logger = logging.getLogger(__name__)


COSTS_DICT_PATH: Path = Path(__file__).parent / "costs_dict.yaml"

ModelCostsDict: TypeAlias = dict[str, float]
CostsDict: TypeAlias = dict[str, ModelCostsDict]


class UsageTracker(BaseModel):
    source_id: str
    costs_dict_path: str | Path = COSTS_DICT_PATH
    costs_dict: CostsDict | None = None
    total_usage: Usage = Field(default_factory=Usage)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.costs_dict = self.load_costs_dict()

    def load_costs_dict(self) -> CostsDict | None:
        try:
            with Path(self.costs_dict_path).open() as f:
                return yaml.safe_load(f)["costs"]
        except Exception:
            logger.info(f"Failed to load cost dictionary from {self.costs_dict_path}")
            return None

    def _add_cost_to_usage(
        self, usage: Usage, model_costs_dict: ModelCostsDict
    ) -> None:
        in_rate = model_costs_dict["input"]
        out_rate = model_costs_dict["output"]
        cached_discount = model_costs_dict.get("cached_discount")
        input_cost = in_rate * usage.input_tokens
        output_cost = out_rate * usage.output_tokens
        reasoning_cost = (
            out_rate * usage.reasoning_tokens
            if usage.reasoning_tokens is not None
            else 0.0
        )
        cached_cost: float = (
            cached_discount * in_rate * usage.cached_tokens
            if (usage.cached_tokens is not None) and (cached_discount is not None)
            else 0.0
        )
        usage.cost = (input_cost + output_cost + reasoning_cost + cached_cost) / 1e6

    def update(
        self, messages: Sequence[Message], model_name: str | None = None
    ) -> None:
        if model_name is not None and self.costs_dict is not None:
            model_costs_dict = self.costs_dict.get(model_name)
        else:
            model_costs_dict = None

        for message in messages:
            if isinstance(message, AssistantMessage) and message.usage is not None:
                if model_costs_dict is not None:
                    self._add_cost_to_usage(
                        usage=message.usage, model_costs_dict=model_costs_dict
                    )
                self.total_usage += message.usage

    def reset(self) -> None:
        self.total_usage = Usage()

    def print_usage(self) -> None:
        usage = self.total_usage

        logger.debug("\n-------------------")

        token_usage_str = (
            f"Total {self.source_id} I/O/(R)/(C) tokens: "
            f"{usage.input_tokens}/{usage.output_tokens}"
        )
        if usage.reasoning_tokens is not None:
            token_usage_str += f"/{usage.reasoning_tokens}"
        if usage.cached_tokens is not None:
            token_usage_str += f"/{usage.cached_tokens}"
        logger.debug(colored(token_usage_str, "light_grey"))

        if usage.cost is not None:
            logger.debug(
                colored(
                    f"Total {self.source_id} cost: ${usage.cost:.4f}",
                    "light_grey",
                )
            )
