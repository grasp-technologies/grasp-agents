"""Logprobs conversion helpers shared by stream converters."""

from grasp_agents.types.llm_events import (
    output_to_delta_logprobs as to_delta_logprobs,
)
from grasp_agents.types.llm_events import (
    output_to_done_logprobs as to_done_logprobs,
)

__all__ = ["to_delta_logprobs", "to_done_logprobs"]
