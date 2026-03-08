"""Map OpenAI Responses SDK exceptions to LLMError types.

Same SDK as OpenAI Completions — reuses its mapping.
"""

from grasp_agents.llm_providers.openai_completions.error_mapping import (
    map_api_error,
)

__all__ = ["map_api_error"]
