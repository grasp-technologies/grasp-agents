"""Convert grasp-agents items → OpenAI Responses API input params."""

from __future__ import annotations

from collections.abc import Sequence

from openai.types.responses.response_function_web_search_param import (
    ActionOpenPage as ActionOpenPageParam,
)
from openai.types.responses.response_function_web_search_param import (
    ActionSearch as ActionSearchParam,
)
from openai.types.responses.response_function_web_search_param import (
    ActionSearchSource as ActionSearchSourceParam,
)
from openai.types.responses.response_function_web_search_param import (
    ResponseFunctionWebSearchParam,
)
from openai.types.responses.response_input_item_param import (
    ResponseInputItemParam,
)

from grasp_agents.types.items import InputItem, SearchAction, WebSearchCallItem

# Fields added by grasp-agents that are NOT part of the OpenAI Responses API
_GRASP_EXTENSION_FIELDS = {
    "content_parts",
    "output_parts",
    "summary_parts",
    "redacted",
    "provider_specific_fields",
}


def items_to_provider_inputs(
    items: Sequence[InputItem],
) -> list[ResponseInputItemParam]:
    result: list[ResponseInputItemParam] = []
    for item in items:
        if isinstance(item, WebSearchCallItem):
            result.append(_web_search_item_to_param(item))
        else:
            result.append(
                item.model_dump(  # type: ignore[arg-type]
                    exclude=_GRASP_EXTENSION_FIELDS, exclude_none=True, mode="json"
                )
            )
    return result


def _web_search_item_to_param(
    item: WebSearchCallItem,
) -> ResponseFunctionWebSearchParam:
    """Convert WebSearchCallItem to Responses API input param."""
    action = item.action
    api_action: ActionSearchParam | ActionOpenPageParam
    if isinstance(action, SearchAction):
        api_action = ActionSearchParam(
            type="search",
            query=action.queries[0] if action.queries else "",
            queries=action.queries or [],
            sources=[
                ActionSearchSourceParam(type="url", url=s.url)
                for s in (action.sources or [])
            ],
        )
    else:
        api_action = ActionOpenPageParam(type="open_page", url=action.url)
    return ResponseFunctionWebSearchParam(
        type="web_search_call",
        id=item.id,
        status=item.status,
        action=api_action,
    )
