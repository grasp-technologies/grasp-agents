"""Convert grasp-agents items → OpenAI Responses API input params."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Fields added by grasp-agents that are NOT part of the OpenAI Responses API
_GRASP_EXTENSION_FIELDS = {
    "redacted",
    "provider_specific_fields",
    "is_error",
}

# Same, on content/output/summary parts and their annotations — the API rejects
# unknown parameters (e.g. ``mime_type`` on a base64 image part).
_GRASP_PART_EXTENSION_FIELDS = {
    "mime_type",
    "cache_control",
    "provider_specific_fields",
}


def _iter_part_dicts(dumped: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for field in ("content", "output", "summary"):
        parts: object = dumped.get(field)
        if not isinstance(parts, list):
            continue
        for part in cast("list[object]", parts):
            if isinstance(part, dict):
                yield cast("dict[str, Any]", part)


def _scrub_part_fields(dumped: dict[str, Any]) -> None:
    for part in _iter_part_dicts(dumped):
        for key in _GRASP_PART_EXTENSION_FIELDS:
            part.pop(key, None)
        annotations: object = part.get("annotations")
        if not isinstance(annotations, list):
            continue
        for annotation in cast("list[object]", annotations):
            if isinstance(annotation, dict):
                cast("dict[str, Any]", annotation).pop("provider_specific_fields", None)


def items_to_provider_inputs(
    items: Sequence[InputItem],
) -> list[ResponseInputItemParam]:
    result: list[ResponseInputItemParam] = []
    for item in items:
        if isinstance(item, WebSearchCallItem):
            result.append(_web_search_item_to_param(item))
            continue
        dumped = item.model_dump(  # type: ignore[arg-type]
            exclude=_GRASP_EXTENSION_FIELDS, exclude_none=True, mode="json"
        )
        _scrub_part_fields(dumped)
        # The Responses API reads a client-sent message ``id`` as a reference to a
        # stored item and 404s on it (fatally when the message carries an image);
        # our ``msg_`` ids are internal bookkeeping, so don't echo them back.
        if dumped.get("type") == "message":
            dumped.pop("id", None)
        result.append(dumped)  # type: ignore[arg-type]
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
