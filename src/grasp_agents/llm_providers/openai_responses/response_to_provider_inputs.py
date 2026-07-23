"""Convert grasp-agents items → OpenAI Responses API input params."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from openai.types.responses.response_function_web_search_param import (
    ActionFind as ActionFindParam,
)
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

from grasp_agents.types.items import (
    FindInPageAction,
    FunctionToolOutputItem,
    InputItem,
    InputMessageItem,
    SearchAction,
    WebSearchCallItem,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

# Fields added by grasp-agents that are NOT part of the OpenAI Responses API
_GRASP_EXTENSION_FIELDS = {
    "redacted",
    "provider_specific_fields",
    "is_error",
    "cache_control",
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


def _reapply_part_cache_breakpoints(item: InputItem, dumped: dict[str, Any]) -> None:
    """
    Attach an explicit prompt-cache breakpoint (gpt-5.6+) to each part param
    whose source part carries a ``CacheControl``. The TTL comes from the
    request-level ``prompt_cache_options``, so ``CacheControl.ttl`` is
    ignored here.
    """
    if isinstance(item, InputMessageItem):
        parts, part_params = item.content, dumped["content"]
    elif isinstance(item, FunctionToolOutputItem) and isinstance(item.output, list):
        parts, part_params = item.output, dumped["output"]
    else:
        return
    for part, part_param in zip(parts, part_params, strict=True):
        if part.cache_control is not None:
            part_param["prompt_cache_breakpoint"] = {"mode": "explicit"}


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
        dumped = item.model_dump(
            exclude=_GRASP_EXTENSION_FIELDS, exclude_none=True, mode="json"
        )
        _scrub_part_fields(dumped)
        _reapply_part_cache_breakpoints(item, dumped)
        # The Responses API reads a client-sent message ``id`` as a reference to a
        # stored item and 404s on it (fatally when the message carries an image);
        # our ``msg_`` ids are internal bookkeeping, so don't echo them back.
        if dumped.get("type") == "message":
            dumped.pop("id", None)
        result.append(cast("ResponseInputItemParam", dumped))
    return result


def _web_search_item_to_param(
    item: WebSearchCallItem,
) -> ResponseFunctionWebSearchParam:
    """Convert WebSearchCallItem to Responses API input param."""
    action = item.action
    api_action: ActionSearchParam | ActionOpenPageParam | ActionFindParam
    if isinstance(action, SearchAction):
        api_action = ActionSearchParam(type="search")
        if action.queries:
            api_action["query"] = action.queries[0]
            api_action["queries"] = action.queries
        if action.sources:
            api_action["sources"] = [
                ActionSearchSourceParam(type="url", url=s.url)
                for s in action.sources
            ]

    elif isinstance(action, FindInPageAction):
        api_action = ActionFindParam(
            type="find_in_page", url=action.url or "", pattern=action.pattern or ""
        )
    else:
        api_action = ActionOpenPageParam(type="open_page", url=action.url)

    return ResponseFunctionWebSearchParam(
        type="web_search_call",
        id=item.id,
        status=item.status,
        action=api_action,
    )
