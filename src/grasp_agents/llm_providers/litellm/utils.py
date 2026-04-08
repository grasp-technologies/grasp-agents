from __future__ import annotations

from typing import TYPE_CHECKING

from grasp_agents.types.errors import CompletionError
from grasp_agents.types.items import ReasoningItem
from litellm.types.utils import Choices as LiteLLMChoice
from litellm.types.utils import ModelResponse as LiteLLMCompletion
from litellm.types.utils import ModelResponseStream as LiteLLMCompletionChunk
from litellm.types.utils import StreamingChoices as LiteLLMChunkChoice

if TYPE_CHECKING:
    from grasp_agents.llm.llm_stream_converter import ToolCallState
    from grasp_agents.types.items import OutputItem


def validate_completion(completion: LiteLLMCompletion) -> None:
    """Convert an OpenAI Chat Completion → internal Response."""
    if completion.choices is None:  # type: ignore[comparison-overlap]
        raise CompletionError(
            f"Completion API error: {getattr(completion, 'error', None)}"
        )

    if not completion.choices:
        raise CompletionError("No choices in completion")

    if len(completion.choices) > 1:
        raise CompletionError("Multiple choices are not supported")

    choice = completion.choices[0]
    if not isinstance(choice, LiteLLMChoice):
        raise CompletionError("choice is not a LiteLLM Choice")

    if choice.message is None:  # type: ignore[comparison-overlap]
        raise CompletionError(
            f"API returned None for message, finish_reason: {choice.finish_reason}"
        )


def validate_chunk(chunk: LiteLLMCompletionChunk) -> None:
    if chunk.choices is None:  # type: ignore[union-attr]
        raise CompletionError(
            f"Completion chunk API error: {getattr(chunk, 'error', None)}"
        )

    if not chunk.choices:
        raise CompletionError("Completion chunk has no choices")

    if len(chunk.choices) > 1:
        raise CompletionError("Multiple choices are not supported in completion chunk")

    choice = chunk.choices[0]
    if not isinstance(choice, LiteLLMChunkChoice):  # type: ignore[union-attr]
        raise CompletionError("choice in completion chunk is not a LiteLLMChunkChoice")

    if choice.delta is None:  # type: ignore[union-attr]
        raise CompletionError("Chunk choice is missing delta")


def patch_thought_signatures(
    thought_signatures: list[str],
    items: list[OutputItem],
    tool_calls: dict[int, ToolCallState],
) -> None:
    """
    Distribute thought_signatures from provider_specific_fields onto items.

    Fallback for providers that send plain reasoning_content without
    thinking_blocks.  Signatures are matched positionally: first to
    ReasoningItems that lack encrypted_content, then to ToolCallStates
    that lack provider_specific_fields.
    """
    sig_iter = iter(thought_signatures)
    for i, item in enumerate(items):
        if isinstance(item, ReasoningItem) and not item.encrypted_content:
            sig = next(sig_iter, None)
            if sig is None:
                return
            items[i] = item.model_copy(update={"encrypted_content": sig})
    for state in tool_calls.values():
        if not state.provider_specific_fields:
            sig = next(sig_iter, None)
            if sig is None:
                return
            state.provider_specific_fields = {"thought_signature": sig}
