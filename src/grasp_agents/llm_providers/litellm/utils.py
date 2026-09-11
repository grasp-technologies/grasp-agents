from __future__ import annotations

from typing import TYPE_CHECKING, Any

from litellm.types.utils import Choices as LiteLLMChoice
from litellm.types.utils import ModelResponse as LiteLLMCompletion
from litellm.types.utils import ModelResponseStream as LiteLLMCompletionChunk
from litellm.types.utils import StreamingChoices as LiteLLMChunkChoice

from grasp_agents.types.errors import CompletionError

if TYPE_CHECKING:
    from collections.abc import Mapping


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
    # Runtime guard: litellm's annotations promise Choices, but a streaming
    # response can put StreamingChoices here.
    if not isinstance(choice, LiteLLMChoice):  # pyright: ignore[reportUnnecessaryIsInstance]
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


_SIGNED_CALL_ID_MARKER = "__thought__"


def split_signed_call_id(call_id: str) -> tuple[str, str | None]:
    plain_id, marker, signature = call_id.partition(_SIGNED_CALL_ID_MARKER)
    if not marker or not signature:
        return call_id, None
    return plain_id, signature


def tool_call_id_and_fields(
    call_id: str, provider_specific_fields: Mapping[str, Any] | None
) -> tuple[str, dict[str, Any] | None]:
    """
    Plain call id and provider fields for a tool call LiteLLM returned.

    LiteLLM embeds a Gemini thought signature in the call id, which other
    providers reject (too long, invalid characters). The signature moves into
    ``provider_specific_fields`` unless one is already there; other fields on
    the call are kept as they are.
    """
    plain_id, id_signature = split_signed_call_id(call_id)
    fields = dict(provider_specific_fields or {})
    if id_signature:
        fields.setdefault("thought_signature", id_signature)
    return plain_id, fields or None
