from __future__ import annotations

from litellm.types.utils import Choices as LiteLLMChoice
from litellm.types.utils import ModelResponse as LiteLLMCompletion
from litellm.types.utils import ModelResponseStream as LiteLLMCompletionChunk
from litellm.types.utils import StreamingChoices as LiteLLMChunkChoice

from grasp_agents.types.errors import CompletionError


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
