from openai.types.chat import ChatCompletion, ChatCompletionChunk

from grasp_agents.errors import CompletionError


def validate_completion(completion: ChatCompletion) -> None:
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
    if choice.message is None:  # type: ignore[comparison-overlap]
        raise CompletionError(
            f"API returned None for message, finish_reason: {choice.finish_reason}"
        )


def validate_chunk(chunk: ChatCompletionChunk) -> bool:
    """Validate a streaming chunk. Returns False for usage-only chunks."""
    if chunk.choices is None:  # type: ignore[union-attr]
        raise CompletionError(
            f"Completion chunk API error: {getattr(chunk, 'error', None)}"
        )

    if not chunk.choices:
        # Final chunk with only usage data — valid but no content
        return False

    if len(chunk.choices) > 1:
        raise CompletionError("Multiple choices are not supported in completion chunk")

    choice = chunk.choices[0]
    if choice.delta is None:  # type: ignore[union-attr]
        raise CompletionError("Chunk choice is missing delta")

    return True
