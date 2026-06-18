from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceLogprobs as ChatCompletionChunkChoiceLogprobs,
)
from openai.types.responses.response_output_text import Logprob, LogprobTopLogprob


def _encode_token(token: str) -> list[int]:
    return list(token.encode("utf-8"))


def convert_logprobs(
    raw_logprobs: ChatCompletionChoiceLogprobs | ChatCompletionChunkChoiceLogprobs,
) -> list[Logprob]:
    if not raw_logprobs.content:
        return []
    return [
        Logprob(
            token=lp.token,
            bytes=lp.bytes or _encode_token(lp.token),
            logprob=lp.logprob,
            top_logprobs=[
                LogprobTopLogprob(
                    token=tlp.token,
                    bytes=tlp.bytes or _encode_token(tlp.token),
                    logprob=tlp.logprob,
                )
                for tlp in lp.top_logprobs
            ],
        )
        for lp in raw_logprobs.content
    ]
