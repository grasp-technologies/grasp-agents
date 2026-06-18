from openai.types.responses.response_output_text import Logprob
from openai.types.responses.response_text_delta_event import Logprob as DeltaLogprob
from openai.types.responses.response_text_delta_event import (
    LogprobTopLogprob as DeltaTopLogprob,
)
from openai.types.responses.response_text_done_event import Logprob as DoneLogprob
from openai.types.responses.response_text_done_event import (
    LogprobTopLogprob as DoneTopLogprob,
)


def output_to_delta_logprobs(logprobs: list[Logprob]) -> list[DeltaLogprob]:
    """Convert by skipping the 'bytes' field which is not needed for delta logprobs."""
    return [
        DeltaLogprob(
            token=lp.token,
            logprob=lp.logprob,
            top_logprobs=[
                DeltaTopLogprob(token=tlp.token, logprob=tlp.logprob)
                for tlp in lp.top_logprobs
            ]
            if lp.top_logprobs
            else None,
        )
        for lp in logprobs
    ]


def output_to_done_logprobs(logprobs: list[Logprob]) -> list[DoneLogprob]:
    """Convert by skipping the 'bytes' field which is not needed for done logprobs."""
    return [
        DoneLogprob(
            token=lp.token,
            logprob=lp.logprob,
            top_logprobs=[
                DoneTopLogprob(token=tlp.token, logprob=tlp.logprob)
                for tlp in lp.top_logprobs
            ]
            if lp.top_logprobs
            else None,
        )
        for lp in logprobs
    ]
