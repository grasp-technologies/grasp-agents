"""TypeSafeLLM under LLMAgent and ParallelProcessor, against a fake TypeSafe client."""

import pytest

from grasp_agents.agent.llm_agent import LLMAgent
from grasp_agents.processors.parallel_processor import ParallelProcessor

from ._helpers import jev_response, noul
from .test_typesafe_llm import TREND, FakeClient, Learnable, make_llm

MEME = "r/aww: my cat sat in a box."


def make_agent(client: FakeClient) -> LLMAgent[str, Learnable, None]:
    return LLMAgent[str, Learnable, None](
        name="trend_filter", llm=make_llm(client), env_info=False
    )


@pytest.mark.asyncio
async def test_agent_output_is_the_typed_schema() -> None:
    client = FakeClient(jev_response({"learnable": noul(0.93)}))

    packet = await make_agent(client).run(TREND)

    assert packet.payloads == [Learnable(learnable=True)]
    assert client.calls[0]["state"] == TREND


@pytest.mark.asyncio
async def test_parallel_processor_judges_every_input() -> None:
    client = FakeClient(
        jev_response({"learnable": noul(0.93)}),
        jev_response({"learnable": noul(0.02)}),
    )
    parallel = ParallelProcessor[str, Learnable, None](subproc=make_agent(client))

    packet = await parallel.run(in_args=[TREND, MEME])

    judged = {call["state"] for call in client.calls}
    assert judged == {TREND, MEME}
    assert sorted(p.learnable for p in packet.payloads) == [False, True]
