import asyncio
from typing import Any

from grasp_agents.llm_agent import LLMAgent
from grasp_agents.openai.openai_llm import (
    OpenAILLM,
    OpenAILLMSettings,
)
from grasp_agents.typing.io import (
    AgentPayload,
)
from grasp_agents.run_context import RunContextWrapper


class Response(AgentPayload):
    response: str


chatbot = LLMAgent[Any, Response, None](
    agent_id="chatbot",
    llm=OpenAILLM(
        model_name="gpt-4o",
        llm_settings=OpenAILLMSettings(),
    ),
    sys_prompt=None,
    out_schema=Response,
)


@chatbot.parse_output_handler
def output_handler(conversation, ctx, **kwargs) -> Response:
    return Response(response=conversation[-1].content)


async def main():
    ctx = RunContextWrapper(print_messages=True)
    out = await chatbot.run("Hello, agent!", ctx=ctx)
    print(out.payloads[0].response)


asyncio.run(main())
