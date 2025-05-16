import asyncio
from typing import Any

from dotenv import load_dotenv

from grasp_agents.llm_agent import LLMAgent
from grasp_agents.openai.openai_llm import OpenAILLM
from grasp_agents.run_context import RunContextWrapper

load_dotenv()


chatbot = LLMAgent[None, int, None](
    agent_id="chatbot", llm=OpenAILLM(model_name="gpt-4o", api_provider="openai")
)


async def main():
    ctx = RunContextWrapper[Any](print_messages=True)
    out = await chatbot.run(
        "Output a random number from 0 to 10. No conversation.", ctx=ctx
    )
    print(out.payloads[0])


asyncio.run(main())
