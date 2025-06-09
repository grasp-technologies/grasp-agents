import asyncio
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from grasp_agents import BaseTool, LLMAgent, Messages, RunContext
from grasp_agents.grasp_logging import setup_logging
from grasp_agents.openai import OpenAILLM, OpenAILLMSettings

load_dotenv()


# Configure the logger to output to the console and/or a file
setup_logging(
    logs_file_path="grasp_agents_demo.log",
    logs_config_path=Path().cwd() / "configs/logging/default.yaml",
)

sys_prompt_react = """
Your task is to suggest an exciting stats problem to a student.
Ask the student about their education, interests, and preferences, then suggest a problem tailored to them.

# Instructions
* Ask questions one by one.
* Provide your thinking before asking a question and after receiving a reply.
* The problem must be enclosed in <PROBLEM> tags.
"""


class TeacherQuestion(BaseModel):
    question: str = Field(..., description="The question to ask the student.")


StudentReply = str


class AskStudentTool(BaseTool[TeacherQuestion, StudentReply, Any]):
    name: str = "ask_student_tool"
    description: str = "Ask the student a question and get their reply."

    async def run(
        self, inp: TeacherQuestion, ctx: RunContext[Any] | None = None
    ) -> StudentReply:
        return input(inp.question)


Problem = str


teacher = LLMAgent[None, Problem, None](
    name="teacher",
    llm=OpenAILLM(
        model_name="openai:gpt-4.1", llm_settings=OpenAILLMSettings(temperature=0.1)
    ),
    tools=[AskStudentTool()],
    max_turns=20,
    react_mode=True,
    sys_prompt=sys_prompt_react,
    reset_memory_on_run=True,
)


@teacher.exit_tool_call_loop
def exit_tool_call_loop(
    conversation: Messages, ctx: RunContext[Any] | None, **kwargs: Any
) -> bool:
    return r"<PROBLEM>" in str(conversation[-1].content)


@teacher.parse_output
def parse_output(
    conversation: Messages, ctx: RunContext[Any] | None, **kwargs: Any
) -> Problem:
    message = str(conversation[-1].content)
    matches = re.findall(r"<PROBLEM>(.*?)</PROBLEM>", message, re.DOTALL)

    return matches[0]


async def main():
    ctx = RunContext[None](print_messages=True)
    out = await teacher.run(ctx=ctx)
    print(out.payloads[0])
    print(ctx.usage_tracker.total_usage)


asyncio.run(main())
