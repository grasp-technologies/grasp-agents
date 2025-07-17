import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from grasp_agents import BaseTool, LLMAgent, RunContext
from grasp_agents.grasp_logging import setup_logging
from grasp_agents.litellm import LiteLLM, LiteLLMSettings

load_dotenv()


# Configure the logger to output to the console and/or a file
setup_logging(
    logs_file_path="grasp_agents_demo.log",
    logs_config_path=Path().cwd() / "configs/logging/default.yaml",
)

sys_prompt_react = """
Your task is to suggest an exciting stats problem to the student. 
You should first ask the student about their education, interests, and preferences, then suggest a problem tailored specifically to them. 

# Instructions
* Use the provided tool to ask questions.
* Ask questions one by one.
* Provide your thinking before asking a question and after receiving a reply.
* Do not include your exact question as part of your thinking.
* The problem must have all the necessary data.
* Use the final answer tool to provide the problem.
"""


# Tool input must be a Pydantic model to infer the JSON schema used by the LLM APIs
class TeacherQuestion(BaseModel):
    question: str


StudentReply = str


ask_student_tool_description = """
"Ask the student a question and get their reply."

Args:
    question: str
        The question to ask the student.
Returns:
    reply: str
        The student's reply to the question.
"""


class AskStudentTool(BaseTool[TeacherQuestion, StudentReply, Any]):
    name: str = "ask_student"
    description: str = ask_student_tool_description

    async def run(
        self, inp: TeacherQuestion, ctx: RunContext[Any] | None = None
    ) -> StudentReply:
        return input(inp.question)


class Problem(BaseModel):
    problem: str


teacher = LLMAgent[None, Problem, None](
    name="teacher",
    llm=LiteLLM(
        model_name="gpt-4.1",
        llm_settings=LiteLLMSettings(temperature=0.5),
    ),
    tools=[AskStudentTool()],
    react_mode=True,
    final_answer_as_tool_call=True,
    sys_prompt=sys_prompt_react,
)


async def main():
    ctx = RunContext[None](print_messages=True)
    out = await teacher.run("start", ctx=ctx)
    print(out.payloads[0])
    print(ctx.usage_tracker.total_usage)


asyncio.run(main())
