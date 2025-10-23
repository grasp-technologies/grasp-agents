import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from grasp_agents import BaseTool, LLMAgent, Printer, RunContext
from grasp_agents.litellm import LiteLLM, LiteLLMSettings

load_dotenv()

sys_prompt = """
Your task is to suggest an exciting stats problem to the student. 
You should first ask the student about their education, interests, and preferences, then suggest a problem tailored specifically to them. 

# Instructions
* Use the provided tool to ask questions.
* Ask questions one by one.
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


class AskStudentTool(BaseTool[TeacherQuestion, StudentReply, None]):
    name: str = "ask_student"
    description: str = ask_student_tool_description

    async def run(self, inp: TeacherQuestion, **kwargs: Any) -> StudentReply:
        return input(inp.question)


class Problem(BaseModel):
    problem: str


teacher = LLMAgent[None, Problem, None](
    name="teacher",
    llm=LiteLLM(
        model_name="claude-sonnet-4-20250514",
        llm_settings=LiteLLMSettings(reasoning_effort="low"),
    ),
    tools=[AskStudentTool()],
    final_answer_as_tool_call=True,
    sys_prompt=sys_prompt,
)


async def main():
    ctx = RunContext[None](printer=Printer())
    out = await teacher.run("start", ctx=ctx)
    print(out.payloads[0])
    print(ctx.usage_tracker.total_usage)


asyncio.run(main())
