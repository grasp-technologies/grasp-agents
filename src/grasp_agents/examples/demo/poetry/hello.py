import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from grasp_agents import BaseTool, LLMAgent, RunContext
from grasp_agents.litellm import LiteLLM, LiteLLMSettings
from grasp_agents.printer import print_event_stream
from grasp_agents.typing.events import ProcPacketOutEvent

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
        model_name="claude-sonnet-4-5",
        llm_settings=LiteLLMSettings(reasoning_effort="low"),
    ),
    tools=[AskStudentTool()],
    final_answer_as_tool_call=True,
    sys_prompt=sys_prompt,
    stream_llm_responses=True,
)


async def main():
    ctx = RunContext[None]()
    async for event in print_event_stream(teacher.run_stream("start", ctx=ctx)):
        if isinstance(event, ProcPacketOutEvent):
            result = event.data.payloads[0]
            print(f"\n<Suggested Problem>:\n\n{result.problem}\n")


asyncio.run(main())
