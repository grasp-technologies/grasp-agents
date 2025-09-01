import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from grasp_agents import BaseTool, LLMAgent, Printer, RunContext
from grasp_agents.litellm import LiteLLM

load_dotenv()

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


class AskStudentTool(BaseTool[TeacherQuestion, StudentReply, None]):
    name: str = "ask_student"
    description: str = ask_student_tool_description

    async def run(self, inp: TeacherQuestion, **kwargs: Any) -> StudentReply:
        return input(inp.question)


class Problem(BaseModel):
    problem: str


teacher = LLMAgent[None, Problem, None](
    name="teacher",
    llm=LiteLLM(model_name="gpt-4.1"),
    tools=[AskStudentTool()],
    react_mode=True,
    final_answer_as_tool_call=True,
    sys_prompt=sys_prompt_react,
)


async def main():
    ctx = RunContext[None](printer=Printer())
    out = await teacher.run("start", ctx=ctx)
    print(out.payloads[0])
    print(ctx.usage_tracker.total_usage)


asyncio.run(main())
