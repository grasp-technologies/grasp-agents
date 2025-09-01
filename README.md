# Grasp Agents

<br/>
<picture>
  <source srcset="https://raw.githubusercontent.com/grasp-technologies/grasp-agents/master/.assets/grasp-dark.svg" media="(prefers-color-scheme: dark)">
  <img src="https://raw.githubusercontent.com/grasp-technologies/grasp-agents/master/.assets/grasp.svg" alt="Grasp Agents"/>
</picture>
<br/>
<br/>

[![PyPI version](https://badge.fury.io/py/grasp_agents.svg)](https://badge.fury.io/py/grasp-agents)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](https://mit-license.org/)
[![PyPI downloads](https://img.shields.io/pypi/dm/grasp-agents?style=flat-square)](https://pypi.org/project/grasp-agents/)
[![GitHub Stars](https://img.shields.io/github/stars/grasp-technologies/grasp-agents?style=social)](https://github.com/grasp-technologies/grasp-agents/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/grasp-technologies/grasp-agents?style=social)](https://github.com/grasp-technologies/grasp-agents/network/members)

## Overview

**Grasp Agents** is a modular Python framework for building agentic AI pipelines and applications. It is meant to be minimalistic but functional, allowing for rapid experimentation while keeping full and granular low-level control over prompting, LLM handling, and inter-agent communication by avoiding excessive higher-level abstractions.

## Features

- Clean formulation of agents as generic entities over I/O schemas and shared context.
- Transparent implementation of common agentic patterns:
  - Single-agent loops
  - Workflows (static communication topology), including loops
  - Agents-as-tools for task delegation
  - Freeform A2A communication via the in-process actor model
- Built-in parallel processing with flexible retries and rate limiting.
- Support for all popular API providers via LiteLLM.
- Granular event streaming with separate events for standard outputs, thinking, and tool calls.
- Callbacks via decorators or subclassing for straightforward customisation of agentic loops and context management.

## Project Structure

- `processors/`, `llm_agent.py`: Core processor and agent class implementations.
- `packet.py`, `packet_pool.py`, `runner.py`: Communication management.
- `llm_policy_executor.py`: LLM actions and tool call loops.
- `prompt_builder.py`: Tools for constructing prompts.
- `workflow/`: Modules for defining and managing static agent workflows.
- `llm.py`, `cloud_llm.py`: LLM integration and base LLM functionalities.
- `openai/`: Modules specific to OpenAI API integration.
- `litellm/`: Modules specific to LiteLLM integration.
- `memory.py`, `llm_agent_memory.py`: Basic agent memory management.
- `run_context.py`: Shared context management for agent runs.
- `usage_tracker.py`: Tracking of API usage and costs.
- `rate_limiting/`: Basic rate limiting tools.

## Quickstart & Installation Variants (UV Package manager)

> **Note:** You can check this sample project code in the [src/grasp_agents/examples/demo/uv](https://github.com/grasp-technologies/grasp-agents/tree/master/src/grasp_agents/examples/demo/uv) folder. Feel free to copy and paste the code from there to a separate project. There are also [examples](https://github.com/grasp-technologies/grasp-agents/tree/master/src/grasp_agents/examples/demo/) for other package managers.

#### 1. Prerequisites

Install the [UV Package Manager](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. Create Project & Install Dependencies

```bash
mkdir my-test-uv-app
cd my-test-uv-app
uv init .
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Add and sync dependencies:

```bash
uv add grasp_agents
uv sync
```

#### 3. Example Usage

Ensure you have a `.env` file with your OpenAI and Google AI Studio API keys set

```
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

Create a script, e.g., `problem_recommender.py`:

```python
import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from grasp_agents import LLMAgent, BaseTool, RunContext, Printer
from grasp_agents.litellm import LiteLLM, LiteLLMSettings


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
```

Run your script:

```bash
uv run problem_recommender.py
```

You can find more examples in [src/grasp_agents/examples/notebooks/agents_demo.ipynb](https://github.com/grasp-technologies/grasp-agents/tree/master/src/grasp_agents/examples/notebooks/agents_demo.ipynb).
