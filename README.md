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

**Grasp Agents** is a modular Python framework for building agentic AI pipelines and applications. It is meant to be minimalistic but functional, allowing for rapid experimentation while keeping full and granular low-level control over prompting, LLM handling, tool call loops, and inter-agent communication by avoiding excessive higher-level abstractions.

## Features

- Clean formulation of agents as generic entities over I/O schemas and shared context.
- Transparent implementation of common agentic patterns:
  - Single-agent loops
  - Workflows (static communication topology), including loops
  - Agents-as-tools for task delegation
  - Freeform A2A communication via the in-process actor model
- Built-in parallel processing with flexible retries and rate limiting.
- Support for all popular API providers via LiteLLM.
- Granular event streaming with separate events for LLM responses, thinking, and tool calls.
- Callbacks via decorators or subclassing for straightforward customisation of agentic loops and context management.

## Project Structure

- `processors/`, `agent/llm_agent.py`: Core processor and agent class implementations.
- `runner/`: Communication management and multi-agent orchestration (`Runner`, `EventBus`).
- `agent/agent_loop.py`: The agentic loop and tool-call handling.
- `agent/prompt_builder.py`: Tools for constructing prompts.
- `workflow/`: Modules for defining and managing static agent workflows.
- `llm/`: Base LLM abstractions (`LLM`, `CloudLLM`, `FallbackLLM`, resilience).
- `llm_providers/`: Per-provider integrations (OpenAI, Anthropic, Gemini, LiteLLM).
- `memory/`, `agent/llm_agent_transcript.py`: Cross-session memory and per-run conversation transcript.
- `tools/`, `sandbox/`: Built-in tools (file/terminal/code/notebook) and sandboxed execution.
- `run_context.py`: Shared run context management.

## Usage

### Installation

Assuming your project manages dependencies through [uv](https://docs.astral.sh/uv/).

```bash
uv add grasp_agents
uv sync
```

You can of course also install using other managers like poetry or simply pip.

We recommend you use [dotenv](https://pypi.org/project/python-dotenv/) to automatically set enviroment variables from a `.env` file containting the necessary API keys, e.g.,

```
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Try it out

#### Jupyter Notebook Example
[Notebook Link](https://github.com/grasp-technologies/grasp-agents/blob/master/src/grasp_agents/examples/notebooks/agents_demo.ipynb)

#### A Grasp-Agents Powered Web App
[https://grasp.study/](https://grasp.study/)

#### Script Example 

Create a script, e.g., `problem_recommender.py`:

```python
import asyncio
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from grasp_agents import (
    BaseTool,
    LLMAgent,
    ProcPacketOutEvent,
    print_event_stream,
)
from grasp_agents.llm_providers.litellm import LiteLLM, LiteLLMSettings

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

    async def _run(self, inp: TeacherQuestion, **kwargs: Any) -> StudentReply:
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
    stream_llm=True,
)

async def main():
    async for event in print_event_stream(teacher.run_stream("start")):
        if isinstance(event, ProcPacketOutEvent):
            result = event.data.payloads[0]
            print(f"\n<Suggested Problem>:\n\n{result.problem}\n")

asyncio.run(main())
```

Run your script:

```bash
uv run problem_recommender.py
```

You can find more examples in [src/grasp_agents/examples/notebooks/agents_demo.ipynb](https://github.com/grasp-technologies/grasp-agents/tree/master/src/grasp_agents/examples/notebooks/agents_demo.ipynb).
