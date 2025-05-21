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

- Clean formulation of agents as generic entities over:
  - I/O schemas
  - Agent state
  - Shared context
- Transparent implementation of common agentic patterns:
  - Single-agent loops with an optional "ReAct mode" to enforce reasoning between the tool calls
  - Workflows (static communication topology), including loops
  - Agents-as-tools for task delegation
  - Freeform A2A communication via the in-process actor model
- Batch processing support outside of agentic loops
- Simple logging and usage/cost tracking

## Project Structure

- `base_agent.py`, `llm_agent.py`, `comm_agent.py`: Core agent class implementations.
- `agent_message.py`, `agent_message_pool.py`: Messaging and message pool management.
- `llm_agent_state.py`: State management for LLM agents.
- `tool_orchestrator.py`: Orchestration of tools used by agents.
- `prompt_builder.py`: Tools for constructing prompts.
- `workflow/`: Modules for defining and managing agent workflows.
- `cloud_llm.py`, `llm.py`: LLM integration and base LLM functionalities.
- `openai/`: Modules specific to OpenAI API integration.
- `memory.py`: Memory management for agents (currently only message history).
- `run_context.py`: Context management for agent runs.
- `usage_tracker.py`: Tracking of API usage and costs.
- `costs_dict.yaml`: Dictionary for cost tracking (update if needed).
- `rate_limiting/`: Basic rate limiting tools.

## Quickstart & Installation Variants (UV Package manager)

> **Note:** You can check this sample project code in the [src/grasp_agents/examples/demo/uv](src/grasp_agents/examples/demo/uv) folder. Feel free to copy and paste the code from there to a separate project. There are also [examples](src/grasp_agents/examples/demo/) for other package managers.

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
GOOGLE_AI_STUDIO_API_KEY=your_google_ai_studio_api_key
```

Create a script, e.g., `problem_recommender.py`:

```python
import asyncio
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from grasp_agents.grasp_logging import setup_logging
from grasp_agents.llm_agent import LLMAgent
from grasp_agents.openai.openai_llm import OpenAILLM, OpenAILLMSettings
from grasp_agents.run_context import RunContextWrapper
from grasp_agents.typing.message import Conversation
from grasp_agents.typing.tool import BaseTool

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
        self, inp: TeacherQuestion, ctx: RunContextWrapper[Any] | None = None
    ) -> StudentReply:
        return input(inp.question)


Problem = str


teacher = LLMAgent[Any, Problem, None](
    agent_id="teacher",
    llm=OpenAILLM(
        model_name="gpt-4.1",
        api_provider="openai",
        llm_settings=OpenAILLMSettings(temperature=0.1),
    ),
    tools=[AskStudentTool()],
    max_turns=20,
    react_mode=True,
    sys_prompt=sys_prompt_react,
    set_state_strategy="reset",
)


@teacher.exit_tool_call_loop_handler
def exit_tool_call_loop(
    conversation: Conversation, ctx: RunContextWrapper[Any] | None, **kwargs: Any
) -> bool:
    return r"<PROBLEM>" in str(conversation[-1].content)


@teacher.parse_output_handler
def parse_output(
    conversation: Conversation, ctx: RunContextWrapper[Any] | None, **kwargs: Any
) -> Problem:
    message = str(conversation[-1].content)
    matches = re.findall(r"<PROBLEM>(.*?)</PROBLEM>", message, re.DOTALL)

    return matches[0]


async def main():
    ctx = RunContextWrapper[None](print_messages=True)
    out = await teacher.run(ctx=ctx)
    print(out.payloads[0])
    print(ctx.usage_tracker.total_usage)


asyncio.run(main())
```

Run your script:

```bash
uv run problem_recommender.py
```

You can find more examples in [src/grasp_agents/examples/notebooks/agents_demo.ipynb](src/grasp_agents/examples/notebooks/agents_demo.ipynb).
