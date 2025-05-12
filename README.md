# Grasp Agents

<br/>
<img src="./.assets/grasp.svg" alt="Grasp Agents" width="320" />
<br/>
<br/>

[![PyPI version](https://badge.fury.io/py/grasp_agents.svg)](https://badge.fury.io/py/grasp-agents)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](https://mit-license.org/)
[![PyPI downloads](https://img.shields.io/pypi/dm/grasp-agents?style=flat-square)](https://pypi.org/project/grasp-agents/)
[![GitHub Stars](https://img.shields.io/github/stars/grasp-technologies/grasp-agents?style=social)](https://github.com/grasp-technologies/grasp-agents/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/grasp-technologies/grasp-agents?style=social)](https://github.com/grasp-technologies/grasp-agents/network/members)

## Overview

**Grasp Agents** is a modular Python framework for building agentic AI pipelines and applications. It provides reusable agent classes, message handling, LLM integration, memory, and orchestration utilities. The framework is designed for flexibility, composability, and clarity, enabling rapid prototyping and robust development of multi-agent systems.

## Features

- Modular agent base classes
- Message and memory management
- LLM and tool orchestration
- Logging and usage tracking
- Extensible architecture

## Project Structure

- `src/grasp_agents/` — Core framework modules
  - `base_agent.py`, `llm_agent.py`, `comm_agent.py`: Agent classes
  - `agent_message.py`, `agent_message_pool.py`: Messaging
  - `memory.py`: Memory management
  - `cloud_llm.py`, `llm.py`: LLM integration
  - `tool_orchestrator.py`: Tool orchestration
  - `usage_tracker.py`, `grasp_logging.py`: Usage and logging
  - `data_retrieval/`, `openai/`, `typing/`, `workflow/`: Extensions and utilities
- `configs/` — Configuration files
- `data/` — Logs and datasets

## Quickstart & Installation Variants (UV Package manager)

### Option 1: UV Package Manager Project

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

Create a file, e.g., `hello.py`:

Ensure you have a `.env` file with your OpenAI and Google AI Studio API keys set

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_AI_STUDIO_API_KEY=your_google_ai_studio_api_key
```

```python
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

from dotenv import load_dotenv

load_dotenv()

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
```

Run your script:

```bash
uv run hello.py
```
