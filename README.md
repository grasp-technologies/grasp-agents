# grasp_agents

<br/>
<img src="./.assets/grasp.svg" alt="Grasp Agents" width="320" />
<br/>
<br/>

[![PyPI version](https://badge.fury.io/py/grasp_agents.svg)](https://badge.fury.io/py/grasp_agents)
[![Python Versions](https://img.shields.io/pypi/pyversions/grasp_agents?style=flat-square)](https://pypi.org/project/grasp_agents/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

## Overview

grasp-agents is a modular Python framework for building agentic AI pipelines and applications. It provides reusable agent classes, message handling, LLM integration, memory, and orchestration utilities. The framework is designed for flexibility, composability, and clarity, enabling rapid prototyping and robust development of multi-agent systems.

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

## Quickstart & Installation Variants

### Option 1: UV Package Manager Project

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

```python
from grasp_agents.llm_agent import LLMAgent
from grasp_agents.base_agent import BaseAgentConfig

agent = LLMAgent(
    config=BaseAgentConfig(
        model="gpt-4o-mini",
        memory=None,  # or your memory implementation
    )
)

response = agent.run("Hello, agent!")
print(response)
```

Run your script:

```bash
python hello.py
```

---

### Option 2: PIP-only (requirements.txt-based) Project

#### 1. Create Project Folder

```bash
mkdir my-test-pip-app
cd my-test-pip-app
```

#### 2. Install Python 3.11.9 (Recommended)

If using [pyenv](https://github.com/pyenv/pyenv):

```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9
```

Open a new terminal after setting the Python version.

#### 3. Create & Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

If you see `ModuleNotFoundError: No module named 'yaml'`, run:

```bash
pip install pyyaml
```

#### 4. Install Grasp Agents SDK

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install grasp-agents
```

#### 5. Example Usage

Create a file, e.g., `hello.py`, and use the same code as above.

#### 6. Run the App

```bash
python hello.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Development

To develop and test the library locally, follow these steps:

### 1. Install UV Package Manager

Make sure [UV](https://github.com/astral-sh/uv) is installed on your system:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

Create a new virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 3. Test Example

- Install the Jupyter Notebook extension for VS Code.
- Open `src/grasp_agents/examples/notebooks/agents_demo.ipynb` in VS Code.
- Ensure you have a `.env` file with your OpenAI and Google AI Studio API keys set (see `.env.example`).

You're now ready to run and experiment with the example notebook.
