# Grasp Agents Hello App (PIP only)

## Overview

This is a simple hello world app for the Grasp Agents SDK.

### Make sure that UV Package manager is installed on your system.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create a new virtual environment using UV:

```bash
uv venv
source .venv/bin/activate
```

### Install dependencies

```bash
uv sync
```

### Environment Variables

Ensure you have a `.env` file with your OpenAI API key set

```
OPENAI_API_KEY=your_openai_api_key
```

Run the app

```bash
uv run hello.py
```
