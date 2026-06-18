# Grasp Agents Hello App (Poetry)

## Overview

This is a simple hello world app for the Grasp Agents SDK.

### Install Python 3.11.9 via pyenv

```bash
brew install pyenv
```

```bash
pyenv install 3.11.9
```

```bash
pyenv local 3.11.9
```

### Make sure that Poetry Package manager is installed on your system. If you don't have it, install it using the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then open new terminal and proceed to the next step.

### Create a Virtual Environment

`python -m venv .venv`

### Activate the Virtual Environment

`source .venv/bin/activate`

### Make Poerty use Python from the virtual environment

```bash
poetry env use $(which python)
```

### Install the Grasp Agents SDK

```bash
poetry install
```

### Environment Variables

Ensure you have a `.env` file with your OpenAI API key set

```
OPENAI_API_KEY=your_openai_api_key
```

### Run the App

```bash
poetry run python hello.py
```
