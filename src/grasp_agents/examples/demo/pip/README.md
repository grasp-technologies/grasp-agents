# Grasp Agents Hello App (pip)

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

Then open new terminal and proceed to the next step.

### Create a Virtual Environment

`python -m venv .venv`

### Activate the Virtual Environment

`source .venv/bin/activate`

### Install the Grasp Agents SDK

`pip install -r requirements.txt`

### Environment Variables

Ensure you have a `.env` file with your OpenAI API key set

```
OPENAI_API_KEY=your_openai_api_key
```

### Run the App

```bash
python hello.py
```
