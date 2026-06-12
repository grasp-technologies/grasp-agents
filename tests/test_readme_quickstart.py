"""
Guard the README quickstart against drift.

Extracts the first ``python`` fenced block from ``README.md`` and executes it
(minus the final ``asyncio.run(main())``, so no LLM call is made). This catches
dead imports, removed APIs, and broken construction — the exact failure modes
that let the published quickstart rot.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def _first_python_block(md: str) -> str:
    match = re.search(r"```python\n(.*?)```", md, re.DOTALL)
    if match is None:
        pytest.fail("README.md has no ```python code block to smoke-test.")
    return match.group(1)


def test_readme_quickstart_runs() -> None:
    code = _first_python_block(README.read_text(encoding="utf-8"))

    # Drop the lines that would do real I/O: launching the agent (hits the
    # network) and dotenv (needs a .env). Everything else — imports, the tool
    # subclass, the agent construction — runs and is asserted below.
    lines = [
        line
        for line in code.splitlines()
        if not line.startswith(("asyncio.run(", "load_dotenv("))
    ]
    namespace: dict[str, object] = {}
    exec("\n".join(lines), namespace)  # noqa: S102

    from grasp_agents import BaseTool, LLMAgent

    teacher = namespace["teacher"]
    assert isinstance(teacher, LLMAgent)

    tool_cls = namespace["AskStudentTool"]
    assert issubclass(tool_cls, BaseTool)
    tool = tool_cls()
    # The typed-args schema the LLM sees must expose the declared field.
    assert "question" in tool.in_type.model_fields
