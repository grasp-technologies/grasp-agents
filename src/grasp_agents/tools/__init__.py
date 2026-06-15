"""
Built-in tool packages for grasp-agents.

- :mod:`.file_edit` — ``Read`` / ``Write`` / ``Edit`` / ``Delete`` file tools.
- :mod:`.file_search` — ``Glob`` / ``Grep`` read-only search tools.
- :class:`FileToolkit` — one factory bundling all of the above.
- :class:`Bash` — run a shell command (fresh process) via ``ctx.exec_backend``.
- :class:`BashSession` — run a command in a persistent shell session.
- :class:`RunCell` — execute a notebook code cell in a live kernel.
- :class:`RunPython` — run ad-hoc Python in a live kernel (code-interpreter).
- :class:`KillTask` — stop any backgrounded tool call by its ``task_id``.

These are imported lazily (PEP 562) so importing :mod:`grasp_agents.tools`
doesn't pull in the file tools (and their ripgrep-availability checks) or the
exec stack eagerly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_tool import AgentTool, AgentToolInput, AgentToolPromptBuilder
    from .base import BaseTool, ToolProgressCallback
    from .bash import (
        Bash,
        BashInput,
        BashResult,
        bash_tools,
    )
    from .bash_session import BashSession
    from .code_interpreter import RunPython, RunPythonInput
    from .file_toolkit import FileToolkit
    from .function_tool import FunctionTool, function_tool
    from .notebook_exec import RunCell, RunCellInput
    from .processor_tool import ProcessorTool
    from .task_tools import KillTask


_LAZY: dict[str, str] = {
    "AgentTool": "agent_tool",
    "AgentToolInput": "agent_tool",
    "AgentToolPromptBuilder": "agent_tool",
    "BaseTool": "base",
    "ToolProgressCallback": "base",
    "FunctionTool": "function_tool",
    "function_tool": "function_tool",
    "ProcessorTool": "processor_tool",
    "Bash": "bash",
    "BashInput": "bash",
    "BashResult": "bash",
    "bash_tools": "bash",
    "BashSession": "bash_session",
    "RunPython": "code_interpreter",
    "RunPythonInput": "code_interpreter",
    "FileToolkit": "file_toolkit",
    "RunCell": "notebook_exec",
    "RunCellInput": "notebook_exec",
    "KillTask": "task_tools",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    attr = getattr(module, name)
    globals()[name] = attr  # cache for next access
    return attr


__all__ = [
    "AgentTool",
    "AgentToolInput",
    "AgentToolPromptBuilder",
    "BaseTool",
    "Bash",
    "BashInput",
    "BashResult",
    "BashSession",
    "FileToolkit",
    "FunctionTool",
    "KillTask",
    "ProcessorTool",
    "RunCell",
    "RunCellInput",
    "RunPython",
    "RunPythonInput",
    "ToolProgressCallback",
    "bash_tools",
    "function_tool",
]
