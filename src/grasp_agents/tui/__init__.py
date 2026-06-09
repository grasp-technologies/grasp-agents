"""
Local Textual UI for monitoring and debugging grasp-agents runs.

Requires the ``tui`` extra::

    pip install 'grasp_agents[tui]'

Terminal — monitor a stream::

    from grasp_agents.tui import run_tui
    run_tui(agent.run_stream("…", ctx=ctx))

Terminal — interactive (type messages, the agent runs each turn)::

    from grasp_agents.tui import run_tui_interactive
    run_tui_interactive(agent.run_stream, main_agent=agent.name)

Notebook (inline rendering / SVG snapshot)::

    from grasp_agents.tui import render_events_inline, display_screenshot

``render_events_inline`` needs only ``rich`` (no Textual); the rest need the
``tui`` extra. Imports are lazy so the notebook helper works without Textual.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import GraspAgentsApp, run_tui, run_tui_interactive
    from .notebook import display_screenshot, render_events_inline, screenshot

__all__ = [
    "GraspAgentsApp",
    "display_screenshot",
    "render_events_inline",
    "run_tui",
    "run_tui_interactive",
    "screenshot",
]

_LAZY: dict[str, str] = {
    "GraspAgentsApp": "app",
    "run_tui": "app",
    "run_tui_interactive": "app",
    "render_events_inline": "notebook",
    "screenshot": "notebook",
    "display_screenshot": "notebook",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = import_module(f"{__name__}.{module}")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "grasp_agents.tui requires the 'tui' extra: pip install 'grasp_agents[tui]'"
        ) from exc
    return getattr(mod, name)
