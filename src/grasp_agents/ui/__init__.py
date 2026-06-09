"""
User-interface surfaces for grasp-agents event streams.

Two front-ends over one renderer (:mod:`._event_render`, Textual-free):

* :class:`.console.EventConsole` / :func:`.console.stream_events` — a light,
  linear ANSI stream that works in any terminal, a pipe, or a notebook. Needs
  only ``rich`` (a core dependency).
* :mod:`.app` — a full-screen Textual app (monitor / interactive); :mod:`.notebook`
  — inline-render / screenshot helpers for notebooks. These need the ``tui``
  extra: ``pip install 'grasp_agents[tui]'``.

The console symbols are imported eagerly; the Textual ``app`` / ``notebook``
names are resolved lazily so ``import grasp_agents.ui`` never pulls Textual into
the dependency graph. (``render_events_inline`` is the exception — it's
Textual-free, so it works from a bare install.)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .console import EventConsole, stream_events

if TYPE_CHECKING:
    from .app import GraspAgentsApp, run_tui, run_tui_interactive
    from .notebook import display_screenshot, render_events_inline, screenshot

__all__ = [
    "EventConsole",
    "GraspAgentsApp",
    "display_screenshot",
    "render_events_inline",
    "run_tui",
    "run_tui_interactive",
    "screenshot",
    "stream_events",
]

# Name → module it lives in; resolved on first access so the Textual modules
# load only when actually used.
_LAZY = {
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
            "grasp_agents.ui's Textual UI requires the 'tui' extra: "
            "pip install 'grasp_agents[tui]'"
        ) from exc
    return getattr(mod, name)
