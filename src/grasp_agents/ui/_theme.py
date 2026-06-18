"""Color themes for the Textual UI, and last-used-theme persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from textual.theme import Theme

GRASP_DARK = Theme(
    name="grasp-dark",
    primary="#AAACFA",
    secondary="#BEE4F7",
    accent="#BFB53B",
    success="#3BBF69",
    warning="#BFB53B",
    error="#FCA9A9",
    background="#0F0F0F",
    surface="#1A1A1A",
    panel="#222222",
    dark=True,
)
GRASP_LIGHT = Theme(
    name="grasp-light",
    primary="#5B5FD6",
    secondary="#2F7FB0",
    accent="#8A7F10",
    success="#1F8F4D",
    warning="#8A7F10",
    error="#C0392B",
    background="#FAFAFA",
    surface="#EEEEEE",
    panel="#E2E2E2",
    dark=False,
)

DEFAULT_THEME = "catppuccin-macchiato"


def _theme_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "grasp-agents" / "tui.json"


def load_saved_theme() -> str | None:
    """The theme the user last selected, persisted across launches."""
    try:
        data: Any = json.loads(_theme_config_path().read_text())
        name = data.get("theme")
    except Exception:
        return None
    return name if isinstance(name, str) else None


def save_theme(name: str) -> None:
    path = _theme_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"theme": name}))
    except Exception:
        pass
