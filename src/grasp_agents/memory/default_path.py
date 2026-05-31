"""
:func:`default_memdir_path` and the memdir-layout constants.

Resolves the project-local memdir path: ``$GRASP_MEMORY_DIR`` if set,
else ``~/.grasp/projects/<sanitized-cwd>/memory/``. Sanitization
NFC-normalizes and replaces separators / unsafe chars with underscores
so the per-project memdir tree is filesystem-safe.

Lives in its own module so backends + the unified
:class:`MemoryProvider` can both import the constants without depending
on each other's I/O layer.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

GRASP_MEMORY_ENV = "GRASP_MEMORY_DIR"
GRASP_HOME_DIR_NAME = ".grasp"
PROJECTS_DIR_NAME = "projects"
MEMDIR_DIR_NAME = "memory"


def default_memdir_path(cwd: Path | None = None) -> Path:
    """
    Resolve the default memdir path for the current project.

    Resolution order:
    1. ``GRASP_MEMORY_DIR`` environment variable (full path, no expansion).
    2. ``~/.grasp/projects/<sanitized-cwd>/memory/``.
    """
    override = os.environ.get(GRASP_MEMORY_ENV)
    if override:
        return Path(override)
    base = Path.home() / GRASP_HOME_DIR_NAME / PROJECTS_DIR_NAME
    sanitized = _sanitize_path((cwd or Path.cwd()).resolve())

    return base / sanitized / MEMDIR_DIR_NAME


def _sanitize_path(path: Path) -> str:
    """NFC-normalize, replace path separators and unsafe chars with underscores."""
    text = unicodedata.normalize("NFC", str(path))
    text = text.replace("/", "_").replace("\\", "_")
    text = text.replace(":", "_").replace("\x00", "_")
    return text.lstrip("_") or "default"


__all__ = [
    "GRASP_HOME_DIR_NAME",
    "GRASP_MEMORY_ENV",
    "MEMDIR_DIR_NAME",
    "PROJECTS_DIR_NAME",
    "default_memdir_path",
]
