"""
Agent Skills — implementation of the agentskills.io specification.

A *skill* is a folder containing a ``SKILL.md`` file with YAML frontmatter
(``name`` + ``description`` required) and markdown instructions. Skills are
catalog-injected into the system prompt as an ``<available_skills>`` block;
the agent resolves a skill on demand via the ``load_skill`` tool. Slash-style
user invocations are rendered via :meth:`SkillRegistry.render_invocation`.

See ``docs/roadmap/12-skills.md`` for the design and the agentskills.io
specification at https://agentskills.io/specification.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from .injection import (
    render_available_skills_block,
    skills_system_prompt_section,
)
from .loader import discover_skills, load_skill_md, parse_skill_md
from .registry import SkillRegistry
from .slash import (
    ParsedSlashCommand,
    parse_named_args,
    parse_slash_command,
)
from .types import (
    Skill,
    SkillError,
    SkillFormatError,
    SkillFrontmatter,
    SkillNotFoundError,
)

if TYPE_CHECKING:
    # Lazy-loaded — these depend on ``function_tool`` (which transitively
    # imports ``RunContext``), and importing them at package load would
    # short-circuit the run-context construction during ``grasp_agents``
    # startup.
    from .tools import attach_skills, list_skills, load_skill


_LAZY_TOOLS = {"attach_skills", "list_skills", "load_skill"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_TOOLS:
        module = importlib.import_module(".tools", __name__)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ParsedSlashCommand",
    "Skill",
    "SkillError",
    "SkillFormatError",
    "SkillFrontmatter",
    "SkillNotFoundError",
    "SkillRegistry",
    "attach_skills",
    "discover_skills",
    "list_skills",
    "load_skill",
    "load_skill_md",
    "parse_named_args",
    "parse_skill_md",
    "parse_slash_command",
    "render_available_skills_block",
    "skills_system_prompt_section",
]
