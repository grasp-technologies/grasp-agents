from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .types import Skill, SkillFormatError, SkillFrontmatter

logger = logging.getLogger(__name__)

SKILL_FILE_NAME = "SKILL.md"

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?(.*)\Z",
    re.DOTALL,
)


def parse_skill_md(
    text: str, *, path: Path | None = None
) -> tuple[SkillFrontmatter, str]:
    """
    Parse a ``SKILL.md`` source string into ``(frontmatter, body)``.

    Raises :class:`SkillFormatError` on missing frontmatter delimiters,
    invalid YAML, or schema-violating frontmatter.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise SkillFormatError(
            path,
            "SKILL.md must begin with a YAML frontmatter block delimited by '---'",
        )

    raw_yaml, body = match.group(1), match.group(2)

    try:
        loaded: Any = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise SkillFormatError(path, f"Invalid YAML frontmatter: {exc}") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise SkillFormatError(
            path, "Frontmatter must be a YAML mapping (key/value pairs)"
        )

    try:
        frontmatter = SkillFrontmatter.model_validate(loaded)
    except ValidationError as exc:
        raise SkillFormatError(
            path, f"Frontmatter validation failed: {exc}"
        ) from exc

    return frontmatter, body.strip("\n")


def load_skill_md(path: Path) -> Skill:
    """Load a single ``SKILL.md`` file. Verifies parent dir name == frontmatter name."""
    if not path.is_file():
        raise SkillFormatError(path, "SKILL.md not found or not a regular file")

    text = path.read_text(encoding="utf-8")
    frontmatter, body = parse_skill_md(text, path=path)

    parent_name = path.parent.name
    if parent_name != frontmatter.name:
        raise SkillFormatError(
            path,
            f"Frontmatter name {frontmatter.name!r} must match parent directory "
            f"name {parent_name!r}",
        )

    return Skill(frontmatter=frontmatter, body=body, path=path)


def discover_skills(source: Path | str) -> list[Skill]:
    """
    Resolve ``source`` to a list of :class:`Skill` objects.

    ``source`` may be:
    - a path to a ``SKILL.md`` file
    - a path to a single skill directory (containing ``SKILL.md``)
    - a path to a parent directory containing one or more skill subdirectories,
      each with their own ``SKILL.md``

    Skills that fail to load are logged and skipped. The caller decides how
    strict to be about that — :func:`load_skill_md` raises directly when given
    a single explicit path.
    """
    path = Path(source).expanduser()

    if path.is_file():
        return [load_skill_md(path)]

    if not path.is_dir():
        raise SkillFormatError(path, "Skill source path does not exist")

    direct = path / SKILL_FILE_NAME
    if direct.is_file():
        return [load_skill_md(direct)]

    skills: list[Skill] = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / SKILL_FILE_NAME
        if not skill_md.is_file():
            continue
        try:
            skills.append(load_skill_md(skill_md))
        except SkillFormatError:
            logger.exception("Failed to load skill at %s", skill_md)

    return skills
