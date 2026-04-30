from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from .loader import discover_skills
from .types import SkillNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ..run_context import RunContext
    from .types import Skill

logger = logging.getLogger(__name__)


SkillFilter: TypeAlias = Callable[
    ...,
    Sequence["Skill"] | Awaitable[Sequence["Skill"]],
]

INVOCATION_WRAPPER = "[SYSTEM: user invoked skill {name}]"

_NAMED_ARG_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


class SkillRegistry:
    """
    In-memory registry keyed by skill name.

    Frozen at session start. Newly-authored skills are picked up only on the
    next session's rescan; mid-session edits to *existing* skill bodies are
    visible on the next ``load_skill`` call (the tool re-reads the file).

    Optionally carries a relevance filter (:meth:`set_filter`) that the
    skills system-prompt section consults before rendering the catalog.
    """

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._by_name: dict[str, Skill] = {}
        self._sources: list[Path] = []
        self._filter: SkillFilter | None = None
        for skill in skills:
            self.register(skill)

    @classmethod
    def from_path(cls, source: Path | str) -> SkillRegistry:
        """Walk ``source`` for skills via :func:`discover_skills`."""
        registry = cls()
        registry.add_source(source)
        return registry

    @property
    def sources(self) -> list[Path]:
        return list(self._sources)

    def add_source(self, source: Path | str) -> list[Skill]:
        """
        Register every skill found under ``source`` and remember the path.

        Subsequent :meth:`refresh` calls re-walk all known sources.
        """
        path = Path(source).expanduser().resolve()
        if path not in self._sources:
            self._sources.append(path)
        added: list[Skill] = []
        for skill in discover_skills(path):
            self.register(skill)
            added.append(skill)
        return added

    def refresh(self) -> None:
        """Re-walk all known sources and replace the catalog."""
        if not self._sources:
            return
        new_by_name: dict[str, Skill] = {}
        for source in self._sources:
            for skill in discover_skills(source):
                if skill.name in new_by_name:
                    existing = new_by_name[skill.name]
                    if existing.path != skill.path:
                        logger.warning(
                            "Skill name %r is registered twice; later "
                            "definition at %s overrides earlier at %s",
                            skill.name,
                            skill.path,
                            existing.path,
                        )
                new_by_name[skill.name] = skill
        self._by_name = new_by_name

    def register(self, skill: Skill) -> None:
        existing = self._by_name.get(skill.name)
        if existing is not None and existing.path != skill.path:
            logger.warning(
                "Skill name %r is registered twice; later definition at %s "
                "overrides earlier at %s",
                skill.name,
                skill.path,
                existing.path,
            )
        self._by_name[skill.name] = skill

    def get(self, name: str) -> Skill:
        skill = self._by_name.get(name)
        if skill is None:
            raise SkillNotFoundError(f"Skill not found: {name!r}")
        return skill

    def get_optional(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    @property
    def all(self) -> list[Skill]:
        return list(self._by_name.values())

    @property
    def visible(self) -> list[Skill]:
        """Skills the LLM can see and call (model_invocation not disabled)."""
        return [s for s in self._by_name.values() if not s.disable_model_invocation]

    # ---- Relevance filter ----------------------------------------------------

    def set_filter(self, fn: SkillFilter | None) -> None:
        """
        Register a relevance filter consulted by the catalog renderer.

        ``fn(skills, ctx, exec_id)`` receives the visible skills and returns
        a (possibly smaller, possibly reordered) subsequence. Sync or async.
        Pass ``None`` to clear.
        """
        self._filter = fn

    @property
    def filter(self) -> SkillFilter | None:
        return self._filter

    async def apply_filter(
        self,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
    ) -> list[Skill]:
        """Run the filter (if any) and return the resulting skills."""
        skills = self.visible
        if self._filter is None:
            return skills
        result = self._filter(skills=skills, ctx=ctx, exec_id=exec_id)
        if inspect.isawaitable(result):
            result = await result
        return list(result)

    # ---- Slash-command rendering --------------------------------------------

    def render_invocation(
        self,
        name: str,
        *,
        args: str | Mapping[str, str] | None = None,
        wrap: bool = True,
    ) -> str:
        """
        Render a skill body as a user-message string.

        Substitutes ``$ARGUMENTS`` (full args string) and ``$ARG_NAME``
        (named arguments, when ``args`` is a mapping). When ``wrap`` is true
        (default), prepends ``[SYSTEM: user invoked skill <name>]`` so the
        agent can distinguish a slash-command turn from a raw user message.
        """
        skill = self.get(name)
        body = _substitute_args(skill.body, args)
        if not wrap:
            return body
        return f"{INVOCATION_WRAPPER.format(name=name)}\n\n{body}"


def _substitute_args(
    body: str, args: str | Mapping[str, str] | None
) -> str:
    if args is None:
        full = ""
        named: Mapping[str, str] = {}
    elif isinstance(args, str):
        full = args
        named = {}
    else:
        full = " ".join(f"{k}={v}" for k, v in args.items())
        named = args

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "ARGUMENTS":
            return full
        return named.get(key, match.group(0))

    return _NAMED_ARG_RE.sub(_replace, body)
