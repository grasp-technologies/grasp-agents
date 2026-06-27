from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from grasp_agents.context.system_reminder import (
    SYSTEM_REMINDER_TAG,
    wrap_in_system_reminder,
)
from grasp_agents.selector import Selector

from .loader import discover_skills
from .types import SkillNotFoundError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from grasp_agents.run_context import RunContext
    from grasp_agents.types.items import InputItem

    from .types import Skill

logger = logging.getLogger(__name__)


type SkillSelector = Selector[Skill]
"""Relevance selector for the skills catalog. See :class:`Selector`."""

# A user-invoked skill (slash-command) turn is wrapped in a <system-reminder> so
# the agent — and any UI — can tell it apart from a raw user message. This subject
# template is the single source of truth for both rendering the wrapper (via
# ``wrap_in_system_reminder``) and matching it (``_INVOCATION_RE``).
_SKILL_INVOCATION_SUBJECT = "user invoked skill {name}"

_INVOCATION_RE = re.compile(
    rf'^<{SYSTEM_REMINDER_TAG} subject="'
    + _SKILL_INVOCATION_SUBJECT.format(name=r'(?P<name>[^"]+)')
    + '">'
)

_NAMED_ARG_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def match_invocation_wrapper(text: str) -> str | None:
    """
    The skill name if ``text`` is a skill-invocation message, else ``None``.

    Inverse of :meth:`SkillRegistry.render_invocation` (``wrap=True``): lets a UI
    tell a slash-command turn from a raw user message and render it accordingly.
    """
    match = _INVOCATION_RE.match(text.lstrip())
    return match.group("name") if match else None


class SkillRegistry:
    """
    In-memory registry keyed by skill name.

    Frozen at session start. Newly-authored skills are picked up only on the
    next session's rescan; mid-session edits to *existing* skill bodies are
    visible on the next ``load_skill`` call (the tool re-reads the file).

    Optionally carries a relevance selector (:meth:`set_selector`) that the
    skills system-prompt section consults before rendering the catalog.
    """

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._by_name: dict[str, Skill] = {}
        # Skills registered directly (ctor / ``register``) rather than
        # discovered under a source path — preserved across ``refresh``.
        self._programmatic: dict[str, Skill] = {}
        self._sources: list[Path] = []
        self._selector: SkillSelector | None = None
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
            self._register_discovered(skill)
            added.append(skill)
        return added

    def refresh(self) -> None:
        """
        Re-walk all known sources and replace the discovered catalog.

        Programmatically-registered skills are kept. A source that can no
        longer be walked (vanished directory, permission change) is skipped
        with a warning rather than failing the whole refresh.
        """
        if not self._sources:
            return
        new_by_name: dict[str, Skill] = dict(self._programmatic)
        for source in self._sources:
            try:
                discovered = discover_skills(source)
            except Exception:
                logger.warning(
                    "Skipping skill source %s during refresh", source, exc_info=True
                )
                continue
            for skill in discovered:
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
        self._register_discovered(skill)
        self._programmatic[skill.name] = skill

    def _register_discovered(self, skill: Skill) -> None:
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

    # ---- Catalog selector ----------------------------------------------------

    def set_selector(self, fn: SkillSelector | None) -> None:
        """
        Register a relevance selector consulted by the catalog renderer.

        See :class:`Selector` for the call shape. Pass ``None`` to clear.
        """
        self._selector = fn

    @property
    def selector(self) -> SkillSelector | None:
        return self._selector

    async def select_relevant(
        self,
        *,
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
    ) -> list[Skill]:
        """Run the selector (if any) and return the resulting skills."""
        skills = self.visible
        if self._selector is None:
            return skills
        result = self._selector(
            entries=skills, ctx=ctx, exec_id=exec_id, messages=messages
        )
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

        A skill body is a self-contained procedure. If it declares explicit
        injection points — ``$ARGUMENTS`` or named ``$ARG_NAME`` placeholders —
        the user's args are substituted into them; otherwise the args are
        appended as a labelled ``User input:`` block, leaving the procedure
        untouched. When ``wrap`` is true (default), the whole message is enclosed
        in ``<system-reminder subject="user invoked skill <name>">…</system-reminder>``
        so the agent can tell a slash-command turn from a raw user message.
        """
        skill = self.get(name)
        body = apply_invocation_args(skill.body, args)
        if not wrap:
            return body
        return wrap_in_system_reminder(
            body, subject=_SKILL_INVOCATION_SUBJECT.format(name=name)
        )


def substitute_args(body: str, args: str | Mapping[str, str] | None) -> str:
    """
    Substitute ``$ARGUMENTS`` / ``$ARG_NAME`` placeholders in a skill body.

    ``args=None`` (a model-invoked load, with no arguments) resolves
    ``$ARGUMENTS`` to empty rather than leaving the literal token.
    """
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


def _has_injection_point(body: str, args: str | Mapping[str, str] | None) -> bool:
    """True if the body declares a placeholder the given args would fill."""
    names = {m.group(1) for m in _NAMED_ARG_RE.finditer(body)}
    if "ARGUMENTS" in names:
        return True
    if args is not None and not isinstance(args, str):
        return bool(names & args.keys())  # named-args mapping
    return False


def apply_invocation_args(body: str, args: str | Mapping[str, str] | None) -> str:
    """
    Fill a skill body with a user's slash-command arguments.

    A skill body is a self-contained procedure. If it declares explicit
    injection points — ``$ARGUMENTS`` or named ``$ARG_NAME`` placeholders that
    match the supplied mapping — substitute them. Otherwise append the args as a
    labelled ``User input:`` block, leaving the procedure untouched (so a
    placeholder-free skill reads the same whether the content arrives inline
    here or sits in the conversation on a model-invoked load). With no args,
    return the body unchanged.
    """
    if _has_injection_point(body, args):
        return substitute_args(body, args)
    if args is None:
        return body
    full = (
        args
        if isinstance(args, str)
        else " ".join(f"{k}={v}" for k, v in args.items())
    )
    if not full.strip():
        return body
    return f"{body.rstrip()}\n\nUser input:\n{full}"


def strip_arg_placeholders(body: str) -> str:
    """
    Remove slash-command argument placeholders for a model-invoked load.

    ``$ARGUMENTS`` / ``$ARG_NAME`` are filled from the *user's* typed
    slash-command input. A model-invoked ``load_skill`` has no such input — the
    material the skill applies to is the user's request already in the
    conversation, not text injected inline — so the placeholders are dropped
    (along with any line they leave blank) rather than resolved to an empty
    inline slot. Author skills with the placeholder on its own line for the
    cleanest result.
    """
    kept: list[str] = []
    for line in body.splitlines():
        stripped = _NAMED_ARG_RE.sub("", line)
        if not stripped.strip() and line.strip():
            continue  # the line held only placeholder(s) — drop it
        kept.append(stripped)
    return "\n".join(kept).rstrip()
