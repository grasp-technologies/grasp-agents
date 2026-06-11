"""
:class:`Selector` Protocol — relevance / catalog filter shared by skills,
memory, and any other catalog-style section.

This Protocol lives in its own module so it can be imported from
``run_context`` (via ``memory.provider``) without dragging the rest of
``types.hooks`` (which itself imports ``run_context``) into the cycle.
Annotations use ``from __future__ import annotations`` so the
``RunContext`` reference is deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence

    from ..run_context import RunContext
    from .items import InputItem


class Selector[T](Protocol):
    """
    Selection / relevance hook for catalog-style sections (skills, memory).

    Implementations receive the full ``entries`` sequence plus optional
    context (``ctx``, ``exec_id``, and the running agent's ``messages``
    transcript) and return a possibly shorter / reordered sequence. Sync
    or async. For implementations that don't need a particular kwarg,
    absorb the rest with ``**_: Any``::

        def keep_alpha(*, entries, **_) -> Sequence[Skill]:
            return [s for s in entries if s.name == "alpha"]

    The ``messages`` kwarg carries the agent's per-run transcript so a
    relevance selector can extract the current user query, recently-used
    tools, or any other signal it cares about.
    """

    def __call__(
        self,
        *,
        entries: Sequence[T],
        ctx: RunContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
    ) -> Sequence[T] | Awaitable[Sequence[T]]: ...
