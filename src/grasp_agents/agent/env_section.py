"""
Environment-info ``SystemPromptSection`` factory.

Renders a small set of standard runtime facts the agent commonly benefits
from knowing — current date, platform, working directory, model name. Every
field is opt-in via the ``include`` set; all defaults are off-by-default-free
so the section produces a useful block out of the box.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .prompt_builder import SystemPromptSection

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..run_context import RunContext


ENV_INFO_SECTION_NAME = "env_info"

EnvField = str
"""One of: ``"date"``, ``"datetime"``, ``"platform"``, ``"os"``, ``"cwd"``,
``"model"``. Add new ones via ``extra_fields`` on the factory."""

DEFAULT_FIELDS: Final[tuple[EnvField, ...]] = (
    "date",
    "platform",
    "os",
    "cwd",
)


def _render_default_field(
    field: EnvField, model_name: str | None
) -> tuple[str, str] | None:
    if field == "date":
        return "Date", datetime.now(tz=UTC).date().isoformat()
    if field == "datetime":
        return "Datetime", datetime.now().astimezone().isoformat(timespec="seconds")
    if field == "platform":
        return "Platform", platform.system()
    if field == "os":
        return "OS", f"{platform.system()} {platform.release()}"
    if field == "cwd":
        return "CWD", str(Path(os.getcwd()))
    if field == "model":
        if model_name is None:
            return None
        return "Model", model_name
    return None


def make_env_info_section(
    *,
    include: Iterable[EnvField] = DEFAULT_FIELDS,
    model_name: str | None = None,
    extra_fields: dict[str, str] | None = None,
    section_name: str = ENV_INFO_SECTION_NAME,
    cache_break: bool = False,
) -> SystemPromptSection:
    """
    Build a section listing the requested environment facts.

    Default fields: ``date``, ``platform``, ``os``, ``cwd``. Pass an
    ``include`` iterable to choose explicitly; ``"datetime"`` and ``"model"``
    are off by default. ``extra_fields`` is a static label→value mapping
    appended verbatim — useful for application-specific facts (e.g.
    ``{"Project": "grasp-agents"}``).

    The compute is sync; the date is recomputed each turn but the rest is
    process-level constant. Set ``cache_break=True`` if you include
    ``datetime`` and want the section to break the prompt cache every turn.
    """
    selected = tuple(include)
    extra = dict(extra_fields or {})

    def compute(
        *, ctx: RunContext[Any] | None = None, exec_id: str | None = None
    ) -> str | None:
        del ctx, exec_id
        rows: list[str] = []
        for field in selected:
            entry = _render_default_field(field, model_name)
            if entry is None:
                continue
            label, value = entry
            rows.append(f"- {label}: {value}")
        for label, value in extra.items():
            rows.append(f"- {label}: {value}")
        if not rows:
            return None
        return "## Environment\n\n" + "\n".join(rows)

    return SystemPromptSection(
        name=section_name,
        compute=compute,
        cache_break=cache_break,
    )
