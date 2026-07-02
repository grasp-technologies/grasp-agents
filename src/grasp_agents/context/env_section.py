"""
Environment-info injectors.

Two ways to give the agent ambient runtime facts:

* :func:`make_env_info_section` — a cache-stable ``SystemPromptSection`` of
  process-level facts (date, platform, cwd, model). Belongs in the system
  prompt; a per-turn ``datetime`` field would only fragment the cache.
* :func:`make_current_time_attachment` — a live wall-clock stamp as an
  ``InputAttachment`` on the *input* message, so it never invalidates the
  cached prompt prefix. This is how you make an agent time-aware.
"""

from __future__ import annotations

import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from .prompt_builder import InputAttachment, SystemPromptSection

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from grasp_agents.agent.agent_context import AgentContext
    from grasp_agents.session_context import SessionContext
    from grasp_agents.types.content import CacheControl
    from grasp_agents.types.items import InputItem, InputMessageItem


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


def _effective_cwd(ctx: SessionContext[Any] | None) -> str:
    env = getattr(ctx, "environment", None)
    policy = getattr(env, "policy", None)
    roots = getattr(policy, "allowed_roots", None)
    if roots:
        return str(roots[0])
    return str(Path.cwd())


def make_env_info_section(
    *,
    include: Iterable[EnvField] = DEFAULT_FIELDS,
    model_name: str | None = None,
    extra_fields: dict[str, str] | None = None,
    section_name: str = ENV_INFO_SECTION_NAME,
    cache_control: CacheControl | None = None,
) -> SystemPromptSection:
    """
    Build a section listing the requested environment facts.

    Default fields: ``date``, ``platform``, ``os``, ``cwd``. Pass an
    ``include`` iterable to choose explicitly; ``"datetime"`` and ``"model"``
    are off by default. ``extra_fields`` is a static label→value mapping
    appended verbatim — useful for application-specific facts (e.g.
    ``{"Project": "grasp-agents"}``).

    The compute is sync; the date is recomputed each turn but the rest is
    process-level constant. ``cache_control`` marks a prompt-cache
    checkpoint on this block — leave it ``None`` unless the section sits at
    a stable prefix boundary you want cached (a per-turn ``datetime`` field
    would only fragment the cache, so don't pair the two).
    """
    selected = tuple(include)
    extra = dict(extra_fields or {})

    def compute(
        *,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        **_: Any,
    ) -> str | None:
        del exec_id
        cwd = _effective_cwd(ctx)
        rows: list[str] = []
        for field in selected:
            entry = (
                ("CWD", cwd)
                if field == "cwd"
                else _render_default_field(field, model_name)
            )
            if entry is None:
                continue
            label, value = entry
            rows.append(f"- {label}: {value}")
        for label, value in extra.items():
            rows.append(f"- {label}: {value}")
        if not rows:
            return None
        return "<environment>\n" + "\n".join(rows) + "\n</environment>"

    return SystemPromptSection(
        name=section_name,
        compute=compute,
        cache_control=cache_control,
    )


CURRENT_TIME_ATTACHMENT_NAME = "current_time"


def make_current_time_attachment(
    *,
    name: str = CURRENT_TIME_ATTACHMENT_NAME,
    timespec: str = "seconds",
) -> InputAttachment:
    """
    Build an :class:`InputAttachment` that stamps the current local wall-clock
    time onto each input message.

    Unlike the ``datetime`` field of :func:`make_env_info_section`, this rides
    on the *input* (the new user message), not the system prompt — so a live,
    per-turn timestamp never invalidates the cached prompt prefix. It gives a
    time-aware agent a clock for reasoning about deadlines / staleness / "now".
    Stamped once per agent run and frozen in the transcript, so a resume
    replays the original time rather than "now".
    """

    def compute(
        *,
        input_message: InputMessageItem,
        ctx: SessionContext[Any] | None = None,
        exec_id: str | None = None,
        messages: Sequence[InputItem] | None = None,
        agent_ctx: AgentContext | None = None,
        source: str | None = None,
    ) -> str:
        del input_message, ctx, exec_id, messages, agent_ctx, source
        now = datetime.now().astimezone().isoformat(timespec=timespec)
        return f"Current time: {now}"

    return InputAttachment(name=name, compute=compute, wrap_in_system_reminder=True)
