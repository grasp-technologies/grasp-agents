"""Roster / capability advertisement for a team member."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from grasp_agents.processors.processor import Processor


class MemberCard(BaseModel):
    """
    A team member's advertised identity — the roster entry peers see when
    deciding whom to message.

    A trimmed analogue of the Agent2Agent (A2A) protocol's Agent Card: it keeps
    the fields meaningful for in-process discovery (``name`` / ``description`` /
    ``skills`` / accepted input) and omits the transport-specific ones (service URL,
    security schemes, signature) a networked transport would add. It is also the
    per-member team config: ``input_type`` and ``resident`` are read by the host.
    """

    # Holds a model *class* in ``input_type``; never serialized (a live roster).
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    skills: list[str] = Field(default_factory=list)
    version: str = "0.1.0"
    # The body this member accepts. ``str`` (default) → free text / content. A
    # ``BaseModel`` subclass → a typed peer hand-off: a peer may ``SendMessage`` a
    # matching object, revalidated to this model on receipt (surviving a durable
    # round-trip). A *triggered* recipient then renders it through the model's own
    # ``InputRenderable`` / input hook; a *resident* receives the structured body as
    # JSON in its inbox turn.
    input_type: type[BaseModel | str] = str
    # Execution mode: ``True`` runs the member **resident** (a persistent loop
    # consuming its inbox between turns), ``False`` **triggered** (one activation per
    # message). ``None`` (default) lets the host infer — an LLM agent with no static
    # recipients runs resident. Set it explicitly when inference is ambiguous (e.g. a
    # chat agent that only becomes a messenger once given the ``SendMessage`` tool).
    resident: bool | None = None
    # The team's lead: at most one per team, and it must run resident (both
    # validated at host construction). The lead holds the session's
    # environment-rewind right (``SessionContext.session_writer``) — only
    # it snapshots the shared filesystem and may roll it back, and a rewind is
    # announced to the other members. Its messages carry ``LEAD_PRIORITY``,
    # draining ahead of ordinary peer mail (below control-plane mail).
    lead: bool = False
    # Opt-in: give this (resident) member the ``ScheduleWakeup`` tool — a
    # self-addressed timer for acting on its own initiative later (a follow-up,
    # a poll, a no-reply timeout). Off by default: inbound mail already wakes a
    # resident, and models tend to schedule wakeups merely to "wait" for
    # replies that would wake them anyway.
    wakeups: bool = False

    @classmethod
    def from_processor(
        cls,
        processor: Processor[Any, Any, Any],
        *,
        description: str = "",
        skills: Sequence[str] | None = None,
        resident: bool | None = None,
        lead: bool = False,
    ) -> MemberCard:
        """
        Build a card from a processor, taking its ``name`` and deriving ``input_type``
        from its declared input type (``InT``).

        A ``BaseModel`` subclass becomes the advertised typed hand-off shape; ``str``
        (and ``Any`` / an unparameterized member, which accept anything) becomes free
        text. Any other ``InT`` — ``int``, a container, a non-model class — **raises**:
        advertising such a member as text would let a peer send a body its input
        validation then rejects, so build its card explicitly instead.

        ``skills`` defaults to the member's own skill allowlist (the names it is scoped
        to via ``skill_include``), or ``[]`` when it is unscoped or not an agent; pass
        ``skills`` to override. ``description`` / ``resident`` cannot be read off a
        processor and default to empty / host-inferred.
        """
        in_type = processor.in_type
        try:
            is_str = issubclass(in_type, str)
            is_model = issubclass(in_type, BaseModel)
        except TypeError:
            is_str = is_model = False  # a container / union / special form

        if is_model:
            input_type: type[BaseModel | str] = in_type
        elif is_str or in_type is object:
            # ``object`` is how ``Any`` / an unresolved param resolves: accepts
            # anything, so free text validates fine — no false contract.
            input_type = str
        else:
            raise ValueError(
                f"Cannot derive a team input_type for member {processor.name!r}: its "
                f"input type {in_type!r} is neither a str nor a BaseModel subclass. A "
                "team message body must be free text or a typed model — change the "
                "member's input type, or build its MemberCard with an explicit "
                "input_type."
            )

        if skills is None:
            # Local import avoids a module-level dep from this data model into the
            # behavioral role helpers (which _roles already TYPE_CHECKING-imports).
            from ._roles import is_llm_agent  # noqa: PLC0415

            skill_filter = processor.skill_filter if is_llm_agent(processor) else None
            include = skill_filter.include if skill_filter is not None else None
            skills = sorted(include) if include else []

        return cls(
            name=processor.name,
            description=description,
            skills=list(skills),
            input_type=input_type,
            resident=resident,
            lead=lead,
        )

    def render(self) -> str:
        """A roster entry for the ``SendMessage`` tool description / team prompt."""
        lines = [self.name]
        if self.lead:
            lines[0] += " (team lead)"
        if self.description:
            lines[0] += f": {self.description}"
        if self.skills:
            lines.append(f"Skills: {', '.join(self.skills)}")
        if issubclass(self.input_type, BaseModel):
            schema = json.dumps(self.input_type.model_json_schema(), indent=2)
            lines.append(f"Input message schema:\n{schema}")

        return "\n\n".join(lines)
