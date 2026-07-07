"""
Agent team demo — peer members that message each other asynchronously (experimental).

A four-member research team: a **lead** that breaks a request down and delegates, a
**researcher** that answers factual sub-questions, a **writer** that drafts prose,
and an **archivist** — a plain, non-agent ``Processor`` that deterministically files
the final answer to ``notes.md``. They talk by calling a ``SendMessage`` tool that
drops a message into the recipient's mailbox; each member reacts to its mailbox at
its own turn boundary.

The lead also fires off a slow ``index_sources`` job per request — a **background
task** that never gates the answer. Watch its live progress stream into its own
tinted pane (nested under the lead) while the delegation continues; the same output
is mirrored to a ``.grasp/tasks/*.log`` under the mailbox directory, and the lead
gets a ``<task_notification>`` when it finishes.

The archivist shows how a **non-agent member** joins a team: it has no LLM, no human
input, and no ``SendMessage`` tool. It simply reacts to inbound packets (here, by
writing a file) and, if it had recipients, would hand its output off by name. In the
separate-process layout it runs in its own process just like the agents do — driven
off the same shared mailbox by the same actor runtime — so "members" are not
necessarily agents.

Needs ``OPENAI_API_KEY`` in ``.env``; requires the ``tui`` extra. Three ways to run:

1. **All panes at once** (tmux), the easiest separate-process view::

       python -m grasp_agents.examples.tui.team_research tmux

2. **Separate process per member** — one terminal each, sharing a file mailbox. Every
   member, agent or processor, is launched the same way::

       python -m grasp_agents.examples.tui.team_research lead
       python -m grasp_agents.examples.tui.team_research researcher
       python -m grasp_agents.examples.tui.team_research writer
       python -m grasp_agents.examples.tui.team_research archivist

3. **Single process** — the whole team in one window (panes per member), over the
   same durable file mailbox as the separate-process modes, for a quick look::

       python -m grasp_agents.examples.tui.team_research team

In every mode, type your request into the **lead** window (or pane), e.g. "Write a
short, accurate paragraph on why the sky is blue." Watch the lead delegate (its
``SendMessage`` calls), the researcher/writer windows light up as mail arrives, their
replies flow back, the lead synthesize, and the archivist window record the result.
You can type into any window to nudge that member directly. Input is posted to the
member's mailbox, so you can keep typing while it works: pending messages are listed
above the prompt as *queued* until the member takes them, one per turn boundary.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from grasp_agents import LLMAgent, SessionContext
from grasp_agents.agent_team import AgentTeam, MemberCard, MemberHost
from grasp_agents.durability import FileCheckpointStore
from grasp_agents.file_backend.local import LocalFileBackend
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)
from grasp_agents.processors.processor import Processor
from grasp_agents.tools.base import BaseTool
from grasp_agents.types.events import Event, ToolOutputEvent, ToolStreamEvent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

DEFAULT_MODEL = "gpt-5.4-nano"
LEAD_MODEL = "gpt-5.4"
DEFAULT_DIR = Path(".grasp") / "team-demo"

ROSTER = [
    MemberCard(name="lead", description="coordinates the team, synthesizes the answer"),
    MemberCard(name="researcher", description="answers factual research questions"),
    MemberCard(name="writer", description="drafts clear prose from notes"),
    MemberCard(
        name="archivist",
        description="files the final answer to notes.md (a non-agent member)",
    ),
]

# Only role-specific guidance here — the team framing (how to send / receive / wait)
# is injected automatically as a system-prompt section on every resident member.
_SYS = {
    "lead": "You are the team LEAD. When a new user request arrives, first call "
    "`index_sources` with its topic — it runs in the background; never wait for it "
    "or mention it to teammates. Then break the request into parts and "
    "delegate: ask `researcher` for any facts you need and `writer` to draft prose, "
    "one SendMessage at a time. When teammates reply with their parts, synthesize "
    "them into a single final answer. SendMessage that final answer to `archivist` "
    "to file it, then present it to the user (no further SendMessage needed). "
    "Follow this playbook for EVERY user request — even when the answer already "
    "sits in your context from earlier work, still index and route the request "
    "through the team; never answer directly.",
    "researcher": "You are the RESEARCHER. When a teammate asks a question, answer it "
    "concisely and accurately from your knowledge, then SendMessage your findings "
    "back to the sender. Do not write polished prose — just the facts.",
    "writer": "You are the WRITER. When a teammate asks you to draft something, write "
    "clear, well-structured prose using the notes they provide, then SendMessage the "
    "draft back to the sender.",
}


class Archivist(Processor[Any, str, None]):
    """
    A non-agent team member: it files every message it receives to a notes file.

    Deterministic — no LLM, no human input, no ``SendMessage`` tool. It reacts to an
    inbound packet (the team delivers a peer message as its input) and produces a
    side effect; with no recipients it is a sink. This is all a ``Processor`` member
    needs to participate, in-process or in its own process.
    """

    def __init__(
        self, name: str, *, ctx: SessionContext[None], notes_path: Path
    ) -> None:
        super().__init__(name=name, ctx=ctx)  # no recipients -> a sink
        self._notes_path = notes_path

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[Any] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> list[str]:
        del chat_inputs, exec_id, step
        notes = [getattr(a, "text", str(a)) for a in (in_args or [])]
        with self._notes_path.open("a", encoding="utf-8") as f:
            for note in notes:
                f.write(note + "\n\n")
        return notes


class IndexSourcesInput(BaseModel):
    topic: str = Field(description="Topic to (re)index the team's source archive for.")


class IndexSources(BaseTool[IndexSourcesInput, str, None]):
    """
    A slow archive-indexing job the lead fires off per request — it exists to
    show background-task handling end to end: launched immediately
    (``auto_background_at=0``), never gating the answer
    (``blocks_final_answer=False``), streaming progress that the TUI routes to
    the task's own log pane and mirrors to a ``.grasp/tasks`` log
    (``has_progress_log=True``), and a ``<task_notification>`` on completion.
    """

    def __init__(self) -> None:
        super().__init__(
            name="index_sources",
            description=(
                "Refresh the team's source index for a topic. Slow — always runs "
                "as a background task. Do NOT wait for it; keep working. A "
                "<task_notification> arrives when it finishes."
            ),
            auto_background_at=0,
            blocks_final_answer=False,
            has_progress_log=True,
        )

    async def _run(self, inp: IndexSourcesInput, **kwargs: Any) -> str:
        del kwargs
        return f"Source index refreshed for '{inp.topic}' (4 sources, 7 passages)."

    async def _run_stream(
        self, inp: IndexSourcesInput, *, exec_id: str | None = None, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        del kwargs
        sources = [
            "arxiv-notes.md",
            "web-clips.json",
            "textbook-excerpts.md",
            "lecture-notes.md",
        ]
        yield ToolStreamEvent(
            data=f"indexing archive for '{inp.topic}'…\n",
            source=self.name,
            exec_id=exec_id,
        )
        for n, src in enumerate(sources, start=1):
            await asyncio.sleep(1.2)
            yield ToolStreamEvent(
                data=f"[{n}/{len(sources)}] {src} indexed\n",
                source=self.name,
                exec_id=exec_id,
            )
        yield ToolOutputEvent(
            data=await self._run(inp), source=self.name, exec_id=exec_id
        )


def _make_llm(model: str) -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "low", "summary": "auto"}},
        ),
    )


def _make_session(mailbox_dir: Path) -> SessionContext[None]:
    """
    One session over ``mailbox_dir``: the checkpoint store is the durable
    mailbox (and task-record substrate); the file backend is where a background
    task's ``.grasp/tasks`` progress log is mirrored.
    """
    return SessionContext[None](
        state=None,
        checkpoint_store=FileCheckpointStore(root=mailbox_dir),
        file_backend=LocalFileBackend(allowed_roots=[mailbox_dir]),
    )


def _make_member(
    name: str, *, ctx: SessionContext[None], model: str, notes_path: Path
) -> Processor[Any, Any, None]:
    """Build member ``name`` — the archivist is a processor, the rest are agents."""
    if name == "archivist":
        return Archivist(name=name, ctx=ctx, notes_path=notes_path)
    return LLMAgent[Any, Any, None](
        name=name,
        ctx=ctx,
        llm=_make_llm(model),
        sys_prompt=_SYS[name],
        tools=[IndexSources()] if name == "lead" else None,
        stream_llm=True,
    )


def build_member(name: str, *, mailbox_dir: Path = DEFAULT_DIR) -> MemberHost:
    """
    Build one member (agent or processor) sharing a durable mailbox with the
    other member processes — derived automatically from the session's
    ``FileCheckpointStore`` under ``mailbox_dir``.
    """
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    ctx = _make_session(mailbox_dir)

    member = _make_member(
        name,
        ctx=ctx,
        model=LEAD_MODEL if name == "lead" else DEFAULT_MODEL,
        notes_path=mailbox_dir / "notes.md",
    )
    return MemberHost(member, cards=ROSTER)


def build_team(*, mailbox_dir: Path = DEFAULT_DIR) -> AgentTeam[None]:
    """
    Build the whole team in one process over a durable mailbox — derived
    automatically from the session's ``FileCheckpointStore`` under
    ``mailbox_dir`` — so in-flight mail, each member's transcript, and the
    team's hop budget all persist to the same store.
    """
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    ctx = _make_session(mailbox_dir)
    members = [
        _make_member(
            c.name,
            ctx=ctx,
            model=LEAD_MODEL if c.name == "lead" else DEFAULT_MODEL,
            notes_path=mailbox_dir / "notes.md",
        )
        for c in ROSTER
    ]
    return AgentTeam(members, entry="lead", cards=ROSTER, ctx=ctx)


def _launch_tmux(*, mailbox_dir: Path) -> None:
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415, S404
    import sys  # noqa: PLC0415

    tmux = shutil.which("tmux")
    if tmux is None:
        raise SystemExit(
            "tmux not found — open four terminals and run the per-member "
            "commands shown in this module's docstring."
        )

    def pane(name: str) -> str:
        argv = [
            sys.executable,
            "-m",
            "grasp_agents.examples.tui.team_research",
            name,
            "--dir",
            str(mailbox_dir),
        ]
        return " ".join(argv)

    subprocess.run(  # noqa: S603
        [
            tmux,
            "new-session",
            "-s",
            "team-demo",
            pane("lead"),
            ";",
            "split-window",
            "-h",
            pane("researcher"),
            ";",
            "split-window",
            "-v",
            pane("writer"),
            ";",
            "split-window",
            "-v",
            pane("archivist"),
            ";",
            "select-layout",
            "tiled",
        ],
        check=False,
    )


def main() -> None:
    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Agent team research demo.")
    parser.add_argument(
        "who",
        choices=[c.name for c in ROSTER] + ["team", "tmux"],
        help="a member name, 'team' (one process), or 'tmux' (launch all panes)",
    )
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR, help="shared mailbox directory"
    )
    args = parser.parse_args()

    if args.who == "tmux":
        _launch_tmux(mailbox_dir=args.dir)
        return

    if args.who == "team":
        # The whole team serves as a daemon in the background; human input is
        # posted to the lead's mailbox fire-and-forget, so messages queue (shown
        # above the prompt) and are taken at the lead's turn boundaries. The
        # stream is passed as a factory so ``esc`` can stop it (interrupting
        # in-flight turns) and the next message can start a fresh one; the
        # roster pre-creates every member's pane even on a resumed session.
        team = build_team(mailbox_dir=args.dir)

        async def post_to_lead(text: str) -> None:
            await team.submit_message("lead", text)

        run_tui_interactive(
            on_post=post_to_lead,
            events=lambda: team.run_stream(daemon=True),
            main_agent="lead",
            agents=[c.name for c in ROSTER],
            ctx=team.ctx,
        )
        return

    # Human input is posted to the member's mailbox as control-plane mail (it
    # drains ahead of peer messages and queues until the member's next turn
    # boundary); the turns render through ``host.run_stream()`` — passed as a
    # factory, so ``esc`` interrupts the member and a message resumes it.
    host = build_member(args.who, mailbox_dir=args.dir)
    run_tui_interactive(
        on_post=host.submit_message,
        events=host.run_stream,
        main_agent=host.name,
        agents=[host.name],
        ctx=host.ctx,
    )


if __name__ == "__main__":
    main()
