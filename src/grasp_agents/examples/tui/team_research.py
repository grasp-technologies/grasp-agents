"""
Agent team demo — peer members that message each other asynchronously (experimental).

A four-member research team: a **lead** that breaks a request down and delegates, a
**researcher** that answers factual sub-questions, a **writer** that drafts prose,
and an **archivist** — a plain, non-agent ``Processor`` that deterministically files
the final answer to ``notes.md``. They talk by calling a ``SendMessage`` tool that
drops a message into the recipient's mailbox; each member reacts to its mailbox at
its own turn boundary.

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

3. **Single process** — the whole team in one window (panes per member, in-memory
   mailbox), for a quick look::

       python -m grasp_agents.examples.tui.team_research team

For (1) and (2), type your request into the **lead** window, e.g. "Write a short,
accurate paragraph on why the sky is blue." Watch the lead delegate (its
``SendMessage`` calls), the researcher/writer windows light up as mail arrives, their
replies flow back, the lead synthesize, and the archivist window record the result.
You can type into any window to nudge that member directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from grasp_agents import LLMAgent, RunContext
from grasp_agents.agent_team import (
    AgentTeam,
    CheckpointMailboxTransport,
    MemberCard,
    MemberDriver,
)
from grasp_agents.durability import FileCheckpointStore
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)
from grasp_agents.processors.processor import Processor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from grasp_agents.types.events import Event

DEFAULT_MODEL = "gpt-5.4-nano"
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

_SHARED = (
    "You are one member of a small team. Teammates collaborate by SENDING MESSAGES "
    "via the SendMessage tool — there is no shared chat. A message you receive "
    "arrives as a user turn prefixed with its sender. To reply, call SendMessage "
    "addressed back to that sender. Keep messages short and purposeful."
)

_SYS = {
    "lead": _SHARED + " "
    "You are the team LEAD. Given the user's request, break it into parts and "
    "delegate: ask `researcher` for any facts you need and `writer` to draft prose, "
    "one SendMessage at a time. When teammates reply with their parts, synthesize "
    "them into a single final answer. SendMessage that final answer to `archivist` "
    "to file it, then present it to the user (no further SendMessage needed).",
    "researcher": _SHARED + " "
    "You are the RESEARCHER. When a teammate asks a question, answer it concisely "
    "and accurately from your knowledge, then SendMessage your findings back to the "
    "sender. Do not write polished prose — just the facts.",
    "writer": _SHARED + " "
    "You are the WRITER. When a teammate asks you to draft something, write clear, "
    "well-structured prose using the notes they provide, then SendMessage the draft "
    "back to the sender.",
}


class Archivist(Processor[Any, str, None]):
    """
    A non-agent team member: it files every message it receives to a notes file.

    Deterministic — no LLM, no human input, no ``SendMessage`` tool. It reacts to an
    inbound packet (the team delivers a peer message as its input) and produces a
    side effect; with no recipients it is a sink. This is all a ``Processor`` member
    needs to participate, in-process or in its own process.
    """

    def __init__(self, name: str, *, ctx: RunContext[None], notes_path: Path) -> None:
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


def _make_llm(model: str) -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "low", "summary": "auto"}},
        ),
    )


def _make_member(
    name: str, *, ctx: RunContext[None], model: str, notes_path: Path
) -> Processor[Any, Any, None]:
    """Build member ``name`` — the archivist is a processor, the rest are agents."""
    if name == "archivist":
        return Archivist(name=name, ctx=ctx, notes_path=notes_path)
    return LLMAgent[Any, Any, None](
        name=name, ctx=ctx, llm=_make_llm(model), sys_prompt=_SYS[name], stream_llm=True
    )


def build_member(
    name: str, *, mailbox_dir: Path = DEFAULT_DIR, model: str = DEFAULT_MODEL
) -> MemberDriver:
    """
    Build one member (agent or processor) sharing a durable mailbox (a
    FileCheckpointStore under ``mailbox_dir``) with the other member processes.
    """
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    store = FileCheckpointStore(root=mailbox_dir)
    ctx = RunContext[None](state=None, checkpoint_store=store)
    member = _make_member(
        name, ctx=ctx, model=model, notes_path=mailbox_dir / "notes.md"
    )
    return MemberDriver(
        member, cards=ROSTER, transport=CheckpointMailboxTransport(store)
    )


def build_team(
    *, model: str = DEFAULT_MODEL, mailbox_dir: Path = DEFAULT_DIR
) -> AgentTeam[None]:
    """Build the whole team in one process (in-memory mailbox, no file wiring)."""
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext[None](state=None)  # no file backend -> in-memory transport
    members = [
        _make_member(c.name, ctx=ctx, model=model, notes_path=mailbox_dir / "notes.md")
        for c in ROSTER
    ]
    return AgentTeam(members, entry="lead", cards=ROSTER, ctx=ctx)


def _human_handler(driver: MemberDriver) -> Any:
    # Queue human input onto the member's serial inbox (so it serializes with
    # mailbox turns); the turn's events render through ``driver.events()``.
    async def on_submit(text: str) -> AsyncIterator[Event[Any]]:  # noqa: RUF029
        driver.submit_human(text)
        return
        yield  # unreachable: makes this an async generator that yields nothing

    return on_submit


def _launch_tmux(*, mailbox_dir: Path, model: str) -> None:
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
            sys.executable, "-m", "grasp_agents.examples.tui.team_research",
            name, "--dir", str(mailbox_dir), "--model", model,
        ]
        return " ".join(argv)

    subprocess.run(  # noqa: S603
        [
            tmux, "new-session", "-s", "team-demo", pane("lead"),
            ";", "split-window", "-h", pane("researcher"),
            ";", "split-window", "-v", pane("writer"),
            ";", "split-window", "-v", pane("archivist"),
            ";", "select-layout", "tiled",
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.who == "tmux":
        _launch_tmux(mailbox_dir=args.dir, model=args.model)
        return

    if args.who == "team":
        team = build_team(model=args.model, mailbox_dir=args.dir)
        run_tui_interactive(on_submit=team.run_stream, main_agent="lead", ctx=team.ctx)
        return

    driver = build_member(args.who, mailbox_dir=args.dir, model=args.model)
    run_tui_interactive(
        on_submit=_human_handler(driver),
        events=driver.events(),
        main_agent=driver.name,
        ctx=driver.ctx,
    )


if __name__ == "__main__":
    main()
