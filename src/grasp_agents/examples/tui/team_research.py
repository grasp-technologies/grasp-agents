"""
Agent team demo — peer agents that message each other asynchronously (experimental).

A three-member research team: a **lead** that breaks a request down and delegates,
a **researcher** that answers factual sub-questions, and a **writer** that drafts
prose. They talk by calling a ``SendMessage`` tool that drops a message into the
recipient's mailbox; each member reacts to its mailbox at its own turn boundary.

Needs ``OPENAI_API_KEY`` in ``.env``; requires the ``tui`` extra. Three ways to run:

1. **All panes at once** (tmux), the easiest separate-process view::

       python -m grasp_agents.examples.tui.team_research tmux

2. **Separate process per member** — one terminal each, sharing a file mailbox::

       python -m grasp_agents.examples.tui.team_research lead
       python -m grasp_agents.examples.tui.team_research researcher
       python -m grasp_agents.examples.tui.team_research writer

3. **Single process** — the whole team in one window (panes per member, in-memory
   mailbox), for a quick look::

       python -m grasp_agents.examples.tui.team_research team

For (1) and (2), type your request into the **lead** window, e.g. "Write a short,
accurate paragraph on why the sky is blue." Watch the lead delegate (its
``SendMessage`` calls), the researcher/writer windows light up as mail arrives,
their replies flow back, and the lead synthesize. You can type into any window to
nudge that member directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from grasp_agents import LLMAgent, RunContext
from grasp_agents.agent_team import (
    MemberCard,
    AgentTeam,
    CheckpointMailboxTransport,
    MemberDriver,
)
from grasp_agents.durability import FileCheckpointStore
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from grasp_agents.types.events import Event

DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_DIR = Path(".grasp") / "team-demo"

ROSTER = [
    MemberCard(name="lead", description="coordinates the team, synthesizes the answer"),
    MemberCard(name="researcher", description="answers factual research questions"),
    MemberCard(name="writer", description="drafts clear prose from notes"),
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
    "them into a single final answer and present it (no further SendMessage needed).",
    "researcher": _SHARED + " "
    "You are the RESEARCHER. When a teammate asks a question, answer it concisely "
    "and accurately from your knowledge, then SendMessage your findings back to the "
    "sender. Do not write polished prose — just the facts.",
    "writer": _SHARED + " "
    "You are the WRITER. When a teammate asks you to draft something, write clear, "
    "well-structured prose using the notes they provide, then SendMessage the draft "
    "back to the sender.",
}


def _make_llm(model: str) -> OpenAIResponsesLLM:
    return OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "low", "summary": "auto"}},
        ),
    )


def _make_agent(
    name: str, *, ctx: RunContext[None], model: str
) -> LLMAgent[Any, Any, None]:
    return LLMAgent[Any, Any, None](
        name=name, ctx=ctx, llm=_make_llm(model), sys_prompt=_SYS[name], stream_llm=True
    )


def build_member(
    name: str, *, mailbox_dir: Path = DEFAULT_DIR, model: str = DEFAULT_MODEL
) -> MemberDriver:
    """
    Build one member sharing a durable mailbox (a FileCheckpointStore under
    ``mailbox_dir``) with the other member processes.
    """
    mailbox_dir.mkdir(parents=True, exist_ok=True)
    store = FileCheckpointStore(root=mailbox_dir)
    ctx = RunContext[None](state=None, checkpoint_store=store)
    agent = _make_agent(name, ctx=ctx, model=model)
    return MemberDriver(
        agent, cards=ROSTER, transport=CheckpointMailboxTransport(store)
    )


def build_team(*, model: str = DEFAULT_MODEL) -> AgentTeam[None]:
    """Build the whole team in one process (in-memory mailbox, no file wiring)."""
    ctx = RunContext[None](state=None)  # no file backend -> in-memory transport
    members = [_make_agent(c.name, ctx=ctx, model=model) for c in ROSTER]
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
            "tmux not found — open three terminals and run the per-member "
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
            ";", "select-layout", "tiled",
        ],
        check=False,
    )


def main() -> None:
    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Agent team research demo.")
    parser.add_argument(
        "who",
        choices=[*_SYS, "team", "tmux"],
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
        team = build_team(model=args.model)
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
