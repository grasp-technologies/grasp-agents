"""
Interactive data-analysis copilot — a real multi-agent + sandbox demo.

A main *analyst* agent coordinates two sandboxed specialists:

- ``data_engineer`` — generates / cleans / inspects datasets (numpy) in a sandbox
- ``viz_specialist`` — computes stats and renders matplotlib charts (shown inline)

Both run Python in a confined local sandbox (``srt``/``seatbelt``) sharing one
workspace; charts they display stream back as images and render inline in the
TUI. Conversation memory persists across turns (the same analyst instance).

Run (needs ``OPENAI_API_KEY`` in ``.env``; the sandbox kernel needs ``numpy`` +
``matplotlib``, present in the dev env)::

    python -m grasp_agents.examples.tui.data_copilot

Requires the ``tui`` + ``notebook-exec`` extras and a local sandbox backend
(the ``srt`` CLI, or macOS ``sandbox-exec`` for ``seatbelt``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from grasp_agents import AgentTool, LLMAgent, SessionContext
from grasp_agents.llm_providers.openai_responses import (
    OpenAIResponsesLLM,
    OpenAIResponsesLLMSettings,
)
from grasp_agents.sandbox import local_environment
from grasp_agents.tools.bash import Bash, bash_tools
from grasp_agents.tools.code_interpreter import RunPython

DEFAULT_MODEL = "gpt-5.4-nano"

_ANALYST_SYS = """\
You are a data-analysis copilot coordinating two specialists, each running \
Python in a sandbox:
- `data_engineer` — generates, loads, cleans, and inspects datasets.
- `viz_specialist` — computes statistics and renders charts.

For an analytical request: (1) delegate data preparation to `data_engineer`; \
(2) delegate the analysis and charts to `viz_specialist`; (3) synthesize a \
concise answer for the user, citing the key numbers. Do not write code \
yourself — delegate. Briefly state your plan before delegating."""

_ENGINEER_SYS = """\
You prepare datasets in a persistent sandbox, using Bash for shell-level data \
work and RunPython (numpy; pandas is NOT available) for analysis — both share \
one workspace.

When asked to prepare data:
1. Generate it with a single Bash loop that appends rows to a CSV in ~6 \
batches, echoing a progress line per batch and running `sleep 1` between \
batches to mimic real per-batch I/O. \
2. Then load the CSV with RunPython (numpy) and report its shape, fields, \
notable summary statistics, and the file path.
Be concise."""

_VIZ_SYS = """\
You write and run Python in a sandbox (RunPython + Bash) sharing the workspace. \
Only factory Python packages, numpy and matplotlib are available (no pandas/sklearn/etc). \
Load the prepared dataset and compute the requested statistics with numpy. Make at \
least one matplotlib chart. The inline backend is already active; build the figure, \
save it, THEN show it (save before show — the inline backend closes the figure on \
show, so saving afterwards writes a blank image):

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # ...draw on ax...
    fig.savefig("chart.png")  # save BEFORE show
    plt.show()

`plt.show()` returns the figure as an image the user sees. Then report the key \
findings (the actual numbers) in text. Be concise."""

_WORKSPACE = (
    "You and the other specialist share one sandbox workspace — your current "
    "working directory (shown as CWD in the environment). Read and write files "
    "there using relative paths (e.g. `data.npz`); stay within this directory."
)

_WORKDIR = Path("./copilot_workdir").resolve()


def build_copilot(
    workdir: Path,
    *,
    model: str = DEFAULT_MODEL,
    confinement: Literal["none", "seatbelt", "bwrap", "srt", "auto"] = "srt",
) -> tuple[LLMAgent[str, str, None], SessionContext[None]]:
    """Build the analyst agent (with two sandboxed subagents) and its context."""
    workdir.mkdir(parents=True, exist_ok=True)
    llm = OpenAIResponsesLLM(
        model_name=model,
        llm_settings=cast(
            "OpenAIResponsesLLMSettings",
            {"reasoning": {"effort": "medium", "summary": "detailed"}},
        ),
    )
    env = local_environment(
        allowed_roots=[workdir],
        confinement=confinement,
        # A dedicated agent venv at <workdir>/.venv with its own stack, so the
        # demo is self-contained and the agent never touches the host env.
        provision=True,
        packages=["ipykernel", "numpy", "matplotlib"],
        env={"MPLCONFIGDIR": str(workdir / ".mpl")},
        kernel_setup_code="%matplotlib inline",  # opt into inline plotting
    )
    # The sandbox environment supplies a file backend rooted at the workspace;
    # the engineer's backgrounded Bash job mirrors its output to
    # `<workdir>/.grasp/tasks/*.log` (a crash-recoverable, Grep-able trace).
    ctx = SessionContext[None](state=None, environment=env)

    def specialist(
        name: str, description: str, sys_prompt: str, tools: list[Any]
    ) -> AgentTool[None]:
        return AgentTool[None](
            name=name,
            description=description,
            llm=llm,
            tools=tools,
            sys_prompt=f"{sys_prompt}\n\n{_WORKSPACE}",
            max_turns=12,
            stream_llm=True,
        )

    data_engineer = specialist(
        "data_engineer",
        "Generate, load, clean, and inspect datasets in a sandbox; long shell "
        "data jobs run in the background. Returns a concise data summary.",
        _ENGINEER_SYS,
        # backgrounding Bash tools (Bash + KillTask) + Python
        [*bash_tools(auto_background_at=2), RunPython()],
    )
    viz_specialist = specialist(
        "viz_specialist",
        "Compute statistics and render matplotlib charts (shown inline) from the "
        "prepared dataset in the sandbox. "
        "Sandbox has numpy and matplotlib, but NOT pandas/sklearn/etc. "
        "Returns key findings.",
        _VIZ_SYS,
        [RunPython(), Bash()],
    )
    analyst = LLMAgent[str, str, None](
        name="analyst",
        ctx=ctx,
        llm=llm,
        tools=[data_engineer, viz_specialist],
        sys_prompt=_ANALYST_SYS,
        max_turns=24,
        stream_llm=True,
    )
    return analyst, ctx


def main() -> None:
    from grasp_agents.ui import run_tui_interactive  # noqa: PLC0415

    analyst, _ = build_copilot(_WORKDIR)
    run_tui_interactive(analyst)


if __name__ == "__main__":
    main()
