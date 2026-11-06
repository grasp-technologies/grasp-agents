---
name: evals
description: How to use grasp_agents.evals and the grasp-evals CLI to measure an async function, an LLMAgent or any processor against a dataset. Covers scorers, metrics, runs, resume and rescore, comparing runs and regression gates, pairwise A/B, LLM judges with label-based validation, probes and validation gates, the labeling loop and Phoenix sync. Above all it covers the non-obvious semantics behind the numbers: how errors enter metrics, run immutability, sealed splits, example identity and pairing, and output typing on rescore. Use when writing or running evaluations, writing scorers or judges, reading or comparing eval results, gating CI on evals, validating a judge, collecting labels, or debugging a grasp-evals command.
---

# Evaluations with `grasp_agents.evals`

The reference page is `docs/evals.md`, which covers types, storage layout and contracts.
This skill is the how-to plus the traps. More detail is in:

- [cli.md](cli.md): commands, run references, selection order, exit codes, JSON output,
  gating recipes, on-disk layout, Phoenix commands.
- [metrics.md](metrics.md): the outcome taxonomy, default metrics and their names,
  intervals, `compare` statistics and the regression gate.
- [judges.md](judges.md): `ProcessorScorer`, `judge_validation`, probes, the labeling loop,
  `ValidationGate`, pairwise judges.

Import everything from `grasp_agents.evals`; the root package exports none of it.
Reducers live in `grasp_agents.evals.metrics` and `regressions` in
`grasp_agents.evals.compare`.

## Mental model

```
Dataset ─select─▶ Example ─Task─▶ Trial (output | error) ─Scorers─▶ Score(s) ─Metrics─▶ MetricResult
                  × repetitions                                     all stored in one EvaluationRun
```

| Concept | What it is | Code |
|---|---|---|
| `Example` | `id`, `input`, `reference`, `metadata`, `splits`. Frozen. Identity is `id` plus the `content_hash` of the stored record | `evals/types.py` → `Example` |
| `Task` | The system under test: an async function (`FunctionTask`), any `Processor` (`ProcessorTask`), or a `Task` subclass | `evals/task.py` |
| `Trial` | One run of one example at one repetition: a JSON output or an `error`, plus duration, usage, models, trace id and scores | `evals/types.py` → `Trial` |
| `Scorer` | Produces the scores of one trial. Judges are scorers too | `evals/scorer.py`, `evals/judge.py` |
| `Score.value` | `bool` (pass/fail), a number, a `str` label, or `None` (unscored) | `evals/types.py` → `Score` |
| `Metric` | An aggregate with an interval. Example-level by default: repetitions are reduced per example first | `evals/metrics.py` |
| `EvaluationRun` | The on-disk record under `$GRASP_EVALS_DIR` (default `./.evals`) | `evals/types.py`, `evals/store.py` |
| `Evaluation` | An importable spec that the CLI runs: task, dataset, scorers, metrics, settings | `evals/evaluation.py` |

What each action does to runs:

| Action | Result | Original run |
|---|---|---|
| `evaluate(...)` / `grasp-evals run` | A new `evaluation` run with a new id, every time | — |
| Resume an unfinished run (running, cancelled, partial, failed) | Continues in place | Updated |
| Resume a completed run that has failed trials or missing scorer outputs | A `retry` child that copies the OK trials and redoes the rest | Unchanged |
| Resume a completed run with nothing to redo | Returns that same run; nothing new is created | Unchanged |
| `rescore` | A `rescore` child. The task is not re-run | Unchanged |
| `pairwise` | A `pairwise` run whose trials hold both arms' outputs | Unchanged |
| `push` (Phoenix) | No new run | Only the Phoenix link is written to `run.json` |

## Workflow 1: a quick eval from Python

```python
import asyncio

from grasp_agents.evals import Dataset, Example, ScoreContext, evaluate, scorer

data = Dataset(
    [Example(id=f"n{i}", input=i, reference=i * 2) for i in range(6)],
    name="doubling",
)


async def solve(x: int) -> int:
    return x * 2


@scorer(version="1")
def exact(ctx: ScoreContext[int, int, int]) -> bool:  # bool → pass/fail
    return ctx.output == ctx.reference


run = asyncio.run(evaluate(solve, data, [exact], persist=False))
print(run.metric("pass_rate(exact)"))
```

- A scorer can return:
  - a `bool` (pass/fail), a number, or a `str` (a label);
  - a `Score`, a list of scores, or a `{name: value}` mapping;
  - `None` when the score does not apply;
  - `Score.unscored(name, reason)` when it cannot judge.

  It should raise only on a real bug.
- Without `persist=False`, the run lands in `./.evals/runs/<run_id>/`.

## Workflow 2: an `Evaluation` spec and the CLI loop

A dataset file holds one JSON record per line. Lines starting with `//` are comments:

```json
{"id": "q2", "input": {"question": "3+3?", "answer": "5"}, "reference": {"verdict": "incorrect"}, "metadata": {"topic": "math"}, "splits": ["dev"]}
```

The spec module, e.g. `evals/grading.py`:

```python
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from grasp_agents import LLMAgent
from grasp_agents.evals import (
    Evaluation,
    FunctionTask,
    PassHatK,
    ProcessorTask,
    ScoreContext,
    default_metrics,
    scorer,
)

DATA = Path(__file__).parent / "data" / "answers.jsonl"  # never relative to the CWD


class Submission(BaseModel):
    question: str
    answer: str


class TeacherGrade(BaseModel):
    verdict: Literal["correct", "incorrect"]


class Grade(BaseModel):
    verdict: Literal["correct", "incorrect"]
    feedback: str


async def rule_grader(sub: Submission) -> Grade:
    ok = sub.answer in {"4", "Paris", "25"}
    return Grade(verdict="correct" if ok else "incorrect", feedback="checked")


@scorer(version="1")
def agrees(ctx: ScoreContext[Submission, Grade, TeacherGrade]) -> bool:
    assert ctx.reference is not None
    return ctx.output.verdict == ctx.reference.verdict


grader_rules = Evaluation(
    name="grader",
    description="Does the grader agree with the teacher?",
    task=FunctionTask(rule_grader, version="rules-1"),  # not task=rule_grader
    dataset=DATA,
    reference_type=TeacherGrade,  # input_type comes from rule_grader's annotation
    scorers=[agrees],
    metrics=lambda trials: [*default_metrics(trials), PassHatK("agrees", 3)],
    repetitions=3,
    group_by=["topic"],
    sealed_splits=["test"],
    max_error_rate=0.1,
)


def build_grader() -> LLMAgent[Submission, Grade, None]:
    return LLMAgent[Submission, Grade, None](
        name="grader", llm=make_llm(), sys_prompt="Grade the answer."  # make_llm: yours
    )


grader_llm = Evaluation(
    name="grader",
    task=lambda: ProcessorTask(build_grader, version="llm-1", output_type=Grade),
    dataset=DATA,
    input_type=Submission,  # a factory-based task cannot report its input type
    reference_type=TeacherGrade,
    scorers=[agrees],
    sealed_splits=["test"],
)
```

The loop, run from the directory that holds `.evals`:

1. **Validate:** `grasp-evals datasets validate evals/grading.py:grader_rules` checks types and
   `dataset_checks` (exit 1 on problems).
2. **Smoke run:** `grasp-evals run evals/grading.py:grader_rules --split dev --limit 3`.
3. **Real run:** `grasp-evals run evals/grading.py:grader_rules --split dev --json -q`.
   Keep the `"id"` it prints.
4. **Inspect:** `grasp-evals show <id> --failures`, then `grasp-evals show <id> --example q2`
   for the input, reference, scores and transcript.
5. **Change** the system and bump the task's `version=`.
6. **Candidate run plus a gate:** `grasp-evals run evals/grading.py:grader_llm --split dev --baseline <id>
   --fail-on-regression --gate agrees --gate error`.
7. **Compare:** `grasp-evals compare <base-id> <candidate-id>`. Read the warnings before the numbers.
8. **Run the sealed split once, at the end:**
   `grasp-evals run ...:grader_llm --split test --fail-under 'pass_rate(agrees)=0.8'`.

## Workflow 3: evaluate an agent or any processor

```python
task = ProcessorTask(
    build_grader,  # a zero-arg factory (called inside each trial's session) or a template
    version="llm-1",  # bump on every prompt or model change
    output_type=Grade,  # how stored outputs are re-validated on rescore and resume
    # ctx_factory=lambda example: SessionContext(state=...),  # seed per-trial state
    # input_mode="chat",                     # pass the input as chat_inputs, not in_args
    # input_fn=lambda sub: ...,              # transform the example input first
    # output_fn=lambda packet, ctx: ...,     # e.g. read the result from ctx.state
)
```

- **Session.** Each trial gets a fresh `SessionContext` (or the one
  `ctx_factory(example)` returns), and that session is ambient for the whole trial.
- **Template processors.** A template is copied per trial, its agents start with an
  empty transcript, and the template itself never runs. `aclose()` runs after every
  trial.
- **Output.** The output is the single payload, or a list when there are several. When no output
  packet arrives, the trial is a task error.
- **What is recorded.** The trial records usage per agent, the models that answered, and the event stream
  (without token deltas). Cost is recorded only for models with a known price.
- **What scorers see.** `ctx.session` is the final session, which helps when a task
  mutates `state`. It is `None` on rescore.

## Workflow 4: change the instrument, not the task (rescore)

- **CLI.** `grasp-evals rescore <run> --spec evals/grading.py:grader_strict` creates a
  child run.
  - A scorer that is unchanged (same name, version, config and own code) keeps its
    scores. It only fills trials where it failed or did not run.
  - A changed scorer runs again.
  - `--rerun` re-runs every scorer, which also measures a judge's self-consistency.
- **Python.** Prefer `evaluation.rescore(run)`. The bare `rescore()` needs
  `output_type=` (plus `input_type=` and `reference_type=`) and `metrics=`. Without
  them it falls back to plain JSON outputs and default metrics.

## Pitfalls: what the README doesn't tell you

Each item gives the wrong assumption, what actually happens, and what to do.

### Numbers

1. **"Task errors count the same in every metric."**
   - `PassRate`, `PassAtK` and `PassHatK` count a crashed trial as a failure.
   - `Mean`, `Percentile`, `Distribution` and `Proportion` drop it and count it in
     `n_missing`.
   - The default metrics use `PassRate` only for scores that are always `bool`. A
     scorer returning `1.0`/`0.0` gets `Mean`.
   - Example: with 2 of 6 trials crashing and the rest correct, `pass_rate(exact)` is
     0.67 while `mean(closeness)` is 1.0.

   **Do:** return `bool` for pass/fail, or list `PassRate("name")` explicitly (0/1
   floats need no threshold). Always read `error_rate` and `n_missing`. Code:
   `evals/metrics.py` → `default_metrics`, `PassRate`, `Mean`.
2. **"A run with many task errors fails."** A run is marked invalid only when no trial
   ran, every trial failed, or `max_error_rate` is exceeded (default `None`). With 6
   of 8 trials erroring, the run is `completed` and the CLI exits 0. Scorer failures
   never invalidate a run.

   **Do:** set `max_error_rate` on the `Evaluation` (there is no CLI flag) and gate
   with `--fail-under`. Code: `evals/_execution.py` → `Executor._invalid_reason`.
3. **"Raising in a scorer is a fail."**
   - Raising records a `ScorerFailure`: the score is missing, and rescoring retries it.
   - `Score.unscored` is also missing, with a reason.
   - Returning `None` is "not applicable" (`n_na`), except that it becomes missing
     when any other scorer failed on that trial.
   - When the task errors, scorers are skipped. A scorer that sets
     `scores_errors=True` still runs and sees `ctx.output is None`.

   See [metrics.md](metrics.md#outcome-taxonomy).
4. **"`"pass"` or `"true"` is a pass."** Any `str` value is a label. It gets
   `dist(name)`, never a pass rate, and `compare` skips label scores by default. A
   numeric score needs `PassRate(name, threshold=...)` unless its values are only 0
   and 1. Without a threshold, the metric has value `None` and an `error` in `details`.
5. **"`--fail-under` takes a score name."** It takes a metric name such as
   `pass_rate(agrees)`, `mean(x)`, `pass^3(agrees)` or `p95(duration_s)`.
   - With the default metrics, a typo only shows after the whole run ("metric ... not
     found", exit 1).
   - With an explicit `metrics=[...]` list, the name is checked before running (exit 2).
   - Metrics, `compare` and gates use score names (`Score.name`, mapping keys). Scorer
     names only identify components.
6. **"`compare` uses McNemar for pass/fail scores, and `p` is what `significant` reads."**
   - McNemar applies only when every paired per-example value is 0 or 1 and nothing is
     clustered.
   - With repetitions > 1, the per-example means are fractional, so a paired t-test is
     used (a clustered t-test with `cluster_by`).
   - `significant` reads the Holm-adjusted `p_adjusted`, computed within outcome targets
     and separately within resource targets. The displayed `p` is unadjusted.
   - The example is the statistical unit: repetitions never raise `n`.

   See [metrics.md](metrics.md#compare).

### Gates and exit codes

7. **"`--fail-on-regression` catches a broken candidate."** The gate's default targets
   are the numeric scores present in *both* runs, plus `error`. If the candidate's
   scorer fails on every trial, or a score is renamed, that score silently drops out
   and the gate passes.

   **Do:** name the targets (`--gate agrees --gate error`; a named target that is
   missing fails the gate) and add `--fail-under`. `--gate` replaces the default set.
   Code: `evals/compare.py` → `regressions`.
8. **"Exit 1 means a gate flag fired."**
   - The CLI also exits 1 with no flags at all when the run is `partial` (the
     `--max-cost` budget was hit) or invalid.
   - Dataset problems found by `run` exit 2, but `datasets validate` exits 1 for the
     same problems.
   - Any `ValueError` exits 2.
   - A failed `--push` decides the exit code even when the run passed.

   See [cli.md](cli.md#exit-codes).
9. **"`--max-cost` is a hard cap."** It stops *scheduling*, so trials already running
   finish and the total can exceed the cap by roughly one trial per concurrent slot.
   It only sees priced usage. Two things are not limited, and the run only warns and
   lists `unpriced_agents` in its metadata:
   - models without a known price;
   - function tasks that don't fill `trial.usage_by_agent`.

### Runs, resume and identity

10. **"`latest:<name>` is my last evaluation run."** It is the newest run with that run
    name or spec attribute, children included. After `rescore`, `latest:grader_rules`
    resolves to the rescore child, which keeps the parent's name and spec.

    **Do:** in scripts, capture ids from `--json` output.
11. **"Resume just continues."**
    - **Must match, cannot be forced:** the task identity (name, version, config, and
      for a template processor its models, prompts and settings), the scorers, the
      selected examples and the repetitions. So repeat
      `--split/--ids/--sample/--seed/--limit/-r` exactly.
    - **Refused, but `--force` overrides:** a change to the git commit or uncommitted
      diff of the repository *containing the current directory* (even a README commit),
      or to the task's or scorers' source files.
    - **Stops with `ResumeError`:** an agent answers with a model the run never used.
    - `--no-score` followed by `--resume` puts the scores in a `retry` child.

    Code: `evals/runner.py` → `_resume_record`.
12. **"Rescore and pairwise see my output types."** Outputs are stored as JSON.
    - The bare `rescore()` and `pairwise()` default `output_type` to `Any`. Scorers
      receive dicts, `ctx.output.verdict` raises `AttributeError`, and every trial gets
      a scorer failure while the run still reports `completed`.
    - A `ProcessorTask` built from an unannotated factory, or with `output_fn`, also has
      output type `Any`.

    **Do:** use `Evaluation.rescore` or the CLI (both pass the spec's types), or pass
    `output_type=`.
13. **"Scorers see the same context on rescore."** On rescore and resume, `ctx.session`
    is `None`, and events reload as generic `Event` objects whose `data` is JSON, so
    `isinstance(e, ToolCallEvent)`-style checks fail. Match on `e.type` and `e.source`
    instead. Function tasks have no events at all.
14. **"`evaluate()` validates inputs."** It never does: `Dataset.load(path)` without
    `input_type=` yields plain dicts.
    - An `Evaluation` takes `input_type` from a `FunctionTask`'s first-parameter
      annotation or a template processor's `in_type`.
    - A factory-based `ProcessorTask` (or one with `input_fn`) reports `Any`, so inputs
      stay dicts unless you set `Evaluation(input_type=...)`.
15. **"Same content means same example."** The default id is a hash of the input *as
    stored*. Examples built in code leave out model fields that are at their default
    value, so a hand-written file that spells those defaults out gets a different id and
    content hash.
    - Runs pair only on the same id *and* an unchanged content hash. Other examples are
      skipped (`n_changed_examples`).
    - Two id-less examples with the same input are rejected as duplicates.

    **Do:** give explicit ids and write files with `Dataset.save`.
16. **"Rescore notices my scorer changed."** Reuse compares name, version, config and
    the code of the scorer's own function or class. Helpers, data files and mutable
    closure state are not hashed.

    **Do:** bump `version` or pass `--rerun`.

### Specs, CLI and data

17. **"`Evaluation(task=my_async_fn)` works like `evaluate(my_async_fn, ...)`."** In an
    `Evaluation`, a value that is neither a `Task` nor a `Processor` is treated as a
    zero-argument *factory*, so the run fails with
    `missing 1 required positional argument`.

    **Do:** use `FunctionTask(fn)` or `lambda: FunctionTask(fn)`. A `FunctionTask`
    passes the `TrialContext` only if the function has a second required positional
    parameter or a parameter named `trial`.
18. **"Sealed means excluded."** A run without `--split` evaluates sealed splits too,
    and their trials count in the headline metrics.
    - **What sealing does:** it hides per-example results in reports, `show`, `compare`,
      progress lines and Phoenix trial pushes. It also forces a selection to take a
      sealed split whole: `--ids/--limit/--sample` inside it raises
      `SealedSelectionError` (exit 2).
    - **What it isn't:** access control. The dataset file, `examples.jsonl`,
      `trials.jsonl`, traces and `datasets push` all contain the examples.

    **Do:** iterate with `--split dev`.
19. **"Paths are relative to the spec file."** Relative dataset paths resolve against
    the current directory, and so does the store (`./.evals`, unless `$GRASP_EVALS_DIR`
    or `--root` is set). `--root` must come *before* the subcommand. Validation-gate
    lookups and `latest` only see runs in that one store.
20. **"A run remembers its spec portably."**
    - A `file.py:attr` spec is recorded by absolute path. `rescore` and `pairwise`
      without `--spec` reload it from there, so moving the file breaks them (exit 2).
    - Module specs (`pkg.mod:attr`) are portable.
    - A rescore child records its parent's spec, not the `--spec` you passed.
21. **"`--timeout` covers judges."** It is per trial and covers the task only. Scorer
    and judge timeouts come only from `Evaluation.scorer_timeout_s`; there is no CLI
    flag. A sync scorer that times out is abandoned, but its thread keeps running.

### Judges (details in [judges.md](judges.md))

22. **"A structured verdict scores itself."** A `ProcessorScorer` whose judge returns a
    pydantic verdict needs `to_scores=`. Without it, every call is a scorer failure
    ("returned an unsupported value ('passed', True)") and the score is absent from the
    metrics.
23. **"Any type works as an `LLMAgent` type parameter."** They must be real classes.
    `type Judged = JudgedOutput[...]` fails at the first call
    (`TypeError: issubclass() arg 1 must be a class`). Write
    `Judged = JudgedOutput[...]` instead.
24. **"`--split dev` is enough to validate on dev labels."** `judge_validation` seals
    `("test",)` by default. A labels file without a `test` split raises `ValueError`,
    even with `--split dev`.
25. **"A judge that agrees with every label passes its gate."** `ValidationGate`
    thresholds read the *lower interval bound* and allow zero missing verdicts by
    default.
    - A judge that agrees with all 15 test labels has accuracy `ci_low` ≈ 0.77 and κ
      `ci_low` ≈ 0.53.
    - With single-class labels, TPR or TNR is absent. κ is undefined when labels and
      verdicts all share one value.
    - A threshold on a missing metric reads "unknown" and fails.
    - A failing gate blocks the whole run, and rescore too, before any trial runs.

## Before you report a result

1. `status` is `completed` and `invalid_reason` is null.
2. For each metric, `n` is the number of examples you expected, and `n_missing` and
   `error_rate` are small. `counts.scorer_failures > 0` means a scorer broke; check
   `show --failures`.
3. In `compare`, read the warnings first:
   - a scorer's version or code changed;
   - the runs cover different subsets;
   - "identical configuration", which means the differences are noise;
   - task errors in only one run.
4. A non-significant difference whose `mde` is larger than the effect you care about is
   inconclusive, not "no change".
5. Never quote individual results from a sealed split.
