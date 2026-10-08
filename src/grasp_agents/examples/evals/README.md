# grasp-evals demo: a short-answer grader

A complete evaluation loop over a small grasp-agents pipeline, runnable offline.

- `grader_evals.py` — the system under test (a two-step `SequentialWorkflow`:
  analyzer → feedback writer, versions `v1` and `v2`), its scorers, a pairwise
  judge, and three `Evaluation` definitions (`grader_v1`, `grader_v2`,
  `grader_v2_strict`). `llm_grader_evaluation(llm)` swaps in an `LLMAgent`.
- `data/short_answers.jsonl` — 24 student answers with teacher grades, `dev` and
  `test` splits (`test` is sealed), `topic`/`difficulty` strata.
- `walkthrough.sh` — the loop below, end to end.
- `../notebooks/evals_workflow.ipynb` — the same loop through the Python API.

## The loop from the shell

Run from the repository root (runs land in `$GRASP_EVALS_DIR`, default `./.evals`).
`grasp-evals` is the same as `python -m grasp_agents.evals`.

```bash
SPEC=src/grasp_agents/examples/evals/grader_evals.py

# 1. Validate the data: types, plus the evaluation's integrity checks.
grasp-evals datasets validate "${SPEC}:grader_v1"

# 2. Baseline on the dev split (3 repetitions per example → pass^3).
grasp-evals run "${SPEC}:grader_v1" --split dev

# 3. Read the failures behind the numbers.
grasp-evals show latest --failures
grasp-evals show latest --example bio-3        # input, reference, outputs, transcript

# 4. Fix, rerun, compare (paired by example; exit 1 if a score or the error rate
#    regresses significantly).
grasp-evals run "${SPEC}:grader_v2" --split dev --baseline latest:grader_v1 --fail-on-regression
grasp-evals compare latest:grader_v1 latest:grader_v2

# 5. Change the instrument, not the task: rescore stored outputs (a child run).
#    Unchanged scorers keep their scores; the changed one runs again.
grasp-evals rescore latest:grader_v2 --spec "${SPEC}:grader_v2_strict"

# 6. Pairwise A/B with an order-swapped judge (a run of its own).
grasp-evals pairwise latest:grader_v1 latest:grader_v2 --judge "${SPEC}:specific_feedback_judge"

# 7. The held-out test split: aggregates only, and always the whole split.
grasp-evals run "${SPEC}:grader_v2" --split test --fail-under 'pass_rate(agrees_with_teacher)=0.6'

# 8. History, and Phoenix (with PHOENIX_BASE_URL / PHOENIX_API_KEY set): share the
#    dataset, then mirror runs as experiments (sealed trials are withheld).
grasp-evals runs
grasp-evals datasets push src/grasp_agents/examples/evals/data/short_answers.jsonl
grasp-evals push latest:grader_v1
grasp-evals push latest                        # the test run: aggregates only
```

Every command takes `--json` (one JSON document on stdout, also for errors).
Progress goes to stderr; `--progress json` makes it one JSON object per finished
trial. Exit codes: 0 done, 1 a gate failed (invalid or incomplete run, missed
`--fail-under`, significant regression, invalid dataset), 2 usage error, 3
unexpected or Phoenix error. Runs are addressed by id, unique id prefix, run
directory, `latest`, or `latest:<name>` where the name is the run's name or the
spec attribute (`latest:grader_v1`).

## What to look for

- **v1 → v2**: agreement with teachers rises from about 0.46 to about 0.75 on
  `dev`, and naming the student's key issue from 0 to about 0.9. The writer is
  noisy on purpose, so numbers move from run to run, and with 16 dev examples the
  comparison's interval excludes zero in most runs, not all — rerun a few times
  to see the spread. v2 still misgrades confident wrong answers that use the
  right words (`phy-3`, `math-3`, `chem-3`) — the next iteration's work.
- **pass^3 ≤ pass rate**: the writer is noisy on borderline answers, so the
  same answer does not always get the same verdict.
- **Rescoring** with the stricter `feedback_quality` v2 drops a few v2 examples
  whose feedback is vague; the comparison warns that the scorer version
  changed, so the drop is the instrument, not the task.
- **Pairwise**: v2's feedback is at least as specific in every pair — it wins
  about half and ties the rest. Position consistency is trivially perfect here,
  since the stand-in judge is a deterministic function.
- **Sealed test split**: reports, `show`, `compare` and Phoenix pushes give only
  aggregates, and a run that touches the split must include all of it (no
  `--ids`/`--limit`/`--sample` inside it). This keeps an agent working through
  the CLI from fitting to those examples; it is not access control — the dataset
  file and the run records on disk contain them.
