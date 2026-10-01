# grasp-evals demo: a short-answer grader

A complete evaluation loop over a small grasp-agents pipeline, runnable offline.

- `grader_evals.py` — the system under test (a two-step `SequentialWorkflow`:
  analyzer → feedback writer, versions `v1` and `v2`), its evaluators, a pairwise
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
grasp-evals datasets validate $SPEC:grader_v1

# 2. Baseline on the dev split (3 repetitions per example → pass^3).
grasp-evals run $SPEC:grader_v1 --split dev

# 3. Read the failures behind the numbers.
grasp-evals show latest
grasp-evals show latest --example bio-3        # input, reference, outputs, scores

# 4. Fix, rerun, compare (paired by example; exit 1 on a significant regression).
grasp-evals run $SPEC:grader_v2 --split dev --baseline <v1-run-id> --fail-on-regression
grasp-evals compare <v1-run-id> latest

# 5. Change the instrument, not the task: rescore stored outputs (a child run).
grasp-evals rescore <v2-run-id> --spec $SPEC:grader_v2_strict

# 6. Pairwise A/B with an order-swapped judge (a run of its own).
grasp-evals pairwise <v1-run-id> <v2-run-id> --judge $SPEC:specific_feedback_judge

# 7. The held-out test split: aggregates only.
grasp-evals run $SPEC:grader_v2 --split test --fail-under 'pass_rate(agrees_with_teacher)=0.6'

# 8. History, and Phoenix (with PHOENIX_BASE_URL / PHOENIX_API_KEY set).
grasp-evals runs
grasp-evals push latest
```

Every command takes `--json` (one JSON document on stdout, progress on stderr).
Runs are addressed by id, unique id prefix, `latest` or `latest:<name>`.

## What to look for

- **v1 → v2**: agreement with teachers rises from ≈0.46 to ≈0.75 on `dev` with a
  CI that excludes zero; naming the student's key issue goes from 0 to ≈0.9.
  v2 still misgrades confident wrong answers that use the right words
  (`phy-3`, `math-3`, `chem-3`) — the next iteration's work.
- **pass^3 < pass rate**: the writer is noisy on borderline answers, so the
  same answer does not always get the same verdict.
- **Rescoring** with the stricter `feedback_quality` v2 drops a few v2 examples
  whose feedback is vague; the comparison warns that the evaluator version
  changed, so the drop is the instrument, not the task.
- **Pairwise**: v2's feedback is more specific in most pairs, the judge is
  position-consistent, and ties are reported as such.
- **Sealed test split**: reports show no per-example rows.
