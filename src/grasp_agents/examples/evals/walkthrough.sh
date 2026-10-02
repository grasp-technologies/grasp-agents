#!/usr/bin/env bash
# The grasp-evals loop, from the shell. Run from the repository root:
#
#   bash src/grasp_agents/examples/evals/walkthrough.sh
#
# Runs land in $GRASP_EVALS_DIR (default ./.evals). Set PHOENIX_BASE_URL (and
# PHOENIX_API_KEY) to also push the final run to Phoenix.
set -euo pipefail

SPEC=src/grasp_agents/examples/evals/grader_evals.py
evals() { uv run --no-sync python -m grasp_agents.evals "$@"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run_id() { python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])'; }

step "1. Validate the dataset against the evaluation's types and checks"
evals datasets validate "${SPEC}:grader_v1"

step "2. Baseline: grader v1 on the dev split (3 repetitions per example)"
V1=$(evals run "${SPEC}:grader_v1" --split dev --json -q | run_id)
evals show "$V1"

step "3. Read the failures, then one of them in detail"
evals show "$V1" --failures
evals show "$V1" --example bio-3

step "4. Candidate: grader v2 on the same examples, compared with the baseline"
V2=$(evals run "${SPEC}:grader_v2" --split dev --baseline "$V1" \
  --fail-on-regression --json -q | run_id)
evals compare "$V1" "$V2"

step "5. Re-judge v2's stored outputs with a stricter evaluator (no re-run)"
STRICT=$(evals rescore "$V2" --spec "${SPEC}:grader_v2_strict" --json -q | run_id)
evals compare "$V2" "$STRICT"

step "6. Pairwise A/B: whose feedback is more specific? (order-swapped judge)"
evals pairwise "$V1" "$V2" --judge "${SPEC}:specific_feedback_judge" -q

step "7. Final check on the sealed test split (reported in aggregate only)"
TEST=$(evals run "${SPEC}:grader_v2" --split test --json -q \
  --fail-under 'pass_rate(agrees_with_teacher)=0.6' | run_id)
evals show "$TEST"

step "8. Is the feedback judge any good? Agreement with teachers' labels on dev"
J1=$(evals run "${SPEC}:judge_v1_validation" --split dev --json -q | run_id)
J2=$(evals run "${SPEC}:judge_v2_validation" --split dev --json -q | run_id)
evals show "$J2" --failures
evals compare "$J1" "$J2"

step "9. Probe it: degraded feedback must fail, padded feedback must not move"
evals run "${SPEC}:judge_v2_probes" -q

step "10. The judge's final check on its sealed test split, then grading with it"
evals run "${SPEC}:judge_v2_validation" --split test -q
JUDGED=$(evals run "${SPEC}:grader_v2_judged" --split dev --json -q | run_id)
evals show "$JUDGED"

step "11. More labels: sample outputs to label (blind), fill them in, merge"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/grasp-evals-labels.XXXXXX")
LABELS=src/grasp_agents/examples/evals/data/feedback_labels.jsonl
evals labels sample "$JUDGED" --score feedback_quality --against names_key_issue \
  -n 5 --exclude "$LABELS" -o "$WORK/to_label.jsonl"
cp "$LABELS" "$WORK/labels.jsonl"
# A person would fill in each "reference"; here every feedback is marked specific.
python3 - "$WORK/to_label.jsonl" <<'PY'
import json, sys
path = sys.argv[1]
rows = [json.loads(line) for line in open(path)]
with open(path, "w") as f:
    for row in rows:
        f.write(json.dumps({**row, "reference": True}) + "\n")
PY
evals labels import "$WORK/to_label.jsonl" --into "$WORK/labels.jsonl" --labeler demo
evals labels show "$WORK/labels.jsonl"

step "12. Everything recorded so far"
evals runs

if [[ -n "${PHOENIX_BASE_URL:-}" ]]; then
  step "13. Mirror the runs to Phoenix (the sealed test run as aggregates only)"
  evals push "$V2"
  evals push "$TEST"

  step "14. The grader in production, traced into Phoenix, then scored online"
  TELEMETRY_COLLECTOR_HTTP_ENDPOINT="$PHOENIX_BASE_URL/v1/traces" \
    uv run --no-sync python src/grasp_agents/examples/evals/production.py
  sleep 6  # the demo's completion buffer: spans still being exported
  evals online "${SPEC}:grader_online" --since 10m

  step "15. Production inputs as dataset examples"
  evals datasets from-traces "${SPEC}:grader_online" --since 10m \
    -o "$WORK/production_inputs.jsonl"
fi
