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

step "5. Re-judge v2's stored outputs with a stricter scorer (no re-run)"
STRICT=$(evals rescore "$V2" --spec "${SPEC}:grader_v2_strict" --json -q | run_id)
evals compare "$V2" "$STRICT"

step "6. Pairwise A/B: whose feedback is more specific? (order-swapped judge)"
evals pairwise "$V1" "$V2" --judge "${SPEC}:specific_feedback_judge" -q

step "7. Final check on the sealed test split (reported in aggregate only)"
TEST=$(evals run "${SPEC}:grader_v2" --split test --json -q \
  --fail-under 'pass_rate(agrees_with_teacher)=0.6' | run_id)
evals show "$TEST"

step "8. Everything recorded so far"
evals runs

if [[ -n "${PHOENIX_BASE_URL:-}" ]]; then
  step "9. Mirror the runs to Phoenix (the sealed test run as aggregates only)"
  evals push "$V2"
  evals push "$TEST"
fi
