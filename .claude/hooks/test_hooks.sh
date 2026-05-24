#!/bin/bash
# Smoke-test the PreToolUse hooks. Self-contained — encodes paths in
# a way that the bash hook's scanner won't treat the test invocation
# itself as a violation.
set -u
HOOKS="$(dirname "$0")"

# Use printf with %s so the path strings live in stdin, not in argv.
ok=0
fail=0
check() {
  local name="$1"
  local script="$2"
  local input="$3"
  local expect_deny="$4"
  local out
  out=$(printf '%s' "$input" | "$script" 2>&1 || true)
  if [ "$expect_deny" = "deny" ]; then
    if echo "$out" | grep -q '"permissionDecision": *"deny"'; then
      printf '  PASS  %s (denied as expected)\n' "$name"
      ok=$((ok+1))
    else
      printf '  FAIL  %s (expected deny, got: %s)\n' "$name" "$out"
      fail=$((fail+1))
    fi
  else
    if [ -z "$out" ]; then
      printf '  PASS  %s (allowed as expected)\n' "$name"
      ok=$((ok+1))
    else
      printf '  FAIL  %s (expected silent allow, got: %s)\n' "$name" "$out"
      fail=$((fail+1))
    fi
  fi
}

echo "=== Read hook ==="
INSIDE='{"tool_input":{"file_path":"/Users/serge/Grasp/repos/grasp-agents/foo.py"}}'
OUTSIDE='{"tool_input":{"file_path":"/Users/serge/Desktop/foo.txt"}}'
TMP='{"tool_input":{"file_path":"/tmp/x.txt"}}'
check "inside working tree" "$HOOKS/check-read-path.sh" "$INSIDE" allow
check "outside (/Users/serge/Desktop)" "$HOOKS/check-read-path.sh" "$OUTSIDE" deny
check "/tmp" "$HOOKS/check-read-path.sh" "$TMP" allow

echo
echo "=== Bash hook ==="
BASH_OK='{"tool_input":{"command":"ls /Users/serge/Grasp/repos/grasp-agents/src"}}'
BASH_BAD='{"tool_input":{"command":"cat /Users/serge/Desktop/notes.txt"}}'
BASH_NEUTRAL='{"tool_input":{"command":"git status"}}'
check "ls inside tree" "$HOOKS/check-bash-path.sh" "$BASH_OK" allow
check "cat outside tree" "$HOOKS/check-bash-path.sh" "$BASH_BAD" deny
check "no /Users/serge path" "$HOOKS/check-bash-path.sh" "$BASH_NEUTRAL" allow

echo
echo "result: $ok passed, $fail failed"
exit "$fail"
