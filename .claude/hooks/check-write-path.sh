#!/bin/bash
#
# PreToolUse hook: gate Write / Edit tools to the write-allow-list
# declared in .claude/settings.local.json under
# sandbox.filesystem.allowWrite.
#
# Same source-of-truth pattern as check-read-path.sh — edit the
# settings file and this hook follows. Anything that isn't a prefix
# of an allowed path gets denied via `permissionDecision: "deny"`.

set -euo pipefail

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SETTINGS="$PROJECT_DIR/.claude/settings.local.json"
FILE_PATH=$(jq -r '.tool_input.file_path // empty' < /dev/stdin)

# Missing file_path → let the harness handle.
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

if [ ! -f "$SETTINGS" ]; then
  exit 0
fi

# Resolve relative paths against the project dir so prefix matching
# against absolute allowed roots is consistent.
case "$FILE_PATH" in
  /*) abs_path="$FILE_PATH" ;;
  *)  abs_path="$PROJECT_DIR/$FILE_PATH" ;;
esac

ALLOWED=$(jq -r '.sandbox.filesystem.allowWrite[]?' "$SETTINGS")

while IFS= read -r allowed; do
  [ -z "$allowed" ] && continue
  # "." is the kernel sandbox's shorthand for the project dir.
  if [ "$allowed" = "." ]; then
    allowed="$PROJECT_DIR"
  fi
  case "$abs_path" in
    "$allowed"|"$allowed"/*) exit 0 ;;
  esac
done <<< "$ALLOWED"

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: "Write is outside the declared working roots. Ask the user before stepping outside."
  }
}'
