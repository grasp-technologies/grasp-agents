#!/bin/bash
#
# PreToolUse hook: gate the Read tool to the read-allow-list declared
# in .claude/settings.local.json under sandbox.filesystem.allowRead.
#
# Single source of truth — edit settings.local.json and this hook
# follows. Anything that isn't a prefix of one of the allowed paths
# gets denied via `permissionDecision: "deny"`.

set -euo pipefail

SETTINGS="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/.claude/settings.local.json"
FILE_PATH=$(jq -r '.tool_input.file_path // empty' < /dev/stdin)

# Missing file_path → let the harness handle.
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Pull the allow-list, one path per line.
if [ ! -f "$SETTINGS" ]; then
  exit 0
fi
ALLOWED=$(jq -r '.sandbox.filesystem.allowRead[]?' "$SETTINGS")

while IFS= read -r allowed; do
  [ -z "$allowed" ] && continue
  case "$FILE_PATH" in
    "$allowed"|"$allowed"/*) exit 0 ;;
  esac
done <<< "$ALLOWED"

jq -n '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: "Read is outside the declared working roots. Ask the user before stepping outside."
  }
}'
