#!/bin/bash
#
# PreToolUse hook: gate Bash commands so they can't read paths in the
# user's home tree that aren't on the read-allow-list declared in
# .claude/settings.local.json under sandbox.filesystem.allowRead.
#
# Single source of truth — edit settings.local.json and this hook
# follows. System paths (/etc, /var, /private/var/...) are already
# guarded by the kernel-level sandbox.filesystem rules; this hook is
# the extra layer for the user-tree paths the kernel sandbox alone
# won't block.

set -euo pipefail

SETTINGS="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}/.claude/settings.local.json"
COMMAND=$(jq -r '.tool_input.command // empty' < /dev/stdin)

if [ -z "$COMMAND" ]; then
  exit 0
fi

if [ ! -f "$SETTINGS" ]; then
  exit 0
fi

# Pull the allow-list, one path per line.
ALLOWED=$(jq -r '.sandbox.filesystem.allowRead[]?' "$SETTINGS")
if [ -z "$ALLOWED" ]; then
  exit 0
fi

# Extract every /Users/... path token (the user-tree leak surface).
# System paths are handled by the kernel sandbox; we focus on home-tree.
PATHS=$(echo "$COMMAND" | grep -oE '/Users/[^[:space:]"'\'']*' || true)

if [ -z "$PATHS" ]; then
  exit 0
fi

# For each path in the command, check it's a prefix of at least one
# allowed root.
while IFS= read -r path; do
  [ -z "$path" ] && continue
  matched=0
  while IFS= read -r allowed; do
    [ -z "$allowed" ] && continue
    case "$path" in
      "$allowed"|"$allowed"/*) matched=1; break ;;
    esac
  done <<< "$ALLOWED"
  if [ "$matched" -eq 0 ]; then
    jq -n '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: "Bash command references a path outside the declared working tree. Ask the user before reading paths outside it."
      }
    }'
    exit 0
  fi
done <<< "$PATHS"

exit 0
