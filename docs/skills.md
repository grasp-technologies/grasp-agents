# Skills

A skill is a packaged procedure: a folder with a `SKILL.md` file in the
[agentskills.io](https://agentskills.io/specification) format. The session
holds one catalog of skills; an agent with skills enabled sees the catalog's
names and descriptions in its system prompt and loads a skill's full body on
demand with the `load_skill` tool. A user can also invoke a skill directly, as
a slash command. Skills are read-only within a session.

## Key types

| Name | Module | Role |
|---|---|---|
| `SkillRegistry` | `skills/registry.py` (root export) | The session's catalog, keyed by name |
| `Skill`, `SkillFrontmatter` | `skills/types.py` | A loaded skill: frontmatter, body, path |
| `SkillFilter` | `skills/types.py` | Per-agent allow/block view over the catalog |
| `discover_skills`, `load_skill_md`, `parse_skill_md` | `skills/loader.py` | Loading from disk |
| `load_skill`, `list_skills` | `skills/tools.py` | Agent-facing tools |
| `make_skills_section` | `skills/injection.py` | The `skills` system-prompt section |
| `parse_slash_command`, `parse_named_args` | `skills/slash.py` | Slash-command parsing for front-ends |
| `match_invocation_wrapper` | `skills/registry.py` | Detect a rendered skill invocation |

## The SKILL.md format

```markdown
---
name: proofread
description: Use when asked to proofread, edit, or fix the grammar of text.
metadata:
  grasp:
    inject_body: false
    disable_model_invocation: false
---

Proofread the user's text: fix grammar, spelling, and punctuation ...
```

- `name` (required): lowercase letters, digits and single hyphens, at most 64
  characters, and equal to the folder name.
- `description` (required, at most 1024 characters): the only text the model
  uses to decide whether the skill applies.
- Optional: `license`, `compatibility` (at most 500 characters), `metadata`,
  `allowed-tools`. Unknown keys are kept.
- Framework keys live under `metadata.grasp`:

| Key | Effect |
|---|---|
| `disable_model_invocation: true` | Hidden from the catalog and refused by `load_skill`; still slash-invocable by a user |
| `inject_body: true` | The body is appended to the system prompt after the catalog, so the model never needs to load it |

`allowed-tools` is shown to the model as a hint. It is not enforced; enforce
tool permissions with a before-tool hook or approvals (see
[Agent loop](agent-loop.md)).

## Wiring

```python
from grasp_agents import LLMAgent, SessionContext, SkillRegistry

skills = SkillRegistry.from_path("skills/")          # a parent dir, one skill dir, or one SKILL.md
ctx = SessionContext[None](state=None, skills=skills)

with ctx:
    agent = LLMAgent[str, str, None](
        name="assistant",
        llm=llm,
        enable_skills=True,
        skill_exclude={"deploy"},
    )
```

`enable_skills=True` (default `False`) adds:

- the `skills` system-prompt section, which renders the instructions in
  `skills/skill_instructions.md` followed by the `<available_skills>` catalog
  (one `<skill>` per visible skill with `<name>`, `<description>`, and
  `<compatibility>` / `<allowed-tools>` when set), plus any `inject_body`
  bodies. The section renders nothing when `ctx.skills` is unset or no skill is
  visible.
- the `load_skill(name)` tool.

`skill_include` / `skill_exclude` build a `SkillFilter` for this agent only:
`include` is an allowlist, `exclude` a blocklist, both apply the intersection,
`include=set()` hides everything. The filter applies to the section, to
`load_skill` and to `list_skills`. `AgentTool` takes `enable_skills=` for its
child agents too.

`list_skills(refresh=False)` is opt-in: pass it in `tools=` when the agent
should be able to re-list the catalog (with licenses) or re-walk the skill
sources mid-session (`refresh=True` → `SkillRegistry.refresh`).

## Loading

`load_skill(name)` re-reads the skill file from disk on each call, so edits to
an existing skill's body are visible immediately. It returns the body with
argument placeholders (`$ARGUMENTS`, `$NAME`) removed: a model-invoked skill
applies to the user's request already in the conversation. An unknown, hidden
or filtered-out name raises `SkillNotFoundError`, which the model sees as a tool
error.

## Slash commands

The framework does not own an input loop. Front-ends parse user input and
render the invocation:

```python
from grasp_agents.skills import parse_slash_command

cmd = parse_slash_command("/proofread Their going to the store")
if cmd is not None and cmd.name in ctx.skills:
    text = ctx.skills.render_invocation(cmd.name, args=cmd.args)
    out = await agent.run(text)
```

`render_invocation(name, args=None, wrap=True)` fills `$ARGUMENTS` (the whole
argument string) and `$NAME` placeholders (from a mapping, e.g.
`parse_named_args("--lang=fr")`). A body with no matching placeholder gets the
arguments appended as a `User input:` block instead. With `wrap=True` the
message is wrapped in `<system-reminder subject="user invoked skill <name>">`,
which the skill instructions tell the model to follow without calling
`load_skill`. `match_invocation_wrapper(text)` returns the skill name of such a
message, so a UI can render it as a command. The Textual TUI uses these
helpers for its skills palette.

## Where skills come from

Skills are read from the host filesystem with `pathlib`, not through
`ctx.file_backend`. A sandboxed or MCP file backend does not affect them, and
files the agent writes with its file tools are never registered as skills.
Ship skills as package assets (a directory next to your code) and point
`SkillRegistry.from_path` or `add_source` at it. A directory that fails to
load one skill logs the error and skips that skill; a missing source path
raises `SkillFormatError`.

## Contracts

1. The catalog is built once per registry; new skill folders appear only after
   `refresh()` (or `list_skills(refresh=True)`), while body edits are seen by the
   next `load_skill` (`SkillRegistry`, `load_skill`).
2. A skill's folder name must equal its frontmatter `name` (`load_skill_md`).
3. `disable_model_invocation` skills are never shown to or loadable by the
   model (`render_available_skills_block`, `load_skill`).
4. The skills section is cache-stable: it does not change between turns.
5. Skills are trusted content: their bodies enter the system prompt or tool
   results unfenced (see [Security](security.md)).
6. Duplicate names: the later definition overrides the earlier, with a warning.

## Limitations

- No skill-authoring path: the agent cannot create or edit skills through its
  tools.
- No per-turn skill selection: `SkillRegistry.set_selector` stores a selector,
  but the catalog section always renders every visible skill.
- `allowed-tools` is informational only.

## See also

- [Memory](memory.md) (cross-session facts, as opposed to procedures),
  [Security](security.md), [Hooks](hooks.md).
- Code: `skills/`.
- Examples: `src/grasp_agents/examples/tui/skills_copilot.py`,
  `src/grasp_agents/examples/tui/skills/`,
  `src/grasp_agents/examples/notebooks/memory_skills.ipynb`.
