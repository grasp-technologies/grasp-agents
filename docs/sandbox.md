# Sandbox

An agent gets no shell, kernel or filesystem access unless you wire it. Command execution is granted only through an `ExecutionEnvironment` on `SessionContext.environment`, which pairs a file surface and an exec surface over the same filesystem under one `SandboxPolicy`. This page covers the policy and its two enforcement planes, the local confinement modes and what each really enforces, the E2B cloud environment, and the `Bash` / `BashSession` tools.

**`confinement="none"` is not a sandbox.** It is the default for `local_environment()`. It runs plain host subprocesses that can read and write anything your user can and reach the network. Use it only for trusted, personal or development work.

## Key types

| Name | Module | Role |
|---|---|---|
| `SandboxPolicy`, `NetworkPolicy` | `sandbox/policy.py` | One policy for both planes. |
| `ExecutionEnvironment` (ABC) | `sandbox/environment.py` | `policy`, `file_backend`, `exec_backend`, async context manager. |
| `ExecBackend`, `ExecSession` (ABCs) | `sandbox/exec_backend.py` | One-shot commands; a persistent shell. |
| `KernelSession` (ABC) | `sandbox/kernel.py` | A live language kernel (see [Notebooks](notebooks.md)). |
| `SessionCapable`, `KernelCapable`, `SnapshotCapable`, `NetworkUpdatable` | `sandbox/exec_backend.py`, `sandbox/kernel.py`, `sandbox/environment.py` | Optional capabilities: runtime-checkable `Protocol`s, detected with `isinstance`. |
| `local_environment`, `LocalEnvironment` | `sandbox/local/environment.py` | Host filesystem + host subprocesses. |
| `LocalExecBackend`, `SeatbeltExecBackend`, `SrtExecBackend` | `sandbox/local/` | Exec backends per confinement. |
| `e2b_environment`, `E2BEnvironment`, `E2BExecBackend`, `E2BFileBackend`, `E2BSandboxExpiredError` | `sandbox/e2b/` | E2B cloud sandbox (`grasp-agents[e2b]`). |
| `sweep_e2b`, `E2BSweepReport` | `sandbox/e2b/cleanup.py` | Remove what E2B environments left on the account. |
| `EnvironmentConfig`, `load_environment_config`; `E2BEnvironmentConfig`, `E2BTemplateConfig` | `sandbox/config.py`, `sandbox/e2b/config.py` | JSON configs. |
| `Bash`, `BashSession`, `bash_tools` | `tools/bash.py`, `tools/bash_session.py` | Shell tools. |

Import the sandbox names from `grasp_agents.sandbox`, except `ExecSession` / `SessionCapable` (`grasp_agents.sandbox.exec_backend`) and `KernelSession` / `KernelCapable` (`grasp_agents.sandbox.kernel`). The shell tools come from `grasp_agents.tools`.

## Wiring

```python
from grasp_agents import LLMAgent, SessionContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools import FileToolkit, bash_tools

env = local_environment(allowed_roots=["./workspace"], confinement="seatbelt")
ctx = SessionContext(environment=env)          # ctx.file_backend comes from env
agent = LLMAgent[str, str, None](
    "dev", llm=llm, ctx=ctx, tools=[*FileToolkit().tools(), *bash_tools()]
)
```

`ctx.exec_backend` is a read-only property of `ctx.environment`, so an exec surface without its co-located filesystem cannot be expressed. Passing both `environment` and a different `file_backend` raises at construction. A file-only session (just `file_backend=`) is fine.

## `SandboxPolicy`

| Field | Meaning |
|---|---|
| `allowed_roots` | The read+write workspace; also the write allowlist. The first root is the default working directory. |
| `readonly_roots` | Readable, not writable. |
| `deny_read` / `allow_read` | Carve unreadable regions; `allow_read` re-allows within them (**allow wins**). |
| `deny_write` | Carve write-protected regions out of `allowed_roots` (**deny wins**; no carve-back). |
| `include_dotfile_denylist`, `dotfile_overrides` | Credential deny set (`.ssh`, `.aws`, `.env`, …) and explicit opt-ins. |
| `network`, `allowed_domains`, `denied_domains` | Egress policy (`NONE` default, `LOOPBACK`, `ALLOWLIST`, `ALL`); `denied_domains` is checked first. |
| `env`, `env_scrub` | Base environment for commands; `fnmatch` patterns scrubbed from the inherited host environment. |
| `kernel_setup_code`, `kernel_startup_timeout` | Kernel startup snippet and readiness ceiling (120 s). |

### Two planes

- **Tool plane.** File tools call `FileBackend.validate_path(path, access="read" | "write")`: containment in the roots, sensitive paths, carve-outs. File tools are confined to `allowed_roots` + `readonly_roots` for reads too.
- **OS plane.** Local confinements generate an OS profile from the same fields: a Seatbelt profile or `srt` settings. Subprocesses get broad reads minus `deny_read` (OS-level read allowlisting would break ordinary tooling), and writes only to `allowed_roots` and scratch temp. Credential and system denies are emitted after every allow, so they always win.

A path is confined on a surface only if that surface's plane enforces it. Under `none` only the tool plane exists.

## Local confinement modes

| `confinement` | Behavior |
|---|---|
| `"none"` (default) | Plain subprocess. No OS isolation; `readonly_roots` and carve-outs are not enforced for commands; `network` is recorded, not enforced. `env_scrub` still applies. |
| `"seatbelt"` | macOS `sandbox-exec` with a generated profile. Writes confined; reads broad minus denies; network all-or-nothing. Raises `RuntimeError` off macOS or without `sandbox-exec`. |
| `"srt"` | Delegates to the `srt` sandbox runtime CLI (must be on `PATH`). Adds domain-allowlisted egress through its proxy. |
| `"auto"` | macOS → `seatbelt`. Linux with `bwrap` on `PATH` → `bwrap`, which raises `NotImplementedError`. Anything else → `RuntimeWarning` and fallback to `none`. |
| `"bwrap"` | Accepted by the type; raises `NotImplementedError` at construction. |

### Network policy × backend

Unenforceable combinations fail at construction, not on the first command — except under `none`, which enforces nothing and accepts every value.

| `network` | `none` | `seatbelt` | `srt` | E2B |
|---|---|---|---|---|
| `NONE` | not enforced | enforced | enforced (empty allowlist) | enforced (no internet) |
| `LOOPBACK` | not enforced | `ValueError` | `NotImplementedError` | `NotImplementedError` |
| `ALLOWLIST` | not enforced | `ValueError` | enforced (proxy) | enforced (E2B egress rules) |
| `ALL` | open | open | `NotImplementedError` | open (`denied_domains`: `NotImplementedError`) |

### Environment, limits and Python

- `env_scrub` (default `DEFAULT_ENV_SCRUB`: `*_API_KEY`, `*_TOKEN`, `*SECRET*`, `*PASSWORD*`, `*_KEY`, `DATABASE_URL`, `SSH_AUTH_SOCK`, …) removes matching inherited host variables, case-insensitively, under every local confinement. Explicit `env` is never scrubbed; `env_scrub=()` disables it. `inherit_host_env=False` starts from an empty environment.
- `limits={"cpu_timeout": ..., "max_memory_mb": ..., "max_file_size_mb": ...}` (a `ResourceLimits` dict) or `supervisor=ProcessSupervisor(SupervisorLimits(...))` apply `setrlimit` ceilings to one-shot commands. Persistent shells and kernels are exempt. `SupervisorLimits` also sets `overall_timeout` (600 s), `idle_timeout` and `max_output_chars` per stream.
- `python=` selects the interpreter kernels run with. `packages=[...]` are **verified** present at setup; a missing one raises with the `pip install` command to run. The framework does not change your environment. `provision=True` instead creates a dedicated venv at `<first allowed root>/.venv` and installs missing packages there.

### JSON config

```json
{
  "confinement": "srt",
  "filesystem": {"allowed_roots": ["."], "deny_write": ["./.git"]},
  "network": {"policy": "allowlist", "allowed_domains": ["pypi.org"]},
  "exec": {"inherit_host_env": true, "cpu_timeout": 60}
}
```

`load_environment_config("sandbox.json").build()` returns a `LocalEnvironment`. `confinement` defaults to `"none"`, and an omitted `env_scrub` keeps the default denylist.

## E2B environment

```python
from grasp_agents import SessionContext
from grasp_agents.sandbox import NetworkPolicy, e2b_environment

env = e2b_environment(template="my-template", network=NetworkPolicy.NONE, code_interpreter=True)
async with env:                       # creates the sandbox; kills it on exit
    ctx = SessionContext(environment=env)
    ...
```

- `grasp-agents[e2b]` needs the `e2b` SDK 2.41 or later: earlier releases cannot delete the named snapshots the environment takes, nor change a live sandbox's network.
- The sandbox is created on `async with` and killed on exit (`pause_on_exit=True` pauses instead). `E2BEnvironment.from_sandbox(...)` wraps an existing sandbox. `env.started` says whether a sandbox is live.
- **Lazy start.** With `lazy=True` (also an `E2BEnvironmentConfig` field), entry creates nothing: the first file or exec call starts the sandbox, once, even when calls race, so a session that never touches it never pays for one. A failed start fails like a failed entry: later calls raise at once, without creating another sandbox, until the environment is re-entered. Exit, `restore()`, `pause()` and `resume()` wait for a start in flight. A start keeps its sandbox while the environment is still entered and holding no other sandbox, even when every call waiting for it was cancelled (a per-call timeout shorter than the start, say): the next call uses it, and exit kills it. A start that finishes after the environment exited or swapped its sandbox kills the sandbox it created. `pause()` before the start raises.
- **Network.** `network=NetworkPolicy.NONE` (the default) turns the sandbox's internet off and `ALL` on. `ALLOWLIST` lets it reach only `allowed_domains` (also `E2BEnvironmentConfig` fields): exact host names, or `*.`-wildcards for subdomains: `*.pythonhosted.org` covers `files.pythonhosted.org` but not `pythonhosted.org` itself, so list both when you need both. Everything else is blocked, raw IP addresses included. E2B denies only addresses, never hosts, so `denied_domains` (checked first) are enforced by leaving them out of the allowlist; a denied host that an allowed wildcard covers raises `NotImplementedError`, and so does any denied host with `ALL`. `network` takes a `NetworkPolicy` or its string value (`"all"`). Every create the environment makes, a restore's included, carries the network settings.
  - **Changing it while the sandbox runs.** `await env.update_network(network, allowed_domains=..., denied_domains=...)` (`NetworkUpdatable`) applies a new policy to the live sandbox and records it: `env.policy` and `env.exec_backend.policy` report it, and every sandbox the environment creates afterwards (a lazy start, a restore, the replacement of one E2B removed) starts with it. Before a lazy start it only records. A typical use: install pinned packages through an allowlist of `pypi.org` and `files.pythonhosted.org`, then `update_network(NetworkPolicy.NONE)` before running anything else. A sandbox `pause()` paused refuses the change until `resume()`; when E2B no longer has the sandbox, the change is recorded for its replacement and `E2BSandboxExpiredError` is raised. `resume()` of another sandbox keeps that sandbox's own network. Local environments fix their egress when built: build a new one to change it.
- **What it creates is marked.** Every sandbox the environment creates carries the metadata key `"grasp-agents"`, set to its `label` (default `"default"`; also an `E2BEnvironmentConfig` field), next to your own `metadata`; the environment's key wins over one you pass. E2B snapshots take no metadata, so `snapshot()` names each one `grasp-agents-<label>-<UTC time>-<random>`. The name is unique per snapshot: E2B points a reused name at its newest snapshot, which would change what an earlier ref restores. A label is up to 32 lowercase letters, digits or dashes; deployments that share an E2B team give each its own.
- **Cleaning up the account.** A process that ends without exiting its environment leaves its sandbox running until its keep-alive ends, one that exits with `pause_on_exit` leaves it paused until something kills it, and a session whose records are gone leaves its snapshots. `sweep_e2b(older_than=..., keep_snapshots=..., label=...)` first kills the sandboxes tagged with `label` (and matching `metadata=`, if given) that have been paused longer than `older_than`, or running too with `states=("paused", "running")`; then it deletes the snapshots named with `label` taken longer ago than `older_than`, except those in `keep_snapshots`. It changes nothing unless `dry_run=False`, and returns an `E2BSweepReport` of what it killed, deleted, kept, or found still in use, and of snapshots named with the label whose time is not a date (`undated`, never deleted). `states` must name at least one state; a sandbox in any other is left alone.
  - Pass `keep_snapshots=await read_fs_snapshot_refs(store)` over every store your sessions use ([Rollback](rollback.md#ending-a-session)). A snapshot a record holds but the set misses is deleted, and restoring it then fails, so the reader raises on a record it cannot read rather than leave its refs out.
  - A snapshot is in no record until its checkpoint lands, so `older_than` must exceed your longest checkpoint; hours are a safe order. When and how often to sweep, and how long a paused sandbox may wait for its user, are yours to decide.
- `allowed_roots` are paths inside the sandbox (default `/home/user/workspace`), created when the sandbox starts. `deny_read` / `allow_read` / `deny_write` apply on the tool plane; the VM is the OS boundary.
- Commands: non-zero exits are returned as `ExecResult.returncode`, never raised. A call with no timeout gets `default_timeout` (600 s). The backend enforces the deadline on the command's output stream: a command that runs past it returns `OVERALL_TIMEOUT` with the output it produced, and the task awaiting it is not cancelled. The kill then reaches the shell E2B runs the command in (`/bin/bash -l -c`), not the processes that shell started (see Limitations).
- Persistent shells (`BashSession`) are supported. A shell runs until it is closed, a command in it times out (which kills it), or its sandbox is replaced; it has no lifetime cap of its own. Kernels (`RunPython`, `RunCell`) need `code_interpreter=True`.
- **Snapshots vs pause.** `snapshot()` creates a persistent E2B snapshot and returns its id; `restore(ref)` spawns a fresh sandbox from it — a true rewind, used by `fs_snapshot_policy` ([Rollback](rollback.md)). A snapshot is self-contained: one taken in a sandbox restored from another survives that one's deletion. `pause()` / `resume()` suspend and reconnect the *same* sandbox; no rewind. `resume()` reconnects with the class the environment creates sandboxes with (the code-interpreter one with `code_interpreter=True`) and its configured API key, domain and keep-alive timeout; it does not create the allowed roots, so a sandbox started elsewhere must have them.
- **Swapping sandboxes.** `restore()` kills the previous sandbox when the environment owns it. `resume()` of another sandbox lets go of the previous one as exit would: it kills an owned one, or pauses it with `pause_on_exit`, and the shells and kernels bound to it close. A kill or pause that fails there is logged, not raised. Cancelling `restore()` or `resume()` does not cancel a create or connect E2B may already be running: the sandbox a cancelled restore creates is killed, and one a cancelled resume connects is let go as exit would, unless it is the one the environment holds. Both raise `RuntimeError` outside `async with env:`, where nothing would kill the sandbox they start.
- **Fresh ref.** Before a lazy start, `snapshot()` returns `FRESH_SNAPSHOT_REF` without starting anything. Restoring that ref drops a lazy environment's sandbox until the next call (and forgets a failed start), starts a new sandbox from the configured template on an eager one, and raises on a `from_sandbox` environment, which has nothing to create one from.
- **Shells and kernels follow their sandbox.** A shell or kernel is bound to the sandbox it was opened on. Once `restore()` or exit replaces that sandbox, it reports `closed` and never addresses the new one, where its pid could belong to another process. The next `BashSession`, `RunPython` or `RunCell` call opens a new one on the live sandbox, starting a lazy one, and tells the model its state was lost. After a rollback that restored a snapshot, a kernel re-attaches to the code context the snapshot kept, with no reset notice; a kernel whose seeded context the live sandbox does not have opens a fresh one and reports the reset.
- **Streams across a pause.** `pause()`, `snapshot()` (E2B pauses the sandbox while it takes one) and `resume()` drop the sandbox's open connections, command streams included; its processes survive. A shell reattaches to its output before its next command and keeps its state; if it cannot, it kills the old shell, starts a new one and the result tells the model its shell state was lost. Kernels need nothing: each call is a request of its own, so `resume()` to the same sandbox keeps them.
- **A sandbox E2B removes.** When E2B no longer has the live sandbox (it reached its keep-alive `sandbox_timeout` or the plan's lifetime cap, or was killed from elsewhere), the call that finds out fails with `E2BSandboxExpiredError`, saying the sandbox's files, shells and kernels are gone; the call is not retried. E2B reports such calls like timeouts, so a failure that could mean either is checked against the sandbox's info first. The environment drops the dead sandbox, and the next call starts a new one from the create params, lazy or eager; shells and kernels bound to the dead one reopen with their reset notices. Nothing is restored from a snapshot. An environment from `from_sandbox` has nothing to create one from, so its calls keep raising the error.
- Python packages are baked into a template; the framework never installs into a sandbox. `E2BTemplateConfig` describes a template (base image + build steps) and builds it idempotently by alias; `E2BEnvironmentConfig` / `load_e2b_config` describe the whole environment.

## Shell tools

**`Bash`** runs each command as a fresh `/bin/sh -c` through `ctx.exec_backend`. It is the primary shell tool.

- Non-interactive: no TTY. Environment and shell variables do not persist. The working directory does: it is round-tripped through `agent_ctx.shell_state`, so `cd` sticks across calls. A per-call `cwd` overrides it.
- `timeout` from the model defaults to 120 s and is clamped to 600 s. A leading bare `sleep` is rejected (`block_leading_sleep=True`).
- With a `progress_callback` (direct use), a heartbeat fires after 2 s of silence and every 10 s after.
- `auto_background_at` (default `None` on `Bash`; 120 s via `bash_tools()`) moves a long command to the background. Its output streams to a `.grasp/tasks/<call_id>.log`, and `KillTask` stops it (see [Background tasks](background-tasks.md)). `blocks_final_answer` defaults to `True`.
- `max_inline_result_chars` defaults to 8,000: larger output is excerpted and spilled to a file.
- Claims global write exclusivity (`"/"`), so it never runs concurrently with file writers in one batch. Output is fenced as untrusted.

`bash_tools(...)` returns `[Bash, KillTask]` with backgrounding on.

**`BashSession`** keeps one persistent shell per agent (`agent_ctx.session_holder`), so `cd`, exports and shell variables carry across calls. Treat it as experimental; prefer `Bash`.

- Needs a `SessionCapable` backend (local, Seatbelt, `srt` and E2B all are).
- Serial and foreground-only: no backgrounding. A timeout closes the whole session; there is no per-command interrupt.
- When the shell has to be reopened (it died, a timeout killed it, the E2B sandbox it ran in was replaced, or its output could not be reattached after an E2B pause or snapshot), the next result tells the model its shell state was lost.
- Outside an agent loop each call opens a throwaway session.

## Contracts

1. Exec is reachable only through `ctx.environment`; a divergent standalone `file_backend` raises (`SessionContext._reconcile_environment`).
2. One `SandboxPolicy` feeds both planes; never build the file and exec surfaces from different policies (`local_environment`, `e2b_environment`).
3. Co-location is a construction contract: an `ExecBackend` must execute against the filesystem its paired `FileBackend` addresses. The filesystem can stand alone; exec cannot (`ExecBackend` docstring).
4. Unenforceable network policies fail at construction under `seatbelt`, `srt` and E2B (`_build_exec_backend`, `build_srt_settings`, `e2b_environment`).
5. Primary interfaces are ABCs (`ExecutionEnvironment`, `ExecBackend`, `ExecSession`, `KernelSession`, `FileBackend`). Optional capabilities are `Protocol`s checked with `isinstance`; implement only what your backend supports.
6. One-shot resource limits never apply to persistent shells or kernels (`SupervisorLimits`).
7. An E2B command timeout ends the wait for the command, not the task awaiting it: the backend passes E2B no command deadline, enforces its own on the output stream, and kills the shell running the command (`E2BExecBackend.stream`).
8. An E2B shell or kernel addresses only the sandbox it was opened on, compared by sandbox id; once another sandbox or none is live, it reports `closed` (`SandboxHandle.live`, `E2BExecSession`, `E2BKernel`).
9. E2B's `pause`, `snapshot`, `resume` and `restore` mark every open command stream as dropped; a shell on the same sandbox reattaches to its output before its next command, or replaces itself and reports the reset through `ExecSession.take_reset` (`SandboxHandle.drop_streams`, `E2BExecSession`).
10. A lazy E2B environment runs one start at a time, and exit, `restore`, `pause`, `resume` and `update_network` hold the same lock as the start. A start keeps its sandbox whenever the environment is still entered and holding no other sandbox, whether or not a caller still awaits it; otherwise it kills the sandbox and raises (`SandboxHandle.ensure`).
11. An E2B environment never leaves an owned sandbox it stops holding, or one it obtained for a cancelled call, running unheld: `restore` kills the previous one, `resume` of another sandbox lets go of the previous one as exit would, and a cancelled `restore` or `resume` finishes in a task of its own and releases what it obtained (`E2BEnvironment.restore`, `resume`).
12. A call that fails because E2B no longer has the sandbox raises `E2BSandboxExpiredError` and drops that sandbox from the environment, which starts a new one on the next call when it has create params; a failure the sandbox survived surfaces unchanged (`SandboxHandle.raise_if_expired`).
13. `sweep_e2b` touches only sandboxes tagged with its label and snapshots named with it, kills sandboxes before it deletes snapshots, deletes only snapshots older than `older_than` that `keep_snapshots` does not hold, and changes nothing in a dry run (`sweep_e2b`).
14. `update_network` changes the live sandbox's egress and records the policy in one step under the environment's lock, so no sandbox the environment creates afterwards starts with the old one; a policy E2B cannot enforce raises and changes nothing (`E2BEnvironment.update_network`).

## Limitations

- `bwrap` (Linux) is not implemented, and `auto` on Linux raises when `bwrap` is installed. Use `srt` or `none` there.
- `NetworkPolicy.LOOPBACK` is accepted only by `none`, which does not enforce it.
- Seatbelt cannot express per-domain egress; use `srt`.
- Only E2B offers filesystem snapshots; there is no local snapshot backend.
- An E2B `restore()` loses persistent shell state; kernel state survives only when a rollback re-attaches the context its snapshot kept.
- On E2B, a command timeout or a `KillTask` kills the shell running the command, and processes it started may keep running in the sandbox. The local backend kills the command's whole process group.
- On E2B, a one-shot command still running when `pause()` or `snapshot()` drops the sandbox's connections loses the rest of its output: its call fails or times out, while the command itself may keep running. Only persistent shells reattach.
- `sweep_e2b` dates a snapshot by the time in its name, so snapshots without such a name (taken by other tools, or before environments named them) are never swept; delete those by hand.
- When E2B removes a sandbox, only the call that finds out hears that its files are gone. Other agents sharing the environment find out through their shell and kernel reset notices, or not at all if they only use files.
- An E2B allowlist cannot carve a denied host out of an allowed wildcard; list the allowed hosts one by one instead.
- Only E2B changes a live environment's network (`NetworkUpdatable`); a local environment's egress is fixed when it is built.
- No Docker or SSH exec backend ships; implement `ExecutionEnvironment` / `ExecBackend` for other targets.

## See also

[Files](files.md) · [Notebooks](notebooks.md) · [Background tasks](background-tasks.md) · [Rollback](rollback.md) · [Security](security.md) · code: `sandbox/`, `tools/bash.py`, `tools/bash_session.py`
