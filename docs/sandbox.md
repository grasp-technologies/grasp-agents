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
| `SessionCapable`, `KernelCapable`, `SnapshotCapable` | `sandbox/exec_backend.py`, `sandbox/kernel.py`, `sandbox/environment.py` | Optional capabilities: runtime-checkable `Protocol`s, detected with `isinstance`. |
| `local_environment`, `LocalEnvironment` | `sandbox/local/environment.py` | Host filesystem + host subprocesses. |
| `LocalExecBackend`, `SeatbeltExecBackend`, `SrtExecBackend` | `sandbox/local/` | Exec backends per confinement. |
| `e2b_environment`, `E2BEnvironment`, `E2BExecBackend`, `E2BFileBackend` | `sandbox/e2b/` | E2B cloud sandbox (`grasp-agents[e2b]`). |
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
| `ALLOWLIST` | not enforced | `ValueError` | enforced (proxy) | `NotImplementedError` |
| `ALL` | open | open | `NotImplementedError` | open |

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

- The sandbox is created on `async with` and killed on exit (`pause_on_exit=True` pauses instead). `E2BEnvironment.from_sandbox(...)` wraps an existing sandbox.
- `allowed_roots` are paths inside the sandbox (default `/home/user/workspace`), created on entry. `deny_read` / `allow_read` / `deny_write` apply on the tool plane; the VM is the OS boundary.
- Commands: non-zero exits are returned as `ExecResult.returncode`, never raised. A call with no timeout gets `default_timeout` (600 s).
- Persistent shells (`BashSession`) are supported. Kernels (`RunPython`, `RunCell`) need `code_interpreter=True`.
- **Snapshots vs pause.** `snapshot()` creates a persistent E2B snapshot and returns its id; `restore(ref)` spawns a fresh sandbox from it — a true rewind, used by `fs_snapshot_policy` ([Rollback](rollback.md)). `pause()` / `resume()` suspend and reconnect the *same* sandbox; no rewind.
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
- When the shell has to be reopened (it died or hit the provider's lifetime cap), the next result tells the model its shell state was lost.
- Outside an agent loop each call opens a throwaway session.

## Contracts

1. Exec is reachable only through `ctx.environment`; a divergent standalone `file_backend` raises (`SessionContext._reconcile_environment`).
2. One `SandboxPolicy` feeds both planes; never build the file and exec surfaces from different policies (`local_environment`, `e2b_environment`).
3. Co-location is a construction contract: an `ExecBackend` must execute against the filesystem its paired `FileBackend` addresses. The filesystem can stand alone; exec cannot (`ExecBackend` docstring).
4. Unenforceable network policies fail at construction under `seatbelt`, `srt` and E2B (`_build_exec_backend`, `build_srt_settings`, `e2b_environment`).
5. Primary interfaces are ABCs (`ExecutionEnvironment`, `ExecBackend`, `ExecSession`, `KernelSession`, `FileBackend`). Optional capabilities are `Protocol`s checked with `isinstance`; implement only what your backend supports.
6. One-shot resource limits never apply to persistent shells or kernels (`SupervisorLimits`).

## Limitations

- `bwrap` (Linux) is not implemented, and `auto` on Linux raises when `bwrap` is installed. Use `srt` or `none` there.
- `NetworkPolicy.LOOPBACK` is accepted only by `none`, which does not enforce it.
- Seatbelt cannot express per-domain egress; use `srt`.
- Only E2B offers filesystem snapshots; there is no local snapshot backend.
- No Docker or SSH exec backend ships; implement `ExecutionEnvironment` / `ExecBackend` for other targets.

## See also

[Files](files.md) · [Notebooks](notebooks.md) · [Background tasks](background-tasks.md) · [Rollback](rollback.md) · [Security](security.md) · code: `sandbox/`, `tools/bash.py`, `tools/bash_session.py`
