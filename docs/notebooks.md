# Notebooks and code execution

Two ways to run Python against an agent's environment: `RunPython`, a stateful REPL in its own kernel, and the notebook tools (`NotebookRead`, `NotebookEdit`, `RunCell`), which work cell by cell on an `.ipynb` file and execute cells in the notebook's kernel. Both run kernels through the session's exec environment, so they inherit its filesystem and confinement. This page covers the tools, the kernel seam, kernel lifecycle, and the setup each needs.

## Key types

| Name | Module | Role |
|---|---|---|
| `RunPython` | `tools/code_interpreter.py` | Run code in the agent's REPL kernel. |
| `NotebookReadTool` / `NotebookEditTool` (tool names `NotebookRead` / `NotebookEdit`) | `tools/file_edit/notebook.py` | Cell-addressed read and edit of `.ipynb`. |
| `RunCell` | `tools/notebook_exec.py` | Execute one notebook cell; write its outputs back. |
| `KernelHolder` | `tools/notebook_exec.py` | Lazily opens and caches one kernel per agent. |
| `KernelCapable`, `KernelSession`, `CellOutput`, `CellResult` | `sandbox/kernel.py` | The kernel seam. |
| `LocalKernel` | `sandbox/local/kernel.py` | Local `jupyter_client` kernel. |
| `E2BKernel` | `sandbox/e2b/kernel.py` | Kernel on an E2B code-interpreter sandbox. |

## Setup

| Need | Install | Environment |
|---|---|---|
| `RunPython` / `RunCell`, local | `grasp-agents[code-exec]` (`jupyter-client`, `ipykernel`, `ipython`) | `local_environment(...)`; the kernel interpreter needs `ipykernel`. |
| `NotebookRead` / `NotebookEdit` | `grasp-agents[notebook-edit]` (`nbformat`) | Any `file_backend`. |
| All notebook tools | `grasp-agents[notebook]` (both of the above) | |
| E2B kernels | `grasp-agents[e2b]` | `e2b_environment(code_interpreter=True)` |

```python
from grasp_agents import LLMAgent, SessionContext
from grasp_agents.sandbox import local_environment
from grasp_agents.tools import FileToolkit, RunCell, RunPython

env = local_environment(
    allowed_roots=["./work"],
    confinement="srt",
    python="./work/.venv/bin/python",
    packages=["pandas", "matplotlib"],
    kernel_setup_code="%matplotlib inline",
)
agent = LLMAgent[str, str, None](
    "analyst",
    llm=llm,
    ctx=SessionContext(environment=env),
    tools=[*FileToolkit(include_notebook=True).tools(), RunCell(), RunPython()],
)
```

**Verify, don't install.** `python=` selects the interpreter, and `packages=` are only checked at setup: a missing one raises with the install command to run, and your environment is never modified. `provision=True` instead creates a dedicated venv at `<first allowed root>/.venv` and installs into it. On E2B, bake packages into the template.

Local kernels talk to their client over loopback. They work under `confinement="none"`, under `"srt"` (which permits local binding), and under `"seatbelt"` only with `network=ALL`. Seatbelt with `NONE` blocks the kernel's sockets.

## `RunPython`

A persistent REPL: variables and imports survive across calls, so the model loads data once and explores it over several calls. Results come back on two separate channels:

- **Display** — stdout, the last expression's value, tracebacks, and anything the code displays (`plt.show()`, `IPython.display`) return as text plus inline images. To *see* a figure, display it.
- **Artifacts** — paths listed in `artifacts=[...]` are reported by path, MIME type and size, never inlined. To *hand over* a file (a CSV, a checkpoint, a saved figure), list it.

Defaults: 120 s per call (interrupted on timeout), 20,000 chars of text, 10 images, 50 artifact files. Outside an agent loop each call uses a throwaway kernel.

## Notebook tools

- **`NotebookRead`** returns a cell-structured view with stable `cell_id`s, sources, and output previews (2,000 chars per cell). Images are included by default for a single-cell read only. The generic `Read` renders the same cell view for `.ipynb` files.
- **`NotebookEdit`** replaces, inserts (after `cell_id`, or at the start) or deletes one cell, re-serializing the notebook so other cells and metadata survive. Replacing a code cell clears its outputs and execution count. Notebooks without native cell ids get deterministic ones; the first edit persists native ids.
- **`RunCell`** runs one code cell by `cell_id` in the notebook's kernel and writes the outputs and execution count back into the `.ipynb`. It returns the outputs to the model (text plus png / jpeg / gif / webp images).
- `NotebookEdit` and `RunCell` follow the file-edit write rules: the notebook must exist, must have been read this session, and must be unchanged since (see [Files](files.md)). `Write` and `Edit` refuse `.ipynb` files.

**Output sanitization.** Outputs persisted into an `.ipynb` are sanitized by default (`RunCell(sanitize_outputs=True)`): `application/javascript` is dropped and `<script>`, `on*=` handlers and `javascript:` URLs are stripped from HTML. A notebook the agent wrote therefore cannot run agent-authored script in a reviewer's frontend. `RunPython` outputs are never persisted, so they are not sanitized.

## The kernel seam

An exec backend that can host a kernel implements `KernelCapable.open_kernel(cwd=, env=, context_id=)`, which returns a `KernelSession`: stateful, serial, with `execute()` streaming `CellOutput`s (nbformat-shaped) and then one `CellResult`. The kernel launches under the backend's own confinement wrapper, so it shares the environment's roots, network and policy. `LocalExecBackend` (and its Seatbelt / `srt` subclasses) and `E2BExecBackend` implement it.

## Kernel lifecycle

- `RunPython` and `RunCell` each have their own kernel on the agent's `AgentContext` (`ipy_kernel_holder`, `nb_kernel_holder`). They never share one, so notebook state and REPL state stay independent.
- Kernels are session-scoped: opened on first use, kept across runs, closed only by `agent.aclose()`. Copies of an agent (replicas, sub-agents) get fresh kernels.
- A kernel that died (e.g. a timeout that had to kill it) is reopened on the next call, and the result header tells the model that variables and imports were lost. A local kernel that crashed and replaced itself reports the same way.
- **Resume.** A local kernel cannot outlive its process; after a restart it starts fresh and the `.ipynb` is the durable artifact. An E2B kernel lives in a code context on the sandbox. Its `context_id` is checkpointed alongside a filesystem snapshot and re-attached only when the session restored that snapshot (see [Rollback](rollback.md)), so variables survive a resume.

## Contracts

1. `RunPython` and `RunCell` never share a kernel (`AgentContext.ipy_kernel_holder`, `nb_kernel_holder`).
2. Kernels run only through `ctx.exec_backend` and only when it is `KernelCapable`; otherwise the tools raise a clear error.
3. Involuntary kernel restarts are always surfaced to the model (`KernelHolder.take_reset`, `KernelSession.take_reset`).
4. Persisted notebook outputs are sanitized unless explicitly disabled (`cell_output_to_nbformat`, `sanitize_output_data`).
5. Kernel context ids are valid only inside the snapshot they were captured with (`LLMAgent._run_checkpoint_unit`, `load_checkpoint`).
6. Both tools claim global write exclusivity (`"/"`), so they never run concurrently with file writers.

## Limitations

- Image tool outputs need a provider that accepts them in tool results: OpenAI Responses, Anthropic or Gemini. `OpenAILLM` (Chat Completions) and `LiteLLM` raise on an image tool output, so plots from `RunPython` / `RunCell` (and `ReadImage`) fail there.
- Code runs synchronously and blocks its kernel; neither tool can be backgrounded. Run long jobs as a script through `Bash` with `auto_background_at`.
- One-shot resource limits do not apply to kernels; bound them with per-call timeouts or use E2B.
- There is no restart-kernel tool and no run-all; the model re-runs setup cells itself.
- The notebook tools cannot create a new notebook: `Write` refuses `.ipynb` and `NotebookEdit` needs an existing file.

## See also

[Sandbox](sandbox.md) · [Files](files.md) · [Background tasks](background-tasks.md) · [LLM providers](llm.md) · code: `tools/code_interpreter.py`, `tools/notebook_exec.py`, `tools/file_edit/notebook.py`, `tools/cell_output.py`, `sandbox/kernel.py`
