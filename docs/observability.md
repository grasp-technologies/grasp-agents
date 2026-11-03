# Observability

Three independent planes: OpenTelemetry tracing (spans for runs, LLM calls and
tools), stdlib logging (the human-readable diagnostic plane, and the only one
when tracing is off), and usage/cost accounting on the session. Printers and
consoles render the event stream for people.

## Key types

| Name | Module | Role |
|---|---|---|
| `traced`, `SpanKind` | `telemetry/decorators.py` | Span decorator for functions and methods |
| `init_tracing`, `add_exporter`, `add_otlp_http_exporter`, `SessionSpanProcessor` | `telemetry/setup.py` | TracerProvider setup |
| `init_phoenix` | `telemetry/phoenix.py` | Phoenix exporter + provider instrumentation (`phoenix` extra) |
| `FilteringExporter`, `LogScopeExporter`, `NoopExporter` | `telemetry/exporters.py` | Exporter wrappers / debugging |
| `set_run_span_attributes`, `derive_session_span_context` | `telemetry/decorators.py` | Enrich and correlate spans |
| `enable_verbose_stdout_logging`, `setup_logging` | `grasp_logging.py` (root exports) | Opt-in log output |
| `body_for_log`, `log_context`, `install_log_correlation`, `SecretRedactingFilter` | `grasp_logging.py` | Logging conventions |
| `UsageTracker` | `usage_tracker.py` | Per-agent token and cost totals |
| `Printer` / `print_events`, `EventConsole` / `render_events` | `printer.py`, `ui/console.py` (root exports) | Event rendering |

## Tracing

The library depends only on the OpenTelemetry API and SDK. With no
`TracerProvider` configured, every span is a no-op.

```python
from grasp_agents.telemetry import init_tracing
from grasp_agents.telemetry.phoenix import init_phoenix

init_tracing(project_name="my-app")   # global provider + SessionSpanProcessor; no exporter
init_phoenix(project_name="my-app")   # needs TELEMETRY_COLLECTOR_HTTP_ENDPOINT
```

- `init_tracing` installs a `TracerProvider` (service name and
  `openinference.project.name` set to `project_name`) unless one is already
  set, and returns it.
- `add_exporter(exporter, provider=None, batch=True)` attaches any OTel
  `SpanExporter`. `add_otlp_http_exporter(endpoint=, headers=)` does so for
  OTLP/HTTP; it needs `opentelemetry-exporter-otlp-proto-http`, which is not a
  dependency.
- `init_phoenix(batch=False, use_litellm_instr=True, use_llm_provider_instr=True,
  project_name=...)` reads `TELEMETRY_COLLECTOR_HTTP_ENDPOINT` (it logs a warning
  and does nothing when unset), creates the provider if needed, and adds an
  OpenInference span processor (converting OpenLLMetry-style attributes) plus a
  Phoenix HTTP exporter wrapped in `FilteringExporter`. The filter drops spans
  whose `gen_ai.system` / `gen_ai.provider.name` is an LLM provider (those calls
  are already captured by the OpenInference instrumentors) and cloud metadata
  requests. It then applies the OpenInference instrumentors for LiteLLM, OpenAI,
  Anthropic and Google GenAI, each only if installed. Calling it twice on one
  provider is a no-op.

**Spans.** `@traced(name=, version=, span_kind=)` wraps `Processor.run_stream`
(`processor`), `AgentLoop` LLM calls (`generate`,
`force_generate_final_answer`), `BaseTool` runs (`tool`) and
`Runner.run_stream` (`runner_run`). The span kind comes from the instance's
`_span_kind` (`agent` for `LLMAgent`, `workflow` for workflows, `task` by
default, `tool` for tools) and is mirrored to `openinference.span.kind`.
Agent, workflow and tool spans are named `<name>.<entity>[<exec_id>]`.

| Attribute | Value |
|---|---|
| `grasp.span.kind`, `grasp.entity.name`, `grasp.entity.version`, `grasp.workflow.name` | identity |
| `grasp.entity.input`, `grasp.entity.output` | JSON of the call's input and output (content capture) |
| `grasp.processor.failed_attempts` | retries of a processor run (`max_retries`), with each exception recorded |
| `gen_ai.conversation.id` | session key (see below) |

Pass `span_name=` / `span_attributes=` to `run` / `run_stream`, or call
`set_run_span_attributes(**attrs)` mid-run (see [Hooks](hooks.md#run-spans)).

**Sessions.** For a `SessionContext` with a non-default `session_key`, the
outermost run span puts the key on the OTel context and `SessionSpanProcessor`
stamps it on every span of the run tree, provider-instrumentation spans
included, under each key in `GRASP_SESSION_ID_ATTRIBUTES` (comma-separated;
default `gen_ai.conversation.id`; empty stamps none). The key travels as a
context value, not baggage, so it is never sent in outbound request headers.
With `session_trace_grouping=True` (default) every run of the session is
parented to a root derived from the key, so all runs form one trace, across
processes. Set it `False` when the backend groups runs itself and expects one
trace per run. `derive_session_span_context(key)` gives the same parent for
your own spans.

**Turning it off.** `tracing_enabled=False` on a processor or tool emits no
spans for it or anything below it, and suppresses provider SDK
instrumentation inside it, so no orphan LLM spans appear.
`tracing_exclude_input_fields` removes input fields from span attributes; both
settings cascade to children.

| Environment variable | Effect |
|---|---|
| `TELEMETRY_COLLECTOR_HTTP_ENDPOINT` | Phoenix collector URL for `init_phoenix` |
| `GRASP_TRACE_CONTENT` (alias `TRACELOOP_TRACE_CONTENT`) | `false` stops recording inputs/outputs on spans; default `true` |
| `OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT` | truncate recorded content to this many characters, keeping head and tail |
| `GRASP_SESSION_ID_ATTRIBUTES` | span attribute keys for the session id |

## Logging

Standard `logging`, loggers under `grasp_agents.*`. The library attaches no
handlers; the host decides where records go.

```python
from grasp_agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging()   # grasp_agents logger → stdout at DEBUG, idempotent
```

`enable_verbose_stdout_logging(level=logging.DEBUG, quiet_third_party=True)`
adds one stdout handler to the `grasp_agents` logger only (the root logger is
untouched), with secret redaction and the `exec_id` in the format, and sets
noisy SDK loggers (`httpx`, `openai`, `litellm`, ...) to WARNING.
`setup_logging(logs_file_path, logs_config_path)` applies a YAML `dictConfig`
whose handlers include one named `fileHandler`.

**Levels.**

| Level | Used for |
|---|---|
| INFO | one-line summaries with timing: run, LLM call (`llm <model> → N in / M out tok in 1.2s`), tool call; kernel, MCP and checkpoint-resume lifecycle |
| DEBUG | per-turn detail and content snippets |
| WARNING | retries, timeouts, recoverable degradation |

**Bodies.** Request, response and tool bodies are logged only at DEBUG and
only through `body_for_log(text, full=...)`: a one-line head…tail snippet
(200 + 100 characters) by default, the full body (clipped at 4,000 characters)
when its gate is on. Each gate is independent because the payloads differ
widely in size:

| Env var | Module flag | Body |
|---|---|---|
| `GRASP_LOG_LLM_INPUT` | `LOG_LLM_INPUT` | assembled request |
| `GRASP_LOG_LLM_OUTPUT` | `LOG_LLM_OUTPUT` | response text |
| `GRASP_LOG_TOOL_INPUT` | `LOG_TOOL_INPUT` | tool arguments |
| `GRASP_LOG_TOOL_OUTPUT` | `LOG_TOOL_OUTPUT` | tool results |

The module flags are read at each call site, so they can be flipped at runtime
(`grasp_agents.grasp_logging.LOG_LLM_OUTPUT = True`).

**Redaction.** `SecretRedactingFilter` masks secrets in rendered messages. Both
setup functions attach it; add it to your own handlers when embedding.
`GRASP_LOG_REDACT_SECRETS=0` disables it (read once at import).

**Correlation.** `Processor.run_stream` binds `exec_id` and the processor name
with `log_context(...)`. After `install_log_correlation()` (called by both
setup functions; idempotent) every record carries `exec_id`, `proc`,
`otelTraceID` and `otelSpanID` (`"-"` when absent), usable in any format string
or JSON formatter.

**Error logging.** The layer that swallows or translates an exception logs it
with `exc_info`; re-raising layers do not, so a failure produces one traceback.
`utils/errors.py` → `format_error_chain` renders the cause chain. Store and
persistence failures are not downgraded to warnings: they fail the run.

## Usage and cost

`SessionContext.usage_tracker` (`UsageTracker`) accumulates `ResponseUsage`
(input, output, cached, cache-write and reasoning tokens, `cost`) per agent
name. The agent loop updates it after every LLM response, counting both
`usage` and `superseded_usage` (attempts replaced by a retry).

```python
tracker = ctx.usage_tracker
total = tracker.total_usage            # ResponseUsage summed over agents
per_agent = tracker.usages["assistant"]
```

`CloudLLM` stamps `cost` from LiteLLM's pricing data for the endpoint's
pricing identity (see [LLM layer](llm.md#endpoints-platform-vs-api_provider)).
A model LiteLLM does not price logs a warning and leaves `cost` as `None`.
`SessionContext.responses` keeps the most recent responses per agent
(`max_responses_per_agent`, default 16).

## Printers and consoles

- `print_events(stream)` / `Printer`: raw, colored text per message; a
  `Printer` on `SessionContext(printer=...)` prints messages as the loop
  produces them.
- `render_events(stream, show_thinking=, show_tool_args=, ...)` /
  `EventConsole`: Rich rendering with markdown, tool panels and usage; needs
  only the core install.
- The Textual app (`grasp_agents.ui.run_tui`, `run_tui_interactive`; `tui`
  extra): one pane per agent and sub-agent.

Both wrap an event stream and yield every event through. See
[`src/grasp_agents/ui/README.md`](../src/grasp_agents/ui/README.md).

## Contracts

1. The library never configures handlers or a tracer provider on import.
2. Bodies are never logged at INFO, and at DEBUG only via `body_for_log`
   inside an `isEnabledFor(DEBUG)` guard.
3. The session id is a span attribute and context value, never OTel baggage.
4. `tracing_enabled=False` silences the whole subtree, including provider
   instrumentation.

## Limitations

- `init_phoenix` takes no endpoint argument; it reads only
  `TELEMETRY_COLLECTOR_HTTP_ENDPOINT`.
- There are no OTel metrics or logs exports; usage is available in-process
  (`UsageTracker`) and on spans.
- Cost is only as current as the installed LiteLLM pricing table.

## See also

- [LLM layer](llm.md), [Hooks](hooks.md), [Evals](evals.md), [Security](security.md).
- Code: `telemetry/`, `grasp_logging.py`, `usage_tracker.py`, `printer.py`, `ui/`.
