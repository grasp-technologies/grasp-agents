"""
Span names and attributes emitted by grasp-agents.

Span names are for people reading a trace; attributes are for machines
(filters, error reports, online evaluations). Payloads use the OpenInference
keys Phoenix renders.

=====================  ==========================  ===============  ======
Span                   Name                        grasp.span.kind  OI kind
=====================  ==========================  ===============  ======
agent run              ``<agent>``                 agent            AGENT
workflow run           ``<workflow>``              workflow         CHAIN
other processor run    ``<processor>``             processor        CHAIN
runner run             ``<runner>``                runner           CHAIN
agent model call       ``<agent>.generate``        generate         CHAIN
forced final answer    ``<agent>.final_answer``    task             CHAIN
tool call              ``<tool>``                  tool             TOOL
``@traced`` function   its ``name`` or qualname    its kind         by kind
evaluation trial       ``<run>.trial``             trial            CHAIN
=====================  ==========================  ===============  ======

Every span carries ``grasp.span.kind`` and ``openinference.span.kind``;
``input.value`` / ``output.value`` (JSON, ``*.mime_type``) hold the call's
input and result when content tracing is on (``GRASP_TRACE_CONTENT``, default
true): a processor's input and output payloads (one payload as itself,
several as a list — the shape an evaluation passes and expects), a tool's
input and result, a model call's output items, a function's arguments and
result. ``tracing_exclude_input_fields`` keys are left out of inputs (a
model's tool-call arguments are its output and keep them). Payloads longer
than the span attribute limit (``OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT``) are
shortened inside the JSON — long strings first, then long lists and
mappings, keeping head and tail — so they stay parseable;
``grasp.input.truncated`` / ``grasp.output.truncated`` mark it. A failed call
(including a tool call that returned an error) has status ERROR with the
whole cause chain as its description, ``error.type`` set to the root cause's
type and the exception recorded as an event; a cancelled call is marked
``grasp.cancelled``.

Processor runs add ``grasp.processor.*`` (name, class, path, exec_id — shared
with what the run dispatches —, declared version, replica index, failed
attempts) and agents ``grasp.agent.name`` and ``grasp.agent.model``; a
parallel replica is named after its template and numbered by
``grasp.processor.replica``. Model calls add ``grasp.agent.name``,
``grasp.llm.request_model`` / ``grasp.llm.response_model`` and
``grasp.llm.usage.*``; tool calls ``grasp.tool.name`` and the calling agent's
``grasp.agent.name``. The outermost run of a named session stamps its id on
every span below it (``GRASP_SESSION_ID_ATTRIBUTES``, default ``session.id``
and ``gen_ai.conversation.id``), as do attributes set with
:func:`~grasp_agents.telemetry.inherited_span_attributes` — evaluations stamp
``grasp.eval.*`` (run id, run name, example id, repetition, and the scorer
for its own calls) on every span they make.
"""

from enum import StrEnum


class SpanKind(StrEnum):
    AGENT = "agent"
    WORKFLOW = "workflow"
    PROCESSOR = "processor"
    RUNNER = "runner"
    GENERATE = "generate"
    TOOL = "tool"
    TASK = "task"
    TRIAL = "trial"


OPENINFERENCE_SPAN_KINDS: dict[SpanKind, str] = {
    SpanKind.AGENT: "AGENT",
    SpanKind.TOOL: "TOOL",
}
"""OpenInference kinds that differ from ``CHAIN``."""

ATTR_SPAN_KIND = "grasp.span.kind"
ATTR_OI_SPAN_KIND = "openinference.span.kind"

ATTR_INPUT_VALUE = "input.value"
ATTR_INPUT_MIME_TYPE = "input.mime_type"
ATTR_OUTPUT_VALUE = "output.value"
ATTR_OUTPUT_MIME_TYPE = "output.mime_type"
ATTR_INPUT_TRUNCATED = "grasp.input.truncated"
ATTR_OUTPUT_TRUNCATED = "grasp.output.truncated"
JSON_MIME_TYPE = "application/json"

ATTR_ERROR_TYPE = "error.type"
ATTR_CANCELLED = "grasp.cancelled"

ATTR_PROCESSOR_NAME = "grasp.processor.name"
ATTR_PROCESSOR_CLASS = "grasp.processor.class"
ATTR_PROCESSOR_PATH = "grasp.processor.path"
ATTR_PROCESSOR_EXEC_ID = "grasp.processor.exec_id"
ATTR_PROCESSOR_VERSION = "grasp.processor.version"
ATTR_PROCESSOR_REPLICA = "grasp.processor.replica"
ATTR_FAILED_ATTEMPTS = "grasp.processor.failed_attempts"

ATTR_AGENT_NAME = "grasp.agent.name"
ATTR_AGENT_MODEL = "grasp.agent.model"

ATTR_LLM_REQUEST_MODEL = "grasp.llm.request_model"
ATTR_LLM_RESPONSE_MODEL = "grasp.llm.response_model"
ATTR_LLM_INPUT_TOKENS = "grasp.llm.usage.input_tokens"
ATTR_LLM_OUTPUT_TOKENS = "grasp.llm.usage.output_tokens"
ATTR_LLM_REASONING_TOKENS = "grasp.llm.usage.reasoning_tokens"
ATTR_LLM_CACHED_TOKENS = "grasp.llm.usage.cached_tokens"
ATTR_LLM_COST_USD = "grasp.llm.usage.cost_usd"

ATTR_TOOL_NAME = "grasp.tool.name"
ATTR_RUNNER_NAME = "grasp.runner.name"

ATTR_SESSION_ID = "session.id"
ATTR_CONVERSATION_ID = "gen_ai.conversation.id"

ATTR_EVAL_RUN_ID = "grasp.eval.run_id"
ATTR_EVAL_RUN_NAME = "grasp.eval.run_name"
ATTR_EVAL_EXAMPLE_ID = "grasp.eval.example_id"
ATTR_EVAL_REPETITION = "grasp.eval.repetition"
ATTR_EVAL_SCORER = "grasp.eval.scorer"
