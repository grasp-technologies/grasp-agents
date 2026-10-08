import functools
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_type_hints

from pydantic import TypeAdapter

from grasp_agents.processors.processor import Processor
from grasp_agents.session_context import SessionContext
from grasp_agents.types.events import (
    Event,
    LLMStreamEvent,
    ProcPacketOutEvent,
    ToolStreamEvent,
)
from grasp_agents.types.packet import Packet
from grasp_agents.types.response import ResponseUsage

from ._util import (
    canonical_json,
    is_library_code,
    qualified_name,
    short_hash,
    to_jsonable,
)
from .types import ComponentInfo, Example, Usage


@dataclass
class TrialContext:
    """
    Per-trial handle passed to a task: which example/repetition is running,
    and where the task reports what it observed (events, usage, models,
    custom measurements, the final session).
    """

    run_id: str
    example: Example[Any, Any]
    repetition: int
    events: list[Event[Any]] = field(default_factory=list[Event[Any]])
    usage_by_agent: dict[str, Usage] = field(default_factory=dict[str, Usage])
    models: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    measurements: dict[str, float] = field(default_factory=dict[str, float])
    session: SessionContext[Any] | None = None

    def record(self, name: str, value: float) -> None:
        """Record a custom per-trial measurement (usable as a metric target)."""
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Measurement {name!r} must be finite, got {value!r}")
        self.measurements[name] = number

    @property
    def usage(self) -> Usage:
        return sum(self.usage_by_agent.values(), Usage())


class Task[InT, OutT](ABC):
    """
    The system under test: maps one example input to an output.

    Declare ``version`` (and ``config`` for anything that changes behaviour —
    models, prompt versions, flags); both are recorded in every run.
    """

    name: str
    version: str | None = None

    @property
    def input_type(self) -> Any:
        """Type used to validate dataset inputs."""
        return Any

    @property
    def output_type(self) -> Any:
        """Type used to re-validate stored outputs (e.g. when rescoring)."""
        return Any

    def config(self) -> dict[str, Any]:
        return {}

    def describe(self) -> ComponentInfo:
        return ComponentInfo(
            name=self.name,
            kind=qualified_name(type(self)),
            version=self.version,
            config=self.config(),
        )

    def source_objects(self) -> list[Any]:
        """Classes and functions whose source files define this task."""
        return [type(self)]

    @abstractmethod
    async def run(self, input: InT, trial: TrialContext) -> OutT: ...  # ruff: ignore[builtin-argument-shadowing]


type TaskFn[InT, OutT] = (
    Callable[[InT], Awaitable[OutT]] | Callable[[InT, TrialContext], Awaitable[OutT]]
)


def _trial_passing(fn: Callable[..., Any]) -> Literal["keyword", "positional"] | None:
    parameters = list(inspect.signature(fn).parameters.values())
    if any(
        p.name == "trial" and p.kind != inspect.Parameter.POSITIONAL_ONLY
        for p in parameters
    ):
        return "keyword"
    positional = [
        p
        for p in parameters
        if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}
        and p.default is p.empty
    ]
    return "positional" if len(positional) >= 2 else None


class FunctionTask[InT, OutT](Task[InT, OutT]):
    """
    Wraps ``async def fn(input)`` or ``async def fn(input, trial)``; the trial
    is passed when ``fn`` takes a second required positional parameter or a
    parameter named ``trial``.
    """

    def __init__(
        self,
        fn: TaskFn[InT, OutT],
        *,
        name: str | None = None,
        version: str | None = None,
        config: Mapping[str, Any] | None = None,
        output_type: Any = None,
    ) -> None:
        self._fn = fn
        self.name = name or getattr(fn, "__name__", "task")
        self.version = version
        self._config = dict(config or {})
        parameters = list(inspect.signature(fn).parameters)
        self._trial_passing = _trial_passing(fn)
        try:
            hints = get_type_hints(fn)
        except Exception:
            hints = {}
        self._input_type: Any = hints.get(parameters[0], Any) if parameters else Any
        self._output_type: Any = (
            output_type if output_type is not None else hints.get("return", Any)
        )

    @property
    def input_type(self) -> Any:
        return self._input_type

    @property
    def output_type(self) -> Any:
        return self._output_type

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        info = super().describe()
        info.kind = qualified_name(self._fn)
        return info

    def source_objects(self) -> list[Any]:
        return [self._fn]

    async def run(self, input: InT, trial: TrialContext) -> OutT:  # ruff: ignore[builtin-argument-shadowing]
        call = cast("Callable[..., Awaitable[OutT]]", self._fn)
        if self._trial_passing == "keyword":
            return await call(input, trial=trial)
        if self._trial_passing == "positional":
            return await call(input, trial)
        return await call(input)


type ProcessorSource[InT, OutT] = (
    Processor[InT, OutT, Any] | Callable[[], Processor[InT, OutT, Any]]
)


def _keep_event(event: Event[Any]) -> bool:
    # Token deltas and incremental tool output are reproduced by the final
    # items/responses; keeping them would multiply transcript size.
    return not isinstance(event, LLMStreamEvent | ToolStreamEvent)


def _to_usage(usage: ResponseUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        cost_usd=usage.cost,
    )


def iter_processors(
    root: Processor[Any, Any, Any],
) -> Iterator[Processor[Any, Any, Any]]:
    """
    ``root`` and every processor inside it: workflow steps, a parallel
    processor's worker, and processors that agents use as tools.
    """
    seen: set[int] = set()
    stack: list[Processor[Any, Any, Any]] = [root]
    while stack:
        proc = stack.pop()
        if id(proc) in seen:
            continue
        seen.add(id(proc))
        yield proc
        children: list[Any] = []
        subprocs = getattr(proc, "subprocs", None)
        if isinstance(subprocs, Sequence):
            children.extend(cast("Sequence[Any]", subprocs))
        children.append(getattr(proc, "subproc", None))
        tools = getattr(proc, "tools", None)
        if isinstance(tools, Mapping):
            for tool in cast("Mapping[str, Any]", tools).values():
                children.append(getattr(tool, "processor", None))
        for child in reversed(children):
            if isinstance(child, Processor):
                stack.append(cast("Processor[Any, Any, Any]", child))


# Settings that change what is observed or persisted, not what is produced.
_OPERATIONAL = frozenset({"tracing_enabled", "durability_enabled"})


def _public_settings(obj: Any) -> dict[str, Any]:
    # Plain public attributes are a processor's construction-time settings
    # (a custom subclass's thresholds, a declared variant).
    return {
        key: value
        for key, value in cast("dict[str, Any]", vars(obj)).items()
        if not key.startswith("_")
        and key not in _OPERATIONAL
        and isinstance(value, str | int | float | bool)
    }


def _tool_entry(tool: Any) -> dict[str, Any]:
    return {
        "description": short_hash(str(getattr(tool, "description", ""))),
        "input": _type_identity(getattr(tool, "in_type", Any)),
    }


_MAX_WALK_DEPTH = 5


def processor_code(root: Processor[Any, Any, Any]) -> list[Any]:
    """
    The user code a processor tree runs: custom processor and tool classes,
    and the functions registered on it — hooks, input and output builders
    and parsers, prompt sections, tool functions, converters. Code from
    grasp-agents, installed packages and the standard library is left out.
    """
    from grasp_agents.llm.llm import LLM  # noqa: PLC0415
    from grasp_agents.tools.base import BaseTool  # noqa: PLC0415

    found: list[Any] = []
    seen: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        if depth > _MAX_WALK_DEPTH or id(value) in seen:
            return
        seen.add(id(value))
        if value is None or isinstance(value, str | bytes | int | float | bool):
            return
        if isinstance(value, SessionContext | type):
            return
        if isinstance(value, Mapping):
            for item in cast("Mapping[Any, Any]", value).values():
                visit(item, depth + 1)
            return
        if isinstance(value, list | tuple | set | frozenset):
            for item in cast("Iterable[Any]", value):
                visit(item, depth + 1)
            return
        if callable(value) and (
            inspect.isfunction(value)
            or inspect.ismethod(value)
            or isinstance(value, functools.partial)
        ):
            if not is_library_code(value):
                found.append(value)
            return
        if not is_library_code(value):
            # A user object: a callable one (a hook, a section) with its
            # settings, others (a custom processor, tool or LLM) by the code
            # of their class — their data is in the fingerprint.
            hook = callable(value) and not isinstance(value, BaseTool | LLM)
            found.append(cast("Any", value) if hook else type(cast("object", value)))
        if is_library_code(value) or isinstance(value, Processor):
            attributes: Any = getattr(cast("object", value), "__dict__", None)
            if isinstance(attributes, dict):
                for item in cast("dict[str, Any]", attributes).values():
                    visit(item, depth + 1)

    for proc in iter_processors(root):
        visit(proc, 0)
    return found


def _type_identity(tp: Any) -> str:
    try:
        return short_hash(canonical_json(TypeAdapter(tp).json_schema()))
    except Exception:
        return qualified_name(tp) if isinstance(tp, type) else str(tp)


def _describe_processor(proc: Processor[Any, Any, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "type": qualified_name(type(proc)),
        "name": proc.name,
        "settings": _public_settings(proc),
    }
    llm = getattr(proc, "llm", None)
    if llm is not None:
        entry["llm"] = {
            "type": qualified_name(type(llm)),
            "model": getattr(llm, "model_name", None),
            "settings": to_jsonable(getattr(llm, "llm_settings", None)),
        }
    for prompt in ("sys_prompt", "in_prompt"):
        text = getattr(proc, prompt, None)
        if text:
            entry[prompt] = short_hash(str(text))
    max_turns = getattr(proc, "max_turns", None)
    if isinstance(max_turns, int):
        entry["max_turns"] = max_turns
    entry["output"] = _type_identity(getattr(proc, "out_type", Any))
    tools = getattr(proc, "tools", None)
    if isinstance(tools, Mapping):
        entry["tools"] = {
            str(name): _tool_entry(tool)
            for name, tool in sorted(
                cast("Mapping[Any, Any]", tools).items(), key=lambda kv: str(kv[0])
            )
        }
    return entry


def processor_fingerprint(root: Processor[Any, Any, Any]) -> str | None:
    """
    Hash of what a processor tree is made of: types, names, models and their
    settings, system and input prompts, turn limits, output schemas, tools
    (names, descriptions, input schemas) and plain public settings — not its
    code (see :func:`processor_code`). ``None`` when it cannot be read.
    """
    try:
        return short_hash(
            canonical_json([_describe_processor(p) for p in iter_processors(root)])
        )
    except Exception:
        return None


def _reset_transcripts(root: Processor[Any, Any, Any]) -> None:
    # A copy of an agent that already ran would start each trial with that
    # conversation; trials must start from scratch.
    from grasp_agents.agent.llm_agent import (  # ruff: ignore[import-outside-top-level]
        LLMAgent,
    )

    for proc in iter_processors(root):
        if isinstance(proc, LLMAgent):
            proc.transcript.clear()


def _declared_output_type(factory: Callable[[], Any]) -> Any:
    # ``def build() -> Processor[In, Out, Ctx]`` (or an agent/workflow class
    # specialized the same way) declares the output type.
    try:
        returned = get_type_hints(factory).get("return")
    except Exception:
        return Any
    if not (isinstance(returned, type) and issubclass(returned, Processor)):
        return Any
    declared = cast("Any", returned)
    resolved = cast(
        "dict[str, Any]", getattr(declared, "_resolved_instance_attr_types", {})
    )
    return resolved.get("_out_type", Any)


class ProcessorTask[InT, OutT](Task[InT, OutT]):
    """
    Runs any grasp-agents :class:`Processor` — an agent, a workflow, a parallel
    fan-out or a custom subclass — once per trial, in isolation.

    Pass either a *template* processor or a zero-argument *factory*:

    - a template is copied for every trial and the copy is rebound to the
      trial's own :class:`SessionContext` (containers cascade it to their
      children). The template itself is never run, and agents in the copy
      start with an empty transcript;
    - a factory is called inside the trial's session, so everything it builds
      binds to it.

    The whole trial runs with its session ambient, so processors built while
    it runs (inside tools, custom processors) bind to it too — their usage is
    counted and their state isolated.

    ``ctx_factory(example)`` builds that session (e.g. to seed ``state`` or an
    execution environment); by default each trial gets an empty one. The
    example input is passed as ``in_args`` (or as ``chat_inputs`` with
    ``input_mode="chat"``), optionally transformed by ``input_fn``. The output
    is the run's single payload (a list when there are several), or whatever
    ``output_fn(packet, ctx)`` derives — e.g. ``list(packet.payloads)`` for a
    list in every case, or a value from ``ctx.state`` for pipelines whose
    result is a state mutation.

    ``output_type`` re-validates stored outputs when a run is rescored or
    resumed; it defaults to the template's output type or the factory's
    declared return type (``-> Processor[In, Out, Ctx]``). Set it whenever an
    ``output_fn`` is used or the factory is unannotated, or scorers receive
    stored outputs as plain JSON.
    """

    def __init__(
        self,
        processor: ProcessorSource[InT, OutT],
        *,
        name: str | None = None,
        version: str | None = None,
        config: Mapping[str, Any] | None = None,
        ctx_factory: Callable[[Example[Any, Any]], SessionContext[Any]] | None = None,
        input_mode: Literal["in_args", "chat"] = "in_args",
        input_fn: Callable[[InT], Any] | None = None,
        output_fn: Callable[[Packet[Any], SessionContext[Any]], OutT] | None = None,
        output_type: Any = None,
        capture_events: bool = True,
    ) -> None:
        self._template: Processor[InT, OutT, Any] | None
        self._factory: Callable[[], Processor[InT, OutT, Any]] | None

        if isinstance(processor, Processor):
            template = cast("Processor[InT, OutT, Any]", processor)
            self._template, self._factory = template, None
            default_name = template.name
            self._in_type: Any = template.in_type
            declared: Any = template.out_type
            self._kind = qualified_name(type(template))
        else:
            factory = processor
            self._template, self._factory = None, factory
            default_name = getattr(factory, "__name__", "processor")
            self._in_type = Any
            declared = _declared_output_type(factory)
            self._kind = qualified_name(factory)

        if output_type is not None:
            self._out_type: Any = output_type
        elif output_fn is not None:
            self._out_type = Any
        else:
            self._out_type = declared

        self.name = name or default_name
        self.version = version
        self._config = dict(config or {})
        self._ctx_factory = ctx_factory
        self._input_mode = input_mode
        self._input_fn = input_fn
        self._output_fn = output_fn
        self._capture_events = capture_events

    @property
    def input_type(self) -> Any:
        return Any if self._input_fn is not None else self._in_type

    @property
    def output_type(self) -> Any:
        return self._out_type

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        info = super().describe()
        info.kind = self._kind
        info.fingerprint = (
            processor_fingerprint(self._template)
            if self._template is not None
            else None
        )
        return info

    def source_objects(self) -> list[Any]:
        if self._factory is not None:
            return [self._factory]
        assert self._template is not None
        return [
            *(type(p) for p in iter_processors(self._template)),
            *processor_code(self._template),
        ]

    def _instantiate(self, ctx: SessionContext[Any]) -> Processor[InT, OutT, Any]:
        if self._template is not None:
            proc = self._template.copy()
            _reset_transcripts(proc)
        else:
            assert self._factory is not None
            proc = self._factory()
        proc.on_adopted(ctx=ctx)
        return proc

    async def run(self, input: InT, trial: TrialContext) -> OutT:  # ruff: ignore[builtin-argument-shadowing]
        ctx: SessionContext[Any] = (
            self._ctx_factory(trial.example)
            if self._ctx_factory is not None
            else SessionContext()
        )
        trial.session = ctx
        packet: Packet[Any] | None = None

        with ctx:
            proc = self._instantiate(ctx)
            try:
                arg: Any = (
                    self._input_fn(input) if self._input_fn is not None else input
                )
                stream = (
                    proc.run_stream(chat_inputs=arg)
                    if self._input_mode == "chat"
                    else proc.run_stream(in_args=arg)
                )
                async for event in stream:
                    if (
                        packet is None
                        and isinstance(event, ProcPacketOutEvent)
                        and event.source == proc.name
                    ):
                        packet = event.data
                    if self._capture_events and _keep_event(event):
                        trial.events.append(event)
            finally:
                await proc.aclose()
                self._collect_session_facts(ctx, trial)

        if packet is None:
            raise RuntimeError(f"Processor {proc.name!r} produced no output packet")

        if self._output_fn is not None:
            return self._output_fn(packet, ctx)

        payloads = list(packet.payloads)
        if len(payloads) == 1:
            return cast("OutT", payloads[0])

        return cast("OutT", payloads or None)

    @staticmethod
    def _collect_session_facts(ctx: SessionContext[Any], trial: TrialContext) -> None:
        for agent, usage in ctx.usage_tracker.usages.items():
            trial.usage_by_agent[agent] = _to_usage(usage)

        for agent, responses in ctx.responses.items():
            models = trial.models.setdefault(agent, [])
            for response in responses:
                if response.model and response.model not in models:
                    models.append(response.model)


def as_task(
    task: "Task[Any, Any] | Processor[Any, Any, Any] | TaskFn[Any, Any]",
) -> Task[Any, Any]:
    """Coerce a processor or an async function into a :class:`Task`."""
    if isinstance(task, Task):
        return cast("Task[Any, Any]", task)

    if isinstance(task, Processor):
        return ProcessorTask(cast("Processor[Any, Any, Any]", task))

    if callable(task):
        return FunctionTask(task)

    raise TypeError(f"Cannot use {task!r} as an evaluation task")
