import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_type_hints

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

from ._util import qualified_name
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
        self.measurements[name] = float(value)

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

    @abstractmethod
    async def run(self, input: InT, trial: TrialContext) -> OutT: ...  # noqa: A002


type TaskFn[InT, OutT] = (
    Callable[[InT], Awaitable[OutT]] | Callable[[InT, TrialContext], Awaitable[OutT]]
)


class FunctionTask[InT, OutT](Task[InT, OutT]):
    """Wraps ``async def fn(input)`` or ``async def fn(input, trial)``."""

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
        self._takes_trial = len(parameters) >= 2
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

    async def run(self, input: InT, trial: TrialContext) -> OutT:  # noqa: A002
        if self._takes_trial:
            fn = cast("Callable[[InT, TrialContext], Awaitable[OutT]]", self._fn)
            return await fn(input, trial)
        fn = cast("Callable[[InT], Awaitable[OutT]]", self._fn)
        return await fn(input)


type ProcessorSource[InT, OutT] = (
    Processor[InT, OutT, Any] | Callable[[], Processor[InT, OutT, Any]]
)


def _keep_event(event: Event[Any]) -> bool:
    # Token deltas and incremental tool output are reproduced by the final
    # items/responses; keeping them would multiply transcript size.
    return not isinstance(event, LLMStreamEvent | ToolStreamEvent)


def _as_in_args(arg: Any) -> Any:
    # A falsy scalar (0, "", False) passed alone reads as "no input"; as a
    # one-element argument list it is unambiguous.
    wrapped: Any = arg
    if arg is not None and not arg and not isinstance(arg, list):
        wrapped = [arg]
    return wrapped


def _to_usage(usage: ResponseUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        cached_tokens=usage.input_tokens_details.cached_tokens,
        cost_usd=usage.cost,
    )


class ProcessorTask[InT, OutT](Task[InT, OutT]):
    """
    Runs any grasp-agents :class:`Processor` — an agent, a workflow, a parallel
    fan-out or a custom subclass — once per trial, in isolation.

    Pass either a *template* processor or a zero-argument *factory*:

    - a template is copied for every trial and the copy is rebound to the
      trial's own :class:`SessionContext` (containers cascade it to their
      children). The template itself is never run — build it fresh;
    - a factory is called inside ``with trial_ctx:``, so everything it builds
      binds to the trial's session.

    ``ctx_factory(example)`` builds that session (e.g. to seed ``state`` or an
    execution environment); by default each trial gets an empty one. The
    example input is passed as ``in_args`` (or as ``chat_inputs`` with
    ``input_mode="chat"``), optionally transformed by ``input_fn``. The output
    is the run's single payload (a list when there are several), or whatever
    ``output_fn(packet, ctx)`` derives — e.g. from ``ctx.state`` for pipelines
    whose result is a state mutation.
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
        capture_events: bool = True,
    ) -> None:
        self._template: Processor[InT, OutT, Any] | None
        self._factory: Callable[[], Processor[InT, OutT, Any]] | None
        kind_cls: type[Any]
        if isinstance(processor, Processor):
            template = cast("Processor[InT, OutT, Any]", processor)
            self._template, self._factory = template, None
            default_name = template.name
            self._in_type: Any = template.in_type
            self._out_type: Any = template.out_type
            kind_cls = type(template)
        else:
            self._template, self._factory = None, processor
            default_name = getattr(processor, "__name__", "processor")
            self._in_type = Any
            self._out_type = Any
            kind_cls = type(processor)
        self.name = name or default_name
        self.version = version
        self._config = dict(config or {})
        self._kind = qualified_name(kind_cls)
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
        return Any if self._output_fn is not None else self._out_type

    def config(self) -> dict[str, Any]:
        return dict(self._config)

    def describe(self) -> ComponentInfo:
        info = super().describe()
        info.kind = self._kind
        return info

    def _instantiate(self, ctx: SessionContext[Any]) -> Processor[InT, OutT, Any]:
        if self._template is not None:
            proc = self._template.copy()
        else:
            assert self._factory is not None
            with ctx:
                proc = self._factory()
        proc.on_adopted(ctx=ctx)
        return proc

    async def run(self, input: InT, trial: TrialContext) -> OutT:  # noqa: A002
        ctx: SessionContext[Any] = (
            self._ctx_factory(trial.example)
            if self._ctx_factory is not None
            else SessionContext()
        )
        trial.session = ctx
        proc = self._instantiate(ctx)
        arg: Any = self._input_fn(input) if self._input_fn is not None else input
        if self._input_mode == "in_args":
            arg = _as_in_args(arg)
        packet: Packet[Any] | None = None
        try:
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
