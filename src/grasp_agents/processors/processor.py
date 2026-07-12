import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from copy import deepcopy
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Self,
    cast,
    final,
    get_origin,
)
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from grasp_agents import grasp_logging
from grasp_agents.durability.checkpoint_mixin import CheckpointPersistMixin
from grasp_agents.durability.checkpoints import CheckpointKind
from grasp_agents.hooks import RecipientSelector
from grasp_agents.session_context import (
    DEFAULT_SESSION_KEY,
    SessionContext,
    current_session_context,
)
from grasp_agents.telemetry import traced
from grasp_agents.types.errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
    ProcRunError,
)
from grasp_agents.types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
    ProcStreamingErrorData,
    ProcStreamingErrorEvent,
)
from grasp_agents.types.io import ProcName
from grasp_agents.types.packet import Packet
from grasp_agents.utils.callbacks import is_method_overridden
from grasp_agents.utils.generics import AutoInstanceAttributesMixin

if TYPE_CHECKING:
    from grasp_agents.tools.processor_tool import ProcessorTool

logger = logging.getLogger(__name__)


def with_retry[F: Callable[..., AsyncIterator[Event[Any]]]](func: F) -> F:
    @wraps(func)
    async def wrapper(
        self: "Processor[Any, Any, Any]", *args: Any, **kwargs: Any
    ) -> AsyncIterator[Event[Any]]:
        exec_id: str | None = kwargs.get("exec_id")

        n_attempt = 0
        while n_attempt <= self.max_retries:
            try:
                async for event in func(self, *args, **kwargs):
                    yield event
                return

            except Exception as err:
                n_attempt += 1

                err_data = ProcStreamingErrorData(error=err, exec_id=exec_id)
                yield ProcStreamingErrorEvent(
                    data=err_data,
                    source=self.name,
                    exec_id=exec_id,
                )
                err_message = (
                    f"Processor run failed [proc_name={self.name}; exec_id={exec_id}]"
                )
                if n_attempt > self.max_retries:
                    raise ProcRunError(
                        proc_name=self.name,
                        exec_id=exec_id,
                        message=err_message
                        + f" after {n_attempt - 1} retries"
                        + f" (caused by {type(err).__name__})",
                    ) from err

                logger.warning(
                    f"{err_message} -> retrying (attempt {n_attempt}):\n{err}"
                )
                self._prepare_retry()  # pyright: ignore[reportPrivateUsage]

    return cast("F", wrapper)


class Processor[InT, OutT, CtxT](
    AutoInstanceAttributesMixin,
    CheckpointPersistMixin,
):
    """
    Base computation unit in the framework. Supports typed input/output validation,
    recipient-based routing, retry, and streaming. Subclasses override ``_process``
    or ``_process_stream`` to implement custom logic.
    """

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    # Base ``Processor`` has no checkpoint machinery, so the default is
    # ``None`` and ``_checkpoint_store_key`` returns ``None``. Subclasses
    # that support resuming override this with a ``CheckpointKind`` value.
    _checkpoint_kind: ClassVar[CheckpointKind | None] = None

    name: str

    max_retries: int
    recipients: Sequence[ProcName] | None
    tracing_enabled: bool
    tracing_exclude_input_fields: set[str] | None
    durability_enabled: bool

    def __init__(
        self,
        name: ProcName,
        *,
        ctx: SessionContext[CtxT] | None = None,
        max_retries: int = 0,
        recipients: Sequence[ProcName] | None = None,
        path: list[str] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        durability_enabled: bool = True,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        if not name or name in {".", ".."} or any(c in name for c in "/\\\x00"):
            raise ValueError(
                f"Invalid processor name {name!r}: must be non-empty, not '.'/'..', "
                "and free of '/', backslash, and null bytes — a processor name "
                "becomes a store-key path segment (recipient / checkpoint path)."
            )
        self.name = name

        self.max_retries = max_retries
        self.recipients = recipients
        self.tracing_enabled = tracing_enabled
        self.tracing_exclude_input_fields = tracing_exclude_input_fields
        self.durability_enabled = durability_enabled

        self._checkpoint_number: int = 0
        self._ctx: SessionContext[CtxT] = (
            ctx if ctx is not None else current_session_context()  # type: ignore
        )
        self._path: list[str] = [self.name] if path is None else list(path)
        # Whether this processor lives inside a container or another agent's
        # turn (adopted by a parent, or built under an explicit lineage) —
        # contained processors don't own session persistence unless declared
        # (see ``SessionContext.session_writer``).
        self._contained: bool = path is not None and len(self._path) > 1
        self._propagate_to_children()

    # --- Session lifecycle ---

    async def aclose(self) -> None:
        """
        Release session-scoped resources (shells, kernels, background tasks).

        Runs never tear these down — they live for the processor's whole
        session and are released only here. The base processor holds none, so
        this is a no-op; ``LLMAgent`` closes its execution resources, and
        composite processors (workflows, parallel) cascade to their children.
        """
        return

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # --- Identity & utilities ---

    @property
    def in_type(self) -> type[InT]:
        return self._in_type

    @property
    def out_type(self) -> type[OutT]:
        return self._out_type

    @property
    def path(self) -> list[str]:
        return list(self._path)

    @property
    def checkpoint_number(self) -> int:
        return self._checkpoint_number

    @property
    def checkpoint_kind(self) -> "CheckpointKind | None":
        return self._checkpoint_kind

    @property
    def ctx(self) -> SessionContext[CtxT]:
        """Session bound at construction. Immutable after init."""
        return self._ctx

    def _trace_session_info(self) -> tuple[str, bool] | None:
        """
        Session identity for tracing: ``(session_id, group)`` or ``None``.

        Read by the tracing layer (only when this run span would be a trace
        root). ``None`` for the default (unnamed) session. Otherwise the session
        id is stamped on the run's spans as the session attribute, and ``group``
        (``ctx.session_trace_grouping``) decides whether every run of the session
        is also parented into one shared trace.
        """
        ctx = self._ctx
        if ctx.session_key == DEFAULT_SESSION_KEY:
            return None
        return ctx.session_key, ctx.session_trace_grouping

    # --- Session persistence ---

    @property
    def is_resumable(self) -> bool:
        """
        True if the bound ``ctx`` has a checkpoint store wired up and this
        processor participates in it (``durability_enabled``).
        """
        return self._ctx.checkpoint_store is not None and self.durability_enabled

    def on_adopted(
        self,
        parent: Any = None,
        *,
        ctx: SessionContext[CtxT] | None = None,
        path: Sequence[str] | None = None,
    ) -> None:
        """
        Bind this processor's session, then cascade it to children.

        Usually called as ``child.on_adopted(parent)`` when a container
        attaches a subproc: the child inherits the parent's ctx, path lineage
        (``[*parent.path, self.name]``), and tracing settings. ``ctx`` / ``path``
        override what the parent would supply — e.g. a tool dispatching a
        fresh copy under a per-call ``path``, or a test re-pointing a built
        tree at a new ``ctx``. Whatever ends up set is cascaded to every
        subproc / tool (a leaf has none, so the cascade is a no-op there).

        ``parent`` is duck-typed: a container exposes ``ctx`` + ``path`` (so
        :class:`Runner`, not itself a :class:`Processor`, can adopt too); a
        :class:`BaseTool` parent contributes only tracing settings, with
        ``ctx`` / ``path`` passed explicitly.
        """
        if parent is not None:
            self._inherit_tracing(parent)
            self._inherit_durability(parent)
            if ctx is None:
                ctx = getattr(parent, "ctx", None)
            if path is None:
                parent_path = getattr(parent, "path", None)
                if parent_path is not None:
                    path = [*parent_path, self.name]

        if ctx is not None:
            self._ctx = ctx

        if path is not None:
            self._path = list(path)

        if parent is not None or (path is not None and len(self._path) > 1):
            # A bare ctx rebind is not containment; being attached to a parent
            # (whatever its path shape) or dispatched under a lineage is.
            self._contained = True

        self._propagate_to_children()

    def _inherit_tracing(self, parent: Any) -> None:
        # Tracing restrictions propagate downward: a parent that disables
        # tracing, or masks an input field, forces the same on every
        # descendant (a child can't widen what an ancestor restricted).
        # ``getattr`` keeps it robust to parents (e.g. ``Runner``) that don't
        # carry these attributes. ``on_adopted`` runs this before the cascade
        # so grandchildren see the merged settings.
        if not getattr(parent, "tracing_enabled", True):
            self.tracing_enabled = False
        parent_fields = getattr(parent, "tracing_exclude_input_fields", None)
        if parent_fields:
            self.tracing_exclude_input_fields = (
                self.tracing_exclude_input_fields or set()
            ) | set(parent_fields)

    def _inherit_durability(self, parent: Any) -> None:
        # Same downward restriction as tracing: a parent that opted out of
        # the checkpoint store forces the same on every descendant (a child
        # can't re-enable what an ancestor disabled).
        if not getattr(parent, "durability_enabled", True):
            self.durability_enabled = False

    def _propagate_to_children(self) -> None:
        """
        Override in container processors to (re-)adopt child subprocs.

        No-op for leaf processors. Container overrides iterate subprocs and
        call ``child.on_adopted(self)``, which re-derives a child's ctx,
        path, and tracing settings from the parent in one walk.
        """

    # ``_checkpoint_store_key`` / ``_deserialize_checkpoint`` /
    # ``_serialize_checkpoint`` are provided by ``CheckpointPersistMixin``.

    # --- Input / output validation ---

    def validate_inputs(
        self,
        exec_id: str,
        chat_inputs: Any | None = None,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
    ) -> list[InT] | None:
        err_kwargs = {"proc_name": self.name, "exec_id": exec_id}

        num_non_null_inputs = sum(
            x is not None for x in [chat_inputs, in_args, in_packet]
        )

        if num_non_null_inputs > 1:
            raise ProcInputValidationError(
                message=(
                    "Only one of chat_inputs, in_args, or in_packet must be provided"
                ),
                **err_kwargs,
            )

        if self.in_type is not type(None) and num_non_null_inputs == 0:
            raise ProcInputValidationError(
                message="One of chat_inputs, in_args, or in_message must be provided",
                **err_kwargs,
            )

        if in_packet is not None and not in_packet.payloads:
            raise ProcInputValidationError(
                message="in_packet must contain at least one payload", **err_kwargs
            )

        if in_args is not None and not in_args:
            raise ProcInputValidationError(
                message="in_args must contain at least one argument", **err_kwargs
            )

        if chat_inputs is not None:
            # 1) chat_inputs are provided -> no need to validate further
            return None

        resolved_args: list[InT]

        if in_packet is not None:
            # 2) in_packet is provided -> its payloads are the arguments. Checked
            # before in_args (the inputs are mutually exclusive) so that a None
            # in_args — the unused parameter — is never mistaken for a single
            # argument when InT admits None (e.g. Any / object).
            resolved_args = list(in_packet.payloads)

        elif self._is_single_in_arg(in_args):
            # 3) Single in_args of the declared type is provided
            resolved_args = [cast("InT", in_args)]

        elif isinstance(in_args, list):
            # 4) List of in_args is provided
            resolved_args = cast("list[InT]", in_args)

        else:
            # 5) A single non-list arg (coerced below — e.g. a raw dict from a
            # resumed checkpoint), or, for a None-typed processor with no input
            # at all, a single None argument.
            resolved_args = [cast("InT", in_args)]

        # Validate AND coerce: a resumed checkpoint round-trips payloads
        # through JSON, so they arrive as raw dicts — the coerced values
        # (not the raw inputs) must be what downstream code receives.
        adapter: TypeAdapter[InT] = TypeAdapter(self._in_type)
        try:
            resolved_args = [adapter.validate_python(args) for args in resolved_args]
        except PydanticValidationError as err:
            raise ProcInputValidationError(message=str(err), **err_kwargs) from err

        return resolved_args

    def _is_single_in_arg(self, in_args: Any) -> bool:
        """
        Whether ``in_args`` is one argument of the declared input type
        (rather than a list of arguments).

        ``isinstance`` raises TypeError for parameterized generics
        (``list[int]``); fall back to the runtime origin — a ``list``
        passed to a list-typed processor is one argument, with element
        validation left to the coercion step.
        """
        try:
            return isinstance(in_args, self.in_type)
        except TypeError:
            origin = get_origin(self.in_type)
            if isinstance(origin, type):
                return isinstance(in_args, origin)
            return False

    def validate_output(self, out_payload: OutT, exec_id: str) -> OutT:
        if out_payload is None:
            return out_payload

        try:
            return TypeAdapter(self.out_type).validate_python(out_payload)
        except PydanticValidationError as err:
            raise ProcOutputValidationError(
                schema=self.out_type,
                proc_name=self.name,
                exec_id=exec_id,
            ) from err

    # --- Recipient selection ---

    def _validate_recipients(
        self, recipients: Sequence[ProcName] | None, exec_id: str
    ) -> None:
        for r in recipients or []:
            if r not in (self.recipients or []):
                raise PacketRoutingError(
                    proc_name=self.name,
                    exec_id=exec_id,
                    selected_recipient=r,
                    allowed_recipients=cast("list[str]", self.recipients),
                )

    def select_recipients_impl(
        self, output: OutT, *, exec_id: str
    ) -> Sequence[ProcName]:
        raise NotImplementedError

    def add_recipient_selector(
        self, func: RecipientSelector[OutT]
    ) -> RecipientSelector[OutT]:
        self.select_recipients_impl = func

        return func

    @final
    def select_recipients(self, output: OutT, exec_id: str) -> Sequence[ProcName]:
        if is_method_overridden("select_recipients_impl", self, Processor):
            recipients = self.select_recipients_impl(output=output, exec_id=exec_id)
            self._validate_recipients(recipients, exec_id=exec_id)
            return recipients

        return cast("list[ProcName]", self.recipients)

    # --- Processing ---

    async def _process(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> list[OutT]:
        """
        Process inputs and return outputs.

        Subclasses can override either ``_process`` or ``_process_stream``:

        - Override ``_process`` only → ``_process_stream`` wraps outputs in events.
        - Override ``_process_stream`` only → ``_process`` collects payload events.
        - Override both → each uses its own logic.
        - Override neither → passthrough (returns ``in_args``).

        Subclasses read the bound session via :attr:`ctx`.
        """
        # If _process_stream is overridden (and we're the base _process — which
        # we must be, since an overriding subclass wouldn't reach this code),
        # derive by collecting payload events from the stream.
        if is_method_overridden("_process_stream", self, Processor):
            outputs: list[OutT] = []
            async for event in self._process_stream(
                chat_inputs=chat_inputs,
                in_args=in_args,
                exec_id=exec_id,
                step=step,
            ):
                if isinstance(event, ProcPayloadOutEvent) and event.source == self.name:
                    outputs.append(event.data)
            return outputs

        return cast("list[OutT]", in_args)

    async def _process_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_args: list[InT] | None = None,
        exec_id: str,
        step: int | None = None,
    ) -> AsyncIterator[Event[Any]]:
        """Stream events for inputs. See ``_process`` docstring for override rules."""
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=in_args,
            exec_id=exec_id,
            step=step,
        )
        for output in outputs:
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

    def _build_packet(
        self,
        outputs: list[OutT],
        exec_id: str,
    ) -> Packet[OutT]:
        for output in outputs:
            self.validate_output(output, exec_id=exec_id)

        routings: list[Sequence[ProcName]] | None = []
        if self.recipients is not None:
            for output in outputs:
                routings.append(self.select_recipients(output=output, exec_id=exec_id))

        joined_routing = [r for r in routings] if routings else None

        return Packet(sender=self.name, payloads=outputs, routing=joined_routing)

    # --- Run ---

    def _prepare_retry(self) -> None:
        """
        Hook run by ``with_retry`` before a retry attempt re-enters the
        stream. Subclasses use it to carry the failed attempt's settled state
        into the retry (see ``LLMAgent``); the base processor retries from
        scratch.
        """

    def generate_exec_id(self, exec_id: str | None) -> str:
        if exec_id is None:
            return str(uuid4())[:6] + "_" + self.name
        return exec_id

    def copy(self) -> Self:
        return deepcopy(self)

    @final
    @traced(name="processor")
    @with_retry
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        exec_id: str | None = None,
        step: int | None = None,
        span_name: str | None = None,
        span_attributes: Mapping[str, str | float | bool] | None = None,
    ) -> AsyncIterator[Event[Any]]:
        # span_name / span_attributes enrich this run's span; the @traced
        # decorator reads them off the call kwargs — unused in the body.
        del span_name, span_attributes
        exec_id = self.generate_exec_id(exec_id)

        with grasp_logging.log_context(exec_id=exec_id, proc=self.name):
            # Session-scoped restore (ctx.state + shared filesystem) — the
            # ctx makes it a no-op for every run after the first on this ctx,
            # so adopted subprocessors and tool-dispatched clones don't redo
            # (or clobber) what the outermost run restored.
            await self._ctx.load_checkpoint()

            val_in_args = self.validate_inputs(
                exec_id=exec_id,
                chat_inputs=chat_inputs,
                in_packet=in_packet,
                in_args=in_args,
            )

            outputs: list[OutT] = []
            async for event in self._process_stream(
                chat_inputs=chat_inputs,
                in_args=val_in_args,
                exec_id=exec_id,
                step=step,
            ):
                if isinstance(event, ProcPayloadOutEvent) and event.source == self.name:
                    outputs.append(event.data)
                else:
                    yield event

            out_packet = self._build_packet(outputs=outputs, exec_id=exec_id)
            yield ProcPacketOutEvent(
                id=out_packet.id,
                data=out_packet,
                source=self.name,
                exec_id=exec_id,
            )

    async def run(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        exec_id: str | None = None,
        step: int | None = None,
        span_name: str | None = None,
        span_attributes: Mapping[str, str | float | bool] | None = None,
    ) -> Packet[OutT]:
        result = None

        async for event in self.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            exec_id=exec_id,
            step=step,
            span_name=span_name,
            span_attributes=span_attributes,
        ):
            if result is not None:
                continue

            if isinstance(event, ProcPacketOutEvent) and event.source == self.name:
                result = event.data

        if result is None:
            raise RuntimeError("Processor run did not yield a ProcPacketOutputEvent")

        return result

    @final
    def as_tool(
        self,
        tool_name: str,
        tool_description: str,
        reset_transcript_on_run: bool = True,
        auto_background_at: float | None = None,
        blocks_final_answer: bool = True,
        max_inline_result_chars: int | None = None,
    ) -> "ProcessorTool[InT, OutT, CtxT]":  # type: ignore[return-value]
        from grasp_agents.tools.processor_tool import (  # noqa: PLC0415
            ProcessorTool as _ProcessorTool,
        )

        if not issubclass(self.in_type, BaseModel):
            raise TypeError(
                "Cannot create a tool from an agent with "
                f"non-BaseModel input type: {self.in_type}"
            )

        return _ProcessorTool[InT, OutT, CtxT](  # type: ignore[type-var]
            processor=self,  # InT bound validated above
            name=tool_name,
            description=tool_description,
            auto_background_at=auto_background_at,
            blocks_final_answer=blocks_final_answer,
            max_inline_result_chars=max_inline_result_chars,
            reset_transcript_on_run=reset_transcript_on_run,
        )
