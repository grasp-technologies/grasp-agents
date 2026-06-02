import logging
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Self,
    TypeVar,
    cast,
    final,
)
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from grasp_agents.telemetry import traced
from grasp_agents.types.errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
    ProcRunError,
)

from ..durability.checkpoints import CheckpointKind
from ..durability.persist import CheckpointPersistMixin
from ..packet import Packet
from ..run_context import CtxT, RunContext
from ..types.events import (
    Event,
    ProcPacketOutEvent,
    ProcPayloadOutEvent,
    ProcStreamingErrorData,
    ProcStreamingErrorEvent,
)
from ..types.hooks import RecipientSelector
from ..types.io import InT, OutT, ProcName
from ..utils.callbacks import is_method_overridden
from ..utils.generics import AutoInstanceAttributesMixin

if TYPE_CHECKING:
    from ..agent.processor_tool import ProcessorTool

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., AsyncIterator[Event[Any]]])


def with_retry(func: F) -> F:
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
                        message=err_message + f" after {n_attempt - 1} retries",
                    ) from err

                logger.warning(
                    f"{err_message} -> retrying (attempt {n_attempt}):\n{err}"
                )

    return cast("F", wrapper)


class Processor(
    AutoInstanceAttributesMixin,
    CheckpointPersistMixin,
    Generic[InT, OutT, CtxT],
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

    def __init__(
        self,
        name: ProcName,
        *,
        ctx: RunContext[CtxT] | None = None,
        max_retries: int = 0,
        recipients: Sequence[ProcName] | None = None,
        path: list[str] | None = None,
        session_metadata: dict[str, Any] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        self.name = name

        self.max_retries = max_retries
        self.recipients = recipients
        self.tracing_enabled = tracing_enabled
        self.tracing_exclude_input_fields = tracing_exclude_input_fields

        self._session_metadata: dict[str, Any] = session_metadata or {}
        self._checkpoint_number: int = 0
        # Records whether the caller handed us a ctx (vs. the fresh
        # placeholder below). Container processors read this on their
        # children to decide whose session to share — see
        # :func:`~grasp_agents.run_context.shared_child_ctx`.
        self._ctx_explicit: bool = ctx is not None
        # ``ctx`` is an immutable instance attribute (passing a different
        # ``ctx`` to ``run`` / ``run_stream`` is unsupported). Set both
        # session axes, then cascade to any children built before us.
        self._ctx: RunContext[CtxT] = (
            ctx if ctx is not None else RunContext[CtxT](state=None)  # type: ignore
        )
        self._path: list[str] = [self.name] if path is None else list(path)
        self._propagate_to_children()

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
    def ctx(self) -> RunContext[CtxT]:
        """Session bound at construction. Immutable after init."""
        return self._ctx

    # --- Session persistence ---

    @property
    def is_resumable(self) -> bool:
        """True if the bound ``ctx`` has a checkpoint store wired up."""
        return self._ctx.checkpoint_store is not None

    def on_adopted(
        self,
        parent: Any = None,
        *,
        ctx: RunContext[CtxT] | None = None,
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
            if ctx is None:
                ctx = getattr(parent, "ctx", None)
            if path is None:
                parent_path = getattr(parent, "path", None)
                if parent_path is not None:
                    path = [*parent_path, self.name]
        if ctx is not None:
            self._ctx = ctx
            self._ctx_explicit = True
        if path is not None:
            self._path = list(path)
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

        if isinstance(in_args, self.in_type):
            # 2) Single in_args of correct type is provided
            resolved_args = [in_args]

        elif isinstance(in_args, list):
            # 3) List of in_args is provided
            resolved_args = cast("list[InT]", in_args)

        elif in_args is not None:
            raise ProcInputValidationError(
                message=f"in_args are neither of type {self.in_type} "
                f"nor a list of {self.in_type}.",
                **err_kwargs,
            )

        else:
            # 4) in_packet is provided
            resolved_args = list(cast("Packet[InT]", in_packet).payloads)

        try:
            for args in resolved_args:
                TypeAdapter(self._in_type).validate_python(args)
        except PydanticValidationError as err:
            raise ProcInputValidationError(message=str(err), **err_kwargs) from err

        return resolved_args

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
    ) -> AsyncIterator[Event[Any]]:
        exec_id = self.generate_exec_id(exec_id)

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
    ) -> Packet[OutT]:
        result = None

        async for event in self.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            exec_id=exec_id,
            step=step,
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
        background: bool = False,
    ) -> "ProcessorTool[InT, OutT, CtxT]":  # type: ignore[return-value]
        from ..agent.processor_tool import (  # noqa: PLC0415
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
            background=background,
            reset_transcript_on_run=reset_transcript_on_run,
        )
