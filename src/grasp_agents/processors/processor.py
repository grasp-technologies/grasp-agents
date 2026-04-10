import logging
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, cast, final
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from grasp_agents.tracing_decorators import workflow
from grasp_agents.types.errors import (
    PacketRoutingError,
    ProcInputValidationError,
    ProcOutputValidationError,
    ProcRunError,
)

from ..memory import DummyMemory, Memory
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
    from ..durability.checkpoints import ProcessorCheckpoint

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., AsyncIterator[Event[Any]]])
CpT = TypeVar("CpT", bound="ProcessorCheckpoint")


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


class Processor(AutoInstanceAttributesMixin, Generic[InT, OutT, CtxT]):
    """
    Base computation unit in the framework. Supports typed input/output validation,
    recipient-based routing, retry, and streaming. Subclasses override ``_process``
    or ``_process_stream`` to implement custom logic.
    """

    _generic_arg_to_instance_attr_map: ClassVar[dict[int, str]] = {
        0: "_in_type",
        1: "_out_type",
    }

    def __init__(
        self,
        name: ProcName,
        max_retries: int = 0,
        memory: Memory | None = None,
        recipients: Sequence[ProcName] | None = None,
        tracing_enabled: bool = True,
        tracing_exclude_input_fields: set[str] | None = None,
        session_id: str | None = None,
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._in_type: type[InT]
        self._out_type: type[OutT]

        super().__init__()

        self._name = name
        self._max_retries = max_retries

        self._memory: Memory = memory or DummyMemory()

        self.recipients = recipients

        self.tracing_enabled = tracing_enabled
        self.tracing_exclude_input_fields = tracing_exclude_input_fields

        self._session_id: str | None = session_id
        self._session_metadata: dict[str, Any] = session_metadata or {}
        self._checkpoint_number: int = 0

    # --- Identity & utilities ---

    @property
    def in_type(self) -> type[InT]:
        return self._in_type

    @property
    def out_type(self) -> type[OutT]:
        return self._out_type

    @property
    def name(self) -> ProcName:
        return self._name

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def checkpoint_number(self) -> int:
        return self._checkpoint_number

    @property
    def max_retries(self) -> int:
        return self._max_retries

    # --- Session persistence ---

    @property
    def resumable(self) -> bool:
        return self._session_id is not None

    def setup_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._checkpoint_number = 0

    @property
    def _checkpoint_store_key(self) -> str | None:
        return None

    async def _deserialize_checkpoint(
        self, ctx: RunContext[CtxT], checkpoint_type: type[CpT]
    ) -> "CpT | None":
        store = ctx.store
        if store is None or self._checkpoint_store_key is None:
            return None

        data = await store.load(self._checkpoint_store_key)
        if data is None:
            return None

        try:
            checkpoint = checkpoint_type.model_validate_json(data)
        except Exception:
            logger.warning(
                "Corrupt checkpoint %s for %s, starting fresh",
                self._session_id,
                self.name,
                exc_info=True,
            )
            return None

        self._checkpoint_number = checkpoint.checkpoint_number

        return checkpoint

    async def _serialize_checkpoint(
        self,
        ctx: RunContext[CtxT],
        checkpoint: "ProcessorCheckpoint",
    ) -> None:
        store = ctx.store
        if store is None or self._checkpoint_store_key is None:
            return

        self._checkpoint_number += 1
        checkpoint.checkpoint_number = self._checkpoint_number

        await store.save(
            self._checkpoint_store_key,
            checkpoint.model_dump_json().encode("utf-8"),
        )

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
        self, output: OutT, *, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        raise NotImplementedError

    def add_recipient_selector(
        self, func: RecipientSelector[OutT, CtxT]
    ) -> RecipientSelector[OutT, CtxT]:
        self.select_recipients_impl = func

        return func

    @final
    def select_recipients(
        self, output: OutT, ctx: RunContext[CtxT], exec_id: str
    ) -> Sequence[ProcName]:
        if is_method_overridden("select_recipients_impl", self, Processor):
            recipients = self.select_recipients_impl(
                output=output, ctx=ctx, exec_id=exec_id
            )
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
        ctx: RunContext[CtxT],
        resume: bool = False,
    ) -> list[OutT]:
        """
        Process inputs and return outputs.

        Subclasses can override either ``_process`` or ``_process_stream``:

        - Override ``_process`` only → ``_process_stream`` wraps outputs in events.
        - Override ``_process_stream`` only → ``_process`` collects payload events.
        - Override both → each uses its own logic.
        - Override neither → passthrough (returns ``in_args``).
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
                ctx=ctx,
                resume=resume,
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
        ctx: RunContext[CtxT],
        resume: bool = False,
    ) -> AsyncIterator[Event[Any]]:
        """
        Stream events for inputs. See ``_process`` docstring for override rules.
        """
        outputs = await self._process(
            chat_inputs=chat_inputs,
            in_args=in_args,
            exec_id=exec_id,
            ctx=ctx,
            resume=resume,
        )
        for output in outputs:
            yield ProcPayloadOutEvent(data=output, source=self.name, exec_id=exec_id)

    def _build_packet(
        self,
        outputs: list[OutT],
        exec_id: str,
        ctx: RunContext[CtxT],
    ) -> Packet[OutT]:
        for output in outputs:
            self.validate_output(output, exec_id=exec_id)

        routings: list[Sequence[ProcName]] | None = []
        if self.recipients is not None:
            for output in outputs:
                routings.append(
                    self.select_recipients(output=output, ctx=ctx, exec_id=exec_id)
                )

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
    @workflow(name="processor")  # type: ignore
    @with_retry
    async def run_stream(
        self,
        chat_inputs: Any | None = None,
        *,
        in_packet: Packet[InT] | None = None,
        in_args: InT | list[InT] | None = None,
        exec_id: str | None = None,
        ctx: RunContext[CtxT] | None = None,
        resume: bool = False,
    ) -> AsyncIterator[Event[Any]]:
        ctx = ctx or RunContext[CtxT](state=None)  # type: ignore
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
            ctx=ctx,
            resume=resume,
        ):
            if isinstance(event, ProcPayloadOutEvent) and event.source == self.name:
                outputs.append(event.data)
            else:
                yield event

        out_packet = self._build_packet(outputs=outputs, exec_id=exec_id, ctx=ctx)
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
        ctx: RunContext[CtxT] | None = None,
        resume: bool = False,
    ) -> Packet[OutT]:
        result = None

        async for event in self.run_stream(
            chat_inputs=chat_inputs,
            in_packet=in_packet,
            in_args=in_args,
            exec_id=exec_id,
            ctx=ctx,
            resume=resume,
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
        reset_memory_on_run: bool = True,
        background: bool = False,
    ) -> "ProcessorTool[InT, OutT, CtxT]":  # type: ignore[return-value]
        from ..agent.processor_tool import ProcessorTool as _ProcessorTool

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
            reset_memory_on_run=reset_memory_on_run,
        )
