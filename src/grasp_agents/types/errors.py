from typing import Any

# --- Processor errors ---


class ProcRunError(Exception):
    def __init__(
        self, proc_name: str, exec_id: str | None = None, message: str | None = None
    ) -> None:
        super().__init__(
            message
            or f"Processor run failed [proc_name: {proc_name}; exec_id: {exec_id}]."
        )
        self.proc_name = proc_name
        self.exec_id = exec_id


class ProcInputValidationError(ProcRunError):
    pass


class ProcOutputValidationError(ProcRunError):
    def __init__(
        self,
        schema: object,
        proc_name: str,
        exec_id: str | None = None,
        message: str | None = None,
    ):
        super().__init__(
            proc_name=proc_name,
            exec_id=exec_id,
            message=message
            or (
                "Processor output validation failed "
                f"[proc_name: {proc_name}; exec_id: {exec_id}]. "
                f"Expected type:\n{schema}"
            ),
        )


class AgentFinalAnswerError(ProcRunError):
    def __init__(
        self, proc_name: str, exec_id: str | None = None, message: str | None = None
    ) -> None:
        super().__init__(
            proc_name=proc_name,
            exec_id=exec_id,
            message=message
            or "Final answer tool call did not return a final answer message "
            f"[proc_name={proc_name}; exec_id={exec_id}]",
        )
        self.message = message


class TranscriptInvariantError(Exception):
    """
    A tool call in the transcript isn't resolved by its result in place.

    Providers require an assistant turn's ``tool_calls`` to be followed by
    their matching ``tool_result``s before any user / system message, and
    none may dangle unresolved. The agent loop maintains this by
    construction; this is raised (before the provider 400s) if a custom
    hook, converter, or bug violates it.
    """


class WorkflowConstructionError(Exception):
    pass


class PacketRoutingError(ProcRunError):
    def __init__(
        self,
        proc_name: str,
        exec_id: str | None = None,
        selected_recipient: str | None = None,
        allowed_recipients: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        default_message = (
            f"Selected recipient '{selected_recipient}' is not in the allowed "
            f"recipients: {allowed_recipients} "
            f"[proc_name={proc_name}; exec_id={exec_id}]"
        )
        super().__init__(
            proc_name=proc_name, exec_id=exec_id, message=message or default_message
        )
        self.selected_recipient = selected_recipient
        self.allowed_recipients = allowed_recipients


class RunnerError(Exception):
    pass


class PromptBuilderError(Exception):
    def __init__(self, proc_name: str, message: str | None = None) -> None:
        super().__init__(message or f"Prompt builder failed [proc_name={proc_name}]")
        self.proc_name = proc_name
        self.message = message


class SystemPromptBuilderError(PromptBuilderError):
    def __init__(self, proc_name: str, message: str | None = None) -> None:
        super().__init__(
            proc_name=proc_name,
            message=message
            or "System prompt builder failed to make system prompt "
            f"[proc_name={proc_name}]",
        )
        self.message = message


class InputPromptBuilderError(PromptBuilderError):
    def __init__(self, proc_name: str, message: str | None = None) -> None:
        super().__init__(
            proc_name=proc_name,
            message=message
            or "Input prompt builder failed to make input content "
            f"[proc_name={proc_name}]",
        )
        self.message = message


class PyJSONStringParsingError(Exception):
    def __init__(self, s: str, message: str | None = None) -> None:
        super().__init__(
            message
            or "Both ast.literal_eval and json.loads failed to parse the following "
            f"JSON/Python string:\n{s}"
        )
        self.s = s


class JSONSchemaValidationError(Exception):
    def __init__(self, s: str, schema: object, message: str | None = None) -> None:
        super().__init__(
            message
            or f"JSON schema validation failed for:\n{s}\nExpected type: {schema}"
        )
        self.s = s
        self.schema = schema


class CompletionError(Exception):
    pass


class LLMToolCallValidationError(Exception):
    """
    Raised when one or more tool calls in an LLM response reference an
    unknown tool or carry invalid arguments.

    Carries the bad ``response`` so the caller can commit the failed
    assistant turn to the transcript and synthesize matching
    ``FunctionToolOutputItem``s — one per failure — so the next LLM call
    sees the validation errors as tool results and can correct itself.

    ``failed_calls`` lists every offending tool call as a 3-tuple of
    ``(call_id, tool_name, error_message)``; ``message`` summarizes them.
    """

    def __init__(
        self,
        message: str,
        *,
        response: Any = None,
        failed_calls: list[tuple[str, str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.response = response
        self.failed_calls: list[tuple[str, str, str]] = failed_calls or []


class LLMResponseValidationError(JSONSchemaValidationError):
    def __init__(self, s: str, schema: object, message: str | None = None) -> None:
        super().__init__(
            s,
            schema,
            message
            or f"Failed to validate LLM response:\n{s}\nExpected type: {schema}",
        )
