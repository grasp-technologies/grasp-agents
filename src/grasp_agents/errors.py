# from openai import APIResponseValidationError


class ProcRunError(Exception):
    def __init__(
        self, proc_name: str, call_id: str, message: str | None = None
    ) -> None:
        super().__init__(
            message
            or f"Processor run failed [proc_name: {proc_name}; call_id: {call_id}]."
        )
        self.proc_name = proc_name
        self.call_id = call_id


class ProcInputValidationError(ProcRunError):
    pass


class ProcOutputValidationError(ProcRunError):
    def __init__(
        self, schema: object, proc_name: str, call_id: str, message: str | None = None
    ):
        super().__init__(
            proc_name=proc_name,
            call_id=call_id,
            message=message
            or (
                "Processor output validation failed "
                f"[proc_name: {proc_name}; call_id: {call_id}]. "
                f"Expected type:\n{schema}"
            ),
        )


class AgentFinalAnswerError(ProcRunError):
    def __init__(
        self, proc_name: str, call_id: str, message: str | None = None
    ) -> None:
        super().__init__(
            proc_name=proc_name,
            call_id=call_id,
            message=message
            or "Final answer tool call did not return a final answer message "
            f"[proc_name={proc_name}; call_id={call_id}]",
        )
        self.message = message


class WorkflowConstructionError(Exception):
    pass


class PacketRoutingError(ProcRunError):
    def __init__(
        self,
        proc_name: str,
        call_id: str,
        selected_recipient: str | None = None,
        allowed_recipients: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        default_message = (
            f"Selected recipient '{selected_recipient}' is not in the allowed "
            f"recipients: {allowed_recipients} "
            f"[proc_name={proc_name}; call_id={call_id}]"
        )
        super().__init__(
            proc_name=proc_name, call_id=call_id, message=message or default_message
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


class CombineCompletionChunksError(Exception):
    pass


class LLMToolCallValidationError(Exception):
    def __init__(
        self, tool_name: str, tool_args: str, message: str | None = None
    ) -> None:
        super().__init__(
            message
            or f"Failed to validate tool call '{tool_name}' with arguments:"
            f"\n{tool_args}."
        )
        self.tool_name = tool_name
        self.tool_args = tool_args


class LLMResponseValidationError(JSONSchemaValidationError):
    def __init__(self, s: str, schema: object, message: str | None = None) -> None:
        super().__init__(
            s,
            schema,
            message
            or f"Failed to validate LLM response:\n{s}\nExpected type: {schema}",
        )
