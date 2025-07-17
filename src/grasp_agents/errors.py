# from openai import APIResponseValidationError
class CompletionError(Exception):
    pass


class CombineCompletionChunksError(Exception):
    pass


class ProcInputValidationError(Exception):
    pass


class ProcOutputValidationError(Exception):
    pass


class AgentFinalAnswerError(Exception):
    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message or "Final answer tool call did not return a final answer message."
        )
        self.message = message


class WorkflowConstructionError(Exception):
    pass


class PacketRoutingError(Exception):
    def __init__(
        self,
        selected_recipient: str,
        allowed_recipients: list[str],
        message: str | None = None,
    ) -> None:
        default_message = (
            f"Selected recipient '{selected_recipient}' is not in the allowed "
            f"recipients: {allowed_recipients}"
        )
        super().__init__(message or default_message)
        self.selected_recipient = selected_recipient
        self.allowed_recipients = allowed_recipients


class SystemPromptBuilderError(Exception):
    pass


class InputPromptBuilderError(Exception):
    pass


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
