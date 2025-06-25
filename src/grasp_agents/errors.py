class InputValidationError(Exception):
    pass


class StringParsingError(Exception):
    pass


class CompletionError(Exception):
    pass


class CombineCompletionChunksError(Exception):
    pass


class ToolValidationError(Exception):
    pass


class OutputValidationError(Exception):
    pass


class WorkflowConstructionError(Exception):
    pass


class SystemPromptBuilderError(Exception):
    pass


class InputPromptBuilderError(Exception):
    pass
