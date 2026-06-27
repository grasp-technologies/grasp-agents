"""
Prompt / context construction.

Everything that shapes what the model sees: the :class:`PromptBuilder` and its
section/attachment vocabulary (:class:`SystemPromptSection`,
:class:`InputAttachment`), plus the built-in section providers
(:func:`make_env_info_section`, :func:`make_untrusted_content_section`).
Future context-window management (compaction) lands here too.
"""

from .compaction import (
    CollapseToolOutputsProjector,
    Compaction,
    ContextBudget,
    LLMSummarizer,
    Summarizer,
    SummarizingCompactor,
    collapse_tool_outputs,
)
from .env_section import (
    CURRENT_TIME_ATTACHMENT_NAME,
    ENV_INFO_SECTION_NAME,
    make_current_time_attachment,
    make_env_info_section,
)
from .prompt_builder import (
    InputAttachment,
    InputAttachmentCompute,
    PromptBuilder,
    SectionCompute,
    SystemPromptSection,
)
from .system_reminder import SYSTEM_REMINDER_TAG, wrap_in_system_reminder
from .untrusted_content import (
    UNTRUSTED_CONTENT_INSTRUCTION,
    UNTRUSTED_CONTENT_SECTION_NAME,
    UNTRUSTED_CONTENT_TAG,
    make_untrusted_content_section,
    unwrap_untrusted,
    wrap_untrusted,
)

__all__ = [
    "CURRENT_TIME_ATTACHMENT_NAME",
    "ENV_INFO_SECTION_NAME",
    "SYSTEM_REMINDER_TAG",
    "UNTRUSTED_CONTENT_INSTRUCTION",
    "UNTRUSTED_CONTENT_SECTION_NAME",
    "UNTRUSTED_CONTENT_TAG",
    "CollapseToolOutputsProjector",
    "Compaction",
    "ContextBudget",
    "InputAttachment",
    "InputAttachmentCompute",
    "LLMSummarizer",
    "PromptBuilder",
    "SectionCompute",
    "Summarizer",
    "SummarizingCompactor",
    "SystemPromptSection",
    "collapse_tool_outputs",
    "make_current_time_attachment",
    "make_env_info_section",
    "make_untrusted_content_section",
    "unwrap_untrusted",
    "wrap_in_system_reminder",
    "wrap_untrusted",
]
