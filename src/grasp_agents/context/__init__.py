"""
Prompt / context construction.

Everything that shapes what the model sees: the :class:`PromptBuilder` and its
section/attachment vocabulary (:class:`SystemPromptSection`,
:class:`InputAttachment`), plus the built-in section providers
(:func:`make_env_info_section`, :func:`make_untrusted_content_section`).
Future context-window management (compaction) lands here too.
"""

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
    "UNTRUSTED_CONTENT_INSTRUCTION",
    "UNTRUSTED_CONTENT_SECTION_NAME",
    "UNTRUSTED_CONTENT_TAG",
    "InputAttachment",
    "InputAttachmentCompute",
    "PromptBuilder",
    "SectionCompute",
    "SystemPromptSection",
    "make_current_time_attachment",
    "make_env_info_section",
    "make_untrusted_content_section",
    "unwrap_untrusted",
    "wrap_untrusted",
]
