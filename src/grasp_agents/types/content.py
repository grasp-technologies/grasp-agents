import base64
import mimetypes
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

from openai.types.responses import (
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputText,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import Annotation, Logprob
from openai.types.responses.response_output_text import (
    AnnotationURLCitation as ResponseAnnotationURLCitation,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from pydantic import BaseModel, Field, model_validator

ImageDetail = Literal["low", "medium", "high", "ultra_high", "auto"]


BASE64_DATA_PREFIX = "data:{mime_type};base64,"

_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP
    (b"<svg", "image/svg+xml"),
]


def _detect_mime_from_base64(base64_encoding: str) -> str:
    try:
        header = base64.b64decode(base64_encoding[:16] + "==")
    except Exception:
        return "image/jpeg"
    for magic, mime in _MAGIC_BYTES:
        if header.startswith(magic):
            return mime
    return "image/jpeg"


class InputImage(ResponseInputImage):
    """
    Image content in a user/system/developer message.

    Supports URL, base64, or file_id.
    """

    # OpenResponses fields (InputImageContent):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: ImageDetail = "auto"  # pyright: ignore[reportIncompatibleVariableOverride] — extended with "medium", "ultra_high"

    # OpenAI-specific fields:
    file_id: str | None = None

    # grasp-agents fields:

    mime_type: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _check_exclusive_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        image_url = data.get("image_url")
        file_id = data.get("file_id")

        if (image_url is not None) and (file_id is not None):
            raise ValueError("InputImageContent cannot have both image_url and file_id")
        if (image_url is None) and (file_id is None):
            raise ValueError("InputImageContent must have either image_url or file_id")

        return data

    def model_post_init(self, _context: Any) -> None:
        if self.mime_type is None and self.image_url is not None:
            match = re.match(r"data:([^;]+);base64,", self.image_url)
            if match:
                self.mime_type = match.group(1)

    @property
    def is_base64(self) -> bool:
        return (
            self.image_url is not None
            and self.mime_type is not None
            and self.image_url.startswith(
                BASE64_DATA_PREFIX.format(mime_type=self.mime_type)
            )
        )

    @property
    def is_url(self) -> bool:
        return self.image_url is not None and not self.image_url.startswith("data:")

    @property
    def is_file_id(self) -> bool:
        return self.file_id is not None

    def to_str(self) -> str:
        if self.file_id is not None:
            return self.file_id
        if self.image_url is not None:
            return self.image_url
        raise ValueError(
            "Invalid InputImageContent: must have either image_url or file_id"
        )

    @classmethod
    def from_base64(
        cls,
        b64_encoding: str,
        *,
        mime_type: str | None = None,
        detail: ImageDetail = "auto",
    ) -> "InputImage":
        if mime_type is None:
            mime_type = _detect_mime_from_base64(b64_encoding)

        return cls(
            image_url=f"{BASE64_DATA_PREFIX.format(mime_type=mime_type)}{b64_encoding}",
            mime_type=mime_type,
            detail=detail,
        )

    @classmethod
    def from_path(
        cls, img_path: str | Path, *, detail: ImageDetail = "auto"
    ) -> "InputImage":
        img_bytes = Path(img_path).read_bytes()
        mime_type, _ = mimetypes.guess_type(str(img_path))
        b64_encoding = base64.b64encode(img_bytes).decode("utf-8")

        if mime_type is None:
            mime_type = _detect_mime_from_base64(b64_encoding)

        return cls(
            image_url=f"{BASE64_DATA_PREFIX.format(mime_type=mime_type)}{b64_encoding}",
            mime_type=mime_type,
            detail=detail,
        )

    @classmethod
    def from_url(cls, img_url: str, *, detail: ImageDetail = "auto") -> "InputImage":
        return cls(image_url=img_url, detail=detail)

    @classmethod
    def from_file_id(
        cls, file_id: str, *, detail: ImageDetail = "auto"
    ) -> "InputImage":
        return cls(file_id=file_id, detail=detail)


class InputText(ResponseInputText):
    """Text content in a user/system/developer message."""

    # OpenResponses fields (InputTextContent):
    type: Literal["input_text"] = "input_text"
    text: str


class InputFile(ResponseInputFile):
    """
    File content in a user/system/developer message.

    Supports base64 data, URL, or file_id.
    """

    # OpenResponses fields (InputFileContent):
    type: Literal["input_file"] = "input_file"
    filename: str | None = None
    file_data: str | None = None
    file_url: str | None = None

    # OpenAI-specific fields:
    file_id: str | None = None

    # grasp-agents fields:

    # check if only one of file_data, file_url, or file_id is provided
    @model_validator(mode="before")
    @classmethod
    def _check_exclusive_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        file_data = data.get("file_data")
        file_url = data.get("file_url")
        file_id = data.get("file_id")

        provided_fields = [f for f in [file_data, file_url, file_id] if f is not None]
        if len(provided_fields) != 1:
            raise ValueError(
                "InputFileContent must have exactly one of file_data, "
                "file_url, or file_id"
            )

        return data

    @property
    def is_base64(self) -> bool:
        return self.file_data is not None

    @property
    def is_url(self) -> bool:
        return self.file_url is not None

    @property
    def is_file_id(self) -> bool:
        return self.file_id is not None

    def to_str(self) -> str:
        if self.filename is not None:
            return self.filename
        if self.file_url is not None:
            return self.file_url
        if self.file_id is not None:
            return self.file_id
        raise ValueError(
            "Invalid InputFileContent: must have either filename, file_url, or file_id"
        )

    @classmethod
    def from_base64(cls, data: str, *, filename: str | None = None) -> "InputFile":
        return cls(file_data=data, filename=filename)

    @classmethod
    def from_url(cls, url: str, *, filename: str | None = None) -> "InputFile":
        return cls(file_url=url, filename=filename)

    @classmethod
    def from_file_id(cls, file_id: str, *, filename: str | None = None) -> "InputFile":
        return cls(file_id=file_id, filename=filename)

    @classmethod
    def from_path(cls, file_path: str | Path) -> "InputFile":
        path = Path(file_path)
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(file_data=data, filename=path.name)


class UrlCitation(ResponseAnnotationURLCitation):
    """URL citation with source excerpt and/or grounded response text."""

    # OpenResponses fields (UrlCitationBody):

    type: Literal["url_citation"] = "url_citation"

    url: str
    """The URL of the web resource."""

    title: str
    """The title of the web resource."""

    start_index: int
    """The index of the first character of the URL citation in the message."""

    end_index: int
    """The index of the last character of the URL citation in the message."""

    # grasp-agents fields:

    provider_specific_fields: dict[str, Any] | None = None

    # cited_text: str | None = None
    # """Text excerpted from the source (Anthropic web search citations)"""

    # grounded_text: str | None = None
    # """Part of the generated response grounded by this source (Gemini)"""

    # encrypted_index: str | None = None
    # """Anthropic-specific"""


Citation = Annotated[UrlCitation, Field(discriminator="type")]


class OutputMessageText(ResponseOutputText):
    """Text content produced by the model in an output message."""

    # OpenResponses fields (OutputTextContent):

    type: Literal["output_text"] = "output_text"
    annotations: list[Annotation] = Field(default_factory=list[Annotation])
    logprobs: list[Logprob] | None = None
    text: str

    # grasp-agents fields:

    citations: list[Citation] = Field(default_factory=list[Citation])

    @model_validator(mode="before")
    @classmethod
    def _sync_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if "citations" in data and "annotations" not in data:
            data["annotations"] = data["citations"]
        elif "annotations" in data and "citations" not in data:
            data["citations"] = data["annotations"]

        return data


class OutputMessageRefusal(ResponseOutputRefusal):
    """Refusal content when the model declines to respond."""

    # OpenResponses fields (RefusalContent):
    type: Literal["refusal"] = "refusal"
    refusal: str


class ReasoningText(ResponseReasoningContent):
    """Reasoning content produced by the model."""

    # OpenResponses fields (ReasoningText):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ReasoningSummary(ResponseReasoningSummary):
    """Summary block within a reasoning item."""

    # OpenResponses fields (ReasoningSummaryContentParam):
    type: Literal["summary_text"] = "summary_text"
    text: str


InputPart = Annotated[InputText | InputImage | InputFile, Field(discriminator="type")]

OutputMessagePart = Annotated[
    OutputMessageText | OutputMessageRefusal, Field(discriminator="type")
]

# "Content" in OpenResponses / Responses API is taken to be anything an LLM generates
# directly, token by token. Anything derived from the generated tokens that underwent
# parsing (e.g. tool calls, annotations) or other post-processing (e.g. reasoning
# summaries) is not considered "Content" in this sense.

OutputContentPart = Annotated[
    OutputMessageText | OutputMessageRefusal | ReasoningText,
    Field(discriminator="type"),
]

OutputPart = Annotated[
    OutputMessageText | OutputMessageRefusal | ReasoningText | ReasoningSummary,
    Field(discriminator="type"),
]


class Content(BaseModel):
    parts: list[InputPart]

    @classmethod
    def from_formatted_prompt(
        cls,
        prompt_template: str,
        /,
        **prompt_args: str | int | bool | InputImage | None,
    ) -> "Content":
        prompt_args = prompt_args or {}
        image_args = {
            arg_name: arg_val
            for arg_name, arg_val in prompt_args.items()
            if isinstance(arg_val, InputImage)
        }
        text_args = {
            arg_name: arg_val
            for arg_name, arg_val in prompt_args.items()
            if isinstance(arg_val, (str, int, float))
        }

        if not image_args:
            prompt_with_args = prompt_template.format(**text_args)
            return cls(parts=[InputText(text=prompt_with_args)])

        pattern = r"({})".format("|".join([r"\{" + s + r"\}" for s in image_args]))
        input_prompt_chunks = re.split(pattern, prompt_template)

        content_parts: list[InputPart] = []
        for chunk in input_prompt_chunks:
            stripped_chunk = chunk.strip(" \n")
            if re.match(pattern, stripped_chunk):
                content_part = image_args[stripped_chunk[1:-1]]
            else:
                text_data = stripped_chunk.format(**text_args)
                content_part = InputText(text=text_data)
            content_parts.append(content_part)

        return cls(parts=content_parts)

    @classmethod
    def from_text(cls, text: str) -> "Content":
        return cls(parts=[InputText(text=text)])

    @classmethod
    def from_image(cls, image: InputImage) -> "Content":
        return cls(parts=[image])

    @classmethod
    def from_images(cls, images: Iterable[InputImage]) -> "Content":
        return cls(parts=list(images))

    @classmethod
    def from_content_parts(cls, content_parts: Iterable[str | InputImage]) -> "Content":
        parts: list[InputPart] = []
        for part in content_parts:
            if isinstance(part, str):
                parts.append(InputText(text=part))
            else:
                parts.append(part)

        return cls(parts=parts)
