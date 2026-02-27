import base64
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from openai.types.responses import (
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputText,
    ResponseOutputRefusal,
    ResponseOutputText,
)
from openai.types.responses.response_output_text import Annotation, Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary as ResponseReasoningSummary,
)
from pydantic import AnyUrl, BaseModel, Field

ImageDetail: TypeAlias = Literal["low", "high", "auto"]

BASE64_DATA_URL_PREFIX = "data:image/jpeg;base64,"


class InputImageContent(ResponseInputImage):
    """
    Image content in a user/system/developer message.

    Supports URL, base64, or file_id.
    """

    # OpenResponses fields:
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: ImageDetail = "auto"

    # OpenAI-specific fields:
    file_id: str | None = None

    # grasp-agents fields:

    @property
    def is_base64(self) -> bool:
        return self.image_url is not None and self.image_url.startswith("data:")

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
        return ""

    @classmethod
    def from_base64(
        cls, base64_encoding: str, *, detail: ImageDetail = "auto", **kwargs: Any
    ) -> "InputImageContent":
        return cls(
            image_url=f"{BASE64_DATA_URL_PREFIX}{base64_encoding}",
            detail=detail,
            **kwargs,
        )

    @classmethod
    def from_path(
        cls, img_path: str | Path, *, detail: ImageDetail = "auto", **kwargs: Any
    ) -> "InputImageContent":
        img_bytes = Path(img_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return cls(image_url=f"{BASE64_DATA_URL_PREFIX}{b64}", detail=detail, **kwargs)

    @classmethod
    def from_url(
        cls, img_url: str, *, detail: ImageDetail = "auto", **kwargs: Any
    ) -> "InputImageContent":
        return cls(image_url=img_url, detail=detail, **kwargs)

    @classmethod
    def from_file_id(
        cls, file_id: str, *, detail: ImageDetail = "auto", **kwargs: Any
    ) -> "InputImageContent":
        return cls(file_id=file_id, detail=detail, **kwargs)


class InputTextContent(ResponseInputText):
    """Text content in a user/system/developer message."""

    # OpenResponses fields:
    type: Literal["input_text"] = "input_text"
    text: str


class InputFileContent(ResponseInputFile):
    """
    File content in a user/system/developer message.

    Supports base64 data, URL, or file_id.
    """

    # OpenResponses fields:
    type: Literal["input_file"] = "input_file"
    filename: str | None = None
    file_data: str | None = None
    file_url: str | None = None

    # OpenAI-specific fields:
    file_id: str | None = None

    # grasp-agents fields:

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
        return ""

    @classmethod
    def from_base64(
        cls, data: str, *, filename: str | None = None, **kwargs: Any
    ) -> "InputFileContent":
        return cls(file_data=data, filename=filename, **kwargs)

    @classmethod
    def from_url(
        cls, url: str, *, filename: str | None = None, **kwargs: Any
    ) -> "InputFileContent":
        return cls(file_url=url, filename=filename, **kwargs)

    @classmethod
    def from_file_id(
        cls, file_id: str, *, filename: str | None = None, **kwargs: Any
    ) -> "InputFileContent":
        return cls(file_id=file_id, filename=filename, **kwargs)

    @classmethod
    def from_path(cls, file_path: str | Path, **kwargs: Any) -> "InputFileContent":
        path = Path(file_path)
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return cls(file_data=data, filename=path.name, **kwargs)


InputContent = Annotated[
    InputTextContent | InputImageContent | InputFileContent, Field(discriminator="type")
]


class OutputTextContent(ResponseOutputText):
    """Text content produced by the model in an output message."""

    # OpenResponses fields:
    type: Literal["output_text"] = "output_text"
    annotations: list[Annotation] = Field(default_factory=list[Annotation])
    logprobs: list[Logprob] | None = None
    text: str


class OutputRefusalContent(ResponseOutputRefusal):
    """Refusal content when the model declines to respond."""

    # OpenResponses fields:
    type: Literal["refusal"] = "refusal"
    refusal: str


OutputContent = Annotated[
    OutputTextContent | OutputRefusalContent, Field(discriminator="type")
]


class ReasoningTextContent(ResponseReasoningContent):
    """Reasoning content produced by the model in an output message."""

    # OpenResponses fields:
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str


class ReasoningSummaryContent(ResponseReasoningSummary):
    """Summary block within a reasoning item."""

    # OpenResponses fields (ReasoningSummaryContentParam):
    type: Literal["summary_text"] = "summary_text"
    text: str


# --- Legacy types (kept for backward compatibility) ---


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ImageData(BaseModel):
    type: Literal["url", "base64"]
    url: AnyUrl | None = None
    base64: str | None = None

    detail: ImageDetail = "high"

    @classmethod
    def from_base64(cls, base64_encoding: str, **kwargs: Any) -> "ImageData":
        return cls(type="base64", base64=base64_encoding, **kwargs)

    @classmethod
    def from_path(cls, img_path: str | Path, **kwargs: Any) -> "ImageData":
        image_bytes = Path(img_path).read_bytes()
        base64_encoding = base64.b64encode(image_bytes).decode("utf-8")
        return cls(type="base64", base64=base64_encoding, **kwargs)

    @classmethod
    def from_url(cls, img_url: str, **kwargs: Any) -> "ImageData":
        return cls(type="url", url=img_url, **kwargs)  # type: ignore

    def to_str(self) -> str:
        if self.type == "url":
            return str(self.url)
        if self.type == "base64":
            return str(self.base64)
        raise ValueError(f"Unsupported image data type: {self.type}")


class ContentPartText(BaseModel):
    type: Literal[ContentType.TEXT] = ContentType.TEXT
    data: str


class ContentPartImage(BaseModel):
    type: Literal[ContentType.IMAGE] = ContentType.IMAGE
    data: ImageData


ContentPart = Annotated[ContentPartText | ContentPartImage, Field(discriminator="type")]


class Content(BaseModel):
    parts: list[ContentPart]

    @classmethod
    def from_formatted_prompt(
        cls,
        prompt_template: str,
        /,
        **prompt_args: str | int | bool | ImageData | None,
    ) -> "Content":
        prompt_args = prompt_args or {}
        image_args = {
            arg_name: arg_val
            for arg_name, arg_val in prompt_args.items()
            if isinstance(arg_val, ImageData)
        }
        text_args = {
            arg_name: arg_val
            for arg_name, arg_val in prompt_args.items()
            if isinstance(arg_val, (str, int, float))
        }

        if not image_args:
            prompt_with_args = prompt_template.format(**text_args)
            return cls(parts=[ContentPartText(data=prompt_with_args)])

        pattern = r"({})".format("|".join([r"\{" + s + r"\}" for s in image_args]))
        input_prompt_chunks = re.split(pattern, prompt_template)

        content_parts: list[ContentPart] = []
        for chunk in input_prompt_chunks:
            stripped_chunk = chunk.strip(" \n")
            if re.match(pattern, stripped_chunk):
                image_data = image_args[stripped_chunk[1:-1]]
                content_part = ContentPartImage(data=image_data)
            else:
                text_data = stripped_chunk.format(**text_args)
                content_part = ContentPartText(data=text_data)
            content_parts.append(content_part)

        return cls(parts=content_parts)

    @classmethod
    def from_text(cls, text: str) -> "Content":
        return cls(parts=[ContentPartText(data=text)])

    @classmethod
    def from_image(cls, image: ImageData) -> "Content":
        return cls(parts=[ContentPartImage(data=image)])

    @classmethod
    def from_images(cls, images: Iterable[ImageData]) -> "Content":
        return cls(parts=[ContentPartImage(data=image) for image in images])

    @classmethod
    def from_content_parts(cls, content_parts: Iterable[str | ImageData]) -> "Content":
        parts: list[ContentPart] = []
        for part in content_parts:
            if isinstance(part, str):
                parts.append(ContentPartText(data=part))
            else:
                parts.append(ContentPartImage(data=part))

        return cls(parts=parts)
