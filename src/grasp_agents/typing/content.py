import base64
import re
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import AnyUrl, BaseModel, Field


class ContentType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


ImageDetail: TypeAlias = Literal["low", "high", "auto"]


class ImageData(BaseModel):
    type: Literal["url", "base64"]
    url: AnyUrl | None = None
    base64: str | None = None

    # Supported by OpenAI API
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
    def from_content_parts(cls, content_parts: Iterable[str | ImageData]) -> "Content":
        parts: list[ContentPart] = []
        for part in content_parts:
            if isinstance(part, str):
                parts.append(ContentPartText(data=part))
            else:
                parts.append(ContentPartImage(data=part))

        return cls(parts=parts)
