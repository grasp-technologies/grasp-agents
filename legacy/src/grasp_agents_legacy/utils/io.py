from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)


def read_txt(file_path: str | Path, encoding: str = "utf-8") -> str:
    return Path(file_path).read_text(encoding=encoding)


def read_contents_from_file(
    file_path: str | Path,
    binary_mode: bool = False,
) -> str | bytes:
    try:
        if binary_mode:
            return Path(file_path).read_bytes()
        return Path(file_path).read_text()
    except FileNotFoundError:
        logger.exception(f"File {file_path} not found.")
        return ""


def get_prompt(prompt_text: str | None, prompt_path: str | Path | None) -> str | None:
    if prompt_text is None:
        return read_contents_from_file(prompt_path) if prompt_path is not None else None  # type: ignore[arg-type]

    return prompt_text
