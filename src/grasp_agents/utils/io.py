from pathlib import Path


def read_txt(file_path: str | Path, encoding: str = "utf-8") -> str:
    return Path(file_path).read_text(encoding=encoding)


def read_contents_from_file(
    file_path: str | Path,
    binary_mode: bool = False,
) -> str | bytes:
    # A missing file propagates: an explicitly-configured prompt path that
    # doesn't exist is a configuration error — returning "" would silently
    # run the agent with an empty system prompt.
    if binary_mode:
        return Path(file_path).read_bytes()
    return Path(file_path).read_text()


def get_prompt(prompt_text: str | None, prompt_path: str | Path | None) -> str | None:
    if prompt_text is None:
        return read_contents_from_file(prompt_path) if prompt_path is not None else None  # type: ignore[arg-type]

    return prompt_text
