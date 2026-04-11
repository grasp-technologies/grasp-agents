import logging
from logging import Formatter, LogRecord
from pathlib import Path

import yaml
from rich.text import Text


class ColorFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        message = super().format(record)
        color: str | None = getattr(record, "color", None)
        if color:
            styled = Text(message, style=color)
            return styled.markup
        return message


def setup_logging(logs_file_path: str | Path, logs_config_path: str | Path) -> None:
    logs_file_path = Path(logs_file_path)
    logs_file_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(logs_config_path).open() as f:
        config = yaml.safe_load(f)

    config["handlers"]["fileHandler"]["filename"] = logs_file_path

    logging.config.dictConfig(config)  # type: ignore

    root = logging.getLogger()
    for handler in root.handlers:
        if handler.formatter is not None:
            fmt_str = handler.formatter._fmt  # noqa: SLF001
            handler.setFormatter(ColorFormatter(fmt_str))


logger = logging.getLogger(__name__)
