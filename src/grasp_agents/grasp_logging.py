import contextlib
import logging
import logging.config
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from logging import Formatter, Handler, LogRecord
from pathlib import Path
from typing import Any

import yaml
from rich.text import Text

try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # opentelemetry is an optional dependency
    _otel_trace = None

# The library never attaches handlers on import — the host application owns
# output config. ``setup_logging`` / ``enable_verbose_stdout_logging`` are the
# opt-in entry points; everything below those is inert until one is called.
PACKAGE_LOGGER_NAME = "grasp_agents"
package_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

logger = logging.getLogger(__name__)


# --- Color formatter ---


class ColorFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        message = super().format(record)
        color: str | None = getattr(record, "color", None)
        if color:
            styled = Text(message, style=color)
            return styled.markup
        return message


# --- Body-logging gates ---


def _env_flag(name: str, *, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"", "0", "false", "no", "off"}


# Full request/response bodies can carry prompts, user data and tool payloads,
# so they are logged (at DEBUG) only when explicitly enabled. The four payloads
# are gated independently because they differ wildly in size: tool *input* (args)
# and LLM *output* are usually small and worth logging, while tool *output* and
# LLM *input* (the assembled transcript) are often very large. Read live at each
# call site so they can also be toggled at runtime:
#   import grasp_agents.grasp_logging as glog; glog.LOG_LLM_OUTPUT = True
LOG_LLM_INPUT: bool = _env_flag("GRASP_LOG_LLM_INPUT")
LOG_LLM_OUTPUT: bool = _env_flag("GRASP_LOG_LLM_OUTPUT")
LOG_TOOL_INPUT: bool = _env_flag("GRASP_LOG_TOOL_INPUT")
LOG_TOOL_OUTPUT: bool = _env_flag("GRASP_LOG_TOOL_OUTPUT")

MAX_BODY_LOG_CHARS = 4000
SNIPPET_HEAD_CHARS = 200
SNIPPET_TAIL_CHARS = 100


def truncate_for_log(text: str, limit: int = MAX_BODY_LOG_CHARS) -> str:
    """Clip an oversized body to keep a single log line bounded."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} chars]"


def snippet(
    text: str, head: int = SNIPPET_HEAD_CHARS, tail: int = SNIPPET_TAIL_CHARS
) -> str:
    """
    A compact one-line head…tail preview of a body.

    Whitespace is collapsed so the preview stays on a single log line. Used when
    the corresponding full-body gate is off — a peek at what flowed without the
    bulk.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= head + tail:
        return collapsed
    omitted = len(collapsed) - head - tail
    return f"{collapsed[:head]} …[{omitted} chars]… {collapsed[-tail:]}"


def body_for_log(text: str, *, full: bool) -> str:
    """Full (clipped) body when its gate is on, else a compact snippet."""
    return truncate_for_log(text) if full else snippet(text)


# --- Secret redaction ---

# Snapshot at import: a tool-driven ``export GRASP_LOG_REDACT_SECRETS=0`` partway
# through a session must not be able to turn redaction off for the rest of it.
_REDACT_ENABLED: bool = _env_flag("GRASP_LOG_REDACT_SECRETS", default=True)

# Bounded quantifiers throughout (no unbounded backtracking) to stay ReDoS-safe
# on large log payloads.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"),  # JWT
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}"),  # Anthropic
    re.compile(r"\b[sr]k-(?:proj-)?[A-Za-z0-9_-]{16,}"),  # OpenAI
    re.compile(r"\bAIza[A-Za-z0-9_-]{16,}"),  # Google
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}"),  # GitHub
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),  # Hugging Face
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}"),  # Groq
)

# ``NAME=secret`` / ``NAME: secret`` and ``Authorization: Bearer …`` — keep the
# name, drop the value.
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Za-z0-9_]*)"
    r"(\s*[=:]\s*)([^\s,;'\"]{4,})"
)
_AUTH_PATTERN = re.compile(
    r"(?i)\b(authorization|bearer)(\s*[=:]?\s+)([A-Za-z0-9._~+/-]{8,}=*)"
)


def _mask(token: str) -> str:
    if len(token) <= 10:
        return "***"
    return f"{token[:6]}…{token[-4:]}"


def _mask_match(match: re.Match[str]) -> str:
    return _mask(match.group(0))


def redact_secrets(text: str) -> str:
    """Mask secret-looking substrings (API keys, tokens, Bearer headers, PEM)."""
    if not _REDACT_ENABLED or not text:
        return text
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(_mask_match, text)
    text = _ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", text)
    return _AUTH_PATTERN.sub(r"\1\2<redacted>", text)


class SecretRedactingFilter(logging.Filter):
    """
    Redact secrets from a record's rendered message.

    Attach to handlers that emit ``grasp_agents`` records (``setup_logging`` and
    ``enable_verbose_stdout_logging`` do this automatically); embedding hosts can
    add it to their own handlers in one line.
    """

    def filter(self, record: LogRecord) -> bool:
        if not _REDACT_ENABLED:
            return True
        with contextlib.suppress(Exception):
            message = record.getMessage()
            redacted = redact_secrets(message)
            if redacted != message:
                record.msg = redacted
                record.args = None
        return True


# --- Correlation context ---

_LOG_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "grasp_agents_log_context", default=None
)


@contextmanager
def log_context(
    *, exec_id: str | None = None, proc: str | None = None
) -> Iterator[None]:
    """
    Bind correlation fields onto every log record emitted within the block.

    Nests: an inner block layers over the outer one. Stamps ``exec_id`` / ``proc``
    (the running processor's name) — and, when a span is recording, ``otelTraceID``
    / ``otelSpanID`` — once :func:`install_log_correlation` has run.
    """
    current = _LOG_CONTEXT.get() or {}
    merged = dict(current)
    if exec_id is not None:
        merged["exec_id"] = exec_id
    if proc is not None:
        merged["proc"] = proc
    token = _LOG_CONTEXT.set(merged)
    try:
        yield
    finally:
        try:
            _LOG_CONTEXT.reset(token)
        except ValueError:
            # Finalized in a different context than it was entered — e.g. an
            # async generator aclose()d / GC'd from another task (an
            # interrupted run_stream). The originating context is discarded
            # anyway, so the correlation metadata needs no reset.
            pass


def _current_span_ids() -> tuple[str, str]:
    if _otel_trace is None:
        return ("-", "-")
    span_context = _otel_trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return ("-", "-")
    return (
        format(span_context.trace_id, "032x"),
        format(span_context.span_id, "016x"),
    )


_ORIG_RECORD_FACTORY = logging.getLogRecordFactory()
_correlation_installed = False


def _record_factory(*args: Any, **kwargs: Any) -> LogRecord:
    record = _ORIG_RECORD_FACTORY(*args, **kwargs)
    current = _LOG_CONTEXT.get() or {}
    trace_id, span_id = _current_span_ids()
    record.__dict__.update(
        exec_id=current.get("exec_id", "-"),
        proc=current.get("proc", "-"),
        otelTraceID=trace_id,
        otelSpanID=span_id,
    )
    return record


def install_log_correlation() -> None:
    """
    Install a record factory so ``exec_id`` / ``proc`` / ``otelTraceID`` /
    ``otelSpanID`` are present on every record (defaulting to ``"-"``).

    Idempotent and purely additive — it only sets new attributes, never mutates
    existing fields — so it is safe to call from an embedding host that wants its
    own format string or JSON formatter to reference these fields.
    """
    global _correlation_installed
    if _correlation_installed:
        return
    logging.setLogRecordFactory(_record_factory)
    _correlation_installed = True


# --- Setup entry points ---


def _instrument_handlers(handlers: list[Handler]) -> None:
    for handler in handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())


def setup_logging(logs_file_path: str | Path, logs_config_path: str | Path) -> None:
    """Configure logging from a YAML ``dictConfig`` file (host/app entry point)."""
    logs_file_path = Path(logs_file_path)
    logs_file_path.parent.mkdir(exist_ok=True, parents=True)
    with Path(logs_config_path).open() as f:
        config = yaml.safe_load(f)

    config["handlers"]["fileHandler"]["filename"] = logs_file_path

    logging.config.dictConfig(config)
    install_log_correlation()

    root = logging.getLogger()
    for handler in root.handlers:
        if handler.formatter is not None:
            fmt_str = handler.formatter._fmt  # noqa: SLF001
            handler.setFormatter(ColorFormatter(fmt_str))
    _instrument_handlers(root.handlers)


_NOISY_THIRD_PARTY = (
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "litellm",
    "LiteLLM",
    "google_genai",
    "jupyter_client",
    "urllib3",
)

_VERBOSE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(exec_id)s] %(message)s"


def enable_verbose_stdout_logging(
    level: int = logging.DEBUG, *, quiet_third_party: bool = True
) -> Handler:
    """
    Route ``grasp_agents`` logs to stdout at ``level`` — a one-liner for local
    debugging. Scoped to the ``grasp_agents`` logger (it does not touch the root
    config, so it is safe to call while embedded). Idempotent.

    Set ``GRASP_LOG_LLM_INPUT`` / ``GRASP_LOG_LLM_OUTPUT`` /
    ``GRASP_LOG_TOOL_INPUT`` / ``GRASP_LOG_TOOL_OUTPUT`` to also surface those
    full bodies at DEBUG (each off by default; they differ widely in size).
    """
    install_log_correlation()
    package_logger.setLevel(level)

    existing = [
        h
        for h in package_logger.handlers
        if getattr(h, "_grasp_verbose_handler", False)
    ]
    if existing:
        handler = existing[0]
        handler.setLevel(level)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler._grasp_verbose_handler = True  # type: ignore[attr-defined]  # noqa: SLF001
        handler.setLevel(level)
        handler.setFormatter(ColorFormatter(_VERBOSE_FORMAT))
        handler.addFilter(SecretRedactingFilter())
        package_logger.addHandler(handler)

    if quiet_third_party:
        for name in _NOISY_THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.WARNING)

    return handler
