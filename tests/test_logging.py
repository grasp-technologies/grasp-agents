"""Tests for the logging infrastructure in grasp_agents.grasp_logging."""

import logging
from collections.abc import Iterator

import pytest

from grasp_agents import grasp_logging
from grasp_agents.grasp_logging import (
    SecretRedactingFilter,
    body_for_log,
    enable_verbose_stdout_logging,
    install_log_correlation,
    log_context,
    redact_secrets,
    snippet,
    truncate_for_log,
)
from grasp_agents.tools.function_tool import function_tool


@pytest.fixture
def _restore_logging_globals() -> Iterator[None]:
    """Snapshot and restore global logging state mutated by the helpers."""
    pkg = logging.getLogger("grasp_agents")
    orig_handlers = list(pkg.handlers)
    orig_level = pkg.level
    orig_propagate = pkg.propagate
    orig_factory = logging.getLogRecordFactory()
    orig_installed = grasp_logging._correlation_installed
    third_party = {
        n: logging.getLogger(n).level
        for n in grasp_logging._NOISY_THIRD_PARTY
    }
    try:
        yield
    finally:
        pkg.handlers[:] = orig_handlers
        pkg.setLevel(orig_level)
        pkg.propagate = orig_propagate
        logging.setLogRecordFactory(orig_factory)
        grasp_logging._correlation_installed = orig_installed
        for name, lvl in third_party.items():
            logging.getLogger(name).setLevel(lvl)


# --- Secret redaction ---


class TestRedaction:
    def test_masks_openai_key(self) -> None:
        secret = "sk-proj-abcdefghijklmnop1234567890"
        out = redact_secrets(f"using {secret} now")
        assert secret not in out

    @pytest.mark.parametrize(
        "secret",
        [
            "sk-ant-api03-abcdefghijklmnop1234567890",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "AIzaSyA1234567890abcdefghijklmnopqrstuv",
            "hf_abcdefghijklmnopqrstuvwxyz123456",
            "xoxb-1234567890-abcdefghij",
            "gsk_abcdefghijklmnopqrstuvwxyz123456",
        ],
    )
    def test_masks_provider_tokens(self, secret: str) -> None:
        assert secret not in redact_secrets(f"token is {secret} ok")

    def test_masks_assignment_keeps_name(self) -> None:
        out = redact_secrets("MY_API_KEY=supersecretvalue123")
        assert "supersecretvalue123" not in out
        assert "MY_API_KEY" in out
        assert "<redacted>" in out

    def test_masks_bearer_header(self) -> None:
        out = redact_secrets("Authorization: Bearer abcdef1234567890xyz")
        assert "abcdef1234567890xyz" not in out

    def test_masks_jwt(self) -> None:
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NQ.SflKxwRJSMeKKF2QT4"
        assert jwt not in redact_secrets(f"jwt={jwt}")

    def test_masks_pem_block(self) -> None:
        pem = (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIEvQIBADANBgkq\n"
            "-----END PRIVATE KEY-----"
        )
        assert "MIIEvQIBADANBgkq" not in redact_secrets(pem)

    def test_leaves_ordinary_text_untouched(self) -> None:
        text = "agent 'planner' run finished: 3 turns in 4.20s"
        assert redact_secrets(text) == text

    def test_respects_disable_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(grasp_logging, "_REDACT_ENABLED", False)
        secret = "sk-proj-abcdefghijklmnop1234567890"
        assert redact_secrets(secret) == secret


class TestSecretRedactingFilter:
    def test_redacts_record_message(self) -> None:
        record = logging.LogRecord(
            name="grasp_agents.x",
            level=logging.INFO,
            pathname="f.py",
            lineno=1,
            msg="connecting with key=%s",
            args=("sk-proj-abcdefghijklmnop1234567890",),
            exc_info=None,
        )
        assert SecretRedactingFilter().filter(record) is True
        assert "sk-proj-abcdefghijklmnop1234567890" not in record.getMessage()


# --- Snippets / truncation ---


class TestSnippet:
    def test_short_text_unchanged(self) -> None:
        assert snippet("hello world") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert snippet("a\n\n  b\tc") == "a b c"

    def test_head_tail_with_omission(self) -> None:
        out = snippet("X" * 1000, head=10, tail=5)
        assert out.startswith("X" * 10)
        assert out.endswith("X" * 5)
        assert "chars]" in out

    def test_truncate_for_log(self) -> None:
        assert truncate_for_log("abc", limit=10) == "abc"
        out = truncate_for_log("a" * 100, limit=10)
        assert out.startswith("a" * 10)
        assert "+90 chars" in out

    def test_body_for_log_full_vs_snippet(self) -> None:
        long = "lorem ipsum dolor sit amet " * 50
        full = body_for_log(long, full=True)
        snip = body_for_log(long, full=False)
        assert full != snip
        assert len(snip) < len(full)


# --- Body-logging gates ---


def test_body_gates_are_bool_and_default_off() -> None:
    flags = ("LOG_LLM_INPUT", "LOG_LLM_OUTPUT", "LOG_TOOL_INPUT", "LOG_TOOL_OUTPUT")
    for flag in flags:
        assert isinstance(getattr(grasp_logging, flag), bool)


def test_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRASP_TEST_FLAG", raising=False)
    assert grasp_logging._env_flag("GRASP_TEST_FLAG") is False
    assert grasp_logging._env_flag("GRASP_TEST_FLAG", default=True) is True
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("GRASP_TEST_FLAG", truthy)
        assert grasp_logging._env_flag("GRASP_TEST_FLAG") is True
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("GRASP_TEST_FLAG", falsy)
        assert grasp_logging._env_flag("GRASP_TEST_FLAG") is False


# --- Correlation context ---


@pytest.mark.usefixtures("_restore_logging_globals")
class TestCorrelation:
    def test_stamps_fields_inside_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        install_log_correlation()
        log = logging.getLogger("grasp_agents.testcorr")
        with caplog.at_level(logging.INFO, logger="grasp_agents.testcorr"):
            with log_context(exec_id="abc123", proc="planner"):
                log.info("inside")
        rec = caplog.records[-1]
        assert rec.exec_id == "abc123"  # type: ignore[attr-defined]
        assert rec.proc == "planner"  # type: ignore[attr-defined]

    def test_defaults_outside_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        install_log_correlation()
        log = logging.getLogger("grasp_agents.testcorr2")
        with caplog.at_level(logging.INFO, logger="grasp_agents.testcorr2"):
            log.info("outside")
        rec = caplog.records[-1]
        assert rec.exec_id == "-"  # type: ignore[attr-defined]
        assert rec.proc == "-"  # type: ignore[attr-defined]

    def test_nesting_inherits_outer(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        install_log_correlation()
        log = logging.getLogger("grasp_agents.testcorr3")
        with caplog.at_level(logging.INFO, logger="grasp_agents.testcorr3"):
            with log_context(exec_id="outer", proc="runner"):
                with log_context(exec_id="inner"):
                    log.info("nested")
                log.info("restored")
        nested, restored = caplog.records[-2], caplog.records[-1]
        assert nested.exec_id == "inner"  # type: ignore[attr-defined]
        assert nested.proc == "runner"  # type: ignore[attr-defined]  # inherited
        assert restored.exec_id == "outer"  # type: ignore[attr-defined]


# --- Verbose helper ---


@pytest.mark.usefixtures("_restore_logging_globals")
class TestVerboseHelper:
    def test_idempotent_scoped_and_quiets_third_party(self) -> None:
        h1 = enable_verbose_stdout_logging(logging.DEBUG)
        h2 = enable_verbose_stdout_logging(logging.INFO)
        pkg = logging.getLogger("grasp_agents")
        verbose = [
            h for h in pkg.handlers if getattr(h, "_grasp_verbose_handler", False)
        ]
        assert len(verbose) == 1  # not added twice
        assert h1 is h2
        assert pkg.level == logging.INFO  # second call updated the level
        assert logging.getLogger("httpx").level == logging.WARNING


# --- Operational summaries (end-to-end through a tool) ---


class TestToolSummaryLogging:
    @pytest.mark.asyncio
    async def test_tool_success_logs_info_with_timing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        @function_tool
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        with caplog.at_level(logging.INFO, logger="grasp_agents.tools.base"):
            result = await add(a=2, b=3)

        assert result == 5
        messages = [r.getMessage() for r in caplog.records]
        assert any("tool add ok in" in m for m in messages)
