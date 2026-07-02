"""
File / exec tool behavior: CRLF-preserving edits, output-overflow guards,
secret redaction, write-conflict key detection, and exec-tool write
exclusivity.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import nbformat
import pytest
from nbformat import v4

from grasp_agents.file_backend import LocalFileBackend
from grasp_agents.session_context import SessionContext
from grasp_agents.tools.base import _keys_overlap, batch_has_concurrency_conflict
from grasp_agents.tools.bash import Bash
from grasp_agents.tools.bash_session import BashSession
from grasp_agents.tools.code_interpreter import RunPython
from grasp_agents.tools.file_edit.fuzzy_match import fuzzy_find_and_replace
from grasp_agents.tools.file_edit.notebook import (
    NotebookEditInput,
    NotebookEditTool,
    NotebookReadInput,
    NotebookReadResult,
    NotebookReadTool,
)
from grasp_agents.tools.file_search.grep import GrepError, _run_rg
from grasp_agents.tools.notebook_exec import RunCell
from grasp_agents.utils.io import read_contents_from_file

# ---------- Item 32: CRLF files keep their line endings ----------


class TestCrlfPreserved:
    def test_crlf_region_keeps_crlf(self) -> None:
        content = "line one\r\nline two\r\nline three\r\n"
        new_content, count, _, error = fuzzy_find_and_replace(
            content, "line two", "line 2a\nline 2b"
        )
        assert error is None
        assert count == 1
        assert new_content == "line one\r\nline 2a\r\nline 2b\r\nline three\r\n"
        assert "\n" not in new_content.replace("\r\n", "")

    def test_multiline_old_string_with_lf(self) -> None:
        content = "a\r\nb\r\nc\r\n"
        # The model types LF in old_string; line-trimmed strategy matches.
        new_content, count, _, error = fuzzy_find_and_replace(content, "a\nb", "x\ny")
        assert error is None
        assert count == 1
        assert new_content == "x\r\ny\r\nc\r\n"

    def test_lf_file_untouched(self) -> None:
        content = "one\ntwo\nthree\n"
        new_content, _, _, error = fuzzy_find_and_replace(content, "two", "2a\n2b")
        assert error is None
        assert new_content == "one\n2a\n2b\nthree\n"


# ---------- Item 33: grep output capped before buffering ----------


@pytest.mark.skipif(shutil.which("rg") is None, reason="requires ripgrep")
class TestGrepOutputCap:
    @pytest.mark.asyncio
    async def test_overflow_raises_and_kills(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import grasp_agents.tools.file_search.grep as grep_mod

        big = tmp_path / "big.txt"
        big.write_text("needle\n" * 50_000)
        monkeypatch.setattr(grep_mod, "MAX_STDOUT_BYTES", 1_000)

        with pytest.raises(GrepError, match="more than"):
            await _run_rg(["--line-number", "needle", str(big)])

    @pytest.mark.asyncio
    async def test_small_output_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("needle\n")
        stdout, _, code = await _run_rg(["--line-number", "needle", str(f)])
        assert code == 0
        assert b"needle" in stdout


# ---------- Item 34: NotebookRead redacts secrets ----------


class TestNotebookReadRedaction:
    @pytest.mark.asyncio
    async def test_secrets_in_source_redacted(self, tmp_path: Path) -> None:
        nb = v4.new_notebook()
        nb.cells.append(
            v4.new_code_cell(
                'OPENAI_API_KEY = "sk-proj-abcdef1234567890abcdef1234567890abcd"'
            )
        )
        nb_path = tmp_path / "x.ipynb"
        nb_path.write_text(nbformat.writes(nb))

        ctx: SessionContext[Any] = SessionContext(
            file_backend=LocalFileBackend(allowed_roots=[tmp_path])
        )
        result = await NotebookReadTool()._run(
            NotebookReadInput(path=str(nb_path)), ctx=ctx
        )
        assert isinstance(result, NotebookReadResult)
        assert "sk-proj-abcdef" not in result.cells[0].source
        assert "REDACTED" in result.cells[0].source


# ---------- Item 35: conflict-key normalization + declarations ----------


class TestConflictKeys:
    def test_dotdot_normalized(self) -> None:
        assert _keys_overlap("/ws/a/../x.md", "/ws/x.md")

    def test_rel_vs_abs_tail_match(self) -> None:
        assert _keys_overlap("notes/x.md", "/workspace/notes/x.md")
        assert not _keys_overlap("other/y.md", "/workspace/notes/x.md")

    def test_root_key_conflicts_with_everything(self) -> None:
        assert _keys_overlap("/", "/anything/at/all")
        assert _keys_overlap("relative/path.md", "/")

    def test_exec_tools_declare_global_exclusivity(self) -> None:
        from grasp_agents.tools.bash_common import BashInput
        from grasp_agents.tools.code_interpreter import (
            RunPythonInput,
        )
        from grasp_agents.tools.notebook_exec import RunCellInput

        bash_inp = BashInput(command="echo hi")
        assert Bash().concurrency_conflict_keys(bash_inp) == ["/"]
        assert BashSession().concurrency_conflict_keys(bash_inp) == ["/"]
        assert RunPython().concurrency_conflict_keys(
            RunPythonInput(code="print(1)")
        ) == ["/"]
        assert RunCell().concurrency_conflict_keys(
            RunCellInput(notebook_path="x.ipynb", cell_id="c1")
        ) == ["/"]

    def test_notebook_edit_declares_target(self) -> None:
        inp = NotebookEditInput(
            notebook_path="/ws/x.ipynb", cell_id="c1", new_source="pass"
        )
        assert NotebookEditTool().concurrency_conflict_keys(inp) == ["/ws/x.ipynb"]

    def test_bash_conflicts_with_file_writer_batch(self) -> None:
        from grasp_agents.tools.bash_common import BashInput
        from grasp_agents.tools.file_edit.write import (
            WriteInput,
            WriteTool,
        )

        bash = Bash()
        write = WriteTool()
        assert batch_has_concurrency_conflict(
            [
                (bash, BashInput(command="make build")),
                (write, WriteInput(path="/ws/out.txt", content="x")),
            ]
        )


# ---------- Item 36: prompt file errors + truncation tail ----------


class TestPromptFileMissing:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_contents_from_file(tmp_path / "nope.md")
