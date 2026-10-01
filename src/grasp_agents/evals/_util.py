import hashlib
import json
import re
import secrets
import shutil
import subprocess  # noqa: S404
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic_core import to_jsonable_python

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def qualified_name(obj: Any) -> str:
    """``module.QualName`` of a class or function, without generic parameters."""
    qualname = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return f"{obj.__module__}.{qualname.split('[', 1)[0]}"


def to_jsonable(obj: Any) -> Any:
    """JSON-compatible form of ``obj``; unserializable leaves become their repr."""
    return to_jsonable_python(obj, fallback=repr, inf_nan_mode="null")


def canonical_json(obj: Any) -> str:
    return json.dumps(
        to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def short_hash(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:length]


def utc_now() -> datetime:
    return datetime.now(UTC)


def slugify(text: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_len].strip("-") or "run"


def new_run_id(name: str, now: datetime | None = None) -> str:
    stamp = (now or utc_now()).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{slugify(name)}-{secrets.token_hex(2)}"


def git_state(
    cwd: Path | None = None,
) -> tuple[str | None, str | None, bool | None, str | None]:
    """
    ``(commit, branch, dirty, diff_hash)`` of the repository containing
    ``cwd``; ``diff_hash`` hashes the uncommitted diff of tracked files.
    """
    git = shutil.which("git")
    if git is None:
        return None, None, None, None

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(  # noqa: S603
                [git, *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    if commit is None:
        return None, None, None, None
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=no")
    dirty = bool(status) if status is not None else None
    diff = run("diff", "HEAD") if dirty else ""
    diff_hash = None if diff is None else (short_hash(diff) if diff else None)
    return commit, branch, dirty, diff_hash
