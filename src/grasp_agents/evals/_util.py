import base64
import dataclasses
import hashlib
import inspect
import json
import math
import re
import secrets
import shutil
import subprocess  # noqa: S404
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Prefix of the module names given to spec files imported by path.
SPEC_MODULE_PREFIX = "_grasp_evals_"


def module_label(module: str) -> str:
    """
    A module's name as recorded in run identities. Spec files imported by
    path are named by their stem, so the same file yields the same name
    wherever it is checked out.
    """
    if module.startswith(SPEC_MODULE_PREFIX):
        return module.removeprefix(SPEC_MODULE_PREFIX).rsplit("_", 1)[0]
    return module


def qualified_name(obj: Any) -> str:
    """``module.QualName`` of a class or function, without generic parameters."""
    qualname = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    module = getattr(obj, "__module__", None) or type(obj).__module__
    return f"{module_label(module)}.{qualname.split('[', 1)[0]}"


def to_jsonable(obj: Any) -> Any:
    """
    JSON-compatible form of ``obj`` for storage: bytes become base64,
    non-finite floats ``null``, unserializable leaves their repr.
    """
    return to_jsonable_python(
        obj, fallback=repr, inf_nan_mode="null", bytes_mode="base64"
    )


def _dumps(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _canonical(obj: Any) -> Any:
    if obj is None or isinstance(obj, str | bool | int):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return {"$float": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, BaseModel):
        return _canonical(obj.model_dump(mode="python"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canonical(
            {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
        )
    if isinstance(obj, Mapping):
        mapping = cast("Mapping[Any, Any]", obj)
        if all(isinstance(k, str) for k in mapping):
            return {str(k): _canonical(v) for k, v in mapping.items()}
        pairs = [[_canonical(k), _canonical(v)] for k, v in mapping.items()]
        return {"$map": sorted(pairs, key=_dumps)}
    if isinstance(obj, set | frozenset):
        members = cast("Iterable[Any]", obj)
        return {"$set": sorted((_canonical(v) for v in members), key=_dumps)}
    if isinstance(obj, list | tuple):
        return [_canonical(v) for v in cast("Iterable[Any]", obj)]
    if isinstance(obj, bytes | bytearray):
        return {"$bytes": base64.b64encode(bytes(obj)).decode("ascii")}
    try:
        return to_jsonable_python(obj)
    except Exception as exc:
        raise TypeError(
            f"Cannot hash a {type(obj).__name__} value: it is not JSON-serializable"
        ) from exc


def canonical_json(obj: Any) -> str:
    """
    Deterministic JSON for hashing: sorted keys and set members, non-finite
    floats and bytes kept distinct from other values. Raises ``TypeError``
    for values with no stable serialization.
    """
    return _dumps(_canonical(obj))


def short_hash(*parts: str, length: int = 12) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:length]


def code_hash(obj: Any) -> str | None:
    """Hash of a function's or class's source code (``None`` when unavailable)."""
    try:
        return short_hash(inspect.getsource(inspect.unwrap(obj)))
    except (OSError, TypeError):
        return None


def source_hash(objects: Iterable[Any]) -> str | None:
    """
    Hash of the source files defining ``objects`` (classes, functions or
    instances), including files git does not track.
    """
    digests: set[str] = set()
    for obj in objects:
        target: Any = inspect.unwrap(obj) if callable(obj) else obj
        if not (inspect.isclass(target) or inspect.isroutine(target)):
            target = cast("Any", type(target))
        try:
            path = inspect.getsourcefile(target)
        except TypeError:
            continue
        if path is None:
            continue
        try:
            digests.add(hashlib.sha256(Path(path).read_bytes()).hexdigest())
        except OSError:
            continue
    return short_hash(*sorted(digests)) if digests else None


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
