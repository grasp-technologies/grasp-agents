import base64
import dataclasses
import functools
import hashlib
import inspect
import json
import math
import re
import secrets
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sysconfig
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


_PLAIN = (str, int, float, bool, type(None))
_MAX_CODE_DEPTH = 4


def _plain_text(value: Any) -> str:
    try:
        return canonical_json(value)
    except TypeError:
        return qualified_name(type(value))


def _is_constant(value: Any) -> bool:
    if isinstance(value, _PLAIN):
        return True
    if isinstance(value, tuple | frozenset):
        return all(_is_constant(v) for v in cast("Iterable[Any]", value))
    return False


def _value_text(value: Any, depth: int) -> str:
    # A value a function closes over, defaults to or is bound with: code for
    # callables, constants as they are. Other objects (lists, caches,
    # clients) are state, not configuration: only their type counts.
    if callable(value) and not isinstance(value, type):
        return _code_text(value, depth + 1)
    if _is_constant(value):
        return _plain_text(value)
    return qualified_name(type(value))


def _function_text(fn: Any, depth: int) -> str:
    parts = [inspect.getsource(fn)]
    parts.extend(_value_text(v, depth) for v in fn.__defaults__ or ())
    keyword_defaults = cast("dict[str, Any]", fn.__kwdefaults__ or {})
    parts.extend(f"{k}={_value_text(v, depth)}" for k, v in keyword_defaults.items())
    for cell in fn.__closure__ or ():
        try:
            parts.append(_value_text(cell.cell_contents, depth))
        except ValueError:  # an empty cell
            parts.append("<empty>")
    return "\x00".join(parts)


def _code_text(obj: Any, depth: int = 0) -> str:
    if depth > _MAX_CODE_DEPTH:
        return qualified_name(obj)
    if isinstance(obj, functools.partial):
        call = cast("functools.partial[Any]", obj)
        return "\x00".join(
            [
                _code_text(call.func, depth + 1),
                *(_value_text(a, depth) for a in call.args),
                *(f"{k}={_value_text(v, depth)}" for k, v in call.keywords.items()),
            ]
        )
    if inspect.ismethod(obj):
        return _code_text(obj.__func__, depth + 1)
    target = inspect.unwrap(obj)
    if inspect.isfunction(target):
        return _function_text(target, depth)
    if inspect.isclass(target):
        return inspect.getsource(target)
    # A callable object: its class's code and its constant public attributes.
    attributes = cast("dict[str, Any]", getattr(target, "__dict__", {}))
    settings = {
        k: v for k, v in attributes.items() if not k.startswith("_") and _is_constant(v)
    }
    cls = cast("type[Any]", type(target))
    return inspect.getsource(cls) + _plain_text(settings)


def code_hash(obj: Any) -> str | None:
    """
    Hash of the code of a function, class, method, ``functools.partial`` or
    callable object, including the constants (numbers, strings, tuples of
    them) it closes over, defaults to or is bound with, and the code of the
    functions among them (``None`` when its source is unavailable).
    """
    try:
        return short_hash(_code_text(obj))
    except (OSError, TypeError):
        return None


def _library_roots() -> tuple[Path, ...]:
    paths = sysconfig.get_paths()
    roots = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    return tuple(Path(root).resolve() for root in roots if root)


def is_library_code(obj: Any) -> bool:
    """
    Whether ``obj`` (a function, class or instance) comes from grasp-agents
    (other than its examples), an installed package or the standard library.
    """
    target: Any = obj
    if isinstance(obj, functools.partial):
        target = cast("functools.partial[Any]", obj).func
    if inspect.ismethod(target):
        target = target.__func__
    if not (inspect.isfunction(target) or inspect.isclass(target)):
        target = cast("type[Any]", type(target))
    module: str = getattr(target, "__module__", None) or ""
    if module.partition(".")[0] == "grasp_agents":
        return not module.startswith("grasp_agents.examples")
    try:
        path = inspect.getsourcefile(target)
    except TypeError:
        return True  # built-in
    if path is None:
        return True
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(root) for root in _library_roots())


def code_identity(obj: Any) -> str:
    """:func:`code_hash`, or a stable description when there is no source."""
    found = code_hash(obj)
    if found is not None:
        return found
    text = repr(obj)
    return text if " at 0x" not in text else qualified_name(obj)


def user_code_hash(objects: Iterable[Any]) -> str | None:
    """
    Combined :func:`code_identity` of ``objects`` that are user code — not
    grasp-agents, installed packages or the standard library (``None`` when
    there are none) — so upgrading a dependency does not change it.
    """
    identities = dict.fromkeys(
        code_identity(obj)
        for obj in objects
        if obj is not None and not is_library_code(obj)
    )
    return short_hash(*identities) if identities else None


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
            proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
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
