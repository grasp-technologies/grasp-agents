import ast
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
import textwrap
import types
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel
from pydantic_core import PydanticUndefined, to_jsonable_python

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
_MAX_DATA_DEPTH = 32
_MAX_WRAPPERS = 32
# Tag of the code-identity scheme, so hashes recorded by an older scheme are
# recognized rather than reported as code changes.
CODE_SCHEME = "v2"
# The recorded source of user code whose text cannot be read: it never
# matches, so such code always counts as possibly changed.
UNKNOWN_SOURCE = "unknown"


def code_scheme(identity: str | None) -> str | None:
    """The scheme a recorded code hash was made with (``None`` when unknown)."""
    if identity is None or identity == UNKNOWN_SOURCE:
        return None
    scheme, sep, _ = identity.partition(":")
    return scheme if sep else "v1"


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


def _is_data(value: Any, depth: int = 0) -> bool:
    if depth > _MAX_DATA_DEPTH:
        return False
    if isinstance(value, (*_PLAIN, bytes)):
        return True
    if isinstance(value, list | tuple | set | frozenset):
        return all(_is_data(v, depth + 1) for v in cast("Iterable[Any]", value))
    if isinstance(value, dict):
        mapping = cast("dict[Any, Any]", value)
        return all(
            isinstance(k, _PLAIN) and _is_data(v, depth + 1) for k, v in mapping.items()
        )
    return False


def _is_constant_name(name: str) -> bool:
    return name.lstrip("_").isupper()


def _is_frozen_record(value: Any) -> bool:
    # Immutable configuration objects: frozen pydantic models and dataclasses.
    if isinstance(value, BaseModel):
        return bool(type(value).model_config.get("frozen"))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        params = getattr(type(value), "__dataclass_params__", None)
        return bool(getattr(params, "frozen", False))
    return False


def _config_text(value: Any, name: str | None = None) -> str | None:
    """
    The text of a value that configures code: constants, compiled patterns,
    frozen records, and — under a constant's name (``RUBRIC``, ``_WORDS``) —
    lists, dicts and sets of plain data. ``None`` for anything else: other
    mutable values are state (caches, clients), not configuration.
    """
    if _is_constant(value):
        return _plain_text(value)
    if isinstance(value, re.Pattern):
        pattern = cast("re.Pattern[Any]", value)
        return f"re:{_plain_text(pattern.pattern)}:{pattern.flags}"
    if _is_frozen_record(value):
        return _plain_text(value)
    if name is not None and _is_constant_name(name) and _is_data(value):
        return _plain_text(value)
    return None


def _ast_text(node: Any) -> str:
    # Like ``ast.dump``, but without empty fields, whose presence varies
    # between Python versions.
    if isinstance(node, ast.AST):
        fields = [
            f"{name}={_ast_text(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None and not (isinstance(value, list) and not value)
        ]
        return f"{type(node).__name__}({','.join(fields)})"
    if isinstance(node, list):
        return f"[{','.join(_ast_text(v) for v in cast('list[Any]', node))}]"
    return repr(node)


def _parsed_text(source: str) -> str:
    """
    Source as parsed, without docstrings, comments or layout; the text itself
    when it does not parse on its own (e.g. a lambda inside a call).
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]
    return _ast_text(tree)


def _names(code: types.CodeType) -> set[str]:
    """Global and attribute names a code object (and code nested in it) uses."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names(const)
    return names


def _member_function(member: Any) -> types.FunctionType | None:
    if isinstance(member, staticmethod | classmethod):
        member = cast("Any", member).__func__
    if isinstance(member, property):
        member = member.fget
    return member if inspect.isfunction(member) else None


def _layers(obj: Any) -> list[Any]:
    """``obj`` and what it wraps (``__wrapped__``), outermost first."""
    layers = [obj]
    seen = {id(obj)}
    while len(layers) < _MAX_WRAPPERS:
        inner = getattr(layers[-1], "__wrapped__", None)
        if inner is None or inspect.isclass(layers[-1]) or id(inner) in seen:
            break
        seen.add(id(inner))
        layers.append(inner)
    return layers


def _is_user_code(obj: Any) -> bool:
    if isinstance(obj, functools.partial) or inspect.ismethod(obj):
        return True  # judged by what they bind
    return any(not is_library_code(layer) for layer in _layers(obj))


class _CodeText:
    """
    The text a code hash is made of. Starting from one object, it collects —
    breadth first, a few levels deep — every user-code function, class and
    object it reaches, each with its parsed source and what it is bound
    with: defaults, closures, class attributes, base classes, decorators,
    and the globals and module members its code names. Configuration values
    (see :func:`_config_text`) count by value; library code (grasp-agents,
    installed packages, the standard library) by name only, so upgrading a
    dependency does not change the hash.
    """

    def __init__(self) -> None:
        self._index: dict[int, int] = {}
        self._nodes: list[tuple[Any, int]] = []

    def of(self, root: Any) -> str:
        self._add(root, 0)
        texts: list[str] = []
        position = 0
        while position < len(self._nodes):
            obj, depth = self._nodes[position]
            if position == 0:
                texts.append(self._text(obj, depth))
            else:
                try:
                    texts.append(self._text(obj, depth))
                except (OSError, TypeError):
                    texts.append(f"?{qualified_name(obj)}")
            position += 1
        return "\x00".join(texts)

    def _add(self, obj: Any, depth: int) -> str:
        index = self._index.get(id(obj))
        if index is None:
            index = len(self._nodes)
            self._index[id(obj)] = index
            self._nodes.append((obj, depth))
        return f"@{index}"

    def _ref(self, value: Any, depth: int, name: str | None = None) -> str:
        """How code at ``depth`` refers to ``value``."""
        config = _config_text(value, name)
        if config is not None:
            return config
        if inspect.ismodule(value):
            return f"module:{value.__name__}"
        is_code = callable(value) or inspect.isclass(value)
        if not _is_user_code(value):
            return qualified_name(value if is_code else type(value))
        if id(value) in self._index:
            return f"@{self._index[id(value)]}"
        if depth + 1 > _MAX_CODE_DEPTH:
            return qualified_name(value if is_code else type(value))
        return self._add(value, depth + 1)

    def _text(self, obj: Any, depth: int) -> str:
        if isinstance(obj, functools.partial):
            call = cast("functools.partial[Any]", obj)
            return "\x00".join(
                [
                    f"partial:{self._ref(call.func, depth)}",
                    *(self._ref(a, depth) for a in call.args),
                    *(f"{k}={self._ref(v, depth)}" for k, v in call.keywords.items()),
                ]
            )
        if inspect.ismethod(obj):
            return (
                f"method:{self._ref(obj.__func__, depth)}:"
                f"{self._ref(obj.__self__, depth)}"
            )
        layers = _layers(obj)
        if len(layers) > 1:
            # A decorated callable: each user-code layer and what it wraps.
            parts = [
                self._function(layer, depth)
                if inspect.isfunction(layer) and not is_library_code(layer)
                else qualified_name(layer if inspect.isfunction(layer) else type(layer))
                for layer in layers[:-1]
            ]
            return "\x00".join([*parts, f"wraps:{self._ref(layers[-1], depth)}"])
        if inspect.isfunction(obj):
            return self._function(obj, depth)
        if inspect.isclass(obj):
            return self._class(cast("type[Any]", obj), depth)
        return self._instance(obj, depth)

    def _function(self, fn: types.FunctionType, depth: int) -> str:
        # The function's own code: ``inspect.getsource(fn)`` would read what
        # it wraps instead.
        source = _parsed_text(inspect.getsource(fn.__code__))
        return "\x00".join([source, *self._bindings(fn, depth)])

    def _bindings(self, fn: types.FunctionType, depth: int) -> list[str]:
        parts = [self._ref(v, depth) for v in fn.__defaults__ or ()]
        keyword_defaults: dict[str, Any] = fn.__kwdefaults__ or {}
        parts.extend(f"{k}={self._ref(v, depth)}" for k, v in keyword_defaults.items())
        for cell in fn.__closure__ or ():
            try:
                parts.append(self._ref(cell.cell_contents, depth))
            except ValueError:  # an empty cell
                parts.append("<empty>")
        parts.extend(self._references(fn.__code__, fn.__globals__, depth))
        return parts

    def _class(self, cls: type[Any], depth: int) -> str:
        parts = [_parsed_text(inspect.getsource(cls))]
        parts.extend(
            f"base:{self._ref(base, depth)}"
            for base in cls.__mro__[1:]
            if not is_library_code(base)
        )
        for name, member in vars(cls).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            if isinstance(member, cls):
                continue  # an enum member, defined in the source
            function = _member_function(member)
            target = member if function is None else function
            if (
                callable(target)
                and not inspect.isclass(target)
                and len(_layers(target)) > 1
            ):
                # A decorated method: the decorator's code is not in the class.
                parts.append(f"{name}={self._ref(target, depth)}")
                continue
            if function is not None:
                parts.extend(
                    f"{name}:{part}" for part in self._bindings(function, depth)
                )
                continue
            config = _config_text(member, name)
            if config is not None:
                parts.append(f"{name}={config}")
            elif _is_user_code(member):
                parts.append(f"{name}={self._ref(member, depth)}")
        if issubclass(cls, BaseModel):
            for name, field in cls.model_fields.items():
                if field.default is not PydanticUndefined:
                    parts.append(f"{name}={self._ref(field.default, depth)}")
        return "\x00".join(parts)

    def _instance(self, obj: Any, depth: int) -> str:
        # An object: its class's code and its public settings.
        parts = [f"instance:{self._ref(type(obj), depth)}"]
        attributes: dict[str, Any] = getattr(obj, "__dict__", {})
        for name, value in attributes.items():
            if name.startswith("_"):
                continue
            config = _config_text(value)
            if config is not None:
                parts.append(f"{name}={config}")
            elif (callable(value) or inspect.isclass(value)) and _is_user_code(value):
                parts.append(f"{name}={self._ref(value, depth)}")
        return "\x00".join(parts)

    def _references(
        self, code: types.CodeType, scope: dict[str, Any], depth: int
    ) -> list[str]:
        names = _names(code)
        found: list[str] = []
        for name in sorted(names):
            if name not in scope:
                continue
            value = scope[name]
            if inspect.ismodule(value):
                found.extend(self._module_members(name, value, names, depth, set()))
            else:
                found.append(f"{name}={self._ref(value, depth, name)}")
        return found

    def _module_members(
        self,
        path: str,
        module: types.ModuleType,
        names: set[str],
        depth: int,
        visited: set[str],
    ) -> list[str]:
        # ``module.helper(...)`` and ``pkg.sub.helper(...)``: the members of a
        # user module that the code names.
        if _is_library_module(module) or module.__name__ in visited:
            return []
        visited.add(module.__name__)
        found: list[str] = []
        for attribute in sorted(names):
            member = getattr(module, attribute, None)
            if member is None:
                continue
            if inspect.ismodule(member):
                found.extend(
                    self._module_members(
                        f"{path}.{attribute}", member, names, depth, visited
                    )
                )
            else:
                text = self._ref(member, depth, attribute)
                found.append(f"{path}.{attribute}={text}")
        return found


def _code_text(obj: Any) -> str:
    return _CodeText().of(obj)


def _tagged(digest: str) -> str:
    return f"{CODE_SCHEME}:{digest}"


def code_hash(obj: Any) -> str | None:
    """
    Hash of the code of a function, class, method, ``functools.partial`` or
    callable object: its source without docstrings, comments or layout, and
    — followed a few levels deep — the user code it reaches: functions and
    classes it calls or names (through modules too), base classes,
    decorators, and the values it is bound with (defaults, closures, class
    and public instance attributes, module globals). Constants, compiled
    patterns and frozen records count by value, as do lists, dicts and sets
    of plain data under a constant's name (``RUBRIC``); other mutable values
    count by type only. ``None`` when its source is unavailable.
    """
    try:
        return _tagged(short_hash(_code_text(obj)))
    except (OSError, TypeError):
        return None


def source_identity(obj: Any) -> str | None:
    """
    What a component records as its ``source``: :func:`code_hash` of user
    code, ``None`` for library code (its name and version identify it), and
    :data:`UNKNOWN_SOURCE` when user code has no readable source.
    """
    if not _is_user_code(obj):
        return None
    return code_hash(obj) or UNKNOWN_SOURCE


def identity_hash(parts: Iterable[Any]) -> str | None:
    """
    Hash of declared identity parts: a string as text, a :class:`Path` by
    the contents of its file (or of every file under a directory, caches and
    hidden files aside), a function or class by its code (see
    :func:`code_hash`; library code by name), other values as canonical
    JSON. ``None`` when there are none.
    """
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(f"text:{part}")
        elif isinstance(part, Path):
            texts.append(f"path:{_path_digest(part)}")
        elif callable(part) and not _is_user_code(part):
            texts.append(f"name:{qualified_name(part)}")
        elif callable(part):
            try:
                texts.append(f"code:{_code_text(part)}")
            except (OSError, TypeError) as exc:
                raise ValueError(
                    f"Cannot hash identity part {part!r}: its source is "
                    "unavailable. Declare a string or a file instead."
                ) from exc
        else:
            texts.append(f"value:{canonical_json(part)}")
    return _tagged(short_hash(*texts)) if texts else None


def _path_digest(path: Path) -> str:
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and not _is_cache_or_hidden(p.relative_to(path))
        )
        return short_hash(
            *(f"{p.relative_to(path).as_posix()}:{_file_digest(p)}" for p in files)
        )
    if not path.exists():
        raise FileNotFoundError(f"Identity file {path} does not exist")
    return _file_digest(path)


def _is_cache_or_hidden(relative: Path) -> bool:
    return relative.suffix == ".pyc" or any(
        part == "__pycache__" or part.startswith(".") for part in relative.parts
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def _library_roots() -> tuple[Path, ...]:
    paths = sysconfig.get_paths()
    roots = {paths[key] for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    return tuple(Path(root).resolve() for root in roots if root)


@functools.cache
def _grasp_agents_dirs() -> tuple[Path, Path]:
    package = Path(__file__).resolve().parents[1]
    return package, package / "examples"


def _library_file(path: str | None) -> bool:
    if path is None:
        return True  # built-in or namespace package
    resolved = Path(path).resolve()
    package, examples = _grasp_agents_dirs()
    if resolved.is_relative_to(package):
        return not resolved.is_relative_to(examples)
    return any(resolved.is_relative_to(root) for root in _library_roots())


def _grasp_agents_library(module: str) -> bool:
    return module.partition(".")[0] == "grasp_agents" and not module.startswith(
        "grasp_agents.examples"
    )


def _is_library_module(module: types.ModuleType) -> bool:
    if module.__name__.partition(".")[0] == "grasp_agents":
        return _grasp_agents_library(module.__name__)
    return _library_file(getattr(module, "__file__", None))


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
        return _grasp_agents_library(module)
    try:
        path = inspect.getsourcefile(target)
    except TypeError:
        return True  # built-in
    if path is not None:
        return _library_file(path)
    try:
        filename = inspect.getfile(target)
    except TypeError:
        return True
    if filename.startswith("<"):
        # Compiled from a string or typed at a prompt (user code), unless
        # frozen into the interpreter.
        return filename.startswith("<frozen")
    return _library_file(filename)  # a compiled extension


def user_code_hash(objects: Iterable[Any]) -> str | None:
    """
    Combined :func:`code_hash` of ``objects`` that are user code — not
    grasp-agents, installed packages or the standard library (``None`` when
    there are none, :data:`UNKNOWN_SOURCE` when one has no readable source)
    — so upgrading a dependency does not change it.
    """
    identities: dict[str, None] = {}
    for obj in objects:
        if obj is None or not _is_user_code(obj):
            continue
        found = code_hash(obj)
        if found is None:
            return UNKNOWN_SOURCE
        identities.setdefault(found, None)
    return _tagged(short_hash(*identities)) if identities else None


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
