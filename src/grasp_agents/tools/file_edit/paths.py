"""
Path resolution, ``allowed_roots`` enforcement, and sensitive-path
deny list for the file-edit toolkit.

Three layers of path policy, checked in this order by the Read/Write/
Edit tools:

1. :func:`is_blocked_device` — literal-path check that rejects paths
   which would hang the reader process (``/dev/zero``, ``/dev/stdin``,
   etc.). Uses literal strings because ``Path.resolve`` follows symlinks
   all the way through (e.g. ``/dev/stdin`` → ``/proc/self/fd/0`` →
   ``/dev/pts/N``) and defeats the check.

2. :func:`resolve_safe` — expands ``~``, resolves symlinks, and enforces
   that the result lies under at least one ``allowed_roots`` entry.
   Default toolkit config uses ``[Path.cwd()]`` as a single-root
   sandbox; consumers opt in to more roots explicitly.

3. :func:`check_sensitive_path` — deny list layered on top of the root
   check. Two tiers:

   - **System paths** (always denied): ``/etc/``, ``/boot/``,
     ``/usr/lib/systemd/``, ``/private/etc/``, selected ``/private/var/``
     subpaths, Docker sockets.

   - **User-dotfile additions** (default on, configurable): ``.ssh/``,
     ``.aws/``, ``.gnupg/``, ``.kube/``, ``.docker/``, ``.env`` (and
     ``.env.*``), ``.netrc``, ``.pypirc``. Covers the credential surfaces
     an agent routinely encounters under ``~``. A per-session
     ``dotfile_overrides`` set carries explicit opt-ins.

:func:`has_binary_extension` is a cheap pre-read guard that refuses
known-binary file extensions (images, archives, executables). Returning
binary blobs to the model wastes tokens and can crash the serializer.
"""

from __future__ import annotations

from pathlib import Path


class PathAccessError(ValueError):
    """
    Raised when a path cannot be accessed under the toolkit's policy.

    Inherits ``ValueError`` (not a dedicated class) so Pydantic/pyright
    see it as a validation-shaped exception; the file-edit tools convert
    this to a structured tool error at the boundary.
    """


# ---------------------------------------------------------------------------
# System-path deny list
# ---------------------------------------------------------------------------

_SYSTEM_SENSITIVE_PREFIXES: tuple[str, ...] = (
    "/etc/",
    "/boot/",
    "/usr/lib/systemd/",
    "/private/etc/",
    # macOS-specific /private/var/* subpaths. Deliberately narrower than
    # the whole ``/private/var/`` tree because that also contains the
    # per-user TMPDIR (``/private/var/folders/``) and world-temp
    # (``/private/var/tmp/``), which are legitimate write targets.
    "/private/var/db/",
    "/private/var/log/",
    "/private/var/root/",
)

_SYSTEM_SENSITIVE_EXACT: frozenset[str] = frozenset(
    {
        "/var/run/docker.sock",
        "/run/docker.sock",
    }
)


# ---------------------------------------------------------------------------
# User-dotfile deny list — grasp-agents default-on
# ---------------------------------------------------------------------------

# Filenames that match exactly or with a suffix (``.env`` matches
# ``.env``, ``.env.local``, ``.env.prod`` via the startswith-plus-dot
# rule in check_sensitive_path).
_DOTFILE_DENY_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".netrc",
        ".pypirc",
    }
)

# Directory names: any path with a component equal to one of these is
# denied (so ``~/.ssh/config``, ``/opt/app/.aws/credentials`` etc. all
# hit). Deliberately narrower than "any dotfile dir" — ``.config`` for
# instance hosts routine editable content (``.config/fish/``,
# ``.config/nvim/``) and is not in the list.
_DOTFILE_DENY_DIRS: frozenset[str] = frozenset(
    {
        ".ssh",
        ".aws",
        ".gnupg",
        ".kube",
        ".docker",
    }
)


# ---------------------------------------------------------------------------
# Device-path blocklist
# ---------------------------------------------------------------------------

_BLOCKED_DEVICE_PATHS: frozenset[str] = frozenset(
    {
        # Infinite output — never reach EOF.
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/full",
        # Blocks waiting for input.
        "/dev/stdin",
        "/dev/tty",
        "/dev/console",
        # Nonsensical to read.
        "/dev/stdout",
        "/dev/stderr",
        # fd aliases
        "/dev/fd/0",
        "/dev/fd/1",
        "/dev/fd/2",
    }
)


# ---------------------------------------------------------------------------
# Binary-extension guard
# ---------------------------------------------------------------------------

# Conservative set covering the obvious binary families. Consumers who
# want to read/write more exotic binary formats can extend via toolkit
# config (exposed when FileEditToolkit lands in Step 2).
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".ico",
        # Audio / video
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        # Archives
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Compiled / binary executables
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".o",
        ".a",
        ".class",
        ".pyc",
        ".pyo",
        # Fonts / documents
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".pdf",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_safe(
    path: str | Path,
    allowed_roots: list[Path],
    *,
    must_exist: bool = False,
) -> Path:
    """
    Resolve ``path`` and confirm it lies under at least one allowed root.

    Args:
        path: Path as provided by the caller / model. Can contain ``~``
            and be relative; will be expanded and resolved.
        allowed_roots: List of root directories the path must be under.
            Each entry is ``expanduser``-expanded and resolved before
            comparison.
        must_exist: If True, the path itself must exist (uses
            ``resolve(strict=True)``). Use True for ``Read`` and ``Edit``
            of existing files; False for ``Write`` creating a new file.

    Returns:
        Resolved absolute path.

    Raises:
        PathAccessError: If resolution fails or the path is outside all
            allowed roots.

    """
    if not allowed_roots:
        raise PathAccessError("No allowed_roots configured for file-edit toolkit.")

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise PathAccessError(f"Path does not exist: {path}") from exc
    except OSError as exc:
        raise PathAccessError(f"Cannot resolve path {path!r}: {exc}") from exc

    for root in allowed_roots:
        root_resolved = root.expanduser().resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue
        else:
            return resolved

    root_strs = ", ".join(str(r) for r in allowed_roots)
    raise PathAccessError(f"Path {resolved} is outside allowed roots [{root_strs}]")


def is_blocked_device(path: str | Path) -> bool:
    """
    Return True for device/pseudo-fs paths that would hang a reader.

    Uses the *literal* path without ``Path.resolve()``: symlink
    following through ``/proc/self/fd/*`` would escape the check.
    """
    s = str(Path(path).expanduser())
    if s in _BLOCKED_DEVICE_PATHS:
        return True
    # Linux stdio aliases under /proc/.../fd/{0,1,2}.
    return s.startswith("/proc/") and s.endswith(("/fd/0", "/fd/1", "/fd/2"))


def has_binary_extension(path: str | Path) -> bool:
    """Return True if the path has an extension classified as binary."""
    return Path(path).suffix.lower() in _BINARY_EXTENSIONS


def check_sensitive_path(
    resolved_path: Path,
    *,
    include_dotfiles: bool = True,
    session_overrides: set[Path] | None = None,
) -> str | None:
    """
    Return an error message if ``resolved_path`` targets a sensitive
    location, or None if it's safe to write to.

    Args:
        resolved_path: An already-resolved ``Path`` (post
            :func:`resolve_safe`).
        include_dotfiles: Stack the user-dotfile additions on top of the
            system-path baseline. Default ``True``.
        session_overrides: Paths the user has explicitly allowed for
            this session; bypasses the dotfile deny list. System-path
            entries are *not* overridable — a session-scoped opt-in is
            not the right level of authorization for ``/etc/sudoers``.

    """
    s = str(resolved_path)

    # System baseline — not overridable per session.
    for prefix in _SYSTEM_SENSITIVE_PREFIXES:
        if s.startswith(prefix):
            return f"Refusing to write to sensitive system path: {resolved_path}"
    if s in _SYSTEM_SENSITIVE_EXACT:
        return f"Refusing to write to sensitive system path: {resolved_path}"

    if not include_dotfiles:
        return None

    if session_overrides and resolved_path in session_overrides:
        return None

    # Dotfile additions — session-overridable.
    name = resolved_path.name
    for pattern in _DOTFILE_DENY_NAMES:
        if name == pattern or name.startswith(pattern + "."):
            return (
                f"Refusing to write to credential-like file: {resolved_path}. "
                f"Add to session overrides if intentional."
            )

    for part in resolved_path.parts:
        if part in _DOTFILE_DENY_DIRS:
            return (
                f"Refusing to write inside credential-sensitive directory "
                f"{part!r}: {resolved_path}."
            )

    return None
