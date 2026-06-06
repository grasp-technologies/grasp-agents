"""
Atomic file writes via tmpfile + ``os.replace``.

Pattern (standard POSIX "write, rename" dance):

1. Create a temporary file in the target's parent directory via
   :func:`tempfile.mkstemp`. Same-directory placement guarantees the
   rename stays on one filesystem, so ``os.replace`` is atomic.
2. Write the payload to the temp file and ``fsync`` it.
3. ``os.replace`` the temp file onto the target. Atomic on POSIX and on
   Windows (Python 3.3+).

On any failure the temp file is cleaned up so partial / torn files never
appear at the target location.

Security notes:

- ``mkstemp`` creates with ``O_CREAT | O_EXCL | O_RDWR`` and mode ``0o600``.
  Callers who care about mode (beyond 0o600 which we want most of the time)
  can pass it in; we ``chmod`` after creation.
- Caller is responsible for having already resolved the target path under
  a trusted root — this helper does not re-check sandbox policy. The
  file-edit tools feed it paths that have already passed :func:`resolve_safe`.

This helper is intentionally sync. Async wrappers (``asyncio.to_thread``)
live at the tool-call boundary so the helper stays testable without an
event loop and is directly reusable by the upcoming
``FileCheckpointStore`` in B2 (which is sync-only on its persistence path).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    overwrite: bool = True,
    fsync: bool = True,
) -> None:
    """
    Write ``data`` to ``path`` atomically.

    Args:
        path: Final destination. Parent directory must exist.
        data: Payload bytes.
        mode: File permissions applied before rename. Default ``0o600``.
        overwrite: If False, raise :class:`FileExistsError` when the
            target already exists. If True (default), replace atomically.
        fsync: If True (default), ``fsync`` the temp file before rename.
            Set False to skip the durability flush in performance-sensitive
            cases where a crash-window reorder is acceptable.

    Raises:
        FileExistsError: When ``overwrite=False`` and the target exists.
        FileNotFoundError: When the parent directory doesn't exist.

    Notes:
        Other ``OSError`` subclasses from the underlying I/O calls
        propagate; in every failure branch the temp file is cleaned up
        before re-raising.

    """
    if not overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    tmp_path = Path(tmp_name)
    try:
        Path(tmp_path).chmod(mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        Path(tmp_path).replace(path)
    except BaseException:
        # Best-effort cleanup — if the rename happened before the
        # failure, tmp_path is already gone and unlink raises FileNotFoundError.
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
    overwrite: bool = True,
    fsync: bool = True,
) -> None:
    """Text convenience wrapper over :func:`atomic_write_bytes`."""
    atomic_write_bytes(
        path=path,
        data=content.encode(encoding),
        mode=mode,
        overwrite=overwrite,
        fsync=fsync,
    )
