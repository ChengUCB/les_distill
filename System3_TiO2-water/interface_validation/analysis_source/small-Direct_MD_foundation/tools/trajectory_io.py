"""Helpers for archived trajectory files.

Text XYZ trajectories may stay gzip-compressed and are resolved transparently.
ASE's binary ``.traj`` reader is not gzip-safe, so ``materialize_traj`` expands
those archives into a temporary file for the duration of a context manager.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


def resolve_compressed(path: str | Path) -> Path:
    """Return an existing path, falling back from ``name`` to ``name.gz``."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    compressed = Path(f"{candidate}.gz")
    return compressed if compressed.exists() else candidate


def open_text(path: str | Path) -> TextIO:
    """Open plain or gzip-compressed text for reading."""
    resolved = resolve_compressed(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt")
    return resolved.open("rt")


def open_binary(path: str | Path) -> BinaryIO:
    """Open plain or gzip-compressed bytes for reading."""
    resolved = resolve_compressed(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rb")
    return resolved.open("rb")


def logical_path(path: str | Path) -> Path:
    """Return the uncompressed logical name used by provenance metadata."""
    resolved = resolve_compressed(path)
    return resolved.with_suffix("") if resolved.suffix == ".gz" else resolved


def uncompressed_size(path: str | Path) -> int:
    """Return payload size without expanding a gzip archive.

    The gzip ISIZE field is modulo 2**32. Every archive in this workspace is
    below that limit, so it exactly recovers the original file size.
    """
    resolved = resolve_compressed(path)
    if resolved.suffix != ".gz":
        return resolved.stat().st_size
    with resolved.open("rb") as handle:
        handle.seek(-4, 2)
        return int.from_bytes(handle.read(4), "little")


def source_identity(path: str | Path) -> tuple[str, int, int]:
    """Return logical path, uncompressed size, and preserved mtime."""
    resolved = resolve_compressed(path)
    return (
        str(logical_path(resolved).resolve()),
        uncompressed_size(resolved),
        resolved.stat().st_mtime_ns,
    )


@contextmanager
def materialize_traj(path: str | Path) -> Iterator[Path]:
    """Yield a plain ``.traj`` path, temporarily expanding ``.traj.gz``."""
    resolved = resolve_compressed(path)
    if resolved.suffix != ".gz":
        yield resolved
        return
    if not resolved.name.endswith(".traj.gz"):
        raise ValueError(f"expected a .traj or .traj.gz file: {resolved}")
    with tempfile.TemporaryDirectory(prefix="ase-traj-") as directory:
        target = Path(directory) / resolved.name[:-3]
        with gzip.open(resolved, "rb") as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        yield target
