"""Small cross-platform inter-process locks for operational file updates."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class LockUnavailable(RuntimeError):
    """Raised when another process still owns a requested lock."""


def _try_lock(handle) -> bool:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        return False
    return True


def _unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def interprocess_lock(
    path: Path,
    *,
    timeout_seconds: float = 0.0,
    poll_seconds: float = 0.25,
    description: str = "resource",
) -> Iterator[None]:
    """Hold an OS-backed exclusive lock without treating file existence as busy.

    The small lock file intentionally remains on disk. Ownership is maintained
    by the operating system and is released automatically if a process exits or
    crashes, avoiding the stale-lock problem of ``O_EXCL`` marker files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    # Windows byte-range locks require the byte being locked to exist.
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()

    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    acquired = False
    announced_wait = False
    try:
        while True:
            acquired = _try_lock(handle)
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise LockUnavailable(
                    f"Another process is using {description}: {path}"
                )
            if not announced_wait:
                print(f"Waiting for {description}: {path}", flush=True)
                announced_wait = True
            time.sleep(max(poll_seconds, 0.01))
        yield
    finally:
        if acquired:
            _unlock(handle)
        handle.close()


def cache_lock_path(data_path: Path) -> Path:
    """Return the persistent lock-file name associated with one cache file."""
    return data_path.with_name(f".{data_path.name}.lock")
