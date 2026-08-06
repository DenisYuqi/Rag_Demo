"""Exclusive ownership for the embedded single-writer data root."""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock, Timeout


class DataRootWriterLockError(RuntimeError):
    """Raised when another process already owns the embedded data root."""

    code = "data_root_writer_locked"

    def __init__(self) -> None:
        super().__init__(self.code)


class DataRootWriterLock:
    """Hold one non-blocking OS-backed lock for an application's lifetime.

    The lock file is intentionally retained after release. Removing a lock file can
    create two independently lockable filesystem objects during a hand-off race.
    Ownership is represented by the operating-system lock, not file existence.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        # App factories and ASGI lifespan hooks may execute on different threads
        # (Starlette's TestClient does this too), so ownership cannot be thread-local.
        self._lock = FileLock(str(self.path), thread_local=False)

    @property
    def acquired(self) -> bool:
        return self._lock.is_locked

    def acquire(self) -> None:
        """Acquire immediately or reject the competing writer."""

        if self.acquired:
            return
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except Timeout:
            raise DataRootWriterLockError from None

    def release(self) -> None:
        """Release ownership idempotently."""

        if self.acquired:
            self._lock.release()

    def __enter__(self) -> DataRootWriterLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()
